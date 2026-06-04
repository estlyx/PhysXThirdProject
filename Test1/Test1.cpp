#include <vector>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <ctime>

#include <PxPhysicsAPI.h>
#include "snippetrender/SnippetRender.h"
#include "snippetrender/SnippetCamera.h"

// NvCloth SDK headers
// lib/include/NvCloth/  <-- Factory.h Cloth.h Fabric.h Solver.h Callbacks.h Allocator.h Range.h PhaseConfig.h ps/
// lib/bin/Debug/        <-- NvCloth_x64.lib  NvCloth_x64.dll
#include <NvCloth/Callbacks.h>
#include <NvCloth/Factory.h>
#include <NvCloth/Fabric.h>
#include <NvCloth/Cloth.h>
#include <NvCloth/Solver.h>

using namespace physx;

// =========================================================
// Lab 3: Flag Banner – NvCloth Cloth Simulation
//
// Controls:
//   W / S / A / D  -- free-fly camera (Snippets default)
//   ESC            -- quit
// =========================================================

// --- PhysX foundation (allocator / error only) -----------
static PxDefaultAllocator     gAlloc;
static PxDefaultErrorCallback gErr;
static PxFoundation* gFoundation = nullptr;

// --- NvCloth objects -------------------------------------
static nv::cloth::Factory* gClothFactory = nullptr;
static nv::cloth::Solver* gClothSolver = nullptr;
static nv::cloth::Fabric* gClothFabric = nullptr;
static nv::cloth::Cloth* gCloth = nullptr;

// --- Flag mesh dimensions --------------------------------
static const int   FLAG_COLS = 10;    // vertices along width
static const int   FLAG_ROWS = 8;     // vertices along height  (80 total >= 50)
static const float FLAG_W = 3.0f;  // width  (m)
static const float FLAG_H = 2.0f;  // height (m)
static const PxVec3 FLAG_TL(0.0f, 4.0f, 0.0f); // top-left corner in world space

static const int FLAG_VERTS = FLAG_COLS * FLAG_ROWS;
static const int FLAG_TRIS = (FLAG_COLS - 1) * (FLAG_ROWS - 1) * 2;

// NvCloth's AVX solver reads particle positions in 8-float batches.
// Providing 8 extra dummy particles after the real ones prevents reading past the buffer end.
static const int FLAG_VERTS_PADDED = FLAG_VERTS + 8;

// Two-colour stripe split
static const int ROWS_HALF = FLAG_ROWS / 2;
static const int TRIS_A = ROWS_HALF * (FLAG_COLS - 1) * 2;
static const int TRIS_B = (FLAG_ROWS - 1 - ROWS_HALF) * (FLAG_COLS - 1) * 2;

// Index buffers (built once at startup)
static PxU32 gTriIdx[FLAG_TRIS * 3]; // full mesh  (normals + wind triangles)
static PxU32 gTriIdxA[TRIS_A * 3];  // upper stripe (red)
static PxU32 gTriIdxB[TRIS_B * 3];  // lower stripe (white)

// Per-vertex normals recomputed every frame
static PxVec4 gNormals[FLAG_VERTS];

// --- Wind ------------------------------------------------
static float gWindTime = 0.0f;
static float gWindStrength = 0.0f;
static float gWindAngle = 0.0f;

// --- Camera ----------------------------------------------
static Snippets::Camera* gCamera = nullptr;

// --- Colours ---------------------------------------------
static const PxVec3 COL_GROUND(0.26f, 0.52f, 0.20f);
static const PxVec3 COL_POLE(0.62f, 0.50f, 0.30f);
static const PxVec3 COL_FLAG_A(0.85f, 0.12f, 0.12f); // red   (upper half)
static const PxVec3 COL_FLAG_B(0.94f, 0.94f, 0.90f); // white (lower half)

// =========================================================
// Helpers
// =========================================================

static inline int vidx(int c, int r) { return r * FLAG_COLS + c; }

static void buildIndexBuffers()
{
    int nAll = 0, nA = 0, nB = 0;
    for (int r = 0; r < FLAG_ROWS - 1; ++r)
        for (int c = 0; c < FLAG_COLS - 1; ++c)
        {
            PxU32 tl = (PxU32)vidx(c, r);
            PxU32 tr = (PxU32)vidx(c + 1, r);
            PxU32 bl = (PxU32)vidx(c, r + 1);
            PxU32 br = (PxU32)vidx(c + 1, r + 1);

            PxU32 t0[3] = { tl, bl, tr };
            PxU32 t1[3] = { bl, br, tr };

            for (int k = 0; k < 3; ++k) gTriIdx[nAll++] = t0[k];
            for (int k = 0; k < 3; ++k) gTriIdx[nAll++] = t1[k];

            if (r < ROWS_HALF)
            {
                for (int k = 0; k < 3; ++k) gTriIdxA[nA++] = t0[k];
                for (int k = 0; k < 3; ++k) gTriIdxA[nA++] = t1[k];
            }
            else
            {
                for (int k = 0; k < 3; ++k) gTriIdxB[nB++] = t0[k];
                for (int k = 0; k < 3; ++k) gTriIdxB[nB++] = t1[k];
            }
        }
}

static void computeNormals(const PxVec4* pts)
{
    for (int i = 0; i < FLAG_VERTS; ++i)
        gNormals[i] = PxVec4(0.0f, 0.0f, 0.0f, 0.0f);

    for (int t = 0; t < FLAG_TRIS; ++t)
    {
        PxU32 i0 = gTriIdx[t * 3 + 0];
        PxU32 i1 = gTriIdx[t * 3 + 1];
        PxU32 i2 = gTriIdx[t * 3 + 2];
        PxVec3 e0(pts[i1].x - pts[i0].x, pts[i1].y - pts[i0].y, pts[i1].z - pts[i0].z);
        PxVec3 e1(pts[i2].x - pts[i0].x, pts[i2].y - pts[i0].y, pts[i2].z - pts[i0].z);
        PxVec3 n = e0.cross(e1);
        PxVec4 nv(n.x, n.y, n.z, 0.0f);
        gNormals[i0] += nv;
        gNormals[i1] += nv;
        gNormals[i2] += nv;
    }

    for (int i = 0; i < FLAG_VERTS; ++i)
    {
        PxVec3 n(gNormals[i].x, gNormals[i].y, gNormals[i].z);
        n = n.getNormalized();
        gNormals[i] = PxVec4(n.x, n.y, n.z, 0.0f);
    }
}

// =========================================================
// Fabric construction (replaces NvClothCookFabricFromMesh)
//
// NvCloth fabric layout (see Factory::createFabric):
//   phases[]    -- map phase -> set index (0, 1, 2, ...), NOT constraint type
//   sets[]      -- inclusive prefix sum of rest-value count per set
//   restVals[]  -- rest length per constraint
//   indices[]   -- two particle indices per constraint
// =========================================================

// Constraints are split into graph-coloured sets so that within each set
// no two constraints share a particle. This is required by NvCloth's AVX
// solver which processes constraints in each set in parallel.
// Each set is also padded to a multiple of 8 (AVX batch size).
// Padding uses the two pinned corners (invMass=0 on both → zero displacement).
static void buildConstraints(
    const std::vector<PxVec3>& pos,
    std::vector<PxU32>& phases,   // phase -> set index
    std::vector<PxU32>& sets,     // cumulative end index per set
    std::vector<float>& restVals,
    std::vector<PxU32>& indices)
{
    const float restPad = (pos[FLAG_COLS - 1] - pos[0]).magnitude();

    auto add = [&](int i0, int i1)
        {
            indices.push_back((PxU32)i0);
            indices.push_back((PxU32)i1);
            restVals.push_back((pos[i1] - pos[i0]).magnitude());
        };

    // Finish current set: pad to multiple of 8, record cumulative end.
    auto finishSet = [&](size_t setStart)
        {
            while ((restVals.size() - setStart) % 8 != 0)
            {
                indices.push_back(0);
                indices.push_back((PxU32)(FLAG_COLS - 1)); // both pinned → Δpos = 0
                restVals.push_back(restPad);
            }
            phases.push_back((PxU32)sets.size()); // phase index -> set index
            sets.push_back((PxU32)restVals.size());
        };

    size_t s;

    // --- Horizontal stretch  (eHORIZONTAL = 2) ---
    // 2-colour: even-c set (40) and odd-c set (32)
    s = restVals.size();
    for (int r = 0; r < FLAG_ROWS; ++r)
        for (int c = 0; c < FLAG_COLS - 1; c += 2)
            add(vidx(c, r), vidx(c + 1, r));
    finishSet(s);

    s = restVals.size();
    for (int r = 0; r < FLAG_ROWS; ++r)
        for (int c = 1; c < FLAG_COLS - 1; c += 2)
            add(vidx(c, r), vidx(c + 1, r));
    finishSet(s);

    // --- Vertical stretch ---
    s = restVals.size();
    for (int c = 0; c < FLAG_COLS; ++c)
        for (int r = 0; r < FLAG_ROWS - 1; r += 2)
            add(vidx(c, r), vidx(c, r + 1));
    finishSet(s);

    s = restVals.size();
    for (int c = 0; c < FLAG_COLS; ++c)
        for (int r = 1; r < FLAG_ROWS - 1; r += 2)
            add(vidx(c, r), vidx(c, r + 1));
    finishSet(s);

    // --- Shear diagonal 1  (c,r)→(c+1,r+1) ---
    s = restVals.size();
    for (int r = 0; r < FLAG_ROWS - 1; ++r)
        for (int c = 0; c < FLAG_COLS - 1; c += 2)
            add(vidx(c, r), vidx(c + 1, r + 1));
    finishSet(s);

    s = restVals.size();
    for (int r = 0; r < FLAG_ROWS - 1; ++r)
        for (int c = 1; c < FLAG_COLS - 1; c += 2)
            add(vidx(c, r), vidx(c + 1, r + 1));
    finishSet(s);

    // --- Shear diagonal 2  (c+1,r)→(c,r+1) ---
    s = restVals.size();
    for (int r = 0; r < FLAG_ROWS - 1; ++r)
        for (int c = 0; c < FLAG_COLS - 1; c += 2)
            add(vidx(c + 1, r), vidx(c, r + 1));
    finishSet(s);

    s = restVals.size();
    for (int r = 0; r < FLAG_ROWS - 1; ++r)
        for (int c = 1; c < FLAG_COLS - 1; c += 2)
            add(vidx(c + 1, r), vidx(c, r + 1));
    finishSet(s);

    // --- Bending horizontal  (c,r)→(c+2,r) ---
    for (int col = 0; col < 3; ++col)
    {
        s = restVals.size();
        for (int r = 0; r < FLAG_ROWS; ++r)
            for (int c = col; c < FLAG_COLS - 2; c += 3)
                add(vidx(c, r), vidx(c + 2, r));
        finishSet(s);
    }

    // --- Bending vertical  (c,r)→(c,r+2) ---
    for (int col = 0; col < 3; ++col)
    {
        s = restVals.size();
        for (int c = 0; c < FLAG_COLS; ++c)
            for (int r = col; r < FLAG_ROWS - 2; r += 3)
                add(vidx(c, r), vidx(c, r + 2));
        finishSet(s);
    }
    // Total: 14 sets, ~424 constraints
}

// =========================================================
// Cloth setup
// =========================================================

static void buildCloth()
{
    buildIndexBuffers();

    // Rest-pose positions: flat vertical grid, top edge at FLAG_TL
    std::vector<PxVec3> restPos(FLAG_VERTS);
    const float dw = FLAG_W / (FLAG_COLS - 1);
    const float dh = FLAG_H / (FLAG_ROWS - 1);
    for (int r = 0; r < FLAG_ROWS; ++r)
        for (int c = 0; c < FLAG_COLS; ++c)
            restPos[vidx(c, r)] = PxVec3(FLAG_TL.x + c * dw,
                FLAG_TL.y - r * dh,
                FLAG_TL.z);

    // Build constraint data (NvCloth copies it internally)
    std::vector<PxU32>  phases, sets, indices;
    std::vector<float>  restVals;
    buildConstraints(restPos, phases, sets, restVals, indices);

    // Create fabric directly (no NvClothExt cooker needed)
    gClothFabric = gClothFactory->createFabric(
        (PxU32)FLAG_VERTS_PADDED,
        nv::cloth::Range<const PxU32>(phases.data(), phases.data() + phases.size()),
        nv::cloth::Range<const PxU32>(sets.data(), sets.data() + sets.size()),
        nv::cloth::Range<const float>(restVals.data(), restVals.data() + restVals.size()),
        nv::cloth::Range<const float>(),  // no per-constraint stiffness → use PhaseConfig
        nv::cloth::Range<const PxU32>(indices.data(), indices.data() + indices.size()),
        nv::cloth::Range<const PxU32>(),  // no tethers
        nv::cloth::Range<const float>(),  // no tether lengths
        nv::cloth::Range<const PxU32>(gTriIdx, gTriIdx + FLAG_TRIS * 3));

    // Initial particles: xyz = rest position, w = inverse mass (0 = pinned)
    // Extra FLAG_VERTS_PADDED - FLAG_VERTS dummy particles at origin pad the
    // NvCloth position buffer so AVX 8-float batch reads never go past the end.
    std::vector<PxVec4> initPts(FLAG_VERTS_PADDED, PxVec4(0.0f, 0.0f, 0.0f, 0.0f));
    for (int r = 0; r < FLAG_ROWS; ++r)
        for (int c = 0; c < FLAG_COLS; ++c)
        {
            const PxVec3& p = restPos[vidx(c, r)];
            float invMass = 1.0f / 78.0f; // ~1 kg total cloth mass
            if (r == 0 && (c == 0 || c == FLAG_COLS - 1))
                invMass = 0.0f; // pin top-left and top-right corners (banner)
            initPts[vidx(c, r)] = PxVec4(p.x, p.y, p.z, invMass);
        }

    gCloth = gClothFactory->createCloth(
        nv::cloth::Range<PxVec4>(initPts.data(), initPts.data() + FLAG_VERTS_PADDED),
        *gClothFabric);

    gCloth->setGravity(PxVec3(0.0f, -9.81f, 0.0f));
    gCloth->setDamping(PxVec3(0.05f, 0.05f, 0.05f));
    gCloth->setFriction(0.25f);
    gCloth->setSolverFrequency(120.0f);

    const PxU32 numPhases = gClothFabric->getNumPhases();
    std::vector<nv::cloth::PhaseConfig> phaseCfg(numPhases);
    for (PxU32 i = 0; i < numPhases; ++i)
    {
        phaseCfg[i].mPhaseIndex = (uint16_t)i;
        phaseCfg[i].mStiffness = 0.8f;
        phaseCfg[i].mStiffnessMultiplier = 1.0f;
        phaseCfg[i].mCompressionLimit = 1.0f;
        phaseCfg[i].mStretchLimit = 1.05f;
    }
    gCloth->setPhaseConfig(nv::cloth::Range<nv::cloth::PhaseConfig>(
        phaseCfg.data(), phaseCfg.data() + numPhases));

    // Ground plane y = 0 keeps the flag from falling through the floor
    {
        const PxVec4 groundPlane(0.0f, 1.0f, 0.0f, 0.0f);
        gCloth->setPlanes(nv::cloth::Range<const PxVec4>(&groundPlane, &groundPlane + 1), 0, 1);
        const PxU32 groundMask = 1u;
        gCloth->setConvexes(nv::cloth::Range<const PxU32>(&groundMask, &groundMask + 1), 0, 1);
    }

    gClothSolver->addCloth(gCloth);

    // Diagnostic: print initial positions of pinned corners and a free particle
    {
        auto pts = nv::cloth::readCurrentParticles(*gCloth);
        const int topRight = vidx(FLAG_COLS - 1, 0);
        std::printf("[INIT] numParticles=%u  numPhases=%u  numSets=%u\n",
            gCloth->getNumParticles(), gClothFabric->getNumPhases(), gClothFabric->getNumSets());
        std::printf("[INIT] particle[0]  = (%.3f, %.3f, %.3f)  w=%.3f (top-left)\n",
            pts[0].x, pts[0].y, pts[0].z, pts[0].w);
        std::printf("[INIT] particle[%d] = (%.3f, %.3f, %.3f)  w=%.3f (top-right)\n",
            topRight, pts[topRight].x, pts[topRight].y, pts[topRight].z, pts[topRight].w);
        std::printf("[INIT] particle[5]  = (%.3f, %.3f, %.3f)  w=%.3f\n",
            pts[5].x, pts[5].y, pts[5].z, pts[5].w);
    }
}

// =========================================================
// Wind
// =========================================================

static void updateWind(float dt)
{
    gWindTime += dt;
    gWindAngle = gWindTime * 0.15f;
    gWindStrength = 1.0f + 0.5f * PxSin(gWindTime * 0.4f);

    PxVec3 dir(PxCos(gWindAngle),
        0.02f * PxSin(gWindTime * 0.3f),
        PxSin(gWindAngle));
    dir = dir.getNormalized();

    gCloth->setWindVelocity(dir * gWindStrength);
    gCloth->setDragCoefficient(0.015f);
    gCloth->setLiftCoefficient(0.008f);
}

// =========================================================
// Physics init
// =========================================================

void initPhysics()
{
    gFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, gAlloc, gErr);

    nv::cloth::InitializeNvCloth(&gAlloc, &gErr, nullptr, nullptr);
    gClothFactory = NvClothCreateFactoryCPU();
    gClothSolver = gClothFactory->createSolver();

    buildCloth();

    std::printf("=== NvCloth Flag Banner -- Lab Work #3 ===\n");
    std::printf("  Mesh  : %d x %d = %d vertices  (%d triangles)\n",
        FLAG_COLS, FLAG_ROWS, FLAG_VERTS, FLAG_TRIS);
    std::printf("  Pinned: top-left + top-right corners (banner)\n");
    std::printf("  Wind direction and strength change over time.\n\n");
}

// =========================================================
// Input
// =========================================================

static void customKeyboard(unsigned char key, int x, int y)
{
    switch (key)
    {
    case 27: exit(0); break; // ESC
    default:
        gCamera->handleKey(key, x, y);
        break;
    }
}

void keyPress(unsigned char /*key*/, const PxTransform& /*cam*/) {}

// =========================================================
// Render
// =========================================================

void renderCallback()
{
    const float dt = 1.0f / 60.0f;

    updateWind(dt);

    gClothSolver->beginSimulation(dt);
    for (int i = 0; i < gClothSolver->getSimulationChunkCount(); ++i)
        gClothSolver->simulateChunk(i);
    gClothSolver->endSimulation();

    Snippets::startRender(gCamera);

    // Ground quad
    {
        static const PxVec3 gv[4] = {
            {-9.0f, 0.0f, -6.0f}, { 9.0f, 0.0f, -6.0f},
            { 9.0f, 0.0f,  6.0f}, {-9.0f, 0.0f,  6.0f} };
        static const PxU32  gi[6] = { 0, 2, 1,  0, 3, 2 };
        static const PxVec3 gn[4] = {
            {0,1,0},{0,1,0},{0,1,0},{0,1,0} };
        Snippets::renderMesh(4, gv, 2, gi, COL_GROUND, gn);
    }

    // Flagpole: vertical post + horizontal crossbar
    {
        Snippets::DrawLine(
            PxVec3(FLAG_TL.x, 0.0f, FLAG_TL.z),
            PxVec3(FLAG_TL.x, FLAG_TL.y + 0.25f, FLAG_TL.z), COL_POLE);
        Snippets::DrawLine(
            PxVec3(FLAG_TL.x, FLAG_TL.y, FLAG_TL.z),
            PxVec3(FLAG_TL.x + FLAG_W, FLAG_TL.y, FLAG_TL.z), COL_POLE);
    }

    {
        auto pts = nv::cloth::readCurrentParticles(*gCloth);

        static int dbgFrame = 0;
        if (++dbgFrame <= 10 || dbgFrame % 60 == 0)
        {
            const PxVec4& p = pts[5];
            std::printf("[SIM] frame %3d  p[5]=(%.3f,%.3f,%.3f)\n",
                dbgFrame, p.x, p.y, p.z);
            if (!PxIsFinite(p.x) || !PxIsFinite(p.y) || !PxIsFinite(p.z))
                std::printf("[SIM] ERROR: NaN detected at frame %d\n", dbgFrame);
        }

        computeNormals(pts.begin());

        Snippets::renderMesh(FLAG_VERTS, pts.begin(), TRIS_A, gTriIdxA, COL_FLAG_A, gNormals);
        Snippets::renderMesh(FLAG_VERTS, pts.begin(), TRIS_B, gTriIdxB, COL_FLAG_B, gNormals);
    }

    // HUD
    {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "Wind: %.1f m/s   dir: %.0f deg   |   [W/S/A/D] camera   [ESC] quit",
            gWindStrength, gWindAngle * 180.0f / PxPi);
        Snippets::print(buf);
    }

    Snippets::finishRender();
}

// =========================================================
// Cleanup
// =========================================================

void exitCallback()
{
    delete gCamera; gCamera = nullptr;

    if (gCloth && gClothSolver) gClothSolver->removeCloth(gCloth);
    if (gClothFactory)
    {
        NvClothDestroyFactory(gClothFactory);
        gClothFactory = nullptr;
    }
    gCloth = nullptr; gClothFabric = nullptr; gClothSolver = nullptr;

    if (gFoundation) { gFoundation->release(); gFoundation = nullptr; }
}

// =========================================================
// Entrypoint
// =========================================================

int main()
{
    PxVec3 flagCenter(FLAG_TL.x + FLAG_W * 0.5f,
        FLAG_TL.y - FLAG_H * 0.5f,
        FLAG_TL.z);
    PxVec3 eye = flagCenter + PxVec3(0.0f, 0.5f, 6.5f);
    gCamera = new Snippets::Camera(eye, (flagCenter - eye).getNormalized());

    Snippets::setupDefault("NvCloth Flag Banner",
        gCamera, keyPress, renderCallback, exitCallback);

    glutKeyboardFunc(customKeyboard);

    initPhysics();
    glutMainLoop();
    return 0;
}
