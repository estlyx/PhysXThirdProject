#include <vector>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <ctime>

#include <PxPhysicsAPI.h>
#include "snippetrender/SnippetRender.h"
#include "snippetrender/SnippetCamera.h"

#include <NvCloth/Callbacks.h>
#include <NvCloth/Factory.h>
#include <NvCloth/Fabric.h>
#include <NvCloth/Cloth.h>
#include <NvCloth/Solver.h>

using namespace physx;

// =========================================================
// Lab 3: Flag Banner – NvCloth Cloth Simulation
//
// Flag shape: triangle ABC
//   A = bottom-left (pinned)   B = top-left (pinned)   C = top-right (free)
//   D = bottom-right is excluded (diagonal A→C cuts the rectangle)
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
static nv::cloth::Solver*  gClothSolver  = nullptr;
static nv::cloth::Fabric*  gClothFabric  = nullptr;
static nv::cloth::Cloth*   gCloth        = nullptr;

// --- Flag mesh dimensions --------------------------------
static const int   FLAG_COLS = 10;   // vertices along width
static const int   FLAG_ROWS = 8;    // vertices along height
static const float FLAG_W    = 3.0f; // width  (m)
static const float FLAG_H    = 2.0f; // height (m)
static const PxVec3 FLAG_TL(0.0f, 4.0f, 0.0f); // top-left corner in world space

static const int FLAG_GRID_VERTS = FLAG_COLS * FLAG_ROWS; // 80, grid size
static const int FLAG_MAX_TRIS   = (FLAG_COLS - 1) * (FLAG_ROWS - 1) * 2; // 126, upper bound

static inline int vidx(int c, int r) { return r * FLAG_COLS + c; }

// A vertex at grid position (c, r) is inside triangle ABC when it lies on or
// above the diagonal from A (bottom-left) to C (top-right):
//   c / (FLAG_COLS-1) + r / (FLAG_ROWS-1) <= 1
// Using integer arithmetic: c*(FLAG_ROWS-1) + r*(FLAG_COLS-1) <= (FLAG_COLS-1)*(FLAG_ROWS-1)
static inline bool vertexInFlag(int c, int r)
{
    return c * (FLAG_ROWS - 1) + r * (FLAG_COLS - 1)
           <= (FLAG_COLS - 1) * (FLAG_ROWS - 1);
}

// Compact mapping: grid index -> cloth particle index (-1 if outside triangle)
static int gVMap[FLAG_GRID_VERTS];
static int gClothVerts       = 0; // number of active particles
static int gClothVertsPadded = 0; // gClothVerts + 8  (AVX read-overrun padding)

// Pinned corners (cloth-index space, filled by buildVertexMap)
static int PIN_A = -1; // A = bottom-left
static int PIN_B = -1; // B = top-left

static inline int cidx(int c, int r) { return gVMap[vidx(c, r)]; }

// Fills gVMap, gClothVerts, gClothVertsPadded, PIN_A, PIN_B.
// Vertices are numbered in row-major order over the grid, skipping outside ones.
static void buildVertexMap()
{
    int n = 0;
    for (int r = 0; r < FLAG_ROWS; ++r)
        for (int c = 0; c < FLAG_COLS; ++c)
            gVMap[vidx(c, r)] = vertexInFlag(c, r) ? n++ : -1;

    gClothVerts       = n;
    gClothVertsPadded = gClothVerts + 8;
    PIN_B = gVMap[vidx(0, 0)];            // B: top-left
    PIN_A = gVMap[vidx(0, FLAG_ROWS - 1)]; // A: bottom-left
}

// --- Index buffer (built once at startup) ----------------
static PxU32 gTriIdx[FLAG_MAX_TRIS * 3];
static int   gTriCount = 0;

// --- Per-vertex normals (dynamic size, recomputed every frame) ---
static std::vector<PxVec4> gNormals;

// --- Wind ------------------------------------------------
static float gWindTime     = 0.0f;
static float gWindStrength = 0.0f;
static float gWindAngle    = 0.0f;

// --- Camera ----------------------------------------------
static Snippets::Camera* gCamera = nullptr;

// --- Colours ---------------------------------------------
static const PxVec3 COL_GROUND(0.26f, 0.52f, 0.20f);
static const PxVec3 COL_POLE  (0.62f, 0.50f, 0.30f);
static const PxVec3 COL_FLAG  (0.94f, 0.94f, 0.90f);

// =========================================================
// Helpers
// =========================================================

// Generates triangles only for cells where at least 3 corners are inside the
// triangle ABC. Cells split by the diagonal A→C contribute one triangle each;
// fully interior cells contribute two.
static void buildIndexBuffers()
{
    int n = 0;
    for (int r = 0; r < FLAG_ROWS - 1; ++r)
        for (int c = 0; c < FLAG_COLS - 1; ++c)
        {
            const bool tl = vertexInFlag(c,     r    );
            const bool tr = vertexInFlag(c + 1, r    );
            const bool bl = vertexInFlag(c,     r + 1);
            const bool br = vertexInFlag(c + 1, r + 1);

            if (tl && tr && bl && br)
            {
                // Full cell: two triangles
                gTriIdx[n++] = (PxU32)cidx(c,     r    );
                gTriIdx[n++] = (PxU32)cidx(c,     r + 1);
                gTriIdx[n++] = (PxU32)cidx(c + 1, r    );

                gTriIdx[n++] = (PxU32)cidx(c,     r + 1);
                gTriIdx[n++] = (PxU32)cidx(c + 1, r + 1);
                gTriIdx[n++] = (PxU32)cidx(c + 1, r    );
            }
            else if (tl && tr && bl)
            {
                // Diagonal cuts off br: one triangle (upper-left half)
                gTriIdx[n++] = (PxU32)cidx(c,     r    );
                gTriIdx[n++] = (PxU32)cidx(c,     r + 1);
                gTriIdx[n++] = (PxU32)cidx(c + 1, r    );
            }
            // Cells with fewer than 3 vertices inside the flag are skipped
        }
    gTriCount = n / 3;
}

static void computeNormals(const PxVec4* pts, int vertCount)
{
    gNormals.assign((size_t)vertCount, PxVec4(0.0f));

    for (int t = 0; t < gTriCount; ++t)
    {
        const PxU32 i0 = gTriIdx[t * 3 + 0];
        const PxU32 i1 = gTriIdx[t * 3 + 1];
        const PxU32 i2 = gTriIdx[t * 3 + 2];
        if (i0 >= (PxU32)vertCount || i1 >= (PxU32)vertCount || i2 >= (PxU32)vertCount)
            continue;
        PxVec3 e0(pts[i1].x - pts[i0].x, pts[i1].y - pts[i0].y, pts[i1].z - pts[i0].z);
        PxVec3 e1(pts[i2].x - pts[i0].x, pts[i2].y - pts[i0].y, pts[i2].z - pts[i0].z);
        PxVec3 n = e0.cross(e1);
        PxVec4 nv(n.x, n.y, n.z, 0.0f);
        gNormals[i0] += nv;
        gNormals[i1] += nv;
        gNormals[i2] += nv;
    }

    for (int i = 0; i < vertCount; ++i)
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
    std::vector<float>&  restVals,
    std::vector<PxU32>& indices)
{
    const float restPad = (pos[PIN_A] - pos[PIN_B]).magnitude();

    // Add a constraint between grid cells (c0,r0) and (c1,r1).
    // Skipped automatically when either vertex falls outside triangle ABC.
    auto add = [&](int c0, int r0, int c1, int r1)
    {
        if (!vertexInFlag(c0, r0) || !vertexInFlag(c1, r1))
            return;
        const int i0 = cidx(c0, r0);
        const int i1 = cidx(c1, r1);
        indices.push_back((PxU32)i0);
        indices.push_back((PxU32)i1);
        restVals.push_back((pos[i1] - pos[i0]).magnitude());
    };

    // Finish current set: pad to multiple of 8, record cumulative end.
    auto finishSet = [&](size_t setStart)
    {
        while ((restVals.size() - setStart) % 8 != 0)
        {
            indices.push_back((PxU32)PIN_B);
            indices.push_back((PxU32)PIN_A); // both pinned → Δpos = 0
            restVals.push_back(restPad);
        }
        phases.push_back((PxU32)sets.size()); // phase index -> set index
        sets.push_back((PxU32)restVals.size());
    };

    size_t s;

    // --- Horizontal stretch ---
    // 2-colour: even-c and odd-c sets
    s = restVals.size();
    for (int r = 0; r < FLAG_ROWS; ++r)
        for (int c = 0; c < FLAG_COLS - 1; c += 2)
            add(c, r, c + 1, r);
    finishSet(s);

    s = restVals.size();
    for (int r = 0; r < FLAG_ROWS; ++r)
        for (int c = 1; c < FLAG_COLS - 1; c += 2)
            add(c, r, c + 1, r);
    finishSet(s);

    // --- Vertical stretch ---
    s = restVals.size();
    for (int c = 0; c < FLAG_COLS; ++c)
        for (int r = 0; r < FLAG_ROWS - 1; r += 2)
            add(c, r, c, r + 1);
    finishSet(s);

    s = restVals.size();
    for (int c = 0; c < FLAG_COLS; ++c)
        for (int r = 1; r < FLAG_ROWS - 1; r += 2)
            add(c, r, c, r + 1);
    finishSet(s);

    // --- Shear diagonal 1  (c,r)→(c+1,r+1) ---
    s = restVals.size();
    for (int r = 0; r < FLAG_ROWS - 1; ++r)
        for (int c = 0; c < FLAG_COLS - 1; c += 2)
            add(c, r, c + 1, r + 1);
    finishSet(s);

    s = restVals.size();
    for (int r = 0; r < FLAG_ROWS - 1; ++r)
        for (int c = 1; c < FLAG_COLS - 1; c += 2)
            add(c, r, c + 1, r + 1);
    finishSet(s);

    // --- Shear diagonal 2  (c+1,r)→(c,r+1) ---
    s = restVals.size();
    for (int r = 0; r < FLAG_ROWS - 1; ++r)
        for (int c = 0; c < FLAG_COLS - 1; c += 2)
            add(c + 1, r, c, r + 1);
    finishSet(s);

    s = restVals.size();
    for (int r = 0; r < FLAG_ROWS - 1; ++r)
        for (int c = 1; c < FLAG_COLS - 1; c += 2)
            add(c + 1, r, c, r + 1);
    finishSet(s);

    // --- Bending horizontal  (c,r)→(c+2,r) ---
    for (int col = 0; col < 3; ++col)
    {
        s = restVals.size();
        for (int r = 0; r < FLAG_ROWS; ++r)
            for (int c = col; c < FLAG_COLS - 2; c += 3)
                add(c, r, c + 2, r);
        finishSet(s);
    }

    // --- Bending vertical  (c,r)→(c,r+2) ---
    for (int col = 0; col < 3; ++col)
    {
        s = restVals.size();
        for (int c = 0; c < FLAG_COLS; ++c)
            for (int r = col; r < FLAG_ROWS - 2; r += 3)
                add(c, r, c, r + 2);
        finishSet(s);
    }
    // Total: 14 sets
}

// =========================================================
// Cloth setup
// =========================================================

static void buildCloth()
{
    buildVertexMap();
    buildIndexBuffers();

    // Rest-pose: flat vertical grid, only triangle ABC vertices
    std::vector<PxVec3> restPos(gClothVerts);
    const float dw = FLAG_W / (FLAG_COLS - 1);
    const float dh = FLAG_H / (FLAG_ROWS - 1);
    for (int r = 0; r < FLAG_ROWS; ++r)
        for (int c = 0; c < FLAG_COLS; ++c)
            if (vertexInFlag(c, r))
                restPos[cidx(c, r)] = PxVec3(FLAG_TL.x + c * dw,
                                              FLAG_TL.y - r * dh,
                                              FLAG_TL.z);

    // Build constraint data (NvCloth copies it internally)
    std::vector<PxU32> phases, sets, indices;
    std::vector<float> restVals;
    buildConstraints(restPos, phases, sets, restVals, indices);

    // Create fabric directly (no NvClothExt cooker needed)
    gClothFabric = gClothFactory->createFabric(
        (PxU32)gClothVertsPadded,
        nv::cloth::Range<const PxU32>(phases.data(),   phases.data()   + phases.size()),
        nv::cloth::Range<const PxU32>(sets.data(),     sets.data()     + sets.size()),
        nv::cloth::Range<const float> (restVals.data(), restVals.data() + restVals.size()),
        nv::cloth::Range<const float>(),  // no per-constraint stiffness → use PhaseConfig
        nv::cloth::Range<const PxU32>(indices.data(),  indices.data()  + indices.size()),
        nv::cloth::Range<const PxU32>(),  // no tethers
        nv::cloth::Range<const float>(),  // no tether lengths
        nv::cloth::Range<const PxU32>(gTriIdx, gTriIdx + gTriCount * 3));

    // Count free (unpinned) vertices to distribute mass evenly
    int freeVerts = 0;
    for (int r = 0; r < FLAG_ROWS; ++r)
        for (int c = 0; c < FLAG_COLS; ++c)
            if (vertexInFlag(c, r) && !(c == 0 && (r == 0 || r == FLAG_ROWS - 1)))
                ++freeVerts;

    // Initial particles: xyz = rest position, w = inverse mass (0 = pinned).
    // Slots [gClothVerts .. gClothVertsPadded-1] stay at zero (AVX padding).
    std::vector<PxVec4> initPts(gClothVertsPadded, PxVec4(0.0f, 0.0f, 0.0f, 0.0f));
    const float invMass = 1.0f / (float)freeVerts; // total cloth mass ≈ 1 kg
    for (int r = 0; r < FLAG_ROWS; ++r)
        for (int c = 0; c < FLAG_COLS; ++c)
            if (vertexInFlag(c, r))
            {
                const PxVec3& p = restPos[cidx(c, r)];
                float w = invMass;
                if (c == 0 && (r == 0 || r == FLAG_ROWS - 1))
                    w = 0.0f; // pin A (bottom-left) and B (top-left)
                initPts[cidx(c, r)] = PxVec4(p.x, p.y, p.z, w);
            }

    gCloth = gClothFactory->createCloth(
        nv::cloth::Range<PxVec4>(initPts.data(), initPts.data() + gClothVertsPadded),
        *gClothFabric);

    gCloth->setGravity(PxVec3(0.0f, -9.81f, 0.0f));
    gCloth->setDamping(PxVec3(0.05f, 0.05f, 0.05f));
    gCloth->setFriction(0.25f);
    gCloth->setSolverFrequency(120.0f);

    const PxU32 numPhases = gClothFabric->getNumPhases();
    std::vector<nv::cloth::PhaseConfig> phaseCfg(numPhases);
    for (PxU32 i = 0; i < numPhases; ++i)
    {
        phaseCfg[i].mPhaseIndex          = (uint16_t)i;
        phaseCfg[i].mStiffness           = 0.8f;
        phaseCfg[i].mStiffnessMultiplier = 1.0f;
        phaseCfg[i].mCompressionLimit    = 1.0f;
        phaseCfg[i].mStretchLimit        = 1.05f;
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
        std::printf("[INIT] numParticles=%u  numPhases=%u  numSets=%u\n",
            gCloth->getNumParticles(), gClothFabric->getNumPhases(), gClothFabric->getNumSets());
        std::printf("[INIT] particle[%d] = (%.3f, %.3f, %.3f)  w=%.3f (B, top-left)\n",
            PIN_B, pts[PIN_B].x, pts[PIN_B].y, pts[PIN_B].z, pts[PIN_B].w);
        std::printf("[INIT] particle[%d] = (%.3f, %.3f, %.3f)  w=%.3f (A, bottom-left)\n",
            PIN_A, pts[PIN_A].x, pts[PIN_A].y, pts[PIN_A].z, pts[PIN_A].w);
        if (gClothVerts > 5)
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
    gWindAngle    = gWindTime * 0.15f;
    gWindStrength = 200.0f + 100.0f * PxSin(gWindTime * 0.4f); // 100–300

    PxVec3 dir(PxCos(gWindAngle),
               0.02f * PxSin(gWindTime * 0.3f),
               PxSin(gWindAngle));
    dir = dir.getNormalized();

    gCloth->setWindVelocity(dir * gWindStrength);
    gCloth->setDragCoefficient(0.022f);
    gCloth->setLiftCoefficient(0.012f);
}

// =========================================================
// Physics init
// =========================================================

void initPhysics()
{
    gFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, gAlloc, gErr);

    nv::cloth::InitializeNvCloth(&gAlloc, &gErr, nullptr, nullptr);
    gClothFactory = NvClothCreateFactoryCPU();
    gClothSolver  = gClothFactory->createSolver();

    buildCloth();

    std::printf("=== NvCloth Flag Banner -- Lab Work #3 ===\n");
    std::printf("  Shape : triangle ABC (%d x %d grid → %d cloth verts, %d tris)\n",
        FLAG_COLS, FLAG_ROWS, gClothVerts, gTriCount);
    std::printf("  Pinned: left edge — A (bottom-left) + B (top-left)\n");
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

    // Flagpole: vertical post along pinned left edge
    Snippets::DrawLine(
        PxVec3(FLAG_TL.x, 0.0f,           FLAG_TL.z),
        PxVec3(FLAG_TL.x, FLAG_TL.y + 0.25f, FLAG_TL.z), COL_POLE);

    {
        auto particles = nv::cloth::readCurrentParticles(*gCloth);

        // Copy only active (triangle) vertices into a contiguous render buffer
        std::vector<PxVec4> renderPts(gClothVerts);
        const int copyCount = PxMin(gClothVerts, (int)particles.size());
        for (int i = 0; i < copyCount; ++i)
            renderPts[i] = particles[(PxU32)i];

        static int dbgFrame = 0;
        if (++dbgFrame <= 10 || dbgFrame % 60 == 0)
        {
            if (gClothVerts > 5)
            {
                const PxVec4& p = renderPts[5];
                std::printf("[SIM] frame %3d  p[5]=(%.3f,%.3f,%.3f)\n",
                    dbgFrame, p.x, p.y, p.z);
                if (!PxIsFinite(p.x) || !PxIsFinite(p.y) || !PxIsFinite(p.z))
                    std::printf("[SIM] ERROR: NaN detected at frame %d\n", dbgFrame);
            }
        }

        computeNormals(renderPts.data(), gClothVerts);

        Snippets::renderMesh((PxU32)gClothVerts, renderPts.data(),
            gTriCount, gTriIdx, COL_FLAG, gNormals.data());
    }

    // HUD
    {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "Wind: %.0f   dir: %.0f deg   |   [W/S/A/D] camera   [ESC] quit",
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
