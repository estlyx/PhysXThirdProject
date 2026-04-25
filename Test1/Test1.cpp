#include <iostream>
#include <vector>
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <ctime>

#include <PxPhysicsAPI.h>
#include "snippetrender/SnippetRender.h"
#include "snippetrender/SnippetCamera.h"

using namespace physx;

// =========================================================
// Lab 2: Shooting Simulation
//
// Controls:
//   W / S   -- move forward / backward
//   A / D   -- strafe left / right
//   J / L   -- rotate aim left / right
//   I / K   -- grenade pitch up / down
//   F       -- shoot (raycast, 3 deg spread)
//   G       -- throw grenade (fuse 3 s)
//   R       -- reset enemy
//   ESC     -- quit
// =========================================================

// --- PhysX core ------------------------------------------
static PxDefaultAllocator      gAlloc;
static PxDefaultErrorCallback  gErr;
static PxFoundation* gFoundation = nullptr;
static PxPhysics* gPhysics = nullptr;
static PxDefaultCpuDispatcher* gDispatcher = nullptr;
static PxScene* gScene = nullptr;
static PxPvd* gPvd = nullptr;
static PxPvdTransport* gTransport = nullptr;

// --- Materials -------------------------------------------
static PxMaterial* gMatGround = nullptr;
static PxMaterial* gMatBox = nullptr;
static PxMaterial* gMatActor = nullptr;   // shared for enemy + player capsule
static PxMaterial* gMatGrenade = nullptr;

// --- Scene objects ---------------------------------------
static PxRigidStatic* gGround = nullptr;
static const int       NUM_BOXES = 5;
static PxRigidStatic* gBoxes[NUM_BOXES] = {};
static PxRigidDynamic* gEnemy = nullptr;
static PxRigidDynamic* gPlayerCapsule = nullptr;  // kinematic, visual only

// --- Player state ----------------------------------------
static PxVec3      gPlayerPos(0.0f, 0.0f, -12.0f);
static const float PLAYER_EYE_H = 1.7f;
static const float CAPSULE_RADIUS = 0.4f;
static const float CAPSULE_HALFH = 0.7f;
// Center of capsule above player feet = radius + halfH
static const float CAPSULE_CENTER_Y = CAPSULE_RADIUS + CAPSULE_HALFH;  // 1.1 m
static const float MOVE_STEP = 0.5f;
static const float FIELD_LIMIT = 18.0f;

static const PxVec3 ENEMY_START(0.0f, CAPSULE_CENTER_Y, 12.0f);

// --- Aim -------------------------------------------------
static float gAimYaw = 0.0f;   // radians, horizontal
static float gAimPitch = 35.0f;  // degrees, for grenade arc

// --- Bullet trails ---------------------------------------
struct BulletTrail { PxVec3 start, end; float lifetime; };
static std::vector<BulletTrail> gTrails;
static const float TRAIL_LIFE = 2.5f;

// --- Grenade ---------------------------------------------
static PxRigidDynamic* gGrenade = nullptr;
static float           gGrenadeTimer = 0.0f;
static bool            gGrenadeActive = false;
static const float     GRENADE_FUSE = 3.0f;
static const float     GRENADE_RADIUS = 6.0f;
static const float     GRENADE_SPEED = 15.0f;

// --- Spread ----------------------------------------------
static const float SPREAD_RAD = 3.0f * PxPi / 180.0f;

// --- Camera ----------------------------------------------
static Snippets::Camera* gCamera = nullptr;
static const float CAM_BACK = 14.0f;
static const float CAM_UP = 11.0f;

// --- Colours ---------------------------------------------
static const PxVec3 COL_GROUND(0.30f, 0.30f, 0.30f);
static const PxVec3 COL_BOX(0.55f, 0.38f, 0.15f);
static const PxVec3 COL_ENEMY(0.85f, 0.15f, 0.10f);
static const PxVec3 COL_PLAYER(0.20f, 0.60f, 1.00f);
static const PxVec3 COL_GRENADE(0.90f, 0.55f, 0.05f);
static const PxVec3 COL_AIM(1.00f, 0.20f, 0.20f);
static const PxVec3 COL_TRAIL(1.00f, 0.95f, 0.00f);

// =========================================================
// Direction helpers
// =========================================================

static PxVec3 aimDirH()   // horizontal unit vector in aim direction
{
    return PxVec3(PxSin(gAimYaw), 0.0f, PxCos(gAimYaw)).getNormalized();
}

// Right vector for strafing.
// Camera looks slightly downward so the screen-right vector is -X when
// facing +Z. We negate the geometric right to match what the player sees.
static PxVec3 strafeRight()
{
    return PxVec3(-PxCos(gAimYaw), 0.0f, PxSin(gAimYaw)).getNormalized();
}

static PxVec3 grenadeLaunchDir()
{
    float p = gAimPitch * PxPi / 180.0f;
    return PxVec3(PxSin(gAimYaw) * PxCos(p),
        PxSin(p),
        PxCos(gAimYaw) * PxCos(p)).getNormalized();
}

static float randF() { return ((rand() % 2001) - 1000) / 1000.0f; }

static void clampPlayer()
{
    gPlayerPos.x = PxClamp(gPlayerPos.x, -FIELD_LIMIT, FIELD_LIMIT);
    gPlayerPos.z = PxClamp(gPlayerPos.z, -FIELD_LIMIT, FIELD_LIMIT);
}

// =========================================================
// Object creation
// =========================================================

static PxRigidStatic* createStaticBox(PxVec3 pos, PxVec3 half)
{
    PxShape* s = gPhysics->createShape(PxBoxGeometry(half), *gMatBox, true);
    PxRigidStatic* a = gPhysics->createRigidStatic(PxTransform(pos));
    a->attachShape(*s);
    s->release();
    gScene->addActor(*a);
    return a;
}

// Capsule with vertical axis: PhysX default capsule is along X,
// rotate 90 deg around Z so the long axis points along Y.
static PxRigidDynamic* createCapsule(PxVec3 center, bool kinematic)
{
    PxShape* s = gPhysics->createShape(
        PxCapsuleGeometry(CAPSULE_RADIUS, CAPSULE_HALFH), *gMatActor, true);
    s->setLocalPose(PxTransform(PxQuat(PxHalfPi, PxVec3(0, 0, 1))));

    if (kinematic)
    {
        // Player capsule: visual only, no collision, no raycasts
        s->setFlag(PxShapeFlag::eSIMULATION_SHAPE, false);
        s->setFlag(PxShapeFlag::eSCENE_QUERY_SHAPE, false);
    }

    PxRigidDynamic* a = gPhysics->createRigidDynamic(PxTransform(center));
    a->attachShape(*s);
    s->release();

    if (kinematic)
    {
        a->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC, true);
        // Dummy mass required even for kinematics
        a->setMass(1.0f);
        a->setMassSpaceInertiaTensor(PxVec3(1.0f));
    }
    else
    {
        PxRigidBodyExt::updateMassAndInertia(*a, 85.0f);  // ~80 kg enemy
        a->setLinearDamping(0.1f);
        a->setAngularDamping(0.5f);
    }

    gScene->addActor(*a);
    return a;
}

static PxRigidDynamic* createGrenadeSphere(PxVec3 pos)
{
    PxShape* s = gPhysics->createShape(PxSphereGeometry(0.12f), *gMatGrenade, true);
    PxRigidDynamic* a = gPhysics->createRigidDynamic(PxTransform(pos));
    a->attachShape(*s);
    s->release();
    PxRigidBodyExt::updateMassAndInertia(*a, 220.0f);
    a->setLinearDamping(0.02f);
    gScene->addActor(*a);
    return a;
}

// =========================================================
// Game actions
// =========================================================

static void shoot()
{
    PxVec3 base = aimDirH();
    PxVec3 rgt = -strafeRight();  // geometric right for spread offset
    PxVec3 up(0, 1, 0);
    PxVec3 dir = (base
        + rgt * (randF() * SPREAD_RAD)
        + up * (randF() * SPREAD_RAD)).getNormalized();

    PxVec3 origin = gPlayerPos + PxVec3(0, PLAYER_EYE_H, 0);
    const float RANGE = 60.0f;
    PxVec3 hitPos = origin + dir * RANGE;

    PxRaycastBuffer hit;
    PxQueryFilterData fd;
    fd.flags = PxQueryFlag::eSTATIC | PxQueryFlag::eDYNAMIC;

    if (gScene->raycast(origin, dir, RANGE, hit, PxHitFlag::eDEFAULT, fd))
    {
        hitPos = hit.block.position;
        if (hit.block.actor == gEnemy)
        {
            std::printf("[HIT] Bullet hit the enemy!\n");
            gEnemy->addForce(dir * 500.0f, PxForceMode::eIMPULSE);
        }
        else
        {
            std::printf("[MISS] Bullet hit an obstacle.\n");
        }
    }
    else
    {
        std::printf("[MISS] Bullet hit nothing.\n");
    }

    gTrails.push_back({ origin, hitPos, TRAIL_LIFE });
}

static void explodeGrenade()
{
    if (!gGrenade) return;

    PxVec3 center = gGrenade->getGlobalPose().p;
    std::printf("[EXPLOSION] Detonated at (%.1f, %.1f, %.1f)\n",
        center.x, center.y, center.z);

    gScene->removeActor(*gGrenade);
    gGrenade->release();
    gGrenade = nullptr;
    gGrenadeActive = false;

    if (!gEnemy) return;

    PxVec3 toEnemy = gEnemy->getGlobalPose().p - center;
    float  dist = toEnemy.magnitude();

    if (dist >= GRENADE_RADIUS)
    {
        std::printf("[EXPLOSION] Enemy out of blast radius (%.2f m).\n", dist);
        return;
    }

    // LOS: only check static geometry (walls / boxes).
    // If something static sits between blast center and enemy, damage is blocked.
    PxRaycastBuffer los;
    PxQueryFilterData fdS;
    fdS.flags = PxQueryFlag::eSTATIC;
    PxVec3 losDir = toEnemy.getNormalized();

    if (gScene->raycast(center, losDir, dist - 0.2f, los, PxHitFlag::eDEFAULT, fdS))
    {
        std::printf("[EXPLOSION] Enemy behind cover -- damage blocked! "
            "(wall at %.2f m)\n", los.block.distance);
        return;
    }

    float t = 1.0f - dist / GRENADE_RADIUS;
    float damage = t * 100.0f;
    float force = t * 3500.0f;
    std::printf("[EXPLOSION] Enemy hit! Damage: %.1f/100  (dist %.2f m)\n", damage, dist);

    PxVec3 impulseDir = (losDir + PxVec3(0, 0.4f, 0)).getNormalized();
    gEnemy->addForce(impulseDir * force, PxForceMode::eIMPULSE);
}

static void throwGrenade()
{
    if (gGrenadeActive) { std::printf("[GRENADE] Already in flight!\n"); return; }

    PxVec3 spawn = gPlayerPos + aimDirH() * 0.5f + PxVec3(0, PLAYER_EYE_H - 0.3f, 0);
    gGrenade = createGrenadeSphere(spawn);
    gGrenade->setLinearVelocity(grenadeLaunchDir() * GRENADE_SPEED);
    gGrenadeTimer = GRENADE_FUSE;
    gGrenadeActive = true;
    std::printf("[GRENADE] Thrown! Fuse: %.1f s  (yaw %.0f deg, pitch %.0f deg)\n",
        GRENADE_FUSE, gAimYaw * 180.0f / PxPi, gAimPitch);
}

static void resetEnemy()
{
    if (gEnemy) { gScene->removeActor(*gEnemy); gEnemy->release(); }
    gEnemy = createCapsule(ENEMY_START, false);
    std::printf("[RESET] Enemy returned to start.\n");
}

// =========================================================
// Physics init
// =========================================================

void initPhysics()
{
    std::srand((unsigned)std::time(nullptr));

    gFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, gAlloc, gErr);
    gPvd = PxCreatePvd(*gFoundation);
    gTransport = PxDefaultPvdSocketTransportCreate("127.0.0.1", 5425, 10000);
    if (gPvd && gTransport)
        gPvd->connect(*gTransport, PxPvdInstrumentationFlag::eALL);

    gPhysics = PxCreatePhysics(PX_PHYSICS_VERSION, *gFoundation,
        PxTolerancesScale(), true, gPvd);

    PxSceneDesc sd(gPhysics->getTolerancesScale());
    sd.gravity = PxVec3(0, -9.81f, 0);
    gDispatcher = PxDefaultCpuDispatcherCreate(2);
    sd.cpuDispatcher = gDispatcher;
    sd.filterShader = PxDefaultSimulationFilterShader;
    gScene = gPhysics->createScene(sd);

    gMatGround = gPhysics->createMaterial(0.7f, 0.7f, 0.20f);
    gMatBox = gPhysics->createMaterial(0.5f, 0.5f, 0.10f);
    gMatActor = gPhysics->createMaterial(0.5f, 0.5f, 0.05f);
    gMatGrenade = gPhysics->createMaterial(0.4f, 0.4f, 0.55f);

    gGround = PxCreatePlane(*gPhysics, PxPlane(0, 1, 0, 0), *gMatGround);
    gScene->addActor(*gGround);

    // 5 obstacle boxes
    gBoxes[0] = createStaticBox(PxVec3(-6.0f, 1.5f, 0.0f), PxVec3(1.5f, 1.5f, 1.5f));
    gBoxes[1] = createStaticBox(PxVec3(5.5f, 1.5f, 2.5f), PxVec3(1.0f, 1.5f, 2.0f));
    gBoxes[2] = createStaticBox(PxVec3(-2.5f, 1.5f, 7.5f), PxVec3(2.0f, 1.5f, 1.0f));
    gBoxes[3] = createStaticBox(PxVec3(4.0f, 1.5f, -4.5f), PxVec3(1.2f, 1.5f, 1.2f));
    gBoxes[4] = createStaticBox(PxVec3(0.0f, 1.5f, 3.5f), PxVec3(1.0f, 1.5f, 3.5f));

    // Enemy: dynamic capsule (red)
    gEnemy = createCapsule(ENEMY_START, false);

    // Player: kinematic capsule (blue), no collision or raycast interaction
    gPlayerCapsule = createCapsule(
        gPlayerPos + PxVec3(0, CAPSULE_CENTER_Y, 0), true);

    std::printf("=== PhysX Shooter -- Lab Work #2 ===\n");
    std::printf("  W/S   -- move forward / back\n");
    std::printf("  A/D   -- strafe left / right\n");
    std::printf("  J/L   -- rotate aim\n");
    std::printf("  I/K   -- grenade pitch\n");
    std::printf("  F     -- shoot\n");
    std::printf("  G     -- throw grenade\n");
    std::printf("  R     -- reset enemy\n\n");
}

// =========================================================
// Input
// =========================================================

static void customKeyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch (std::toupper(key))
    {
    case 'W': gPlayerPos += aimDirH() * MOVE_STEP; clampPlayer(); break;
    case 'S': gPlayerPos -= aimDirH() * MOVE_STEP; clampPlayer(); break;
    case 'A': gPlayerPos -= strafeRight() * MOVE_STEP; clampPlayer(); break;
    case 'D': gPlayerPos += strafeRight() * MOVE_STEP; clampPlayer(); break;

    case 'J': gAimYaw -= 0.06f; break;
    case 'L': gAimYaw += 0.06f; break;

    case 'I': gAimPitch = PxMin(gAimPitch + 2.0f, 75.0f); break;
    case 'K': gAimPitch = PxMax(gAimPitch - 2.0f, 5.0f); break;

    case 'F': shoot();        break;
    case 'G': throwGrenade(); break;
    case 'R': resetEnemy();   break;

    case 27: exit(0); break;  // ESC
    default:  break;
    }
}

// Stub passed to setupDefault -- never called because we override the handler.
void keyPress(unsigned char /*key*/, const PxTransform& /*cam*/) {}

// =========================================================
// Render
// =========================================================

void renderCallback()
{
    // Move player capsule to follow gPlayerPos (kinematic target)
    if (gPlayerCapsule)
    {
        PxVec3 center = gPlayerPos + PxVec3(0, CAPSULE_CENTER_Y, 0);
        gPlayerCapsule->setKinematicTarget(PxTransform(center));
    }

    // Third-person camera: behind + above the player, always looking at player head
    PxVec3 camEye = gPlayerPos - aimDirH() * CAM_BACK + PxVec3(0, CAM_UP, 0);
    PxVec3 lookAt = gPlayerPos + PxVec3(0, 1.0f, 0);
    gCamera->setPose(camEye, (lookAt - camEye).getNormalized());

    const float dt = 1.0f / 60.0f;
    gScene->simulate(dt);
    gScene->fetchResults(true);

    if (gGrenadeActive) { gGrenadeTimer -= dt; if (gGrenadeTimer <= 0.0f) explodeGrenade(); }

    for (auto& t : gTrails) t.lifetime -= dt;
    gTrails.erase(
        std::remove_if(gTrails.begin(), gTrails.end(),
            [](const BulletTrail& t) { return t.lifetime <= 0.0f; }),
        gTrails.end());

    Snippets::startRender(gCamera);

    // Ground
    { PxRigidActor* a = gGround; Snippets::renderActors(&a, 1, false, COL_GROUND, nullptr, false, false); }

    // Boxes
    {
        PxRigidActor* arr[NUM_BOXES]; PxU32 n = 0;
        for (int i = 0; i < NUM_BOXES; i++) if (gBoxes[i]) arr[n++] = gBoxes[i];
        if (n) Snippets::renderActors(arr, n, true, COL_BOX, nullptr, false, false);
    }

    // Enemy capsule (red)
    if (gEnemy)
    {
        PxRigidActor* a = gEnemy; Snippets::renderActors(&a, 1, true, COL_ENEMY, nullptr, false, false);
    }

    // Player capsule (blue)
    if (gPlayerCapsule)
    {
        PxRigidActor* a = gPlayerCapsule; Snippets::renderActors(&a, 1, true, COL_PLAYER, nullptr, false, false);
    }

    // Grenade (orange sphere)
    if (gGrenade && gGrenadeActive)
    {
        PxRigidActor* a = gGrenade; Snippets::renderActors(&a, 1, true, COL_GRENADE, nullptr, false, false);
    }

    // Aim ray from player eye
    {
        PxVec3 eye = gPlayerPos + PxVec3(0, PLAYER_EYE_H, 0);
        Snippets::DrawLine(eye, eye + aimDirH() * 25.0f, COL_AIM);
    }

    // Bullet trails (yellow)
    for (const auto& t : gTrails) Snippets::DrawLine(t.start, t.end, COL_TRAIL);

    // HUD
    {
        char buf[400];
        std::snprintf(buf, sizeof(buf),
            "[W/S/A/D]=Move  [J/L]=Aim %.0fdeg  [I/K]=Pitch %.0fdeg  "
            "[F]=Shoot  [G]=Grenade  [R]=Reset%s",
            gAimYaw * 180.0f / PxPi, gAimPitch,
            gGrenadeActive ? "  | GRENADE IN FLIGHT!" : "");
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

    if (gGrenade) { gScene->removeActor(*gGrenade);       gGrenade->release();       gGrenade = nullptr; }
    if (gPlayerCapsule) { gScene->removeActor(*gPlayerCapsule); gPlayerCapsule->release(); gPlayerCapsule = nullptr; }
    if (gEnemy) { gScene->removeActor(*gEnemy);         gEnemy->release();         gEnemy = nullptr; }
    for (int i = 0; i < NUM_BOXES; i++)
        if (gBoxes[i]) { gScene->removeActor(*gBoxes[i]);      gBoxes[i]->release();      gBoxes[i] = nullptr; }
    if (gGround) { gScene->removeActor(*gGround);        gGround->release();        gGround = nullptr; }

    if (gScene) { gScene->release();      gScene = nullptr; }
    if (gDispatcher) { gDispatcher->release(); gDispatcher = nullptr; }
    if (gMatGrenade) { gMatGrenade->release(); gMatGrenade = nullptr; }
    if (gMatActor) { gMatActor->release();   gMatActor = nullptr; }
    if (gMatBox) { gMatBox->release();     gMatBox = nullptr; }
    if (gMatGround) { gMatGround->release();  gMatGround = nullptr; }
    if (gPhysics) { gPhysics->release();    gPhysics = nullptr; }
    if (gPvd) { gPvd->release();        gPvd = nullptr; }
    if (gTransport) { gTransport->release();  gTransport = nullptr; }
    if (gFoundation) { gFoundation->release(); gFoundation = nullptr; }
}

// =========================================================
// Entrypoint
// =========================================================

int main()
{
    PxVec3 initEye = PxVec3(0, 0, -12) - PxVec3(0, 0, 1) * CAM_BACK + PxVec3(0, CAM_UP, 0);
    PxVec3 initLook = PxVec3(0, 1, -12);
    gCamera = new Snippets::Camera(initEye, (initLook - initEye).getNormalized());

    Snippets::setupDefault("PhysX Shooter",
        gCamera, keyPress, renderCallback, exitCallback);

    // Override keyboard handler so WASD does player movement, not the Snippets camera.
    glutKeyboardFunc(customKeyboard);

    initPhysics();
    glutMainLoop();
    return 0;
}
