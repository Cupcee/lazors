// ----------------------------------------------------
// tiny LiDAR simulator using Raylib bindings for Zig
// ----------------------------------------------------

const std = @import("std");
const Thread = std.Thread;
const builtin = @import("builtin");
const rand = std.Random;
const rl = @import("raylib"); // ziraylib package
const s = @import("structs.zig");
const mt = @import("multithreading.zig");
const scene = @import("scene.zig");
const CLASS_COUNT = 4;
const WINDOW_WIDTH = 1240;
const WINDOW_HEIGHT = 800;
const JITTER_SCALE = 0.005;

var debugAllocator: std.heap.DebugAllocator(.{}) = .init;

fn sensorDt(sensor: *s.Sensor, dt: f32, debug: *bool) void {
    if (rl.isKeyDown(rl.KeyboardKey.right)) sensor.pos.x -= sensor.velocity * dt;
    if (rl.isKeyDown(rl.KeyboardKey.left)) sensor.pos.x += sensor.velocity * dt;
    if (rl.isKeyDown(rl.KeyboardKey.up)) sensor.pos.z += sensor.velocity * dt;
    if (rl.isKeyDown(rl.KeyboardKey.down)) sensor.pos.z -= sensor.velocity * dt;
    if (rl.isKeyDown(rl.KeyboardKey.k)) sensor.pos.y += sensor.velocity * dt;
    if (rl.isKeyDown(rl.KeyboardKey.j)) sensor.pos.y -= sensor.velocity * dt;
    if (rl.isKeyDown(rl.KeyboardKey.h)) sensor.yaw += sensor.turn_speed * dt;
    if (rl.isKeyDown(rl.KeyboardKey.l)) sensor.yaw -= sensor.turn_speed * dt;
    if (rl.isKeyReleased(rl.KeyboardKey.tab)) debug.* = !debug.*;

    const half_pi: f32 = std.math.pi / 2.0 - 0.001;
    sensor.pitch = std.math.clamp(sensor.pitch, -half_pi, half_pi);

    sensor.fwd = .{
        .x = std.math.sin(sensor.yaw) * std.math.cos(sensor.pitch),
        .y = std.math.sin(sensor.pitch),
        .z = std.math.cos(sensor.yaw) * std.math.cos(sensor.pitch),
    };
    sensor.up = .{ .x = 0, .y = 1, .z = 0 };
    sensor.updateLocalAxes(sensor.fwd, sensor.up);
}

fn initInstanceMats() !struct { [CLASS_COUNT]rl.Material, [CLASS_COUNT]rl.Color } {
    const vsPath = "resources/shaders/glsl330/lighting_instancing_unlit.vs";
    const fsPath = "resources/shaders/glsl330/lighting_unlit.fs";
    const instShader = try rl.loadShader(vsPath, fsPath);
    instShader.locs[@intFromEnum(rl.ShaderLocationIndex.matrix_model)] =
        rl.getShaderLocation(instShader, "instanceTransform");

    const instMatColors: [CLASS_COUNT]rl.Color = .{ rl.Color.black, rl.Color.red, rl.Color.green, rl.Color.yellow };
    var instMats: [CLASS_COUNT]rl.Material = undefined;
    for (&instMats, 0..) |*m, i| {
        m.* = try rl.loadMaterialDefault();
        m.*.shader = instShader;
        m.*.maps[@intFromEnum(rl.MATERIAL_MAP_DIFFUSE)].color = instMatColors[i];
    }
    return .{ instMats, instMatColors };
}

fn initClassTxs(alloc: std.mem.Allocator, max_points: usize) ![CLASS_COUNT][]rl.Matrix {
    var classTxs: [CLASS_COUNT][]rl.Matrix = undefined;
    for (&classTxs) |*slot| {
        slot.* = try alloc.alloc(rl.Matrix, max_points);
    }
    return classTxs;
}

fn initCamera() struct { rl.Camera, rl.CameraMode } {
    const camera = rl.Camera3D{
        .position = .{ .x = 0, .y = 2, .z = -8 },
        .target = .{ .x = 0, .y = 2, .z = 0 },
        .up = .{ .x = 0, .y = 1, .z = 0 },
        .fovy = 60,
        .projection = rl.CameraProjection.perspective,
    };
    const cameraMode = rl.CameraMode.free;
    return .{ camera, cameraMode };
}

fn drawGUI(
    simulation: *s.Simulation,
    classCounter: *[CLASS_COUNT]usize,
    totalHitCount: usize,
    instMatColors: *const [CLASS_COUNT]rl.Color,
) void {
    rl.drawFPS(10, 10);
    rl.drawText(
        "Camera: WASD, Left-CTRL, Space. Sensor: arrow keys, HJKL. TAB: Toggle Points",
        10,
        30,
        20,
        rl.Color.dark_gray,
    );
    if (simulation.debug) {
        rl.drawText(
            rl.textFormat("Total hitCount: %04i", .{totalHitCount}),
            10,
            50,
            20,
            rl.Color.dark_gray,
        );
        for (classCounter, 0..) |count, index| {
            const _c: i32 = @intCast(count);
            const _i: i32 = @intCast(index);
            rl.drawText(
                rl.textFormat("[class %i]: %i", .{ _i, _c }),
                10,
                70 + (_i * 20),
                20,
                instMatColors[index],
            );
        }
    } else {
        rl.drawText("Hit points hidden (Press TAB)", 10, 50, 20, rl.Color.dark_gray);
    }
}

fn draw3D(
    models: []const s.Object,
    sphereMesh: rl.Mesh,
    instMats: *const [CLASS_COUNT]rl.Material,
    classTx: *const [CLASS_COUNT][]rl.Matrix,
    classCounter: *[CLASS_COUNT]usize,
    sensor: *s.Sensor,
    simulation: *s.Simulation,
) void {
    // rl.drawGrid(20, 1);
    for (models) |model| {
        rl.drawModel(model.model, rl.Vector3.zero(), 1, model.color);
    }
    rl.drawSphere(sensor.pos, 0.07, rl.Color.black);

    if (simulation.debug) {
        for (0..CLASS_COUNT) |cls| {
            if (classCounter[cls] > 0) {
                rl.drawMeshInstanced(
                    sphereMesh,
                    instMats[cls],
                    classTx[cls][0..classCounter[cls]],
                );
            }
        }
    }
}

//------------------------------------------------------------------
// MAIN
//------------------------------------------------------------------
pub fn main() !void {
    const alloc, const is_debug = alloc: {
        break :alloc switch (builtin.mode) {
            .Debug, .ReleaseSafe => .{ debugAllocator.allocator(), true },
            .ReleaseFast, .ReleaseSmall => .{ std.heap.smp_allocator, false },
        };
    };
    defer if (is_debug) {
        _ = debugAllocator.deinit();
    };

    rl.initWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "lazors");
    rl.disableCursor();
    rl.setTargetFPS(24);

    var simulation = s.Simulation{};

    var camera, const cameraMode = initCamera();

    const models = try scene.buildScene(50, alloc);
    defer for (models) |*m| rl.unloadModel(m.*.model);

    var sensor = try s.Sensor.init(alloc, 800, 192, 360, 70);
    defer sensor.deinit();
    sensor.updateLocalAxes(sensor.fwd, sensor.up);
    const maxPoints = sensor.res_h * sensor.res_v;

    var sphereMesh = rl.genMeshSphere(0.02, 4, 4);
    rl.uploadMesh(&sphereMesh, false);
    defer rl.unloadMesh(sphereMesh);

    const instMats, const instMatColors = try initInstanceMats();
    var classTx = try initClassTxs(alloc, maxPoints);
    defer for (classTx) |buf| alloc.free(buf);

    var classCounter: [CLASS_COUNT]usize = .{0} ** CLASS_COUNT;

    // const numThreads = mt.getNumThreads();
    const numThreads = blk: {
        const cpu_count = Thread.getCpuCount() catch 1;
        break :blk @max(1, cpu_count);
    };
    std.log.info("Using {} threads for raycasting.", .{numThreads});

    var threads = try alloc.alloc(Thread, numThreads);
    defer alloc.free(threads);

    var contexts = try alloc.alloc(mt.RaycastContext, numThreads);
    defer alloc.free(contexts);

    var threadHitLists = try alloc.alloc(std.ArrayList(mt.ThreadHit), numThreads);
    defer {
        for (threadHitLists) |*list| list.deinit();
        alloc.free(threadHitLists);
    }
    for (threadHitLists) |*list| {
        list.* = std.ArrayList(mt.ThreadHit).init(alloc);
        try list.ensureTotalCapacity(maxPoints / numThreads + 10);
    }

    var threadPrngs = try alloc.alloc(rand.DefaultPrng, numThreads);
    defer alloc.free(threadPrngs);
    for (threadPrngs) |*rng_state| {
        rng_state.* = rand.DefaultPrng.init(@intCast(std.time.nanoTimestamp()));
    }

    while (!rl.windowShouldClose()) {
        rl.updateCamera(&camera, cameraMode);

        const dt = rl.getFrameTime();
        sensorDt(&sensor, dt, &simulation.debug);

        const nRays = maxPoints;
        var totalHitCount: usize = 0;

        for (threadHitLists) |*list| list.clearRetainingCapacity();

        const rays_per_thread = nRays / numThreads;
        var remaining_rays = nRays % numThreads;
        var current_ray_index: usize = 0;

        for (0..numThreads) |i| {
            var chunk_size = rays_per_thread;
            if (remaining_rays > 0) {
                chunk_size += 1;
                remaining_rays -= 1;
            }

            const start_index = current_ray_index;
            const end_index = @min(start_index + chunk_size, nRays);
            const threadPointsSlice = if (start_index < end_index)
                sensor.points[start_index..end_index]
            else
                sensor.points[0..0];
            // break;

            contexts[i] = .{
                .thread_id = i,
                .start_index = start_index,
                .end_index = end_index,
                .sensor = &sensor,
                .models = models,
                .jitter_scale = JITTER_SCALE,
                .thread_prng = &threadPrngs[i],
                .thread_hits = &threadHitLists[i],
                .points_slice = threadPointsSlice,
            };

            threads[i] = try Thread.spawn(.{}, mt.raycastWorker, .{&contexts[i]});

            current_ray_index = end_index;
        }

        for (threads) |t| t.join();

        @memset(&classCounter, 0);
        totalHitCount = 0;
        for (threadHitLists) |list| {
            totalHitCount += list.items.len;
            for (list.items) |hit| {
                const cls: usize = @intCast(hit.hit_class);
                if (cls < CLASS_COUNT) {
                    const current_idx = classCounter[cls];
                    classTx[cls][current_idx] = hit.transform;
                    classCounter[cls] += 1;
                } else {
                    std.log.warn(
                        "Hit with invalid class ID {} encountered during merge.",
                        .{cls},
                    );
                }
            }
        }

        rl.beginDrawing();
        defer rl.endDrawing();
        rl.clearBackground(rl.Color.ray_white);

        // 3D drawing block
        {
            rl.beginMode3D(camera);
            defer rl.endMode3D();

            draw3D(models, sphereMesh, &instMats, &classTx, &classCounter, &sensor, &simulation);
        }

        drawGUI(&simulation, &classCounter, totalHitCount, &instMatColors);
    }

    rl.closeWindow();
}
