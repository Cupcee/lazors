// ----------------------------------------------------
// tiny LiDAR simulator using Raylib bindings for Zig
// runtime-configurable resolution version
// + bounding-box culling and analytic ground plane (steps 1-3)
// ----------------------------------------------------

const std = @import("std");
const Thread = std.Thread;
const builtin = @import("builtin");
const rand = std.Random;
const rl = @import("raylib"); // ziraylib package
const s = @import("structs.zig");
const mt = @import("multithreading.zig");
const scene = @import("scene.zig");

var debugAllocator: std.heap.DebugAllocator(.{}) = .init;

fn sensorDt(sensor: *s.Sensor, dt: f32, debug: *bool) void {
    if (rl.isKeyDown(rl.KeyboardKey.right)) sensor.pos.x -= 5 * dt;
    if (rl.isKeyDown(rl.KeyboardKey.left)) sensor.pos.x += 5 * dt;
    if (rl.isKeyDown(rl.KeyboardKey.up)) sensor.pos.z += 5 * dt;
    if (rl.isKeyDown(rl.KeyboardKey.down)) sensor.pos.z -= 5 * dt;
    if (rl.isKeyDown(rl.KeyboardKey.k)) sensor.pos.y += 5 * dt;
    if (rl.isKeyDown(rl.KeyboardKey.j)) sensor.pos.y -= 5 * dt;
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

    rl.initWindow(1240, 800, "lazors");
    rl.disableCursor();
    rl.setTargetFPS(24);
    var SIM_DEBUG = true;

    var camera = rl.Camera3D{
        .position = .{ .x = 0, .y = 2, .z = -8 },
        .target = .{ .x = 0, .y = 2, .z = 0 },
        .up = .{ .x = 0, .y = 1, .z = 0 },
        .fovy = 60,
        .projection = rl.CameraProjection.perspective,
    };
    const cameraMode = rl.CameraMode.free;

    const models = try scene.buildScene(50, alloc);
    defer for (models) |*m| rl.unloadModel(m.*.model);

    var sensor = try s.Sensor.init(alloc, 800, 192, 360, 70);
    defer sensor.deinit();
    sensor.updateLocalAxes(sensor.fwd, sensor.up);

    var sphereMesh = rl.genMeshSphere(0.02, 4, 4);
    rl.uploadMesh(&sphereMesh, false);
    defer rl.unloadMesh(sphereMesh);

    const vsPath = "resources/shaders/glsl330/lighting_instancing_unlit.vs";
    const fsPath = "resources/shaders/glsl330/lighting_unlit.fs";
    const instShader = try rl.loadShader(vsPath, fsPath);
    instShader.locs[@intFromEnum(rl.ShaderLocationIndex.matrix_model)] =
        rl.getShaderLocation(instShader, "instanceTransform");

    const CLASS_COUNT = 4;
    const instanceMatColors: [CLASS_COUNT]rl.Color = .{ rl.Color.black, rl.Color.red, rl.Color.green, rl.Color.yellow };
    var instMats: [CLASS_COUNT]rl.Material = undefined;
    for (&instMats, 0..) |*m, i| {
        m.* = try rl.loadMaterialDefault();
        m.*.shader = instShader;
        m.*.maps[@intFromEnum(rl.MATERIAL_MAP_DIFFUSE)].color = instanceMatColors[i];
    }

    var classTx: [CLASS_COUNT][]rl.Matrix = undefined;
    for (&classTx) |*slot| {
        slot.* = try alloc.alloc(rl.Matrix, sensor.res_h * sensor.res_v);
    }
    defer for (classTx) |buf| alloc.free(buf);

    var classCount: [CLASS_COUNT]usize = .{0} ** CLASS_COUNT;

    const jitter_scale: f32 = 0.005;

    const num_threads = blk: {
        const cpu_count = std.Thread.getCpuCount() catch 1;
        break :blk @max(1, cpu_count);
    };
    std.log.info("Using {} threads for raycasting.", .{num_threads});

    var threads = try alloc.alloc(Thread, num_threads);
    defer alloc.free(threads);

    var contexts = try alloc.alloc(mt.RaycastContext, num_threads);
    defer alloc.free(contexts);

    var thread_hit_lists = try alloc.alloc(std.ArrayList(mt.ThreadHit), num_threads);
    defer {
        for (thread_hit_lists) |*list| list.deinit();
        alloc.free(thread_hit_lists);
    }
    for (thread_hit_lists) |*list| {
        list.* = std.ArrayList(mt.ThreadHit).init(alloc);
        try list.ensureTotalCapacity(sensor.res_h * sensor.res_v / num_threads + 10);
    }

    var thread_prngs = try alloc.alloc(rand.DefaultPrng, num_threads);
    defer alloc.free(thread_prngs);
    for (thread_prngs) |*rng_state| {
        rng_state.* = rand.DefaultPrng.init(@intCast(std.time.nanoTimestamp()));
    }

    while (!rl.windowShouldClose()) {
        rl.updateCamera(&camera, cameraMode);

        const dt = rl.getFrameTime();
        sensorDt(&sensor, dt, &SIM_DEBUG);

        const nRays = sensor.res_h * sensor.res_v;
        var totalHitCount: usize = 0;

        for (thread_hit_lists) |*list| list.clearRetainingCapacity();

        const rays_per_thread = nRays / num_threads;
        var remaining_rays = nRays % num_threads;
        var current_ray_index: usize = 0;

        for (0..num_threads) |i| {
            var chunk_size = rays_per_thread;
            if (remaining_rays > 0) {
                chunk_size += 1;
                remaining_rays -= 1;
            }

            const start_index = current_ray_index;
            const end_index = @min(start_index + chunk_size, nRays);
            const points_slice = if (start_index < end_index)
                sensor.points[start_index..end_index]
            else
                sensor.points[0..0];

            contexts[i] = .{
                .thread_id = i,
                .start_index = start_index,
                .end_index = end_index,
                .sensor = &sensor,
                .models = models,
                .jitter_scale = jitter_scale,
                .thread_prng = &thread_prngs[i],
                .thread_hits = &thread_hit_lists[i],
                .points_slice = points_slice,
            };

            threads[i] = try Thread.spawn(.{}, mt.raycastWorker, .{&contexts[i]});

            current_ray_index = end_index;
        }

        for (threads) |t| t.join();

        @memset(&classCount, 0);
        totalHitCount = 0;
        for (thread_hit_lists) |list| {
            totalHitCount += list.items.len;
            for (list.items) |hit| {
                const cls: usize = @intCast(hit.hit_class);
                if (cls < CLASS_COUNT) {
                    const current_idx = classCount[cls];
                    classTx[cls][current_idx] = hit.transform;
                    classCount[cls] += 1;
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

            // rl.drawGrid(20, 1);
            for (models) |model| {
                rl.drawModel(model.model, rl.Vector3.zero(), 1, model.color);
            }
            rl.drawSphere(sensor.pos, 0.07, rl.Color.black);

            if (SIM_DEBUG) {
                for (0..CLASS_COUNT) |cls| {
                    if (classCount[cls] > 0) {
                        rl.drawMeshInstanced(
                            sphereMesh,
                            instMats[cls],
                            classTx[cls][0..classCount[cls]],
                        );
                    }
                }
            }
        }

        rl.drawFPS(10, 10);
        rl.drawText(
            "Camera: WASD, Left-CTRL, Space. Sensor: arrow keys, HJKL. TAB: Toggle Points",
            10,
            30,
            20,
            rl.Color.dark_gray,
        );
        if (SIM_DEBUG) {
            rl.drawText(
                rl.textFormat("Total hitCount: %04i", .{totalHitCount}),
                10,
                50,
                20,
                rl.Color.dark_gray,
            );
            for (classCount, 0..) |count, index| {
                const _c: i32 = @intCast(count);
                const _i: i32 = @intCast(index);
                rl.drawText(
                    rl.textFormat("[class %i]: %i", .{ _i, _c }),
                    10,
                    70 + (_i * 20),
                    20,
                    instanceMatColors[index],
                );
            }
        } else {
            rl.drawText("Hit points hidden (Press TAB)", 10, 50, 20, rl.Color.dark_gray);
        }
    }

    rl.closeWindow();
}
