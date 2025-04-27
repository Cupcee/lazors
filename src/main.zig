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
const JITTER_SCALE = 0.002;

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
    const vs_path = "resources/shaders/glsl330/lighting_instancing_unlit.vs";
    const fs_path = "resources/shaders/glsl330/lighting_unlit.fs";
    const inst_shader = try rl.loadShader(vs_path, fs_path);
    inst_shader.locs[@intFromEnum(rl.ShaderLocationIndex.matrix_model)] =
        rl.getShaderLocation(inst_shader, "instanceTransform");

    const inst_mat_colors: [CLASS_COUNT]rl.Color = .{ rl.Color.black, rl.Color.red, rl.Color.green, rl.Color.yellow };
    var inst_mats: [CLASS_COUNT]rl.Material = undefined;
    for (&inst_mats, 0..) |*m, i| {
        m.* = try rl.loadMaterialDefault();
        m.*.shader = inst_shader;
        m.*.maps[@intFromEnum(rl.MATERIAL_MAP_DIFFUSE)].color = inst_mat_colors[i];
    }
    return .{ inst_mats, inst_mat_colors };
}

fn initClassTxs(alloc: std.mem.Allocator, max_points: usize) ![CLASS_COUNT][]rl.Matrix {
    var class_txs: [CLASS_COUNT][]rl.Matrix = undefined;
    for (&class_txs) |*slot| {
        slot.* = try alloc.alloc(rl.Matrix, max_points);
    }
    return class_txs;
}

fn initCamera() struct { rl.Camera, rl.CameraMode } {
    const camera = rl.Camera3D{
        .position = .{ .x = 0, .y = 2, .z = -8 },
        .target = .{ .x = 0, .y = 2, .z = 0 },
        .up = .{ .x = 0, .y = 1, .z = 0 },
        .fovy = 60,
        .projection = rl.CameraProjection.perspective,
    };
    const camera_mode = rl.CameraMode.free;
    return .{ camera, camera_mode };
}

fn drawGUI(
    simulation: *s.Simulation,
    class_counter: *[CLASS_COUNT]usize,
    total_hit_count: usize,
    inst_mat_colors: *const [CLASS_COUNT]rl.Color,
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
            rl.textFormat("Total hitCount: %04i", .{total_hit_count}),
            10,
            50,
            20,
            rl.Color.dark_gray,
        );
        for (class_counter, 0..) |count, index| {
            const _c: i32 = @intCast(count);
            const _i: i32 = @intCast(index);
            rl.drawText(
                rl.textFormat("[class %i]: %i", .{ _i, _c }),
                10,
                70 + (_i * 20),
                20,
                inst_mat_colors[index],
            );
        }
    } else {
        rl.drawText("Hit points hidden (Press TAB)", 10, 50, 20, rl.Color.dark_gray);
    }
}

fn draw3D(
    models: []const s.Object,
    sphere_mesh: rl.Mesh,
    inst_mats: *const [CLASS_COUNT]rl.Material,
    class_tx: *const [CLASS_COUNT][]rl.Matrix,
    class_counter: *[CLASS_COUNT]usize,
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
            if (class_counter[cls] > 0) {
                rl.drawMeshInstanced(
                    sphere_mesh,
                    inst_mats[cls],
                    class_tx[cls][0..class_counter[cls]],
                );
            }
        }
    }
}

//------------------------------------------------------------------
// helpers for threaded ray-cast work in simulation
//------------------------------------------------------------------

/// Slice‐by-slice preparation of the contexts that each worker thread
/// will receive.  All storage lives *outside* the function so there
/// are no hidden allocations.
fn prepareRaycastContexts(
    contexts: []mt.RaycastContext,
    sensor: *s.Sensor,
    models: []const s.Object,
    jitter_scale: f32,
    thread_prngs: []rand.DefaultPrng,
    thread_hit_lists: []std.ArrayList(mt.ThreadHit),
    n_rays: usize,
) void {
    const num_threads = contexts.len;
    const rays_per_thread = n_rays / num_threads;
    var remaining_rays = n_rays % num_threads;
    var ray_idx: usize = 0;

    for (contexts, 0..) |*ctx, i| {
        // clear & reuse the slice that will collect this thread’s hits
        thread_hit_lists[i].clearRetainingCapacity();

        var chunk = rays_per_thread;
        if (remaining_rays > 0) {
            chunk += 1;
            remaining_rays -= 1;
        }

        const start = ray_idx;
        const end = @min(start + chunk, n_rays);

        ctx.* = .{
            .thread_id = i,
            .start_index = start,
            .end_index = end,
            .sensor = sensor,
            .models = models,
            .jitter_scale = jitter_scale,
            .thread_prng = &thread_prngs[i],
            .thread_hits = &thread_hit_lists[i],
            .points_slice = sensor.points[start..end],
        };

        ray_idx = end;
    }
}

/// Spawn workers
/// Returns count of spawned workers
fn launchRaycastWorkers(
    threads: []Thread,
    contexts: []const mt.RaycastContext,
) !usize {
    var spawned: usize = 0;

    for (contexts, 0..) |ctx, i| {
        if (ctx.start_index == ctx.end_index) continue; // nothing to do

        threads[spawned] = try Thread.spawn(.{}, mt.raycastWorker, .{&contexts[i]});
        spawned += 1;
    }
    return spawned;
}

fn waitForWorkers(threads: []Thread, spawnedCount: usize) void {
    for (threads[0..spawnedCount]) |t| t.join();
}

/// Merge per-thread hit‐lists into the per-class instance matrices and
/// return the total number of hits.
fn mergeThreadHits(
    thread_hit_lists: []const std.ArrayList(mt.ThreadHit),
    class_tx: *[CLASS_COUNT][]rl.Matrix,
    class_counter: *[CLASS_COUNT]usize,
) usize {
    @memset(class_counter, 0);

    var total: usize = 0;
    for (thread_hit_lists) |list| {
        total += list.items.len;
        for (list.items) |hit| {
            const cls: usize = @intCast(hit.hit_class);
            if (cls < CLASS_COUNT) {
                const idx = class_counter[cls];
                class_tx[cls][idx] = hit.transform;
                class_counter[cls] += 1;
            } else {
                std.log.warn("Hit with invalid class ID {} encountered.", .{cls});
            }
        }
    }
    return total;
}

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

    // simulation init
    var simulation = s.Simulation{};

    // 3D camera init
    var camera, const camera_mode = initCamera();

    // 3D scene init
    const models = try scene.buildScene(50, alloc);
    defer for (models) |*m| rl.unloadModel(m.*.model);

    // sensor state init
    var sensor = try s.Sensor.init(alloc, 800, 192, 360, 70);
    defer sensor.deinit();
    sensor.updateLocalAxes(sensor.fwd, sensor.up);
    const max_points = sensor.res_h * sensor.res_v;

    // prepare mesh and materials for visualizing hits with instanced rendering
    var sphere_mesh = rl.genMeshSphere(0.02, 4, 4);
    // loads mesh to GPU
    rl.uploadMesh(&sphere_mesh, false);
    defer rl.unloadMesh(sphere_mesh);
    var class_counter: [CLASS_COUNT]usize = .{0} ** CLASS_COUNT;
    const inst_mats, const inst_mat_colors = try initInstanceMats();
    var class_tx = try initClassTxs(alloc, max_points);
    defer for (class_tx) |buf| alloc.free(buf);

    // prepare multithreading for raycasting
    const num_threads = mt.getNumThreads();
    std.log.info("Using {} threads for raycasting.", .{num_threads});
    var threads = try mt.ThreadResources.init(alloc, num_threads, max_points);
    try threads.startWorkers();
    defer threads.deinit(alloc);

    while (!rl.windowShouldClose()) {
        rl.updateCamera(&camera, camera_mode);

        const dt = rl.getFrameTime();
        sensorDt(&sensor, dt, &simulation.debug);

        // 1. Build the contexts for this frame’s ray-casts
        prepareRaycastContexts(
            threads.contexts,
            &sensor,
            models,
            JITTER_SCALE,
            threads.prngs,
            threads.hits,
            max_points,
        );

        // 2. let the pool run them
        const n_jobs = threads.contexts.len;
        threads.dispatch(n_jobs);
        try threads.wait(); // blocks until all rays done

        // 3. Merge results
        const total_hit_count = mergeThreadHits(
            threads.hits,
            &class_tx,
            &class_counter,
        );

        rl.beginDrawing();
        defer rl.endDrawing();
        rl.clearBackground(rl.Color.ray_white);

        // 3D drawing block
        {
            rl.beginMode3D(camera);
            defer rl.endMode3D();

            draw3D(models, sphere_mesh, &inst_mats, &class_tx, &class_counter, &sensor, &simulation);
        }

        drawGUI(&simulation, &class_counter, total_hit_count, &inst_mat_colors);
    }

    rl.closeWindow();
}
