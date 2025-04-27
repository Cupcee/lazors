// ----------------------------------------------------
// tiny LiDAR simulator using Raylib bindings for Zig
// ----------------------------------------------------

const std = @import("std");
const Thread = std.Thread;
const builtin = @import("builtin");
const rand = std.Random;
const rl = @import("raylib"); // ziraylib package
const s = @import("structs.zig");
const rc = @import("raycasting.zig");
const tp = @import("thread_pool.zig");
const scene = @import("scene.zig");
const CLASS_COUNT = rc.CLASS_COUNT;
const WINDOW_WIDTH = 1240;
const WINDOW_HEIGHT = 800;
const JITTER_SCALE = 0.002;

const RayPool = tp.ThreadPool(rc.RaycastContext, rc.raycastWorker);

var debug_allocator: std.heap.DebugAllocator(.{}) = .init;

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

pub fn main() !void {
    const alloc, const is_debug = alloc: {
        break :alloc switch (builtin.mode) {
            .Debug, .ReleaseSafe => .{ debug_allocator.allocator(), true },
            .ReleaseFast, .ReleaseSmall => .{ std.heap.smp_allocator, false },
        };
    };
    defer if (is_debug) {
        _ = debug_allocator.deinit();
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
    const num_threads = rc.getNumThreads();
    var thread_resources = try rc.ThreadResources.init(alloc, num_threads, max_points);
    defer thread_resources.deinit(alloc);
    const thread_ctx = try alloc.alloc(rc.RaycastContext, num_threads);
    // defer for (thread_ctx) |ctx| alloc.free(ctx);
    var pool = try RayPool.init(alloc, num_threads, thread_ctx);
    defer pool.deinit(alloc);
    try pool.startWorkers();

    while (!rl.windowShouldClose()) {
        rl.updateCamera(&camera, camera_mode);

        const dt = rl.getFrameTime();
        sensorDt(&sensor, dt, &simulation.debug);

        // 1. Build the contexts for this frame’s ray-casts
        rc.prepareRaycastContexts(
            thread_ctx,
            &sensor,
            models,
            JITTER_SCALE,
            thread_resources.prngs,
            thread_resources.hits,
            max_points,
        );

        // 2. let the pool run them
        const n_jobs = pool.contexts.len;
        pool.dispatch(n_jobs);
        pool.wait(); // blocks until all rays done

        // 3. Merge results
        const total_hit_count = rc.mergeThreadHits(
            thread_resources.hits,
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
