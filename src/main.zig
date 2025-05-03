// ----------------------------------------------------
// tiny LiDAR simulator using Raylib bindings for Zig
// ----------------------------------------------------

const std = @import("std");
const builtin = @import("builtin");
const rl = @import("raylib"); // ziraylib package
const s = @import("structs.zig");
const rc = @import("raycasting.zig");
const tp = @import("thread_pool.zig");
const scene = @import("scene.zig");
const pcd = @import("pcd_exporter.zig");
const sim = @import("simulation.zig");
const WINDOW_WIDTH = 1240;
const WINDOW_HEIGHT = 800;
const JITTER_SCALE = 0.002;

const RayPool = tp.ThreadPool(rc.RaycastContext, rc.raycastWorker);

var debug_allocator: std.heap.DebugAllocator(.{}) = .init;

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

    var simulation = s.Simulation{};

    // --- DRAW WINDOW ---
    rl.initWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "lazors");
    rl.disableCursor();
    rl.setTargetFPS(simulation.target_fps);
    var camera, const camera_mode = sim.initCamera();

    // --- BUILD SCENE ---
    const models = try scene.buildScene(
        alloc,
        simulation.num_objects,
        simulation.class_count,
        simulation.plane_half_size,
    );
    defer for (models) |*m| {
        m.bvh.deinit();
        rl.unloadModel(m.*.model);
    };

    // --- INIT SENSOR ---
    var sensor = try s.Sensor.init(alloc, 800, 192, 360, 70);
    defer sensor.deinit();
    sensor.updateLocalAxes(sensor.fwd, sensor.up);
    const max_points = sensor.res_h * sensor.res_v;

    // --- COLLISION DRAWING ---
    var collision_mesh = rl.genMeshCube(0.02, 0.02, 0.02);
    rl.uploadMesh(&collision_mesh, false); // loads mesh to GPU for instancing
    defer rl.unloadMesh(collision_mesh);
    const class_counter: []usize = try alloc.alloc(usize, simulation.class_count);
    defer alloc.free(class_counter);
    for (class_counter) |*class| {
        class.* = 0;
    }
    const inst_mats = try sim.initInstanceMats(alloc, @intCast(simulation.class_count));
    // const class_tx = try sim.initClassTxs(alloc, max_points, @intCast(simulation.class_count));
    // defer for (class_tx) |buf| alloc.free(buf);
    const class_tx = try sim.initClassTxLists(
        alloc,
        @intCast(simulation.class_count),
        max_points / simulation.class_count + 32,
    );
    defer for (class_tx) |*list| list.deinit();

    // --- SETUP PCD WRITING ---
    const output_dir = "frames";
    std.fs.Dir.makeDir(std.fs.cwd(), output_dir) catch |e| {
        switch (e) {
            error.PathAlreadyExists => {
                std.log.info("Writing to existing directory '{d}'", .{output_dir});
            },
            else => return e,
        }
    };

    // --- INIT PCD EXPORTER / THREAD ---
    var export_dt: f32 = 0.0;
    var dump_id: u32 = 0;
    var exporter = try pcd.Exporter.create(alloc);
    defer exporter.destroy();

    // --- PREPARE MULTITHREADED RAYCASTING
    const num_threads = rc.getNumThreads();
    var thread_resources = try rc.ThreadResources.init(alloc, num_threads, max_points);
    defer thread_resources.deinit(alloc);
    const thread_ctx = try alloc.alloc(rc.RaycastContext, num_threads);
    var pool = try RayPool.init(alloc, num_threads, thread_ctx);
    defer pool.deinit(alloc);
    try pool.startWorkers();

    // --- SIMULATION LOOP ---
    while (!rl.windowShouldClose()) {
        rl.updateCamera(&camera, camera_mode);

        const dt = rl.getFrameTime();
        sim.sensorDt(&sensor, dt, &simulation.debug);

        for (class_tx) |*dst| dst.clearRetainingCapacity();

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
            class_tx,
            class_counter,
        );

        if (simulation.collect) {
            if (export_dt >= simulation.collect_wait_seconds) {
                try sim.exportPCD(&exporter, class_tx, class_counter, dump_id);
                dump_id += 1;
                export_dt = 0.0;
            } else {
                export_dt += dt;
            }
        }

        rl.beginDrawing();
        defer rl.endDrawing();
        rl.clearBackground(rl.Color.ray_white);

        // 3D drawing block
        {
            rl.beginMode3D(camera);
            defer rl.endMode3D();

            sim.draw3D(models, collision_mesh, inst_mats, class_tx, class_counter, &sensor, &simulation);
        }

        sim.drawGUI(&simulation, class_counter, total_hit_count);
    }

    rl.closeWindow();
}
