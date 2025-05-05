//! ----------------------------------------------------
//! tiny LiDAR simulator using Raylib bindings for Zig
//! ----------------------------------------------------

const std = @import("std");
const builtin = @import("builtin");
const rl = @import("raylib");
const clap = @import("clap");

const structs = @import("structs.zig");
const rc = @import("raycasting.zig");
const sim = @import("simulation.zig");
const state = @import("state.zig");

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
    const params = comptime clap.parseParamsComptime(
        \\-h, --help                   Display this help and exit.
        \\-n, --num_objects <u32>      Number of objects in simulation.
        \\-t, --target_fps <i32>       Number of objects in simulation.
        \\-d, --plane_half_size <f32>      Size of the ground plane divided by two.
        \\--collect
        \\--collect_wait_seconds <f32> How many seconds to wait between collections.         
    );
    var diag = clap.Diagnostic{};
    var _args = clap.parse(clap.Help, &params, clap.parsers.default, .{
        .diagnostic = &diag,
        .allocator = alloc,
        .assignment_separators = "=:",
    }) catch |err| {
        // Report useful error and exit.
        diag.report(std.io.getStdErr().writer(), err) catch {};
        return err;
    };
    defer _args.deinit();
    const args = _args.args;
    // `clap.usage` is a function that can print a simple help message. It can print any `Param`
    // where `Id` has a `value` method (`Param(Help)` is one such parameter).
    if (args.help != 0) return clap.help(std.io.getStdErr().writer(), clap.Help, &params, .{});

    var simulation = structs.Simulation{
        .num_objects = args.num_objects orelse 3,
        .target_fps = args.target_fps orelse 30,
        .plane_half_size = args.plane_half_size orelse 7.5,
        .collect = if (args.collect != 0) true else false,
        .collect_wait_seconds = args.collect_wait_seconds orelse 1.0,
    };

    var ctx = try state.State.init(alloc, &simulation);
    // have to start workers here so they receive correct address for ctx
    try ctx.pool.startWorkers();
    // below clears the whole simulation context
    defer ctx.deinit(alloc);

    // (keep these for the exporter timing)
    var export_dt: f32 = 0.0;
    var dump_id: u32 = 0;

    // --- SIMULATION LOOP ---
    while (!rl.windowShouldClose()) {
        rl.updateCamera(&ctx.camera, ctx.camera_mode);

        const dt = rl.getFrameTime();
        const contact_point = rc.getCameraRayContactPoint(&ctx.camera, ctx.models.items);
        sim.handleKeys(&ctx, contact_point, dt, &simulation.debug);

        for (ctx.class_tx) |*dst| dst.clearRetainingCapacity();

        // 1. Build the contexts for this frame’s ray-casts
        rc.prepareRaycastContexts(&ctx, state.JITTER);

        // 2. let the pool run them
        const n_jobs = ctx.pool.contexts.len;
        ctx.pool.dispatch(n_jobs);
        ctx.pool.wait(); // blocks until all rays done

        // 3. Merge results
        const total_hit_count = rc.mergeThreadHits(ctx.thread_res.hits, ctx.class_tx, ctx.class_counter);

        if (simulation.collect) {
            if (export_dt >= simulation.collect_wait_seconds) {
                try sim.exportPCD(&ctx.exporter, ctx.class_tx, ctx.class_counter, dump_id);
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
            rl.beginMode3D(ctx.camera);
            defer rl.endMode3D();

            sim.draw3D(&ctx, &simulation);
        }

        sim.drawGUI(&ctx, &simulation, total_hit_count);
    }
}
