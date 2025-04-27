const rl = @import("raylib");
const std = @import("std");
const rand = std.Random;
const s = @import("structs.zig");
const Thread = std.Thread;

//------------------------------------------------------------------
// MULTITHREADING STRUCTURES
//------------------------------------------------------------------

/// Represents a single hit detected by a worker thread.
pub const ThreadHit = struct {
    hit_class: u32,
    transform: rl.Matrix,
};

/// Data passed to each worker thread.
pub const RaycastContext = struct {
    thread_id: usize,
    start_index: usize, // First ray index this thread handles
    end_index: usize, // Last ray index + 1 this thread handles
    sensor: *const s.Sensor, // Read-only access needed
    models: []const s.Object, // Read-only access needed
    jitter_scale: f32,
    thread_prng: *rand.DefaultPrng, // Each thread gets its own PRNG state

    // Output - each thread appends its hits here
    thread_hits: *std.ArrayList(ThreadHit),

    // Direct write access to the shared points buffer (safe due to non-overlapping ranges)
    points_slice: []s.RayPoint,
};

/// The function executed by each worker thread.
pub fn raycastWorker(ctx: *RaycastContext) void {
    const rng = ctx.thread_prng.random(); // Use thread-local RNG

    // Process the assigned range of rays
    for (ctx.start_index..ctx.end_index) |i| {
        // Index relative to the thread's points_slice
        const slice_idx = i - ctx.start_index;

        // apply a tiny random offset to break up aliasing
        const dir_local = ctx.sensor.dirs[i];
        // Manually apply rotation part of local_to_world (translation comes from sensor.pos)
        const dir_ws = rl.Vector3.transform(dir_local, ctx.sensor.local_to_world);
        const dir = rl.Vector3.normalize(.{
            .x = dir_ws.x + (rng.float(f32) - 0.5) * ctx.jitter_scale,
            .y = dir_ws.y + (rng.float(f32) - 0.5) * ctx.jitter_scale,
            .z = dir_ws.z, // Assuming jitter only applied in x/y based on original code
        });

        const ray = rl.Ray{ .position = ctx.sensor.pos, .direction = dir };

        var closest: f32 = ctx.sensor.max_range;
        var contact: rl.Vector3 = undefined;
        var hit: bool = false;
        var hit_class: u32 = 0; // Default to 0 or some 'no hit' class if needed

        for (ctx.models) |model| {
            // Bounding box check (early out)
            const bc = rl.getRayCollisionBox(ray, model.bbox_ws);
            if (!bc.hit) continue;
            if (bc.distance >= closest or bc.distance > ctx.sensor.max_range) continue;

            // Precise mesh collision check
            const rc = rl.getRayCollisionMesh(ray, model.model.meshes[0], model.model.transform);

            if (rc.hit and rc.distance < closest) {
                closest = rc.distance;
                contact = rc.point;
                hit = true;
                hit_class = model.class;
            }
        }

        // Write result to the shared points buffer (safe - distinct index `i`)
        ctx.points_slice[slice_idx] = .{ .xyz = contact, .hit = hit, .hitClass = hit_class };

        // If hit, record it for later merging
        if (hit) {
            const transform = rl.Matrix.translate(contact.x, contact.y, contact.z);
            ctx.thread_hits.append(.{ .hit_class = hit_class, .transform = transform }) catch |err| {
                std.log.err("Failed to append hit in thread {}: {s}\n", .{ ctx.thread_id, @errorName(err) });
                return;
            };
        }
    }
}

pub fn getNumThreads() usize {
    return blk: {
        const cpu_count = Thread.getCpuCount() catch 1;
        break :blk @max(1, cpu_count);
    };
}
