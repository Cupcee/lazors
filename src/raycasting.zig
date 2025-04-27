const rl = @import("raylib");
const std = @import("std");
const builtin = @import("builtin");
const rand = std.Random;
const s = @import("structs.zig");
const kd = @import("kdtree.zig");
const Acquire = std.builtin.AtomicOrder.acquire;
const Release = std.builtin.AtomicOrder.release;
const AcqRel = std.builtin.AtomicOrder.acq_rel;
pub const CLASS_COUNT = 4;

// ────────────────────────────────────────────────────────────────
//  DATA TYPES
// ────────────────────────────────────────────────────────────────

pub const ThreadHit = struct {
    hit_class: u32,
    transform: rl.Matrix,
};

pub const RaycastContext = struct {
    thread_id: usize,
    start_index: usize,
    end_index: usize,
    sensor: *const s.Sensor,
    models: []const s.Object,
    kdtree: *const kd.KDTree,
    jitter_scale: f32,
    thread_prng: *rand.DefaultPrng,
    thread_hits: *std.ArrayList(ThreadHit),
    points_slice: []s.RayPoint,
    skip: bool = false, // this allows skipping running the workload, if it is empty
};

/// Worker function for a thread.
pub fn raycastWorker(ctx: *const RaycastContext) void {
    const rng = ctx.thread_prng.random();

    var hits = &ctx.thread_hits.*; // no extra indirections
    var hit_ix = hits.items.len; // we guaranteed capacity
    for (ctx.start_index..ctx.end_index) |global_i| {
        const local_i = global_i - ctx.start_index;

        const dir_local = ctx.sensor.dirs[global_i];
        const dir_ws = rl.Vector3.transform(dir_local, ctx.sensor.local_to_world);
        // Small jitter to break up aliasing
        const dir = rl.Vector3.normalize(.{
            .x = dir_ws.x + (rng.float(f32) - 0.5) * ctx.jitter_scale,
            .y = dir_ws.y + (rng.float(f32) - 0.5) * ctx.jitter_scale,
            .z = dir_ws.z,
        });

        const ray = rl.Ray{ .position = ctx.sensor.pos, .direction = dir };

        // Naive version of looking for closest hits
        // var closest: f32 = ctx.sensor.max_range;
        // var contact: rl.Vector3 = undefined;
        // var hit: bool = false;
        // var hit_class: u32 = 0;
        // for (ctx.models) |model| {
        //     const bc = rl.getRayCollisionBox(ray, model.bbox_ws);
        //     if (!bc.hit or bc.distance >= closest) continue;
        //
        //     const rc = rl.getRayCollisionMesh(ray, model.model.meshes[0], model.model.transform);
        //     if (rc.hit and rc.distance < closest) {
        //         closest = rc.distance;
        //         contact = rc.point;
        //         hit = true;
        //         hit_class = model.class;
        //     }
        // }

        const nearest = ctx.kdtree.closestHit(ray, ctx.models, ctx.sensor.max_range);
        const contact = if (nearest.hit) nearest.point else rl.Vector3.zero();

        ctx.points_slice[local_i] = .{
            .xyz = contact,
            .hit = nearest.hit,
            .hit_class = nearest.hit_class,
        };

        if (nearest.hit) {
            const transform = rl.Matrix.translate(contact.x, contact.y, contact.z);
            // ctx.thread_hits.appendAssumeCapacity(.{
            //     .hit_class = hit_class,
            //     .transform = transform,
            // });
            hits.items[hit_ix] = .{ .hit_class = nearest.hit_class, .transform = transform };
            hit_ix += 1;
        }
    }
    hits.items.len = hit_ix;
}

/// Slice‐by-slice preparation of the contexts that each worker thread
/// will receive.  All storage lives *outside* the function so there
/// are no hidden allocations.
pub fn prepareRaycastContexts(
    contexts: []RaycastContext,
    sensor: *s.Sensor,
    models: []const s.Object,
    kdtree: *const kd.KDTree,
    jitter_scale: f32,
    thread_prngs: []rand.DefaultPrng,
    thread_hit_lists: []std.ArrayList(ThreadHit),
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
            .kdtree = kdtree,
            .jitter_scale = jitter_scale,
            .thread_prng = &thread_prngs[i],
            .thread_hits = &thread_hit_lists[i],
            .points_slice = sensor.points[start..end],
            .skip = start == end,
        };

        ray_idx = end;
    }
}

/// Merge per-thread hit‐lists into the per-class instance matrices and
/// return the total number of hits.
pub fn mergeThreadHits(
    thread_hit_lists: []const std.ArrayList(ThreadHit),
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

/// Keeps arrays that are *indexed by thread*
/// (hit lists, PRNGs, …).
pub const ThreadResources = struct {
    hits: []std.ArrayList(ThreadHit),
    prngs: []rand.DefaultPrng,

    // ───────────────────────────────────────
    //  life-cycle
    // ───────────────────────────────────────
    pub fn init(
        alloc: std.mem.Allocator,
        num_threads: usize,
        /// Worst-case number of ray hits per frame –
        /// lets us pre-size every list once.
        max_points: usize,
    ) !ThreadResources {
        // -------- 1. per-thread hit arrays  --------
        const cap = max_points / num_threads + 32;
        const lists = try alloc.alloc(std.ArrayList(ThreadHit), num_threads);
        for (lists) |*l| {
            l.* = try std.ArrayList(ThreadHit).initCapacity(alloc, cap);
            // try l.ensureTotalCapacity(cap);
        }

        // -------- 2. per-thread PRNGs  --------------
        const prngs = try alloc.alloc(rand.DefaultPrng, num_threads);
        for (prngs, 0..) |*p, i|
            p.* = rand.DefaultPrng.init(@intCast(std.time.nanoTimestamp() + i));

        return ThreadResources{
            .hits = lists,
            .prngs = prngs,
        };
    }

    /// Reset every hit-list ready for a new frame.
    pub fn clearHitLists(self: *ThreadResources) void {
        for (self.hits) |*l| l.clearRetainingCapacity();
    }

    pub fn deinit(self: *ThreadResources, alloc: std.mem.Allocator) void {
        for (self.hits) |*l| l.deinit();
        alloc.free(self.hits);
        alloc.free(self.prngs);
    }
};

pub fn getNumThreads() usize {
    return @max(1, std.Thread.getCpuCount() catch 1);
}
