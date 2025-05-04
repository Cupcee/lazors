const rl = @import("raylib");
const std = @import("std");
const builtin = @import("builtin");
const rand = std.Random;
const s = @import("structs.zig");
const rlsimd = @import("raylib_simd.zig");
const Acquire = std.builtin.AtomicOrder.acquire;
const Release = std.builtin.AtomicOrder.release;
const AcqRel = std.builtin.AtomicOrder.acq_rel;
// pub const CLASS_COUNT = 5;

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
    jitter_scale: f32,
    thread_prng: *rand.DefaultPrng,
    thread_hits: *std.ArrayList(ThreadHit),
    points_slice: []s.RayPoint,
    skip: bool = false, // this allows skipping running the workload, if it is empty
};

const HitResult = struct {
    hit: bool = false,
    distance: f32 = 0,
    point: rlsimd.Vec4f = .{ 0, 0, 0, 1 },
    hit_class: u32 = 0,
};

fn toModelSpaceRaySIMD(ray_ws: rlsimd.RaySIMD, inv: rlsimd.Mat4x4_SIMD) rlsimd.RaySIMD {
    const o_ms = rlsimd.transformSIMD(ray_ws.origin, inv);
    const d_ms = rlsimd.normalizeSIMD(rlsimd.transformSIMD(ray_ws.dir, inv));
    return .{ .origin = o_ms, .dir = d_ms };
}

fn toModelSpaceRay(ray_ws: rl.Ray, inv: rl.Matrix) rl.Ray {
    // position: full 4×4 transform (w = 1)
    const pos = rl.Vector3.transform(ray_ws.position, inv);

    // direction: rotate & scale only (w = 0)
    var dir = rl.Vector3{
        .x = ray_ws.direction.x * inv.m0 + ray_ws.direction.y * inv.m4 + ray_ws.direction.z * inv.m8,
        .y = ray_ws.direction.x * inv.m1 + ray_ws.direction.y * inv.m5 + ray_ws.direction.z * inv.m9,
        .z = ray_ws.direction.x * inv.m2 + ray_ws.direction.y * inv.m6 + ray_ws.direction.z * inv.m10,
    };
    dir = rl.Vector3.normalize(dir); // ***-- keep *t* a distance in model-space
    return rl.Ray{ .position = pos, .direction = dir };
}

fn closestHitSIMD(ray_ws: rlsimd.RaySIMD, models: []const s.Object, max_range: f32) HitResult {
    var best = HitResult{ .distance = max_range };

    for (models) |m| {
        // very fast SIMD slab test against the model’s WS AABB
        if (!rlsimd.getRayCollisionBoxSIMD(ray_ws, m.bbox_ws).hit) continue;

        const ray_ms = toModelSpaceRaySIMD(ray_ws, m.inv_transform_simd);

        const sx = @abs(m.inv_transform_simd.col0[0]);
        const sy = @abs(m.inv_transform_simd.col1[1]);
        const sz = @abs(m.inv_transform_simd.col2[2]);
        const max_scale = @max(@max(sx, sy), sz);
        const t_max = best.distance * max_scale;

        if (m.bvh.intersect(ray_ms, 1e-4, t_max)) |hit| {
            const hit_ms = ray_ms.origin + ray_ms.dir * @as(rlsimd.Vec4f, @splat(hit.t));
            const hit_ws = rlsimd.transformSIMD(hit_ms, m.transform_simd);
            const dist = rlsimd.distanceSIMD(hit_ws, ray_ws.origin);

            if (dist < best.distance) {
                best = .{
                    .hit = true,
                    .distance = dist,
                    .point = hit_ws,
                    .hit_class = m.class,
                };
            }
        }
    }

    return best;
}

fn closestHit(ray_ws: rl.Ray, models: []const s.Object, max_range: f32) HitResult {
    var best = HitResult{ .distance = max_range };
    for (models) |model| {
        // std.debug.print("{d}\n", .{model.class});
        const c = rl.getRayCollisionBox(ray_ws, model.bbox_ws);
        if (!c.hit or c.distance > best.distance) continue;
        const inv = model.inv_transform; // cached
        const ray_ms = toModelSpaceRay(ray_ws, inv);
        const sx = @abs(inv.m0);
        const sy = @abs(inv.m5);
        const sz = @abs(inv.m10);
        const max_scale = @max(@max(sx, sy), sz);
        const t_max = best.distance * max_scale;

        if (model.bvh.intersect(ray_ms, 1e-4, t_max)) |hit| {
            const hit_ms = ray_ms.position.add(ray_ms.direction.scale(hit.t));
            const hit_ws = rl.Vector3.transform(hit_ms, model.model.transform);
            const dist_ws = hit_ws.distance(ray_ws.position);

            if (dist_ws < best.distance) {
                best = .{
                    .hit = true,
                    .distance = dist_ws,
                    .point = hit_ws,
                    .hit_class = model.class,
                };
            }
        }
    }
    return best;
}

/// Worker function for a thread.
pub fn raycastWorker(ctx: *const RaycastContext) void {
    const rng = ctx.thread_prng.random();
    var hits = &ctx.thread_hits.*;
    var ptr = hits.items.ptr;
    var n: usize = 0;

    for (ctx.start_index..ctx.end_index) |gidx| {
        const lidx = gidx - ctx.start_index;

        var dir_ws = rlsimd.transformSIMD(ctx.sensor.dirs[gidx], ctx.sensor.local_to_world_simd);
        dir_ws += .{ (rng.float(f32) - 0.5) * ctx.jitter_scale, (rng.float(f32) - 0.5) * ctx.jitter_scale, 0, 0 };
        dir_ws = rlsimd.normalizeSIMD(dir_ws);

        const ray = rlsimd.RaySIMD{ .origin = ctx.sensor.pos, .dir = dir_ws };
        const nearest = closestHitSIMD(ray, ctx.models, ctx.sensor.max_range);
        const contact = if (nearest.hit) nearest.point else rlsimd.Vec4f{ 0, 0, 0, 1 };

        ctx.points_slice[lidx] = .{ .xyz = contact, .hit = nearest.hit, .hit_class = nearest.hit_class };

        if (nearest.hit) {
            const t = rl.Matrix.translate(contact[0], contact[1], contact[2]);
            ptr[n] = .{ .hit_class = nearest.hit_class, .transform = t };
            n += 1;
        }
    }
    hits.items.len = n;
}

/// Slice‐by-slice preparation of the contexts that each worker thread
/// will receive.  All storage lives *outside* the function so there
/// are no hidden allocations.
pub fn prepareRaycastContexts(
    contexts: []RaycastContext,
    sensor: *s.Sensor,
    models: []const s.Object,
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
            .jitter_scale = jitter_scale,
            .thread_prng = &thread_prngs[i],
            .thread_hits = &thread_hit_lists[i],
            .points_slice = sensor.points[start..end],
            .skip = start == end,
        };

        ray_idx = end;
    }
}

pub fn mergeThreadHits(
    thread_hit_lists: []const std.ArrayList(ThreadHit),
    class_tx: []std.ArrayList(rl.Matrix),
    class_counter: []usize,
) usize {
    for (class_tx) |*dst| dst.clearRetainingCapacity();
    @memset(class_counter, 0);
    var total: usize = 0;

    for (thread_hit_lists) |list| {
        total += list.items.len;
        for (list.items) |hit| {
            const cls: usize = @intCast(hit.hit_class);
            if (cls >= class_tx.len) continue;

            var dst = &class_tx[cls];
            if (dst.items.len == dst.capacity) // grow rarely
                dst.ensureTotalCapacityPrecise(dst.capacity * 2) catch unreachable;
            dst.appendAssumeCapacity(hit.transform);
        }
    }

    // write counters for the UI
    for (class_counter, 0..) |*c, i| c.* = class_tx[i].items.len;
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
