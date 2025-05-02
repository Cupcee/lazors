const rl = @import("raylib");
const std = @import("std");
const builtin = @import("builtin");
const rand = std.Random;
const s = @import("structs.zig");
const rlsimd = @import("raylib_simd.zig");
const Acquire = std.builtin.AtomicOrder.acquire;
const Release = std.builtin.AtomicOrder.release;
const AcqRel = std.builtin.AtomicOrder.acq_rel;
pub const CLASS_COUNT = 5;

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

fn toModelSpaceRaySIMD(ray_ws: rl.Ray, inv_simd: rlsimd.Mat4x4_SIMD) rl.Ray {
    // Load world-space ray vectors into SIMD registers
    const pos_ws_simd = rlsimd.vec3ToVec4W(ray_ws.position, 1.0); // w=1 for position
    const dir_ws_simd = rlsimd.vec3ToVec4W(ray_ws.direction, 0.0); // w=0 for direction

    // Transform position using SIMD
    const pos_ms_simd = rlsimd.transformSIMD(pos_ws_simd, inv_simd);

    // Transform direction using SIMD
    // Note: Direction transform ignores translation, which transformSIMD handles correctly if w_in=0
    const dir_ms_simd_unnormalized = rlsimd.transformSIMD(dir_ws_simd, inv_simd);

    // Normalize direction using SIMD
    const dir_ms_simd = rlsimd.normalizeSIMD(dir_ms_simd_unnormalized);

    return rl.Ray{
        .position = rlsimd.vec4ToVec3(pos_ms_simd),
        .direction = rlsimd.vec4ToVec3(dir_ms_simd),
    };
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

fn closestHitSIMD(ray_ws: rl.Ray, models: []const s.Object, max_range: f32) HitResult {
    var best = HitResult{ .distance = max_range };
    const ray_pos_ws_simd = rlsimd.vec3ToVec4W(ray_ws.position, 1.0); // Load once

    for (models) |model| {
        const c = rlsimd.getRayCollisionBoxSIMD(ray_ws, model.bbox_ws);
        if (!c.hit or c.distance > best.distance) continue;

        const inv_simd = model.inv_transform_simd;
        const transform_simd = model.transform_simd;

        const ray_ms = toModelSpaceRaySIMD(ray_ws, inv_simd);

        // Calculate max scale for t_max adjustment (scalar part, unchanged)
        const sx = @abs(inv_simd.col0[0]); // Access SIMD matrix components
        const sy = @abs(inv_simd.col1[1]);
        const sz = @abs(inv_simd.col2[2]);
        const max_scale = @max(@max(sx, sy), sz);
        const t_max = best.distance * max_scale;

        const maybe_hit = model.bvh.intersect(ray_ms, 1e-4, t_max);
        if (maybe_hit) |hit| {
            const splat_t: rlsimd.Vec4f = @splat(hit.t);
            const hit_ms_simd = rlsimd.vec3ToVec4W(ray_ms.position, 1.0) + rlsimd.vec3ToVec4W(ray_ms.direction, 0.0) * splat_t;
            const hit_ws_simd = rlsimd.transformSIMD(hit_ms_simd, transform_simd); // Use model's world transform SIMD matrix

            // Compare distance
            const dist_ws = rlsimd.distanceSIMD(hit_ws_simd, ray_pos_ws_simd);
            if (dist_ws < best.distance) {
                best = .{
                    .hit = true,
                    .distance = dist_ws,
                    .point = hit_ws_simd,
                    .hit_class = model.class,
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

    var hits = &ctx.thread_hits.*; // no extra indirections
    var hit_ix = hits.items.len; // we guaranteed capacity
    // Pre-calculate sensor SIMD transform matrix if it's constant for the worker
    // const sensor_transform_simd = rlsimd.Mat4x4_SIMD.fromRlMatrix(ctx.sensor.local_to_world);
    for (ctx.start_index..ctx.end_index) |global_i| {
        const local_i = global_i - ctx.start_index;

        const dir_local_simd = ctx.sensor.dirs[global_i];
        var dir_ws_simd = rlsimd.transformSIMD(dir_local_simd, ctx.sensor.local_to_world_simd);
        // Add jitter (using SIMD)
        // Create a jitter vector - Can potentially optimize random generation later
        const jitter_vec = rlsimd.Vec4f{
            (rng.float(f32) - 0.5) * ctx.jitter_scale,
            (rng.float(f32) - 0.5) * ctx.jitter_scale,
            0.0, // No jitter on Z as per original code
            0.0, // W remains 0 for direction
        };
        dir_ws_simd += jitter_vec;

        // Normalize the final direction
        const dir_norm_simd = rlsimd.normalizeSIMD(dir_ws_simd);

        const ray = rl.Ray{ .position = ctx.sensor.pos, .direction = rlsimd.vec4ToVec3(dir_norm_simd) };

        const nearest = closestHitSIMD(ray, ctx.models, ctx.sensor.max_range);
        const contact = if (nearest.hit) nearest.point else rlsimd.Vec4f{ 0, 0, 0, 1 };

        ctx.points_slice[local_i] = .{
            .xyz = contact,
            .hit = nearest.hit,
            .hit_class = nearest.hit_class,
        };

        if (nearest.hit) {
            const transform = rl.Matrix.translate(contact[0], contact[1], contact[2]);
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
