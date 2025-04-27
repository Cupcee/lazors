const rl = @import("raylib");
const std = @import("std");
const builtin = @import("builtin");
const rand = std.Random;
const s = @import("structs.zig");
const Acquire = std.builtin.AtomicOrder.acquire;
const Release = std.builtin.AtomicOrder.release;
const AcqRel = std.builtin.AtomicOrder.acq_rel;

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
};

// ────────────────────────────────────────────────────────────────
//  LOW-LEVEL WORK (unchanged from your original worker)
// ────────────────────────────────────────────────────────────────
pub fn raycastWorker(ctx: *const RaycastContext) void {
    const rng = ctx.thread_prng.random();

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

        var closest: f32 = ctx.sensor.max_range;
        var contact: rl.Vector3 = undefined;
        var hit: bool = false;
        var hit_class: u32 = 0;

        for (ctx.models) |model| {
            const bc = rl.getRayCollisionBox(ray, model.bbox_ws);
            if (!bc.hit or bc.distance >= closest) continue;

            const rc = rl.getRayCollisionMesh(ray, model.model.meshes[0], model.model.transform);
            if (rc.hit and rc.distance < closest) {
                closest = rc.distance;
                contact = rc.point;
                hit = true;
                hit_class = model.class;
            }
        }

        ctx.points_slice[local_i] = .{
            .xyz = contact,
            .hit = hit,
            .hit_class = hit_class,
        };

        if (hit) {
            const transform = rl.Matrix.translate(contact.x, contact.y, contact.z);
            ctx.thread_hits.append(.{
                .hit_class = hit_class,
                .transform = transform,
            }) catch unreachable; // capacity guaranteed by caller
        }
    }
}

// ────────────────────────────────────────────────────────────────
//  THREAD-POOL
// ────────────────────────────────────────────────────────────────
const State = enum(u8) { idle, working, shutdown };

/// All the memory we keep for the lifetime of the program
pub const ThreadResources = struct {
    // user-visible slices (unchanged)
    contexts: []RaycastContext,
    hits: []std.ArrayList(ThreadHit),
    prngs: []rand.DefaultPrng,

    // private pool machinery
    threads: []Thread,
    state: std.atomic.Value(State),
    next_job: std.atomic.Value(usize),
    job_count: std.atomic.Value(usize),
    done_count: std.atomic.Value(usize),

    wg: Thread.WaitGroup,
    workers_started: bool = false,

    // ───── initialise once ─────
    pub fn init(
        alloc: std.mem.Allocator,
        num_threads: usize,
        max_points: usize,
    ) !ThreadResources {
        // allocate storage ----------------------------------------------------
        const tr = ThreadResources{
            .threads = try alloc.alloc(Thread, num_threads),
            .contexts = try alloc.alloc(RaycastContext, num_threads),
            .hits = try alloc.alloc(std.ArrayList(ThreadHit), num_threads),
            .prngs = try alloc.alloc(rand.DefaultPrng, num_threads),

            .state = std.atomic.Value(State).init(.idle),
            .next_job = std.atomic.Value(usize).init(0),
            .job_count = std.atomic.Value(usize).init(0),
            .done_count = std.atomic.Value(usize).init(0),
            .wg = .{},
        };

        // per-thread helpers ---------------------------------------------------
        const cap = max_points / num_threads + 32;
        for (tr.hits) |*l| {
            l.* = std.ArrayList(ThreadHit).init(alloc);
            try l.ensureTotalCapacity(cap);
        }
        for (tr.prngs) |*p| {
            p.* = rand.DefaultPrng.init(@intCast(std.time.nanoTimestamp()));
        }
        return tr;
    }

    pub fn startWorkers(self: *ThreadResources) !void {
        if (self.workers_started) return error.AlreadyStarted;
        for (self.threads) |*t| t.* = try Thread.spawn(.{}, workerLoop, .{self});
        self.workers_started = true;
    }

    // ───── one call per frame ─────
    pub fn dispatch(self: *ThreadResources, job_count: usize) void {
        self.wg.reset();
        if (job_count == 0) return;
        self.wg.startMany(job_count);
        self.next_job.store(0, Release);
        self.state.store(.working, Release);
    }

    /// Busy-wait until every worker has called finish()
    pub fn wait(self: *ThreadResources) !void {
        self.wg.wait();
        self.state.store(.idle, Release);
    }

    // ───── shutdown at exit ─────
    pub fn deinit(self: *ThreadResources, alloc: std.mem.Allocator) void {
        // signal workers to exit
        self.state.store(.shutdown, Release);
        for (self.threads) |t| t.join();

        // free memory
        for (self.hits) |*l| l.*.deinit();
        alloc.free(self.hits);
        alloc.free(self.prngs);
        alloc.free(self.contexts);
        alloc.free(self.threads);
    }
};

/// Per-thread infinite loop
fn workerLoop(pool: *ThreadResources) void {
    while (true) {
        // park until somebody sets state=working
        while (pool.state.load(Acquire) == .idle) {
            Thread.yield() catch {};
        }

        if (pool.state.load(Acquire) == .shutdown) return;

        while (true) {
            const idx = pool.next_job.fetchAdd(1, AcqRel);
            if (idx >= pool.contexts.len) break;

            const ctx = &pool.contexts[idx];
            if (ctx.start_index != ctx.end_index) {
                raycastWorker(ctx); // do real work
            }
            pool.wg.finish();
        }
        std.Thread.yield() catch {};
    }
}

/// Utility: return how many contexts have real work this frame
pub fn countActive(contexts: []const RaycastContext) usize {
    var n: usize = 0;
    for (contexts) |c| {
        if (c.start_index != c.end_index) n += 1;
    }
    return n;
}

/// Handier alias so the caller’s code reads the same as before
pub const Thread = std.Thread;
pub const ThreadSpawn = std.Thread.spawn;

/// Expose same helper as before
pub fn getNumThreads() usize {
    return @max(1, std.Thread.getCpuCount() catch 1);
}
