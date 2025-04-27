const std = @import("std");
const AtomicOrder = std.builtin.AtomicOrder;

/// Create a pool for a particular job type + worker function
///     const MyPool = ThreadPool(MyContext, myWorkerFn);
pub fn ThreadPool(
    /// Per-job “context” type
    comptime Context: type,
    /// Worker callback: `fn (*const Context) void`
    comptime workerFn: fn (*const Context) void,
) type {
    const State = enum(u8) { idle, working, shutdown };

    return struct {
        // ───────── user-visible data ─────────
        /// One context slot per thread.  The caller mutates these
        /// every frame (just like you already do).
        contexts: []Context,
        /// Expose the number of threads so the caller can size
        /// its aux buffers (`hits`, PRNGs, …) however it wants.
        thread_count: usize,

        // ───────── internal machinery ─────────
        threads: []std.Thread,
        state: std.atomic.Value(State),
        next_job: std.atomic.Value(usize),
        wg: std.Thread.WaitGroup,
        started: bool = false,

        // ────────────────────────────────────
        //  life-cycle
        // ────────────────────────────────────
        pub fn init(
            alloc: std.mem.Allocator,
            num_threads: usize,
            /// The slice that the caller owns and will rewrite
            /// each frame (often `allocator.alloc(Context, num_threads)`).
            contexts: []Context,
        ) !@This() {
            return @This(){
                .contexts = contexts,
                .thread_count = num_threads,
                .threads = try alloc.alloc(std.Thread, num_threads),
                .state = std.atomic.Value(State).init(.idle),
                .next_job = std.atomic.Value(usize).init(0),
                .wg = .{},
            };
        }

        /// Spawn `num_threads` background workers (only once).
        pub fn startWorkers(self: *@This()) !void {
            if (self.started) return error.AlreadyStarted;
            for (self.threads) |*t| {
                t.* = try std.Thread.spawn(.{}, workerLoop, .{self});
            }
            self.started = true;
        }

        /// Tell the pool that *`job_count`* contexts contain work
        /// and let the workers rip.  (Call once per frame.)
        pub fn dispatch(self: *@This(), job_count: usize) void {
            self.wg.reset();
            if (job_count == 0) return; // nothing to do this frame
            self.wg.startMany(job_count);
            self.next_job.store(0, AtomicOrder.release);
            self.state.store(.working, AtomicOrder.release);
        }

        /// Block until every worker has called `finish()`.
        pub fn wait(self: *@This()) void {
            self.wg.wait();
            self.state.store(.idle, AtomicOrder.release);
        }

        /// Flush workers, join, free memory.
        pub fn deinit(self: *@This(), alloc: std.mem.Allocator) void {
            self.state.store(.shutdown, AtomicOrder.release);
            for (self.threads) |t| t.join();
            alloc.free(self.threads);
        }

        // ────────────────────────────────────
        //  private
        // ────────────────────────────────────
        fn workerLoop(pool: *@This()) void {
            while (true) {
                // sleep until somebody flips the big switch
                while (pool.state.load(AtomicOrder.acquire) == .idle)
                    std.Thread.yield() catch {};

                if (pool.state.load(AtomicOrder.acquire) == .shutdown)
                    return;

                while (true) {
                    const idx = pool.next_job.fetchAdd(1, AtomicOrder.acq_rel);
                    if (idx >= pool.contexts.len) break;

                    const ctx = &pool.contexts[idx];
                    if (ctx.skip) continue;
                    workerFn(ctx); // ← application work
                    pool.wg.finish();
                }
                std.Thread.yield() catch {};
            }
        }
    };
}
