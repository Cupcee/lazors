// -----------------------------------------------------------------------------
// Lazors – threaded binary‑PCD exporter               MIT licence
// Zig ≥ 0.12.0 – uses only std‑lib primitives
// -----------------------------------------------------------------------------
const std = @import("std");
const rl = @import("raylib");
const rc = @import("raycasting.zig");
// const CLASS_COUNT = rc.CLASS_COUNT;

/// A point aligned to 4‑byte boundaries → 16‑byte records
pub const Point = extern struct {
    x: f32,
    y: f32,
    z: f32,
    class: u32,
};

/// One dump job sent to the worker thread
const Task = struct {
    path: []const u8,
    pts: []Point,
};

/// Data that lives for the whole lifetime of the exporter and is shared
/// by both threads.  Allocated on the heap so its address never changes.
const Shared = struct {
    mutex: std.Thread.Mutex = .{},
    cond: std.Thread.Condition = .{},
    queue: std.ArrayListUnmanaged(Task) = .{},
    running: bool = true,
};

/// Public facade kept in the caller’s scope.  It owns a pointer to the
/// shared data and the worker Thread handle.
pub const Exporter = struct {
    alloc: std.mem.Allocator,
    shared: *Shared,
    worker: std.Thread,

    /// Create the exporter and start the worker thread.
    pub fn create(alloc: std.mem.Allocator) !Exporter {
        const shared = try alloc.create(Shared);
        shared.* = .{ .mutex = .{}, .cond = .{}, .queue = .{}, .running = true };
        const worker = try std.Thread.spawn(.{}, workerMain, .{ alloc, shared });
        return .{
            .alloc = alloc,
            .shared = shared,
            .worker = worker,
        };
    }

    /// Queue a frame for dumping.  Returns as soon as the task is en‑queued.
    pub fn dump(
        self: *Exporter,
        path: []const u8,
        class_tx: [][]rl.Matrix,
        class_counter: []usize,
    ) !void {
        // 1. Gather the points
        const pts = try collectPoints(self.alloc, class_tx, class_counter);

        // const _path = try self.alloc.alloc(u8, path.len);
        // @memcpy(_path, path);
        // 3. Push the task.
        const task = Task{ .path = path, .pts = pts };

        self.shared.mutex.lock();
        defer self.shared.mutex.unlock();
        try self.shared.queue.append(self.alloc, task);
        self.shared.cond.signal(); // wake the worker
    }

    /// Stop the worker and clean everything up.
    pub fn destroy(self: *Exporter) void {
        // Tell the worker to finish up
        self.shared.mutex.lock();
        self.shared.running = false;
        self.shared.cond.signal();
        self.shared.mutex.unlock();

        // Wait for it
        self.worker.join();

        // Free any tasks that might have remained (error paths, etc.)
        for (self.shared.queue.items) |t| {
            self.alloc.free(t.path);
            self.alloc.free(t.pts);
        }
        self.shared.queue.deinit(self.alloc);

        // Free the shared block
        self.alloc.destroy(self.shared);
    }
};

// -----------------------------------------------------------------------------
// Worker thread entry point
// -----------------------------------------------------------------------------
fn workerMain(alloc: std.mem.Allocator, shared: *Shared) !void {
    while (true) {
        shared.mutex.lock();
        while (shared.queue.items.len == 0 and shared.running)
            shared.cond.wait(&shared.mutex);

        if (shared.queue.items.len == 0 and !shared.running) {
            shared.mutex.unlock();
            break; // graceful shutdown
        }

        // Pop a task (LIFO is fine here)
        const task = shared.queue.pop();
        shared.mutex.unlock();

        // Do the actual file writing
        if (task != null) {
            writeBinaryPCD(task.?.path, task.?.pts) catch |err| {
                std.log.err("PCD write failed: {}", .{err});
            };

            // Free the buffers
            alloc.free(task.?.path);
            alloc.free(task.?.pts);
        }
    }
}

// -----------------------------------------------------------------------------
// Helper functions (unchanged except for allocator switch and u32 class field)
// -----------------------------------------------------------------------------
fn collectPoints(
    alloc: std.mem.Allocator,
    class_tx: [][]rl.Matrix,
    class_counter: []usize,
) ![]Point {
    var total: usize = 0;
    for (class_counter) |n| total += n;

    const buf = try alloc.alloc(Point, total);

    var i: usize = 0;
    for (0..class_counter.len) |cls| {
        for (class_tx[cls][0..class_counter[cls]]) |m| {
            buf[i] = .{
                .x = m.m12,
                .y = m.m13,
                .z = m.m14,
                .class = @as(u32, @intCast(cls)),
            };
            i += 1;
        }
    }
    return buf;
}

fn writeBinaryPCD(path: []const u8, pts: []Point) !void {
    var file = try std.fs.cwd().createFile(path, .{ .truncate = true });
    defer file.close();

    var hdr_buf: [256]u8 = undefined;
    const header = try std.fmt.bufPrint(&hdr_buf,
        \\# .PCD v0.7 (binary, xyz, class) – generated by lazors
        \\VERSION 0.7
        \\FIELDS x y z class
        \\SIZE 4 4 4 4
        \\TYPE F F F U
        \\COUNT 1 1 1 1
        \\WIDTH {d}
        \\HEIGHT 1
        \\VIEWPOINT 0 0 0 1 0 0 0
        \\POINTS {d}
        \\DATA binary
        \\
    , .{ pts.len, pts.len });

    try file.writeAll(header);
    try file.writeAll(std.mem.sliceAsBytes(pts));
}
