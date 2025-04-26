// ----------------------------------------------------
// tiny LiDAR simulator using Raylib bindings for Zig
// runtime-configurable resolution version
// + bounding-box culling and analytic ground plane (steps 1-3)
// ----------------------------------------------------

const std = @import("std");
const rand = std.Random;
const rl = @import("raylib"); // ziraylib package
const Thread = std.Thread;

//------------------------------------------------------------------
// GLOBAL ALLOCATOR
//------------------------------------------------------------------
var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const alloc = gpa.allocator();

//------------------------------------------------------------------
// HELPER MATH
//------------------------------------------------------------------
inline fn vecMin(a: rl.Vector3, b: rl.Vector3) rl.Vector3 {
    return .{ .x = @min(a.x, b.x), .y = @min(a.y, b.y), .z = @min(a.z, b.z) };
}
inline fn vecMax(a: rl.Vector3, b: rl.Vector3) rl.Vector3 {
    return .{ .x = @max(a.x, b.x), .y = @max(a.y, b.y), .z = @max(a.z, b.z) };
}

/// Transform an AABB by a general 4×4 matrix (slow but done *once*).
fn transformBBox(b: rl.BoundingBox, m: rl.Matrix) rl.BoundingBox {
    const corners = [_]rl.Vector3{
        .{ .x = b.min.x, .y = b.min.y, .z = b.min.z },
        .{ .x = b.min.x, .y = b.min.y, .z = b.max.z },
        .{ .x = b.min.x, .y = b.max.y, .z = b.min.z },
        .{ .x = b.min.x, .y = b.max.y, .z = b.max.z },
        .{ .x = b.max.x, .y = b.min.y, .z = b.min.z },
        .{ .x = b.max.x, .y = b.min.y, .z = b.max.z },
        .{ .x = b.max.x, .y = b.max.y, .z = b.min.z },
        .{ .x = b.max.x, .y = b.max.y, .z = b.max.z },
    };

    var new_min = rl.Vector3{ .x = 3.4e38, .y = 3.4e38, .z = 3.4e38 };
    var new_max = rl.Vector3{ .x = -3.4e38, .y = -3.4e38, .z = -3.4e38 };
    for (corners) |c| {
        const p = rl.Vector3.transform(c, m);
        new_min = vecMin(new_min, p);
        new_max = vecMax(new_max, p);
    }
    return .{ .min = new_min, .max = new_max };
}

//------------------------------------------------------------------
// BASIC TYPES
//------------------------------------------------------------------
const RayPoint = struct {
    xyz: rl.Vector3 = .{ .x = 0, .y = 0, .z = 0 },
    hit: bool = false,
    hitClass: u32 = 0,
};

const Object = struct {
    model: rl.Model,
    class: u32,
    color: rl.Color,
    /// world-space axis-aligned bounding box (step 1)
    bbox_ws: rl.BoundingBox,
};

//------------------------------------------------------------------
// SENSOR (runtime resolution)
//------------------------------------------------------------------
const Sensor = struct {
    // –– pose and optics ––
    pos: rl.Vector3 = .{ .x = 0, .y = 1, .z = 0 },
    fwd: rl.Vector3 = .{ .x = 0, .y = 0, .z = 1 },
    up: rl.Vector3 = .{ .x = 0, .y = 1, .z = 0 },
    yaw: f32 = 0,
    pitch: f32 = 0,
    turn_speed: f32 = std.math.pi,
    fov_h_deg: f32 = 120,
    fov_v_deg: f32 = 70,
    max_range: f32 = 70,

    // –– runtime resolution ––
    res_h: usize,
    res_v: usize,

    // –– working buffers ––
    dirs: []rl.Vector3, // pre-computed directions (sensor space)
    points: []RayPoint, // collision results (refilled every frame)
    transforms: []rl.Matrix, // used for instanced-draw spheres

    allocator: std.mem.Allocator,
    local_to_world: rl.Matrix,

    //--------------------------------------------------------------
    pub fn init(allocator: std.mem.Allocator, res_h: usize, res_v: usize, fov_h_deg: f32, fov_v_deg: f32) !Sensor {
        var self = Sensor{
            .allocator = allocator,
            .res_h = res_h,
            .res_v = res_v,
            .fov_h_deg = fov_h_deg,
            .fov_v_deg = fov_v_deg,
            .dirs = &.{},
            .points = &.{},
            .transforms = &.{},
            .local_to_world = rl.Matrix.identity(),
        };
        try self.allocateBuffers();
        self.precomputeDirs();
        return self;
    }

    pub fn deinit(self: *Sensor) void {
        self.allocator.free(self.dirs);
        self.allocator.free(self.points);
        self.allocator.free(self.transforms);
    }

    /// Change the grid size on the fly (buffers are re-created).
    pub fn setResolution(self: *Sensor, w: usize, h: usize) !void {
        if (w == self.res_h and h == self.res_v) return;
        self.allocator.free(self.dirs);
        self.allocator.free(self.points);
        self.allocator.free(self.transforms);
        self.res_h = w;
        self.res_v = h;
        try self.allocateBuffers();
        self.precomputeDirs();
    }

    /// Update `fwd`/`up` and build the 3×3 rotation matrix
    pub fn updateLocalAxes(self: *Sensor, fwd: rl.Vector3, up: rl.Vector3) void {
        self.fwd = rl.Vector3.normalize(fwd);
        self.up = rl.Vector3.normalize(up);
        const right = rl.Vector3.normalize(rl.Vector3.crossProduct(self.up, self.fwd));
        self.local_to_world = .{
            .m0 = right.x,
            .m1 = right.y,
            .m2 = right.z,
            .m3 = 0,
            .m4 = self.up.x,
            .m5 = self.up.y,
            .m6 = self.up.z,
            .m7 = 0,
            .m8 = self.fwd.x,
            .m9 = self.fwd.y,
            .m10 = self.fwd.z,
            .m11 = 0,
            .m12 = 0,
            .m13 = 0,
            .m14 = 0,
            .m15 = 1,
        };
    }

    //–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    fn allocateBuffers(self: *Sensor) !void {
        const n = self.res_h * self.res_v;
        self.dirs = try self.allocator.alloc(rl.Vector3, n);
        self.points = try self.allocator.alloc(RayPoint, n);
        self.transforms = try self.allocator.alloc(rl.Matrix, n);
    }

    /// Pay the sin/cos cost only once, when the grid size changes.
    fn precomputeDirs(self: *Sensor) void {
        const deg2rad: f32 = 0.0174532925;
        const res_hf: f32 = @floatFromInt(self.res_h - 1);
        const res_vf: f32 = @floatFromInt(self.res_v - 1);
        const h_step: f32 = (self.fov_h_deg * deg2rad) / res_hf;
        const v_step: f32 = (self.fov_v_deg * deg2rad) / res_vf;

        var idx: usize = 0;
        for (0..self.res_v) |v| {
            const fv: f32 = @floatFromInt(v);
            const el = (fv - (res_vf * 0.5)) * v_step;
            for (0..self.res_h) |h| {
                const fh: f32 = @floatFromInt(h);
                const az = (fh - (res_hf * 0.5)) * h_step;
                self.dirs[idx] = .{
                    .x = std.math.cos(el) * std.math.sin(az),
                    .y = std.math.sin(el),
                    .z = std.math.cos(el) * std.math.cos(az),
                };
                idx += 1;
            }
        }
    }
};

//------------------------------------------------------------------
// SCENE CONSTRUCTION
//------------------------------------------------------------------
fn pushObject(
    mesh: rl.Mesh,
    class: u32,
    color: rl.Color,
    transform: rl.Matrix,
    list: *std.ArrayListAligned(Object, null),
) !void {
    var mdl = try rl.loadModelFromMesh(mesh);
    mdl.transform = transform;
    const local_bb = rl.getMeshBoundingBox(mesh);
    const world_bb = transformBBox(local_bb, transform);
    list.appendAssumeCapacity(
        Object{ .model = mdl, .class = class, .color = color, .bbox_ws = world_bb },
    );
}

fn buildScene(objectCount: usize) ![]const Object {
    // Pre-allocate for all dynamic objects + 1 ground plane
    var list = try std.ArrayList(Object).initCapacity(alloc, objectCount + 1);
    errdefer {
        // On error or exit, unload all models
        for (list.items) |m| rl.unloadModel(m.model);
        list.deinit();
    }

    // Seed our per-scene PRNG (nanoseconds as u64)
    var prng = rand.DefaultPrng.init(@intCast(std.time.nanoTimestamp()));
    const rng = prng.random();

    // Define how far out on the X/Z plane we’ll scatter objects
    const planeHalfSize: f32 = 25.0;

    // Round-robin through Cube, Sphere, Cylinder
    for (0..objectCount) |i| {
        const kind = i % 3;
        var mesh: rl.Mesh = undefined;
        var class: u32 = 0;
        const color = rl.Color.dark_gray;

        switch (kind) {
            0 => {
                mesh = rl.genMeshCube(2, 2, 2);
                class = 1;
            },
            1 => {
                mesh = rl.genMeshSphere(1.5, 12, 12);
                class = 2;
            },
            2 => {
                mesh = rl.genMeshCylinder(2, 4, 12);
                class = 3;
            },
            else => unreachable,
        }

        // Random X and Z on top of ground plane, Y between 0 and 3
        // random x in [-planeHalfSize, planeHalfSize]
        const x = (rng.float(f32) * (planeHalfSize * 2.0)) - planeHalfSize;
        // random z in [0, planeHalfSize*2]
        const z = rng.float(f32) * (planeHalfSize * 2.0);
        // random y in [0, 3]
        const y = rng.float(f32) * 3.0;

        const transform = rl.Matrix.translate(x, y, z);
        try pushObject(mesh, class, color, transform, &list);
    }

    // Finally: a big, flat ground plane
    try pushObject(
        rl.genMeshCube(planeHalfSize * 2, 0.2, planeHalfSize * 2),
        0,
        rl.Color.beige,
        rl.Matrix.translate(0, -0.1, planeHalfSize),
        &list,
    );

    return try list.toOwnedSlice();
}

//------------------------------------------------------------------
// MULTITHREADING STRUCTURES
//------------------------------------------------------------------

/// Represents a single hit detected by a worker thread.
const ThreadHit = struct {
    hit_class: u32,
    transform: rl.Matrix,
};

/// Data passed to each worker thread.
const RaycastContext = struct {
    thread_id: usize,
    start_index: usize, // First ray index this thread handles
    end_index: usize, // Last ray index + 1 this thread handles
    sensor: *const Sensor, // Read-only access needed
    models: []const Object, // Read-only access needed
    jitter_scale: f32,
    thread_prng: *rand.DefaultPrng, // Each thread gets its own PRNG state

    // Output - each thread appends its hits here
    thread_hits: *std.ArrayList(ThreadHit),

    // Direct write access to the shared points buffer (safe due to non-overlapping ranges)
    points_slice: []RayPoint,
};

/// The function executed by each worker thread.
fn raycastWorker(ctx: *RaycastContext) void {
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

//------------------------------------------------------------------
// MAIN
//------------------------------------------------------------------
pub fn main() !void {
    defer _ = gpa.deinit();

    rl.initWindow(1240, 800, "lazors");
    rl.disableCursor();
    rl.setTargetFPS(20);
    var debug = true;

    var camera = rl.Camera3D{
        .position = .{ .x = 0, .y = 2, .z = -8 },
        .target = .{ .x = 0, .y = 2, .z = 0 },
        .up = .{ .x = 0, .y = 1, .z = 0 },
        .fovy = 60,
        .projection = rl.CameraProjection.perspective,
    };
    const cameraMode = rl.CameraMode.free;

    const models = try buildScene(50);
    defer for (models) |*m| rl.unloadModel(m.*.model);

    var sensor = try Sensor.init(alloc, 800, 192, 360, 70);
    defer sensor.deinit();
    sensor.updateLocalAxes(sensor.fwd, sensor.up);

    var sphereMesh = rl.genMeshSphere(0.02, 4, 4);
    rl.uploadMesh(&sphereMesh, false);
    defer rl.unloadMesh(sphereMesh);

    const vsPath = "resources/shaders/glsl330/lighting_instancing_unlit.vs";
    const fsPath = "resources/shaders/glsl330/lighting_unlit.fs";
    const instShader = try rl.loadShader(vsPath, fsPath);
    instShader.locs[@intFromEnum(rl.ShaderLocationIndex.matrix_model)] =
        rl.getShaderLocation(instShader, "instanceTransform");

    const instanceMatColors: [3]rl.Color = .{ rl.Color.black, rl.Color.red, rl.Color.green };
    const CLASS_COUNT = 4;
    var instMats: [CLASS_COUNT]rl.Material = undefined;
    for (&instMats, 0..) |*m, i| {
        m.* = try rl.loadMaterialDefault();
        m.*.shader = instShader;
        m.*.maps[@intFromEnum(rl.MATERIAL_MAP_DIFFUSE)].color = instanceMatColors[i];
    }

    var classTx: [CLASS_COUNT][]rl.Matrix = undefined;
    for (&classTx) |*slot| {
        slot.* = try alloc.alloc(rl.Matrix, sensor.res_h * sensor.res_v);
    }
    defer for (classTx) |buf| alloc.free(buf);

    var classCount: [CLASS_COUNT]usize = .{0} ** CLASS_COUNT;

    const jitter_scale: f32 = 0.005;

    const num_threads = blk: {
        const cpu_count = std.Thread.getCpuCount() catch 1;
        break :blk @max(1, cpu_count);
    };
    std.log.info("Using {} threads for raycasting.", .{num_threads});

    var threads = try alloc.alloc(Thread, num_threads);
    defer alloc.free(threads);

    var contexts = try alloc.alloc(RaycastContext, num_threads);
    defer alloc.free(contexts);

    var thread_hit_lists = try alloc.alloc(std.ArrayList(ThreadHit), num_threads);
    defer {
        for (thread_hit_lists) |*list| list.deinit();
        alloc.free(thread_hit_lists);
    }
    for (thread_hit_lists) |*list| {
        list.* = std.ArrayList(ThreadHit).init(alloc);
        try list.ensureTotalCapacity(sensor.res_h * sensor.res_v / num_threads + 10);
    }

    var thread_prngs = try alloc.alloc(rand.DefaultPrng, num_threads);
    defer alloc.free(thread_prngs);
    for (thread_prngs) |*rng_state| {
        rng_state.* = rand.DefaultPrng.init(@intCast(std.time.nanoTimestamp()));
    }

    while (!rl.windowShouldClose()) {
        rl.updateCamera(&camera, cameraMode);

        const dt = rl.getFrameTime();
        if (rl.isKeyDown(rl.KeyboardKey.right)) sensor.pos.x -= 5 * dt;
        if (rl.isKeyDown(rl.KeyboardKey.left)) sensor.pos.x += 5 * dt;
        if (rl.isKeyDown(rl.KeyboardKey.up)) sensor.pos.z += 5 * dt;
        if (rl.isKeyDown(rl.KeyboardKey.down)) sensor.pos.z -= 5 * dt;
        if (rl.isKeyDown(rl.KeyboardKey.k)) sensor.pos.y += 5 * dt;
        if (rl.isKeyDown(rl.KeyboardKey.j)) sensor.pos.y -= 5 * dt;
        if (rl.isKeyDown(rl.KeyboardKey.h)) sensor.yaw += sensor.turn_speed * dt;
        if (rl.isKeyDown(rl.KeyboardKey.l)) sensor.yaw -= sensor.turn_speed * dt;
        if (rl.isKeyReleased(rl.KeyboardKey.tab)) debug = !debug;

        const half_pi: f32 = std.math.pi / 2.0 - 0.001;
        sensor.pitch = std.math.clamp(sensor.pitch, -half_pi, half_pi);

        sensor.fwd = .{
            .x = std.math.sin(sensor.yaw) * std.math.cos(sensor.pitch),
            .y = std.math.sin(sensor.pitch),
            .z = std.math.cos(sensor.yaw) * std.math.cos(sensor.pitch),
        };
        sensor.up = .{ .x = 0, .y = 1, .z = 0 };
        sensor.updateLocalAxes(sensor.fwd, sensor.up);

        const nRays = sensor.res_h * sensor.res_v;
        var totalHitCount: usize = 0;

        for (thread_hit_lists) |*list| list.clearRetainingCapacity();

        const rays_per_thread = nRays / num_threads;
        var remaining_rays = nRays % num_threads;
        var current_ray_index: usize = 0;

        for (0..num_threads) |i| {
            var chunk_size = rays_per_thread;
            if (remaining_rays > 0) {
                chunk_size += 1;
                remaining_rays -= 1;
            }

            const start_index = current_ray_index;
            const end_index = @min(start_index + chunk_size, nRays);
            const points_slice = if (start_index < end_index)
                sensor.points[start_index..end_index]
            else
                sensor.points[0..0];

            contexts[i] = .{
                .thread_id = i,
                .start_index = start_index,
                .end_index = end_index,
                .sensor = &sensor,
                .models = models,
                .jitter_scale = jitter_scale,
                .thread_prng = &thread_prngs[i],
                .thread_hits = &thread_hit_lists[i],
                .points_slice = points_slice,
            };

            threads[i] = try Thread.spawn(.{}, raycastWorker, .{&contexts[i]});

            current_ray_index = end_index;
        }

        for (threads) |t| t.join();

        @memset(&classCount, 0);
        totalHitCount = 0;
        for (thread_hit_lists) |list| {
            totalHitCount += list.items.len;
            for (list.items) |hit| {
                const cls: usize = @intCast(hit.hit_class);
                if (cls < CLASS_COUNT) {
                    const current_idx = classCount[cls];
                    classTx[cls][current_idx] = hit.transform;
                    classCount[cls] += 1;
                } else {
                    std.log.warn(
                        "Hit with invalid class ID {} encountered during merge.",
                        .{cls},
                    );
                }
            }
        }

        rl.beginDrawing();
        rl.clearBackground(rl.Color.ray_white);

        rl.beginMode3D(camera);

        rl.drawGrid(20, 1);
        for (models) |model| {
            rl.drawModel(model.model, rl.Vector3.zero(), 1, model.color);
        }
        rl.drawSphere(sensor.pos, 0.07, rl.Color.black);

        if (debug) {
            for (0..CLASS_COUNT) |cls| {
                if (classCount[cls] > 0) {
                    rl.drawMeshInstanced(
                        sphereMesh,
                        instMats[cls],
                        classTx[cls][0..classCount[cls]],
                    );
                }
            }
        }

        rl.endMode3D();

        rl.drawFPS(10, 10);
        rl.drawText(
            "Camera: WASD, Left-CTRL, Space. Sensor: arrow keys, HJKL. TAB: Toggle Points",
            10,
            30,
            20,
            rl.Color.dark_gray,
        );
        if (debug) {
            rl.drawText(
                rl.textFormat("Total hitCount: %04i", .{totalHitCount}),
                10,
                50,
                20,
                rl.Color.dark_gray,
            );
            for (classCount, 0..) |count, index| {
                const _c: i32 = @intCast(count);
                const _i: i32 = @intCast(index);
                rl.drawText(
                    rl.textFormat("[class %i]: %i", .{ _i, _c }),
                    10,
                    70 + (_i * 20),
                    20,
                    instanceMatColors[index],
                );
            }
        } else {
            rl.drawText("Hit points hidden (Press TAB)", 10, 50, 20, rl.Color.dark_gray);
        }
        rl.endDrawing();
    }

    rl.closeWindow();
}
