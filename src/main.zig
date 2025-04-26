// ----------------------------------------------------
// tiny LiDAR simulator using Raylib bindings for Zig
// runtime-configurable resolution version
//  + bounding-box culling and analytic ground plane (steps 1-3)
// ----------------------------------------------------

const std = @import("std");
const rand = std.Random;
const rl = @import("raylib"); // ziraylib package

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
    debug: bool = false,

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
    pub fn init(allocator: std.mem.Allocator, res_h: usize, res_v: usize) !Sensor {
        var self = Sensor{
            .allocator = allocator,
            .res_h = res_h,
            .res_v = res_v,
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
fn pushObject(mesh: rl.Mesh, class: u32, color: rl.Color, transform: rl.Matrix, list: *std.ArrayListAligned(Object, null)) !void {
    var mdl = try rl.loadModelFromMesh(mesh);
    mdl.transform = transform;
    const local_bb = rl.getMeshBoundingBox(mesh);
    const world_bb = transformBBox(local_bb, transform);
    list.appendAssumeCapacity(Object{ .model = mdl, .class = class, .color = color, .bbox_ws = world_bb });
}

fn buildScene() ![]const Object {
    var list = try std.ArrayList(Object).initCapacity(alloc, 4);
    errdefer {
        for (list.items) |m| rl.unloadModel(m.model);
        list.deinit();
    }
    // cube
    try pushObject(rl.genMeshCube(2, 2, 2), 1, rl.Color.dark_gray, rl.Matrix.translate(-4, 1, 8), &list);
    // sphere
    try pushObject(rl.genMeshSphere(1.5, 12, 12), 2, rl.Color.dark_gray, rl.Matrix.translate(0, 2, 5), &list);
    // cylinder
    try pushObject(rl.genMeshCylinder(2, 4, 12), 3, rl.Color.dark_gray, rl.Matrix.translate(4, 0.1, 8), &list);
    // ground plane
    try pushObject(rl.genMeshCube(50, 0.2, 50), 0, rl.Color.beige, rl.Matrix.translate(0, -0.1, 25), &list);

    return try list.toOwnedSlice();
}

//------------------------------------------------------------------
// MAIN
//------------------------------------------------------------------
pub fn main() !void {
    defer _ = gpa.deinit();

    var prng = rand.DefaultPrng.init(@intCast(std.time.nanoTimestamp()));
    const rng = prng.random();

    rl.initWindow(1240, 800, "lazors");
    rl.disableCursor();
    rl.setTargetFPS(60);

    var camera = rl.Camera3D{
        .position = .{ .x = 0, .y = 2, .z = -8 },
        .target = .{ .x = 0, .y = 2, .z = 0 },
        .up = .{ .x = 0, .y = 1, .z = 0 },
        .fovy = 60,
        .projection = rl.CameraProjection.perspective,
    };
    const cameraMode = rl.CameraMode.free;

    const models = try buildScene();

    // pick any resolution you like:
    var sensor = try Sensor.init(alloc, 260, 70);
    defer sensor.deinit();
    sensor.updateLocalAxes(sensor.fwd, sensor.up);

    // hit-point sphere (one tiny sphere reused with instancing)
    var sphereMesh = rl.genMeshSphere(0.02, 4, 4);
    rl.uploadMesh(&sphereMesh, false);
    defer rl.unloadMesh(sphereMesh);

    const vsPath = "resources/shaders/glsl330/lighting_instancing_unlit.vs";
    const fsPath = "resources/shaders/glsl330/lighting_unlit.fs";
    const instShader = try rl.loadShader(vsPath, fsPath);
    instShader.locs[@intFromEnum(rl.ShaderLocationIndex.matrix_model)] =
        rl.getShaderLocation(instShader, "instanceTransform");

    // one material per class so we can give each of them a tint
    const instanceMatColors: [3]rl.Color = .{ rl.Color.black, rl.Color.red, rl.Color.green };
    const CLASS_COUNT = 4;
    var instMats: [CLASS_COUNT]rl.Material = undefined;
    for (&instMats, 0..) |*m, i| {
        m.* = try rl.loadMaterialDefault();
        m.*.shader = instShader;
        m.*.maps[@intFromEnum(rl.MATERIAL_MAP_DIFFUSE)].color = instanceMatColors[i];
    }

    // helper buffers: one transform array per class
    var classTx: [CLASS_COUNT][]rl.Matrix = undefined;
    for (&classTx) |*slot| {
        slot.* = try alloc.alloc(rl.Matrix, sensor.res_h * sensor.res_v);
    }
    defer for (classTx) |buf| alloc.free(buf);

    var classCount: [CLASS_COUNT]usize = .{ 0, 0, 0, 0 };

    const jitter_scale: f32 = 0.002;

    //------------------------------------------------------------------
    // GAME LOOP
    //------------------------------------------------------------------
    while (!rl.windowShouldClose()) {
        rl.updateCamera(&camera, cameraMode);

        // –– move the sensor with arrow keys ––
        const dt = rl.getFrameTime();
        if (rl.isKeyDown(rl.KeyboardKey.right)) sensor.pos.x -= 5 * dt;
        if (rl.isKeyDown(rl.KeyboardKey.left)) sensor.pos.x += 5 * dt;
        if (rl.isKeyDown(rl.KeyboardKey.up)) sensor.pos.z += 5 * dt;
        if (rl.isKeyDown(rl.KeyboardKey.down)) sensor.pos.z -= 5 * dt;
        if (rl.isKeyDown(rl.KeyboardKey.h)) sensor.yaw += sensor.turn_speed * dt;
        if (rl.isKeyDown(rl.KeyboardKey.l)) sensor.yaw -= sensor.turn_speed * dt;
        if (rl.isKeyDown(rl.KeyboardKey.k)) sensor.pitch += sensor.turn_speed * dt;
        if (rl.isKeyDown(rl.KeyboardKey.j)) sensor.pitch -= sensor.turn_speed * dt;
        if (rl.isKeyReleased(rl.KeyboardKey.tab)) sensor.debug = !sensor.debug;

        // keep pitch inside (-90°, +90°)
        const half_pi: f32 = std.math.pi / 2.0 - 0.001;
        sensor.pitch = std.math.clamp(sensor.pitch, -half_pi, half_pi);

        sensor.fwd = .{
            .x = std.math.sin(sensor.yaw) * std.math.cos(sensor.pitch),
            .y = std.math.sin(sensor.pitch),
            .z = std.math.cos(sensor.yaw) * std.math.cos(sensor.pitch),
        };
        sensor.up = .{ .x = 0, .y = 1, .z = 0 };
        sensor.updateLocalAxes(sensor.fwd, sensor.up);

        // –– raycast pass ––
        var hitCount: usize = 0;
        const nRays = sensor.res_h * sensor.res_v;
        @memset(&classCount, 0);

        for (0..nRays) |i| {
            // apply a tiny random offset to break up aliasing
            const dir_local = sensor.dirs[i];
            const dir_ws = rl.Vector3.transform(dir_local, sensor.local_to_world);
            const dir = rl.Vector3.normalize(.{
                .x = dir_ws.x + (rng.float(f32) - 0.5) * jitter_scale,
                .y = dir_ws.y + (rng.float(f32) - 0.5) * jitter_scale,
                .z = dir_ws.z,
            });

            const ray = rl.Ray{ .position = sensor.pos, .direction = dir };

            var closest: f32 = sensor.max_range;
            var contact: rl.Vector3 = undefined;
            var hit: bool = false;
            var hitClass: u32 = sensor.points[i].hitClass;

            for (models) |model| {
                const bc = rl.getRayCollisionBox(ray, model.bbox_ws);
                // prune search space by first checking for bbox collision
                if (!bc.hit) continue;
                if (bc.distance >= closest or bc.distance > sensor.max_range) continue;

                const rc = rl.getRayCollisionMesh(ray, model.model.meshes[0], model.model.transform);

                if (rc.hit and rc.distance < closest) {
                    closest = rc.distance;
                    contact = rc.point;
                    hit = true;
                    hitClass = model.class;
                }
            }

            sensor.points[i] = .{ .xyz = contact, .hit = hit, .hitClass = hitClass };
            if (hit) {
                const cls: usize = @intCast(hitClass);
                classTx[cls][classCount[cls]] =
                    rl.Matrix.translate(contact.x, contact.y, contact.z);
                classCount[cls] += 1;
                hitCount += 1;
            }
        }

        // –– render everything ––
        rl.beginDrawing();
        rl.clearBackground(rl.Color.ray_white);

        rl.beginMode3D(camera);

        rl.drawGrid(20, 1);
        for (models) |model| {
            rl.drawModel(model.model, rl.Vector3.zero(), 1, model.color);
            rl.drawModelWires(model.model, rl.Vector3.zero(), 1, rl.Color.black);
        }
        rl.drawSphere(sensor.pos, 0.07, rl.Color.black);

        if (sensor.debug) {
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
        rl.drawText("Camera: WASD, Left-CTRL, Space. Sensor: arrow keys.", 10, 30, 20, rl.Color.dark_gray);
        if (sensor.debug) {
            rl.drawText(rl.textFormat("hitCount: %04i", .{hitCount}), 10, 50, 20, rl.Color.dark_gray);
            for (classCount, 0..) |count, index| {
                const _c: i32 = @intCast(count);
                const _i: i32 = @intCast(index);
                rl.drawText(rl.textFormat("[class %i]: %i", .{ _i, _c }), 10, 70 + (_i * 20), 20, rl.Color.dark_gray);
            }
        }
        rl.endDrawing();
    }

    for (models) |*m| rl.unloadModel(m.*.model);
    rl.closeWindow();
}
