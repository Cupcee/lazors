//! lidar_sim.zig
//! zig run lidar_sim.zig -lc -lraylib
//! ------------------------------------
//! tiny LiDAR simulator using Raylib bindings for Zig
//! ------------------------------------

const std = @import("std");
const rand = std.Random;
const rl = @import("raylib"); // ziraylib package

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const a = gpa.allocator();

pub const Sensor = struct {
    pos: rl.Vector3 = .{ .x = 0, .y = 1.0, .z = 0 },
    fwd: rl.Vector3 = rl.Vector3{ .x = 0, .y = 0, .z = 1 },
    up: rl.Vector3 = rl.Vector3{ .x = 0, .y = 1, .z = 0 },

    fov_h_deg: f32 = 120, // horizontal cone angle
    fov_v_deg: f32 = 70, // vertical cone angle
    comptime res_h: usize = 800, // rays per row   (“density”)
    comptime res_v: usize = 192, // rays per column

    max_range: f32 = 25.0,
    debug: bool = false,

    /// return world-space unit vector for the (h,v) sample
    fn rayDir(self: *const Sensor, h: f32, v: f32) rl.Vector3 {
        const deg2rad = 0.0174532925;
        const res_h: f32 = @floatFromInt(self.res_h);
        const res_v: f32 = @floatFromInt(self.res_v);
        const az: f32 = ((h / (res_h - 1)) - 0.5) * self.fov_h_deg * deg2rad;
        const el: f32 = ((v / (res_v - 1)) - 0.5) * self.fov_v_deg * deg2rad;

        // local axes: forward, right, up
        const right = rl.Vector3.normalize(rl.Vector3.crossProduct(self.up, self.fwd));

        const dir = rl.Vector3.add(rl.Vector3.scale(self.fwd, std.math.cos(el) * std.math.cos(az)), rl.Vector3.add(
            rl.Vector3.scale(self.up, std.math.sin(el)),
            rl.Vector3.scale(right, std.math.cos(el) * std.math.sin(az)),
        ));
        return rl.Vector3.normalize(dir);
    }
};

pub const RayPoint = struct {
    xyz: rl.Vector3 = undefined,
    hit: bool = false,
};

fn buildScene() ![]const rl.Model {
    var list = try std.ArrayList(rl.Model).initCapacity(a, 3);
    errdefer { // run only on failure
        for (list.items) |m| rl.unloadModel(m);
        list.deinit();
    }

    const mesh1 = try rl.loadModelFromMesh(rl.genMeshCube(2.0, 2.0, 2.0));
    list.appendAssumeCapacity(mesh1);
    list.items[0].transform = rl.Matrix.translate(-4.0, 1.0, 8.0);

    const mesh2 = try rl.loadModelFromMesh(rl.genMeshCube(1.5, 3.0, 1.5));
    list.appendAssumeCapacity(mesh2);
    list.items[1].transform = rl.Matrix.translate(3.0, 1.5, 5.0);

    // big thin cube as ground
    const mesh3 = try rl.loadModelFromMesh(rl.genMeshCube(50.0, 0.2, 50.0));
    list.appendAssumeCapacity(mesh3);
    list.items[2].transform = rl.Matrix.translate(0.0, -0.1, 25.0);

    return try list.toOwnedSlice();
}

fn checkPointCollision(h: usize, v: usize, jitter_scale: f32, sensor: *Sensor, models: []const rl.Model, points: []RayPoint, rng: *const std.Random) void {
    const fh: f32 = @floatFromInt(h);
    const fv: f32 = @floatFromInt(v);
    const jh = fh + (rng.float(f32) - 0.5) * jitter_scale;
    const jv = fv + (rng.float(f32) - 0.5) * jitter_scale;
    const dir = sensor.rayDir(jh, jv);
    const ray = rl.Ray{ .position = sensor.pos, .direction = dir };

    var closest = sensor.max_range;
    var contactPoint = RayPoint{};
    // find closest mesh with contact, if any
    for (models) |model| {
        const mesh = model.meshes[0];
        const rc = rl.getRayCollisionMesh(ray, mesh, model.transform);
        if (!rc.hit) continue;
        if (rc.distance < closest and rc.distance < sensor.max_range) {
            closest = rc.distance;
            contactPoint = RayPoint{ .xyz = rc.point, .hit = true };
        }
    }
    const idx = v * sensor.res_h + h;
    points[idx] = contactPoint;
}

pub fn main() anyerror!void {
    defer _ = gpa.deinit();

    var prng = rand.DefaultPrng.init(@intCast(std.time.nanoTimestamp()));
    const rng = prng.random();

    rl.initWindow(1280, 720, "lazors");
    rl.disableCursor();
    rl.setTargetFPS(60);

    var camera = rl.Camera3D{
        .position = .{ .x = 0, .y = 4, .z = -8 },
        .target = .{ .x = 0, .y = 1, .z = 5 },
        .up = .{ .x = 0, .y = 1, .z = 0 },
        .fovy = 45,
        .projection = rl.CameraProjection.perspective,
    };
    const cameraMode = rl.CameraMode.free;
    const colors: [3]rl.Color = .{ rl.Color.red, rl.Color.green, rl.Color.blue };

    var models = try buildScene();
    var sensor = Sensor{};

    // --- once, after InitWindow ----------------------------
    // low-poly sphere; 4×4 rings are plenty at 0.02 m radius
    var sphereMesh = rl.genMeshSphere(0.02, 4, 4);
    rl.uploadMesh(&sphereMesh, false); // once, right after the GenMesh
    defer rl.unloadMesh(sphereMesh);
    const vsPath = "resources/shaders/glsl330/lighting_instancing.vs";
    const fsPath = "resources/shaders/glsl330/lighting.fs";
    const instShader = try rl.loadShader(vsPath, fsPath);
    // a bit of ambient so the spheres aren’t pitch-black
    const ambientLoc = rl.getShaderLocation(instShader, "ambient");
    const ambient = [_]f32{ 0.25, 0.25, 0.25, 1.0 };
    rl.setShaderValue(instShader, ambientLoc, &ambient, rl.ShaderUniformDataType.vec4);

    // build a material that uses that shader
    var instMat = try rl.loadMaterialDefault();
    instMat.shader = instShader;
    instMat.maps[@intFromEnum(rl.MaterialMapIndex.albedo)].color = rl.Color.sky_blue;
    defer rl.unloadMaterial(instMat);

    const MAX_HITS = sensor.res_h * sensor.res_v;
    var transforms: [MAX_HITS]rl.Matrix = undefined;

    // var points = try std.ArrayList(rl.Vector3).initCapacity(a, sensor.res_v * sensor.res_h);
    var points: [MAX_HITS]RayPoint = undefined;

    const jitter_scale: f32 = 1.0;

    while (!rl.windowShouldClose()) {
        rl.updateCamera(&camera, cameraMode);
        // --- update sensor pose (orbit with arrow keys) ----
        const dt = rl.getFrameTime();
        if (rl.isKeyDown(rl.KeyboardKey.left)) sensor.pos.x -= 5 * dt;
        if (rl.isKeyDown(rl.KeyboardKey.right)) sensor.pos.x += 5 * dt;
        if (rl.isKeyDown(rl.KeyboardKey.up)) sensor.pos.z += 5 * dt;
        if (rl.isKeyDown(rl.KeyboardKey.down)) sensor.pos.z -= 5 * dt;
        if (rl.isKeyReleased(rl.KeyboardKey.tab)) sensor.debug = !sensor.debug;

        for (0..sensor.res_v) |v| {
            for (0..sensor.res_h) |h| {
                checkPointCollision(h, v, jitter_scale, &sensor, models[0..], points[0..], &rng);
            }
        }

        // --- draw everything --------------------------------
        rl.beginDrawing();
        rl.clearBackground(rl.Color.ray_white);

        rl.beginMode3D(camera);
        // 1) helpful grid on the ground
        rl.drawGrid(20, 1.0);

        // 3) draw the “real” cubes with tints + wireframes
        for (models, 0..) |model, idx| {
            const tint = colors[idx % colors.len];
            rl.drawModel(model, rl.Vector3.zero(), 1, tint);
            rl.drawModelWires(model, rl.Vector3.zero(), 1, rl.Color.black);
        }
        rl.drawSphere(sensor.pos, 0.07, rl.Color.black); // sensor origin

        var hitCount: usize = 0;
        if (sensor.debug) {
            for (points) |p| {
                if (p.hit) {
                    transforms[hitCount] = rl.Matrix.translate(p.xyz.x, p.xyz.y, p.xyz.z);
                    hitCount += 1;
                }
            }

            if (hitCount > 0)
                rl.drawMeshInstanced(sphereMesh, instMat, transforms[0..hitCount]); // slice with the live instances
        }
        rl.endMode3D();

        rl.drawFPS(10, 10);
        if (sensor.debug) {
            rl.drawText(rl.textFormat("hitCount: %02i", .{hitCount}), 10, 30, 20, rl.Color.dark_gray);
        }
        rl.endDrawing();
    }
    for (models) |*m| rl.unloadModel(m.*);
    rl.closeWindow();
}
