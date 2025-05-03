const std = @import("std");
const rl = @import("raylib"); // ziraylib package
const s = @import("structs.zig");
const pcd = @import("pcd_exporter.zig");
pub const INST_MAT_COLORS: [6]rl.Color = .{ rl.Color.black, rl.Color.red, rl.Color.green, rl.Color.yellow, rl.Color.blue, rl.Color.magenta };

pub fn sensorDt(sensor: *s.Sensor, dt: f32, debug: *bool) void {
    if (rl.isKeyDown(rl.KeyboardKey.right)) sensor.pos.x -= sensor.velocity * dt;
    if (rl.isKeyDown(rl.KeyboardKey.left)) sensor.pos.x += sensor.velocity * dt;
    if (rl.isKeyDown(rl.KeyboardKey.up)) sensor.pos.z += sensor.velocity * dt;
    if (rl.isKeyDown(rl.KeyboardKey.down)) sensor.pos.z -= sensor.velocity * dt;
    if (rl.isKeyDown(rl.KeyboardKey.k)) sensor.pos.y += sensor.velocity * dt;
    if (rl.isKeyDown(rl.KeyboardKey.j)) sensor.pos.y -= sensor.velocity * dt;
    if (rl.isKeyDown(rl.KeyboardKey.h)) sensor.yaw += sensor.turn_speed * dt;
    if (rl.isKeyDown(rl.KeyboardKey.l)) sensor.yaw -= sensor.turn_speed * dt;
    if (rl.isKeyReleased(rl.KeyboardKey.tab)) debug.* = !debug.*;

    const half_pi: f32 = std.math.pi / 2.0 - 0.001;
    sensor.pitch = std.math.clamp(sensor.pitch, -half_pi, half_pi);

    sensor.fwd = .{
        @sin(sensor.yaw) * @cos(sensor.pitch),
        @sin(sensor.pitch),
        @cos(sensor.yaw) * @cos(sensor.pitch),
        0,
    };
    sensor.up = .{ 0, 1, 0, 0 };
    sensor.updateLocalAxes(sensor.fwd, sensor.up);
}

pub fn initInstanceMats(alloc: std.mem.Allocator, num_classes: usize) ![]rl.Material {
    const vs_path = "resources/shaders/glsl330/lighting_instancing_unlit.vs";
    const fs_path = "resources/shaders/glsl330/lighting_unlit.fs";
    const inst_shader = try rl.loadShader(vs_path, fs_path);
    inst_shader.locs[@intFromEnum(rl.ShaderLocationIndex.matrix_model)] =
        rl.getShaderLocation(inst_shader, "instanceTransform");
    if (num_classes > INST_MAT_COLORS.len) {
        return error.TooFewClassesDefined;
    }

    const inst_mats = try alloc.alloc(rl.Material, num_classes);
    for (inst_mats, 0..) |*m, i| {
        m.* = try rl.loadMaterialDefault();
        m.*.shader = inst_shader;
        m.*.maps[@intFromEnum(rl.MATERIAL_MAP_DIFFUSE)].color = INST_MAT_COLORS[i];
    }
    return inst_mats;
}

pub fn initClassTxLists(
    alloc: std.mem.Allocator,
    num_classes: usize,
    reserve: usize,
) ![]std.ArrayList(rl.Matrix) {
    const lists = try alloc.alloc(std.ArrayList(rl.Matrix), num_classes);
    for (lists) |*l| {
        l.* = try std.ArrayList(rl.Matrix).initCapacity(alloc, reserve);
    }
    return lists;
}

pub fn initCamera() struct { rl.Camera, rl.CameraMode } {
    const camera = rl.Camera3D{
        .position = .{ .x = 0, .y = 2, .z = -8 },
        .target = .{ .x = 0, .y = 2, .z = 0 },
        .up = .{ .x = 0, .y = 1, .z = 0 },
        .fovy = 60,
        .projection = rl.CameraProjection.perspective,
    };
    const camera_mode = rl.CameraMode.free;
    return .{ camera, camera_mode };
}

pub fn drawGUI(
    simulation: *s.Simulation,
    class_counter: []usize,
    total_hit_count: usize,
) void {
    rl.drawFPS(10, 10);
    rl.drawText(
        "Camera: WASD, Left-CTRL, Space. Sensor: arrow keys, HJKL. TAB: Toggle Points",
        10,
        30,
        20,
        rl.Color.dark_gray,
    );
    if (simulation.debug) {
        rl.drawText(
            rl.textFormat("Total hitCount: %04i", .{total_hit_count}),
            10,
            50,
            20,
            rl.Color.dark_gray,
        );
        for (class_counter, 0..) |count, index| {
            const _c: i32 = @intCast(count);
            const _i: i32 = @intCast(index);
            rl.drawText(
                rl.textFormat("[class %i]: %i", .{ _i, _c }),
                10,
                70 + (_i * 20),
                20,
                INST_MAT_COLORS[index],
            );
        }
    } else {
        rl.drawText("Hit points hidden (Press TAB)", 10, 50, 20, rl.Color.dark_gray);
    }
}

pub fn draw3D(
    models: []const s.Object,
    collision_mesh: rl.Mesh,
    inst_mats: []rl.Material,
    class_tx: []std.ArrayList(rl.Matrix),
    class_counter: []usize,
    sensor: *s.Sensor,
    simulation: *s.Simulation,
) void {
    for (models) |model| {
        rl.drawModel(model.model, rl.Vector3.zero(), 1, model.color);
    }
    rl.drawSphere(sensor.pos, 0.07, rl.Color.black);

    if (simulation.debug) {
        for (0..class_counter.len) |cls| {
            if (class_counter[cls] > 0) {
                rl.drawMeshInstanced(
                    collision_mesh,
                    inst_mats[cls],
                    class_tx[cls].items,
                );
            }
        }
    }
}

pub fn exportPCD(
    exporter: *pcd.Exporter,
    class_tx: []std.ArrayList(rl.Matrix),
    class_counter: []usize,
    dump_id: u32,
) !void {
    var name_buf: [64]u8 = undefined; // buffer for PCD filenames...
    const fname = try std.fmt.bufPrint(&name_buf, "frames/scan_{d:0>7}.pcd", .{dump_id});
    try exporter.dump(fname, class_tx, class_counter);
}
