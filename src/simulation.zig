const std = @import("std");
const rl = @import("raylib"); // ziraylib package
const s = @import("structs.zig");
const pcd = @import("pcd_exporter.zig");
const state = @import("state.zig");
const rlsimd = @import("raylib_simd.zig"); // ziraylib package
const scene = @import("scene.zig");
pub const INST_MAT_COLORS: [6]rl.Color = .{ rl.Color.black, rl.Color.red, rl.Color.green, rl.Color.yellow, rl.Color.blue, rl.Color.magenta };

pub fn handleKeys(ctx: *state.State, contact_pos: ?rl.Vector3, dt: f32, debug: *bool) void {
    if (rl.isKeyDown(rl.KeyboardKey.right)) ctx.sensor.pos[0] -= ctx.sensor.velocity * dt;
    if (rl.isKeyDown(rl.KeyboardKey.left)) ctx.sensor.pos[0] += ctx.sensor.velocity * dt;
    if (rl.isKeyDown(rl.KeyboardKey.up)) ctx.sensor.pos[2] += ctx.sensor.velocity * dt;
    if (rl.isKeyDown(rl.KeyboardKey.down)) ctx.sensor.pos[2] -= ctx.sensor.velocity * dt;
    if (rl.isKeyDown(rl.KeyboardKey.k)) ctx.sensor.pos[1] += ctx.sensor.velocity * dt;
    if (rl.isKeyDown(rl.KeyboardKey.j)) ctx.sensor.pos[1] -= ctx.sensor.velocity * dt;
    if (rl.isKeyDown(rl.KeyboardKey.h)) ctx.sensor.yaw += ctx.sensor.turn_speed * dt;
    if (rl.isKeyDown(rl.KeyboardKey.l)) ctx.sensor.yaw -= ctx.sensor.turn_speed * dt;
    if (rl.isKeyDown(rl.KeyboardKey.one)) ctx.selected_model = state.ModelPlacerItem.cube;
    if (rl.isKeyDown(rl.KeyboardKey.two)) ctx.selected_model = state.ModelPlacerItem.cylinder;
    if (rl.isKeyDown(rl.KeyboardKey.three)) ctx.selected_model = state.ModelPlacerItem.sphere;
    if (rl.isKeyReleased(rl.KeyboardKey.tab)) debug.* = !debug.*;
    if (rl.isKeyDown(rl.KeyboardKey.left_shift)) {
        if (contact_pos != null) {
            ctx.show_editor = true;
            if (rl.isMouseButtonPressed(rl.MouseButton.left)) {
                const t = rl.Matrix.translate(contact_pos.?.x, contact_pos.?.y, contact_pos.?.z);
                switch (ctx.selected_model) {
                    state.ModelPlacerItem.cube => {
                        scene.pushObject(rl.genMeshCube(1, 1, 1), 1, rl.Color.dark_gray, t, &ctx.models, ctx.alloc) catch unreachable;
                    },
                    state.ModelPlacerItem.cylinder => {
                        scene.pushObject(rl.genMeshCylinder(2, 4, 12), 1, rl.Color.dark_gray, t, &ctx.models, ctx.alloc) catch unreachable;
                    },
                    state.ModelPlacerItem.sphere => {
                        scene.pushObject(rl.genMeshSphere(1.5, 12, 12), 1, rl.Color.dark_gray, t, &ctx.models, ctx.alloc) catch unreachable;
                    },
                    else => unreachable,
                }
            }
        }
    } else {
        ctx.show_editor = false;
    }

    const half_pi: f32 = std.math.pi / 2.0 - 0.001;
    ctx.sensor.pitch = std.math.clamp(ctx.sensor.pitch, -half_pi, half_pi);

    ctx.sensor.fwd = .{
        @sin(ctx.sensor.yaw) * @cos(ctx.sensor.pitch),
        @sin(ctx.sensor.pitch),
        @cos(ctx.sensor.yaw) * @cos(ctx.sensor.pitch),
        0,
    };
    ctx.sensor.up = .{ 0, 1, 0, 0 };
    ctx.sensor.updateLocalAxes(ctx.sensor.fwd, ctx.sensor.up);
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
    ctx: *state.State,
    simulation: *s.Simulation,
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
    rl.drawText(
        rl.textFormat("Camera pos: [X: %.2f Y: %.2f Z: %.2f]", .{ ctx.camera.position.x, ctx.camera.position.y, ctx.camera.position.z }),
        10,
        50,
        20,
        rl.Color.dark_gray,
    );
    if (ctx.show_editor) {
        const mdl_name_cstr: [*c]const u8 = @ptrCast(@tagName(ctx.selected_model));
        rl.drawText(rl.textFormat("Now placing: %s", .{mdl_name_cstr}), state.WIN_W / 2 + 30, state.WIN_H / 2, 20, rl.Color.black);
    }
    if (simulation.debug) {
        rl.drawText(
            rl.textFormat("Total hitCount: %04i", .{total_hit_count}),
            10,
            70,
            20,
            rl.Color.dark_gray,
        );
        for (ctx.class_counter, 0..) |count, index| {
            const _c: i32 = @intCast(count);
            const _i: i32 = @intCast(index);
            rl.drawText(
                rl.textFormat("[class %i]: %i", .{ _i, _c }),
                10,
                90 + (_i * 20),
                20,
                INST_MAT_COLORS[index],
            );
        }
    } else {
        rl.drawText("Hit points hidden (Press TAB)", 10, 70, 20, rl.Color.dark_gray);
    }
}

pub fn draw3D(
    ctx: *state.State,
    simulation: *s.Simulation,
    contact_point: ?rl.Vector3,
) void {
    for (ctx.models.items) |model| {
        rl.drawModel(model.model, rl.Vector3.zero(), 1, model.color);
    }
    rl.drawSphere(rlsimd.vec4ToVec3(ctx.sensor.pos), 0.07, rl.Color.black);

    if (simulation.debug) {
        for (0..ctx.class_counter.len) |cls| {
            if (ctx.class_counter[cls] > 0) {
                rl.drawMeshInstanced(
                    ctx.collision,
                    ctx.inst_mats[cls],
                    ctx.class_tx[cls].items,
                );
            }
        }
    }

    if (contact_point != null) {
        if (ctx.show_editor) {
            switch (ctx.selected_model) {
                state.ModelPlacerItem.cube => {
                    rl.drawCube(contact_point.?, 1.0, 1.0, 1.0, rl.Color.dark_gray);
                },
                state.ModelPlacerItem.cylinder => {
                    rl.drawCylinder(contact_point.?, 2, 2, 4, 12, rl.Color.dark_gray);
                },
                state.ModelPlacerItem.sphere => {
                    rl.drawSphere(contact_point.?, 1.5, rl.Color.dark_gray);
                },
                else => unreachable,
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
