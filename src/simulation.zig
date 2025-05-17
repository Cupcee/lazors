const std = @import("std");
const rl = @import("raylib"); // ziraylib package
const structs = @import("structs.zig");
const pcd = @import("pcd_exporter.zig");
const state = @import("state.zig");
const rlsimd = @import("raylib_simd.zig"); // ziraylib package
const scene = @import("scene.zig");
pub const INST_MAT_COLORS: [6]rl.Color = .{ rl.Color.black, rl.Color.red, rl.Color.green, rl.Color.yellow, rl.Color.blue, rl.Color.magenta };
const EDITOR_TURN_SPEED = 2.0; // rad · s⁻¹
const EDITOR_SCALE_SPEED = 1.5; // 50 % per second
const deg2rad: f32 = 0.0174532925;

// ──────────────────────────────────────────────────────────────
//  helper: lift the mesh so it rests on the hit point
// ──────────────────────────────────────────────────────────────
fn toHitPointTx(item: state.ModelPlacerItem) rl.Matrix {
    return switch (item) {
        .cube => rl.Matrix.translate(0, 0.5, 0), // 1 × 1 × 1  → +½
        .cylinder => rl.Matrix.translate(0, 2.0, 0), // height 4  → +2
        .sphere => rl.Matrix.translate(0, 1.5, 0), // Ø 3       → +1.5
        .grandpa => rl.Matrix.identity(),
        else => rl.Matrix.identity(),
    };
}

/// Build the **local** transform for the preview / newly placed object.
/// Order for row‑vector convention: **scale → rotate → lift**
fn makeEditorTx(yaw: f32, scale: f32, item: state.ModelPlacerItem) rl.Matrix {
    const rot = rl.Matrix.rotateY(yaw);
    const scl = rl.Matrix.scale(scale, scale, scale);
    const lift = toHitPointTx(item); // applied last
    return rl.Matrix.multiply(rl.Matrix.multiply(scl, rot), lift);
}

pub fn handleKeys(ctx: *state.State, contact: ?rl.Vector3, dt: f32, debug: *bool) void {
    // ------------- camera controls (unchanged) -----------------
    if (rl.isKeyDown(rl.KeyboardKey.right)) ctx.sensor.pos[0] -= ctx.sensor.velocity * dt;
    if (rl.isKeyDown(rl.KeyboardKey.left)) ctx.sensor.pos[0] += ctx.sensor.velocity * dt;
    if (rl.isKeyDown(rl.KeyboardKey.up)) ctx.sensor.pos[2] += ctx.sensor.velocity * dt;
    if (rl.isKeyDown(rl.KeyboardKey.down)) ctx.sensor.pos[2] -= ctx.sensor.velocity * dt;
    if (!ctx.show_editor) {
        if (rl.isKeyDown(rl.KeyboardKey.k)) ctx.sensor.pos[1] += ctx.sensor.velocity * dt;
        if (rl.isKeyDown(rl.KeyboardKey.j)) ctx.sensor.pos[1] -= ctx.sensor.velocity * dt;
        if (rl.isKeyDown(rl.KeyboardKey.h)) ctx.sensor.yaw += ctx.sensor.turn_speed * dt;
        if (rl.isKeyDown(rl.KeyboardKey.l)) ctx.sensor.yaw -= ctx.sensor.turn_speed * dt;
    }

    // ------------- choose a primitive to place ----------------
    if (rl.isKeyDown(rl.KeyboardKey.one)) ctx.selected_model = state.ModelPlacerItem.cube;
    if (rl.isKeyDown(rl.KeyboardKey.two)) ctx.selected_model = state.ModelPlacerItem.cylinder;
    if (rl.isKeyDown(rl.KeyboardKey.three)) ctx.selected_model = state.ModelPlacerItem.sphere;
    if (rl.isKeyDown(rl.KeyboardKey.four)) ctx.selected_model = state.ModelPlacerItem.grandpa;

    if (rl.isKeyReleased(rl.KeyboardKey.tab)) debug.* = !debug.*;

    if (rl.isKeyDown(rl.KeyboardKey.left_shift) and contact != null) {
        if (!ctx.show_editor) {
            ctx.show_editor = true;
            // ctx.editor_yaw = 0;
            // ctx.editor_scale = 1.0;
        }

        // accumulate edits
        if (rl.isKeyDown(rl.KeyboardKey.h)) ctx.editor_yaw += EDITOR_TURN_SPEED * dt;
        if (rl.isKeyDown(rl.KeyboardKey.l)) ctx.editor_yaw -= EDITOR_TURN_SPEED * dt;
        if (rl.isKeyDown(rl.KeyboardKey.j)) ctx.editor_scale *= 1.0 + EDITOR_SCALE_SPEED * dt;
        if (rl.isKeyDown(rl.KeyboardKey.k)) ctx.editor_scale /= 1.0 + EDITOR_SCALE_SPEED * dt;

        // rebuild *every* frame – no floating‑point drift
        ctx.editor_tx = makeEditorTx(
            ctx.editor_yaw,
            ctx.editor_scale,
            ctx.selected_model,
        );

        // place the object with LMB
        if (rl.isMouseButtonPressed(rl.MouseButton.left)) {
            const pivotT = rl.Matrix.translate(contact.?.x, contact.?.y, contact.?.z);
            const world_tx = rl.Matrix.multiply(ctx.editor_tx, pivotT);

            switch (ctx.selected_model) {
                .cube => scene.pushObject(
                    rl.genMeshCube(1, 1, 1),
                    1,
                    rl.Color.dark_gray,
                    world_tx,
                    &ctx.models,
                    ctx.alloc,
                ) catch unreachable,
                .sphere => scene.pushObject(
                    rl.genMeshSphere(1.5, 12, 12),
                    2,
                    rl.Color.dark_gray,
                    world_tx,
                    &ctx.models,
                    ctx.alloc,
                ) catch unreachable,
                .cylinder => scene.pushObject(
                    rl.genMeshCylinder(2, 4, 12),
                    3,
                    rl.Color.dark_gray,
                    world_tx,
                    &ctx.models,
                    ctx.alloc,
                ) catch unreachable,
                .grandpa => scene.pushGLTF(
                    "resources/objects/grandpa/scene.gltf",
                    4,
                    world_tx,
                    &ctx.models,
                    ctx.alloc,
                ) catch unreachable,
                else => unreachable,
            }
        }
    } else { // ⇧ released → leave editor mode
        ctx.show_editor = false;
    }

    // ------- sensor orientation (unchanged) -------------------
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
    simulation: *structs.Simulation,
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

pub fn drawModelWithMatrix(model: rl.Model, world_tx: rl.Matrix) void {
    //    DrawModel(model, position,  scale,  tint);
    for (0..@intCast(model.meshCount)) |i| {
        rl.drawMesh(model.meshes[i], model.materials[i], world_tx);
    }
}

pub fn draw3D(
    ctx: *state.State,
    simulation: *structs.Simulation,
    contact: ?rl.Vector3,
) void {
    for (ctx.models.items) |model| {
        if (model.render) rl.drawModel(model.model, rl.Vector3.zero(), 1, model.color orelse rl.Color.ray_white);
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

    rl.drawModel(ctx.biome.object.model, .{ .x = 0, .y = 0, .z = 0 }, 1.0, rl.Color.white);

    // live preview of the placer object
    if (ctx.show_editor and contact != null) {
        const cp = contact.?;
        const pivot_t = rl.Matrix.translate(cp.x, cp.y, cp.z);

        switch (ctx.selected_model) {
            .cube => {
                const world_tx = rl.Matrix.multiply(pivot_t, ctx.editor_tx);
                rl.drawMesh(ctx.preview_meshes.cube, ctx.preview_mat, world_tx);
            },
            .cylinder => {
                const world_tx = rl.Matrix.multiply(pivot_t, ctx.editor_tx);
                rl.drawMesh(ctx.preview_meshes.cylinder, ctx.preview_mat, world_tx);
            },
            .sphere => {
                const world_tx = rl.Matrix.multiply(pivot_t, ctx.editor_tx);
                rl.drawMesh(ctx.preview_meshes.sphere, ctx.preview_mat, world_tx);
            },
            .grandpa => {
                // grandpa needs extra care because of the model file origin
                const T_to_cp = rl.Matrix.translate(cp.x, cp.y, cp.z);
                const T_from_cp = rl.Matrix.translate(-cp.x, -cp.y, -cp.z);

                const S = rl.Matrix.scale(0.01, 0.01, 0.01);
                const R = rl.Matrix.rotateX(std.math.pi / 2.0);

                // editor_tx already contains your yaw-about-Y and possible “lift”
                const SRL = rl.Matrix.multiply(rl.Matrix.multiply(S, R), ctx.editor_tx);

                // build the sandwich:  -P  →  SRL  →  +P
                var tx = rl.Matrix.multiply(T_from_cp, SRL); // -P then SRL
                tx = rl.Matrix.multiply(tx, T_to_cp); // finally +P

                drawModelWithMatrix(ctx.preview_meshes.grandpa, tx);
            },
            else => unreachable,
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
