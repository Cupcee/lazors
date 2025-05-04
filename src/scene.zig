//  scene.zig  —  builds the world; every mesh is accelerated by an SIMD BVH
const rl = @import("raylib");
const std = @import("std");
const rand = std.Random;

const s = @import("structs.zig");
const math = @import("math.zig");
const bvh = @import("bvh.zig");
const simd = @import("raylib_simd.zig");

//──────────────────────────────────────────────────────────────
// Build BVH for a single Mesh
//──────────────────────────────────────────────────────────────
pub fn buildBVHFromMesh(alloc: std.mem.Allocator, mesh: rl.Mesh) !bvh.BVH {
    const vcount: usize = @intCast(mesh.vertexCount);
    const verts_f32 = @as([*]const f32, @ptrCast(mesh.vertices))[0 .. vcount * 3];

    var verts = try alloc.alloc(simd.Vec4f, vcount);
    defer alloc.free(verts);

    var i: usize = 0;
    while (i < vcount) : (i += 1) {
        const x = verts_f32[i * 3 + 0];
        const y = verts_f32[i * 3 + 1];
        const z = verts_f32[i * 3 + 2];
        verts[i] = .{ x, y, z, 1 };
    }

    const icount: usize = @intCast(mesh.triangleCount * 3);
    var indices = try alloc.alloc(u32, icount);
    defer alloc.free(indices);

    if (mesh.indices != null) {
        const src = @as([*]const u16, @ptrCast(mesh.indices))[0..icount];
        for (src, 0..) |idx16, k| indices[k] = idx16;
    } else {
        for (indices, 0..) |*dst, k| dst.* = @intCast(k);
    }

    return bvh.BVH.build(alloc, verts, indices);
}

//──────────────────────────────────────────────────────────────
// Build BVH over *several* meshes (flattened)
//──────────────────────────────────────────────────────────────
pub fn buildBVHFromMeshes(
    alloc: std.mem.Allocator,
    meshes: [*c]rl.Mesh,
    meshCount: usize,
) !bvh.BVH {
    const slice = meshes[0..meshCount];

    var total_v: usize = 0;
    var total_t: usize = 0;
    for (slice) |m| {
        total_v += @intCast(m.vertexCount);
        total_t += @intCast(m.triangleCount);
    }
    const total_i = total_t * 3;

    var verts = try alloc.alloc(simd.Vec4f, total_v);
    defer alloc.free(verts);
    var indices = try alloc.alloc(u32, total_i);
    defer alloc.free(indices);

    var v_off: usize = 0;
    var i_off: usize = 0;

    for (slice) |m| {
        // ---- verts ----
        const vcount: usize = @intCast(m.vertexCount);
        const src_v = @as([*]const f32, @ptrCast(m.vertices))[0 .. vcount * 3];
        var j: usize = 0;
        while (j < vcount) : (j += 1) {
            const x = src_v[j * 3 + 0];
            const y = src_v[j * 3 + 1];
            const z = src_v[j * 3 + 2];
            verts[v_off + j] = .{ x, y, z, 1 };
        }

        // ---- indices ----
        const tri_cnt: usize = @intCast(m.triangleCount);
        const idx_cnt = tri_cnt * 3;

        if (m.indices) |ptr| {
            const src_i = @as([*]const u16, @ptrCast(ptr))[0..idx_cnt];
            for (src_i, 0..) |idx16, k|
                indices[i_off + k] = idx16 + @as(u32, @intCast(v_off));
        } else {
            for (0..idx_cnt) |k|
                indices[i_off + k] = @as(u32, @intCast(v_off + k));
        }

        v_off += vcount;
        i_off += idx_cnt;
    }

    return bvh.BVH.build(alloc, verts, indices);
}

//──────────────────────────────────────────────────────────────
// Bounding boxes (scalar because Raylib needs them)
//──────────────────────────────────────────────────────────────
pub fn getMeshesBoundingBox(meshes: []const rl.Mesh) rl.BoundingBox {
    var bb = rl.getMeshBoundingBox(meshes[0]);
    for (meshes[1..]) |m| {
        const b = rl.getMeshBoundingBox(m);
        bb.min = rl.Vector3.min(bb.min, b.min);
        bb.max = rl.Vector3.max(bb.max, b.max);
    }
    return bb;
}
pub fn getMeshesBoundingBoxPtr(meshes: [*c]rl.Mesh, meshCount: usize) rl.BoundingBox {
    return getMeshesBoundingBox(meshes[0..meshCount]);
}

//──────────────────────────────────────────────────────────────
// Helper to push one Object into the scene list
//──────────────────────────────────────────────────────────────
fn pushObject(
    mesh: rl.Mesh,
    class: u32,
    color: rl.Color,
    transform: rl.Matrix,
    dst: *std.ArrayListAligned(s.Object, null),
    alloc: std.mem.Allocator,
) !void {
    const mesh_bvh = try buildBVHFromMesh(alloc, mesh);

    var mdl = try rl.loadModelFromMesh(mesh);
    mdl.transform = transform;

    const local_bb = rl.getMeshBoundingBox(mesh);
    const world_bb = math.transformBBox(local_bb, transform);
    const inv_tr = rl.Matrix.invert(transform);

    dst.appendAssumeCapacity(.{
        .model = mdl,
        .class = class,
        .color = color,
        .bbox_ws = simd.toBoundingBoxSIMD(world_bb),
        .bvh = mesh_bvh,
        .transform_simd = simd.Mat4x4_SIMD.fromRlMatrix(transform),
        .inv_transform_simd = simd.Mat4x4_SIMD.fromRlMatrix(inv_tr),
    });
}

//──────────────────────────────────────────────────────────────
// Build complete scene
//──────────────────────────────────────────────────────────────
pub fn buildScene(
    alloc: std.mem.Allocator,
    object_count: usize,
    num_classes: usize,
    plane_half_size: f32,
) ![]const s.Object {
    var objs = try std.ArrayList(s.Object).initCapacity(alloc, object_count + 1);
    errdefer {
        for (objs.items) |o| rl.unloadModel(o.model);
        objs.deinit();
    }

    var prng = rand.DefaultPrng.init(@intCast(std.time.nanoTimestamp()));
    const rng = prng.random();

    for (0..object_count) |i| {
        const kind = i % (num_classes - 1);
        const x = (rng.float(f32) * plane_half_size * 2) - plane_half_size;
        const z = rng.float(f32) * plane_half_size * 2;
        const y = rng.float(f32) * 3.0;

        const t = rl.Matrix.translate(x, y, z);

        switch (kind) {
            0 => try pushObject(rl.genMeshCube(2, 2, 2), 1, rl.Color.dark_gray, t, &objs, alloc),
            1 => try pushObject(rl.genMeshSphere(1.5, 12, 12), 2, rl.Color.dark_gray, t, &objs, alloc),
            2 => try pushObject(rl.genMeshCylinder(2, 4, 12), 3, rl.Color.dark_gray, t, &objs, alloc),
            3 => { // “grandpa” GLTF
                var mdl = try rl.loadModel("resources/objects/grandpa/scene.gltf");
                const scale = rl.Matrix.scale(0.01, 0.01, 0.01);
                const rx = rl.Matrix.rotateX(0.0174532925 * 90.0);
                mdl.transform = rl.Matrix.multiply(rx, rl.Matrix.multiply(scale, t));

                const bb_l = getMeshesBoundingBoxPtr(mdl.meshes, @intCast(mdl.meshCount));
                const bb_w = math.transformBBox(bb_l, mdl.transform);
                const mesh_bvh = try buildBVHFromMeshes(alloc, mdl.meshes, @intCast(mdl.meshCount));

                const inv = rl.Matrix.invert(mdl.transform);
                objs.appendAssumeCapacity(.{
                    .model = mdl,
                    .class = 4,
                    .color = rl.Color.dark_gray,
                    .bbox_ws = simd.toBoundingBoxSIMD(bb_w),
                    .bvh = mesh_bvh,
                    .transform_simd = simd.Mat4x4_SIMD.fromRlMatrix(mdl.transform),
                    .inv_transform_simd = simd.Mat4x4_SIMD.fromRlMatrix(inv),
                });
            },
            4 => { // “godzilla” GLTF
                var mdl = try rl.loadModel("resources/objects/godzilla/scene.gltf");
                mdl.transform = rl.Matrix.multiply(rl.Matrix.scale(1, 1, 1), t);

                const bb_l = getMeshesBoundingBoxPtr(mdl.meshes, @intCast(mdl.meshCount));
                const bb_w = math.transformBBox(bb_l, mdl.transform);
                const mesh_bvh = try buildBVHFromMeshes(alloc, mdl.meshes, @intCast(mdl.meshCount));

                const inv = rl.Matrix.invert(mdl.transform);
                objs.appendAssumeCapacity(.{
                    .model = mdl,
                    .class = 5,
                    .color = rl.Color.dark_gray,
                    .bbox_ws = simd.toBoundingBoxSIMD(bb_w),
                    .bvh = mesh_bvh,
                    .transform_simd = simd.Mat4x4_SIMD.fromRlMatrix(mdl.transform),
                    .inv_transform_simd = simd.Mat4x4_SIMD.fromRlMatrix(inv),
                });
            },
            else => unreachable,
        }
    }

    // ground plane
    try pushObject(
        rl.genMeshCube(plane_half_size * 2, 0.2, plane_half_size * 2),
        0,
        rl.Color.beige,
        rl.Matrix.translate(0, -0.1, plane_half_size),
        &objs,
        alloc,
    );

    return try objs.toOwnedSlice();
}
