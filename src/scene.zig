const rl = @import("raylib");
const std = @import("std");
const rand = std.Random;
const s = @import("structs.zig");
const math = @import("math.zig");
const bvh = @import("bvh.zig");

// TODO: support building from multiple meshes
/// Build a BVH for a *single* raylib Mesh.
/// The function copies the data so the BVH stays valid even if the Mesh is unloaded.
///
/// mesh      : pointer to the mesh you want to accelerate
/// alloc     : allocator used both for the copies *and* inside BVH.build()
pub fn buildBVHFromMesh(alloc: std.mem.Allocator, mesh: rl.Mesh) !bvh.BVH {
    // ---- vertices ----------------------------------------------------------
    const vcount: usize = @intCast(mesh.vertexCount);
    const verts_f32 =
        @as([*]const f32, @ptrCast(mesh.vertices))[0 .. vcount * 3];

    var verts = try alloc.alloc(rl.Vector3, vcount);
    var i: usize = 0;
    while (i < vcount) : (i += 1) {
        verts[i] = rl.Vector3{
            .x = verts_f32[i * 3 + 0],
            .y = verts_f32[i * 3 + 1],
            .z = verts_f32[i * 3 + 2],
        };
    }

    // ---- indices -----------------------------------------------------------
    const icount: u32 = @intCast(mesh.triangleCount * 3);
    var indices = try alloc.alloc(u32, icount);

    if (mesh.indices != null) {
        const src = @as([*]const u16, @ptrCast(mesh.indices))[0..icount];
        for (src, 0..) |idx16, k| indices[k] = idx16; // widen to u32
    } else {
        // Non-indexed mesh: triangles are laid out sequentially
        for (indices, 0..) |*dst, k| dst.* = @intCast(k);
    }

    // ---- build BVH ---------------------------------------------------------
    return bvh.BVH.build(alloc, verts, indices);
}

//------------------------------------------------------------------
// SCENE CONSTRUCTION
//------------------------------------------------------------------
fn pushObject(
    mesh: rl.Mesh,
    class: u32,
    color: rl.Color,
    transform: rl.Matrix,
    list: *std.ArrayListAligned(s.Object, null),
    alloc: std.mem.Allocator,
) !void {
    const mesh_bvh = try buildBVHFromMesh(alloc, mesh);
    var mdl = try rl.loadModelFromMesh(mesh);
    mdl.transform = transform;
    const local_bb = rl.getMeshBoundingBox(mesh);
    const world_bb = math.transformBBox(local_bb, transform);
    const obj = s.Object{
        .model = mdl,
        .class = class,
        .color = color,
        .bbox_ws = world_bb,
        .bvh = mesh_bvh,
        .inv_transform = rl.Matrix.invert(mdl.transform),
    };
    list.appendAssumeCapacity(obj);
}

pub fn buildScene(object_count: usize, alloc: std.mem.Allocator) ![]const s.Object {
    // Pre-allocate for all dynamic objects + 1 ground plane
    var list = try std.ArrayList(s.Object).initCapacity(alloc, object_count + 1);
    errdefer {
        for (list.items) |m| rl.unloadModel(m.model);
        list.deinit();
    }

    var prng = rand.DefaultPrng.init(@intCast(std.time.nanoTimestamp()));
    const rng = prng.random();
    const plane_half_size: f32 = 25.0;

    for (0..object_count) |i| {
        const kind = i % 4; // ← now four kinds
        const x = (rng.float(f32) * (plane_half_size * 2.0)) - plane_half_size;
        const z = rng.float(f32) * (plane_half_size * 2.0);
        const y = rng.float(f32) * 3.0;
        const transform = rl.Matrix.translate(x, y, z);

        switch (kind) {
            0 => {
                const mesh = rl.genMeshCube(2, 2, 2);
                try pushObject(mesh, 1, rl.Color.dark_gray, transform, &list, alloc);
            },
            1 => {
                const mesh = rl.genMeshSphere(1.5, 12, 12);
                try pushObject(mesh, 2, rl.Color.dark_gray, transform, &list, alloc);
            },
            2 => {
                const mesh = rl.genMeshCylinder(2, 4, 12);
                try pushObject(mesh, 3, rl.Color.dark_gray, transform, &list, alloc);
            },
            3 => {
                var mdl = try rl.loadModel("resources/objects/scene.gltf");
                // build a scale matrix
                const scale = 0.01;
                const mScale = rl.Matrix.scale(scale, scale, scale);
                // build a translate matrix for your desired world position
                const mTranslate = rl.Matrix.translate(x, y, z);

                // combine them so you scale first, then translate
                mdl.transform = rl.Matrix.multiply(mTranslate, mScale);

                // build BVH off the first mesh in the loaded model:
                const mesh = mdl.meshes[0];
                const local_bb = rl.getMeshBoundingBox(mesh);
                const world_bb = math.transformBBox(local_bb, transform);
                const mesh_bvh = try buildBVHFromMesh(alloc, mesh);

                const obj = s.Object{
                    .model = mdl,
                    .class = 4,
                    .color = rl.Color.dark_gray,
                    .bbox_ws = world_bb,
                    .bvh = mesh_bvh,
                    .inv_transform = rl.Matrix.invert(mdl.transform),
                };
                list.appendAssumeCapacity(obj);
            },
            else => unreachable,
        }
    }

    // ground plane as before
    try pushObject(
        rl.genMeshCube(plane_half_size * 2, 0.2, plane_half_size * 2),
        0,
        rl.Color.beige,
        rl.Matrix.translate(0, -0.1, plane_half_size),
        &list,
        alloc,
    );

    return try list.toOwnedSlice();
}
