const rl = @import("raylib");
const std = @import("std");
const rand = std.Random;
const s = @import("structs.zig");
const math = @import("math.zig");
const bvh = @import("bvh.zig");

/// Build a BVH for a *single* raylib Mesh.
/// The function copies the data so the BVH stays valid even if the Mesh is unloaded.
///
/// mesh      : pointer to the mesh you want to accelerate
/// alloc     : allocator used both for the copies *and* inside BVH.build()
pub fn buildBVHFromMesh(alloc: std.mem.Allocator, mesh: *const rl.Mesh) !bvh.BVH {
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
    const mesh_bvh = try buildBVHFromMesh(alloc, &mesh);
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
        // On error or exit, unload all models
        for (list.items) |m| rl.unloadModel(m.model);
        list.deinit();
    }

    // Seed our per-scene PRNG (nanoseconds as u64)
    var prng = rand.DefaultPrng.init(@intCast(std.time.nanoTimestamp()));
    const rng = prng.random();

    // Define how far out on the X/Z plane we’ll scatter objects
    const plane_half_size: f32 = 25.0;

    // Round-robin through Cube, Sphere, Cylinder
    for (0..object_count) |i| {
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
        const x = (rng.float(f32) * (plane_half_size * 2.0)) - plane_half_size;
        // random z in [0, planeHalfSize*2]
        const z = rng.float(f32) * (plane_half_size * 2.0);
        // random y in [0, 3]
        const y = rng.float(f32) * 3.0;

        const transform = rl.Matrix.translate(x, y, z);
        try pushObject(mesh, class, color, transform, &list, alloc);
    }

    // Finally: a big, flat ground plane
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
