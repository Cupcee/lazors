const rl = @import("raylib");
const std = @import("std");
const rand = std.Random;
const s = @import("structs.zig");
const math = @import("math.zig");
const bvh = @import("bvh.zig");
const rc = @import("raycasting.zig");
const rlsimd = @import("raylib_simd.zig");

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

/// Build one BVH over *all* meshes in a model by flattening their
/// vertex/index data into a single buffer.
///
/// alloc      : memory allocator
/// meshes     : C-pointer to raylib meshes
/// meshCount  : number of meshes the pointer refers to
///
/// The function copies every vertex and **every index** (three per
/// triangle) into new contiguous arrays so the BVH stays valid even
/// after the original meshes are unloaded.
pub fn buildBVHFromMeshes(
    alloc: std.mem.Allocator,
    meshes: [*c]rl.Mesh,
    meshCount: usize,
) !bvh.BVH {
    // Treat the C pointer + count as a Zig slice
    const mesh_slice = meshes[0..meshCount];

    // ────────────────────────────────────────────────
    // 1) Count how much memory we need
    // ────────────────────────────────────────────────
    var total_verts: usize = 0;
    var total_tris: usize = 0;
    for (mesh_slice) |mesh| {
        total_verts += @intCast(mesh.vertexCount);
        total_tris += @intCast(mesh.triangleCount);
    }
    const total_indices: usize = total_tris * 3;

    // ────────────────────────────────────────────────
    // 2) Allocate flat buffers
    // ────────────────────────────────────────────────
    var verts = try alloc.alloc(rl.Vector3, total_verts);
    var indices = try alloc.alloc(u32, total_indices);

    // ────────────────────────────────────────────────
    // 3) Copy vertices + (offset) indices mesh by mesh
    // ────────────────────────────────────────────────
    var v_off: usize = 0; // running vertex offset
    var i_off: usize = 0; // running index  offset

    for (mesh_slice) |mesh| {
        // -------- vertices --------
        const vcount: usize = @intCast(mesh.vertexCount);
        const src_verts =
            @as([*]const f32, @ptrCast(mesh.vertices))[0 .. vcount * 3];

        for (0..vcount) |vi| {
            verts[v_off + vi] = rl.Vector3{
                .x = src_verts[vi * 3 + 0],
                .y = src_verts[vi * 3 + 1],
                .z = src_verts[vi * 3 + 2],
            };
        }

        // -------- indices ---------
        const tri_count: usize = @intCast(mesh.triangleCount);
        const idx_count: usize = tri_count * 3; // *** three per triangle ***

        if (mesh.indices) |idx_ptr| {
            const src_idx16 =
                @as([*]const u16, @ptrCast(idx_ptr))[0..idx_count];

            for (src_idx16, 0..) |idx16, k| {
                // offset each index so it points into the *flat* vertex list
                const idx16c: u32 = @intCast(idx16);
                const voffc: u32 = @intCast(v_off);
                indices[i_off + k] = idx16c + voffc;
            }
        } else {
            // Non-indexed mesh: vertices appear sequentially, already 0..vcount-1
            for (0..idx_count) |k|
                indices[i_off + k] = @intCast(v_off + k);
        }

        // advance offsets for the next mesh
        v_off += vcount;
        i_off += idx_count;
    }

    // ────────────────────────────────────────────────
    // 4) Build a BVH over the combined buffers
    // ────────────────────────────────────────────────
    return bvh.BVH.build(alloc, verts, indices);
}

/// Compute the AABB of a non-empty slice of raylib Meshes (in local space).
pub fn getMeshesBoundingBox(meshes: []const rl.Mesh) rl.BoundingBox {
    // start with the first mesh’s box
    var bb = rl.getMeshBoundingBox(meshes[0]);

    // union in each subsequent mesh
    for (meshes[1..]) |m| {
        const m_bb = rl.getMeshBoundingBox(m);
        bb.min = rl.Vector3.min(bb.min, m_bb.min);
        bb.max = rl.Vector3.max(bb.max, m_bb.max);
    }

    return bb;
}

/// Overload for C-style pointer + count
pub fn getMeshesBoundingBoxPtr(
    meshes: [*c]rl.Mesh,
    meshCount: usize,
) rl.BoundingBox {
    return getMeshesBoundingBox(meshes[0..meshCount]);
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

    const inv_transform = rl.Matrix.invert(mdl.transform);
    const obj = s.Object{
        .model = mdl,
        .class = class,
        .color = color,
        .bbox_ws = world_bb,
        .bvh = mesh_bvh,
        .transform_simd = rlsimd.Mat4x4_SIMD.fromRlMatrix(mdl.transform),
        .inv_transform_simd = rlsimd.Mat4x4_SIMD.fromRlMatrix(inv_transform),
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
    const plane_half_size: f32 = 40.0;

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
                const scale_trans = rl.Matrix.scale(scale, scale, scale);
                mdl.transform = rl.Matrix.multiply(scale_trans, transform);

                const local_bb = getMeshesBoundingBoxPtr(mdl.meshes, @intCast(mdl.meshCount));
                const world_bb = math.transformBBox(local_bb, mdl.transform);
                const mesh_bvh = try buildBVHFromMeshes(alloc, mdl.meshes, @intCast(mdl.meshCount));

                const inv_transform = rl.Matrix.invert(mdl.transform);
                const obj = s.Object{
                    .model = mdl,
                    .class = 4,
                    .color = rl.Color.dark_gray,
                    .bbox_ws = world_bb,
                    .bvh = mesh_bvh,
                    .transform_simd = rlsimd.Mat4x4_SIMD.fromRlMatrix(mdl.transform),
                    .inv_transform_simd = rlsimd.Mat4x4_SIMD.fromRlMatrix(inv_transform),
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
