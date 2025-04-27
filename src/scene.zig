const rl = @import("raylib");
const std = @import("std");
const rand = std.Random;
const s = @import("structs.zig");
const math = @import("math.zig");

//------------------------------------------------------------------
// SCENE CONSTRUCTION
//------------------------------------------------------------------
fn pushObject(
    mesh: rl.Mesh,
    class: u32,
    color: rl.Color,
    transform: rl.Matrix,
    list: *std.ArrayListAligned(s.Object, null),
) !void {
    var mdl = try rl.loadModelFromMesh(mesh);
    mdl.transform = transform;
    const local_bb = rl.getMeshBoundingBox(mesh);
    const world_bb = math.transformBBox(local_bb, transform);
    list.appendAssumeCapacity(
        s.Object{ .model = mdl, .class = class, .color = color, .bbox_ws = world_bb },
    );
}

pub fn buildScene(objectCount: usize, gpa: std.mem.Allocator) ![]const s.Object {
    // Pre-allocate for all dynamic objects + 1 ground plane
    var list = try std.ArrayList(s.Object).initCapacity(gpa, objectCount + 1);
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
