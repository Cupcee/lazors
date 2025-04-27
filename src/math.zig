const rl = @import("raylib");

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
pub fn transformBBox(b: rl.BoundingBox, m: rl.Matrix) rl.BoundingBox {
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
