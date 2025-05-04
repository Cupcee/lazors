const std = @import("std");
const rl = @import("raylib");

/// Four‑wide packed single‑precision vector
pub const Vec4f = @Vector(4, f32);

/// SIMD‑friendly ray (all maths now use this)
pub const RaySIMD = struct {
    origin: Vec4f, // w must be 1
    dir: Vec4f, // w must be 0 (always *normalised*)
};

pub const BoundingBoxSIMD = struct {
    min: Vec4f,
    max: Vec4f,
};

pub inline fn toBoundingBoxSIMD(bbox: rl.BoundingBox) BoundingBoxSIMD {
    return BoundingBoxSIMD{
        .min = vec3ToVec4W(bbox.min, 0),
        .max = vec3ToVec4W(bbox.max, 0),
    };
}

/// Axis‑aligned bounding box (SIMD)
pub const AABB_SIMD = struct {
    min: Vec4f,
    max: Vec4f,

    pub inline fn empty() AABB_SIMD {
        return .{ .min = @splat(std.math.inf(f32)), .max = @splat(-std.math.inf(f32)) };
    }

    pub inline fn expand(self: *AABB_SIMD, p: Vec4f) void {
        self.min = @min(self.min, p);
        self.max = @max(self.max, p);
    }

    pub inline fn unite(a: AABB_SIMD, b: AABB_SIMD) AABB_SIMD {
        return .{ .min = @min(a.min, b.min), .max = @max(a.max, b.max) };
    }

    /// Classic slab test – true when the segment [tMin,tMax] intersects.
    pub inline fn hit(box: AABB_SIMD, ray: RaySIMD, tMin: f32, tMax: f32) bool {
        const invDir = @as(Vec4f, @splat(1.0)) / ray.dir;

        const t0 = (box.min - ray.origin) * invDir;
        const t1 = (box.max - ray.origin) * invDir;

        const tmin_v = @min(t0, t1);
        const tmax_v = @max(t0, t1);

        // ‼️ Ignore lane 3 (w) – we only care about xyz.
        const tNear = @max(@max(tmin_v[0], tmin_v[1]), tmin_v[2]);
        const tFar = @min(@min(tmax_v[0], tmax_v[1]), tmax_v[2]);

        return tNear <= tFar and tFar >= tMin and tNear <= tMax;
    }

    /// Axis (0/1/2) with largest extent.
    pub inline fn largestExtentAxis(box: AABB_SIMD) u8 {
        const e = box.max - box.min;
        return if (e[0] > e[1] and e[0] > e[2]) 0 else if (e[1] > e[2]) 1 else 2;
    }
};

// Helper function to load a Vec3 into a Vec4
// Pad with 'w' (1.0 for position, 0.0 for direction)
pub inline fn vec3ToVec4W(v3: rl.Vector3, w: f32) Vec4f {
    return .{ v3.x, v3.y, v3.z, w };
}

// Helper to extract Vec3 from Vec4 (ignores w)
pub inline fn vec4ToVec3(v4: Vec4f) rl.Vector3 {
    return .{ .x = v4[0], .y = v4[1], .z = v4[2] };
}

// SIMD Matrix type (assuming column-major like OpenGL/Raylib for this example)
pub const Mat4x4_SIMD = extern struct {
    // Store columns as Vec4f for efficient SIMD access
    col0: Vec4f,
    col1: Vec4f,
    col2: Vec4f,
    col3: Vec4f,

    // Load from Raylib Matrix
    pub fn fromRlMatrix(m: rl.Matrix) Mat4x4_SIMD {
        return .{
            .col0 = .{ m.m0, m.m1, m.m2, m.m3 },
            .col1 = .{ m.m4, m.m5, m.m6, m.m7 },
            .col2 = .{ m.m8, m.m9, m.m10, m.m11 },
            .col3 = .{ m.m12, m.m13, m.m14, m.m15 },
        };
    }

    // Convert back to Raylib Matrix (if needed)
    pub fn toRlMatrix(self: Mat4x4_SIMD) rl.Matrix {
        return .{
            .m0 = self.col0[0],
            .m1 = self.col0[1],
            .m2 = self.col0[2],
            .m3 = self.col0[3],
            .m4 = self.col1[0],
            .m5 = self.col1[1],
            .m6 = self.col1[2],
            .m7 = self.col1[3],
            .m8 = self.col2[0],
            .m9 = self.col2[1],
            .m10 = self.col2[2],
            .m11 = self.col2[3],
            .m12 = self.col3[0],
            .m13 = self.col3[1],
            .m14 = self.col3[2],
            .m15 = self.col3[3],
        };
    }
};

// SIMD Vector * Matrix transform function
// v_in should have w=1 for position, w=0 for direction
pub fn transformSIMD(v_in: Vec4f, m: Mat4x4_SIMD) Vec4f {
    // Use broadcast/splat and FMA (fused multiply-add) style calculation
    // result = m.col0 * v_in[0] + m.col1 * v_in[1] + m.col2 * v_in[2] + m.col3 * v_in[3];
    const xxxx: Vec4f = @splat(v_in[0]);
    const yyyy: Vec4f = @splat(v_in[1]);
    const zzzz: Vec4f = @splat(v_in[2]);
    const wwww: Vec4f = @splat(v_in[3]); // w component from input vector

    var result: Vec4f = m.col0 * xxxx;
    result += m.col1 * yyyy;
    result += m.col2 * zzzz;
    result += m.col3 * wwww; // Add translation * w

    return result;
}

// SIMD Dot Product (including W component, useful for length squared)
pub fn dot4SIMD(a: Vec4f, b: Vec4f) f32 {
    const product = a * b;
    return @reduce(.Add, product);
}

// SIMD Dot Product (3 components only)
pub fn dot3SIMD(a: Vec4f, b: Vec4f) f32 {
    const product: Vec4f = a * b;
    return product[0] + product[1] + product[2];
}

// SIMD Normalization (handles zero vector)
pub fn normalizeSIMD(v: Vec4f) Vec4f {
    const epsilon: f32 = 1e-12; // Small value to avoid division by zero
    const dot = dot3SIMD(v, v); // Use dot3 for length of 3D part
    if (dot < epsilon) {
        return @splat(0.0); // Return zero vector if length is near zero
    }
    // Use rsqrt approximation + refinement or direct sqrt
    // Option 1: Direct Sqrt (often mapped to hardware sqrt)
    const inv_length: f32 = 1.0 / @sqrt(dot);

    // Option 2: Reciprocal Sqrt (can be faster on some HW, less precise initially)
    // const rsqrt_approx = @sqrt(f32, dot); // This needs Zig's intrinsic name for rsqrt if available, else use std.math.sqrt
    // // Refinement step (e.g., Newton-Raphson) might be needed depending on required precision
    // const inv_length = rsqrt_approx; // Placeholder

    const inv_length_splat: Vec4f = @splat(inv_length);
    return v * inv_length_splat;
}

// SIMD Distance calculation
pub fn distanceSIMD(a: Vec4f, b: Vec4f) f32 {
    const diff = a - b;
    const dot = dot3SIMD(diff, diff); // Use dot3 for distance in 3D space
    return @sqrt(dot);
}

// SIMD cross product for Vec4f (treating them as 3D vectors)
// a = (ax, ay, az, _)
// b = (bx, by, bz, _)
// r = (ay*bz - az*by, az*bx - ax*bz, ax*by - ay*bx, 0)
pub fn crossSIMD(a: Vec4f, b: Vec4f) Vec4f {
    // Shuffle operands to prepare for component calculations
    // yzx = (ay, az, ax, _)
    const vec1: Vec4f = .{ 1, 2, 0, 3 };
    const a_yzx = @shuffle(f32, a, undefined, vec1);
    const b_yzx = @shuffle(f32, b, undefined, vec1);
    // zxy = (az, ax, ay, _)
    const vec2: Vec4f = .{ 2, 0, 1, 3 };
    const a_zxy = @shuffle(f32, a, undefined, vec2);
    const b_zxy = @shuffle(f32, b, undefined, vec2);

    // Calculate cross product components using SIMD multiply and subtract
    // result = (a_yzx * b_zxy) - (a_zxy * b_yzx)
    return (a_yzx * b_zxy) - (a_zxy * b_yzx);
}

// ─── SIMD Ray–AABB intersection ────────────────────────────────────────────
pub fn getRayCollisionBoxSIMD(ray: RaySIMD, box: BoundingBoxSIMD) rl.RayCollision {
    // xyz → SIMD, w = 0 so it never influences the maths
    const origin = Vec4f{ ray.origin[0], ray.origin[1], ray.origin[2], 0 };
    const dir = Vec4f{ ray.dir[0], ray.dir[1], ray.dir[2], 0 };

    // 1 / dir  (∞ where dir == 0)
    const inv_dir = @as(Vec4f, @splat(1.0)) / dir;

    // parametric distances to the slabs
    const t1 = (box.min - origin) * inv_dir;
    const t2 = (box.max - origin) * inv_dir;

    const tmin_v = @min(t1, t2);
    const tmax_v = @max(t1, t2);

    // use only XYZ lanes for the reduce
    const t_near = @max(@max(tmin_v[0], tmin_v[1]), tmin_v[2]);
    const t_far = @min(@min(tmax_v[0], tmax_v[1]), tmax_v[2]);

    var rc: rl.RayCollision = .{
        .hit = false,
        .distance = 0,
        .point = rl.Vector3{ .x = ray.origin[0], .y = ray.origin[1], .z = ray.origin[2] },
        .normal = .{ .x = 0, .y = 0, .z = 0 },
    };

    // ray starts inside?
    if (t_near < 0.0 and t_far >= 0.0) {
        rc.hit = true;
        return rc; // distance = 0, point already set
    }

    // regular intersection
    if (t_far >= @max(t_near, 0.0)) {
        rc.hit = true;
        rc.distance = t_near;
        rc.point = .{
            .x = ray.origin[0] + ray.dir[0] * t_near,
            .y = ray.origin[1] + ray.dir[1] * t_near,
            .z = ray.origin[2] + ray.dir[2] * t_near,
        };

        // simple face normal (optional – unchanged)
        const eps: f32 = 1e-5;
        const p = rc.point;
        if (@abs(p.x - box.min[0]) < eps) rc.normal = .{ .x = -1, .y = 0, .z = 0 } else if (@abs(p.x - box.max[0]) < eps) rc.normal = .{ .x = 1, .y = 0, .z = 0 } else if (@abs(p.y - box.min[1]) < eps) rc.normal = .{ .x = 0, .y = -1, .z = 0 } else if (@abs(p.y - box.max[1]) < eps) rc.normal = .{ .x = 0, .y = 1, .z = 0 } else if (@abs(p.z - box.min[2]) < eps) rc.normal = .{ .x = 0, .y = 0, .z = -1 } else if (@abs(p.z - box.max[2]) < eps) rc.normal = .{ .x = 0, .y = 0, .z = 1 };
    }

    return rc;
}
