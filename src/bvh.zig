// bvh.zig — BVH over triangle meshes using raylib's Vector3 (Zig 0.14)
// Replaces the standalone Vec3 with `rl.Vector3` from the Zig‑raylib bindings.
//
// Public API
//   const bvh = try BVH.build(allocator, vertices, indices);
//   if (bvh.intersect(ray, 0.0, std.math.inf(f32))) |hit| { ... }
//   bvh.deinit();
//
// Compile/run: `zig run bvh.zig -Ipath/to/raylib/bindings`

const std = @import("std");
const rl = @import("raylib");

///////////////////////////////////////////////////////////////////////////////
// Basic types using rl.Vector3
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
// Axis‑aligned bounding box
///////////////////////////////////////////////////////////////////////////////

pub const AABB = struct {
    min: rl.Vector3,
    max: rl.Vector3,

    pub inline fn empty() AABB {
        return .{
            .min = rl.Vector3.init(std.math.inf(f32), std.math.inf(f32), std.math.inf(f32)),
            .max = rl.Vector3.init(-std.math.inf(f32), -std.math.inf(f32), -std.math.inf(f32)),
        };
    }

    pub inline fn expand(self: *AABB, p: rl.Vector3) void {
        self.min = self.min.min(p);
        self.max = self.max.max(p);
    }

    pub inline fn unite(a: AABB, b: AABB) AABB {
        return .{ .min = a.min.min(b.min), .max = a.max.max(b.max) };
    }

    /// Classic slab test. Returns true if the segment [t_min,t_max] hits the box.
    pub inline fn hit(box: AABB, ray: rl.Ray, t_min: f32, t_max: f32) bool {
        var t0 = t_min;
        var t1 = t_max;
        const inv = rl.Vector3.init(1.0 / ray.direction.x, 1.0 / ray.direction.y, 1.0 / ray.direction.z);
        const orig = ray.position;
        // X
        var tNear = (box.min.x - orig.x) * inv.x;
        var tFar = (box.max.x - orig.x) * inv.x;
        if (tNear > tFar) std.mem.swap(f32, &tNear, &tFar);
        t0 = @max(t0, tNear);
        t1 = @min(t1, tFar);
        if (t0 > t1) return false;
        // Y
        tNear = (box.min.y - orig.y) * inv.y;
        tFar = (box.max.y - orig.y) * inv.y;
        if (tNear > tFar) std.mem.swap(f32, &tNear, &tFar);
        t0 = @max(t0, tNear);
        t1 = @min(t1, tFar);
        if (t0 > t1) return false;
        // Z
        tNear = (box.min.z - orig.z) * inv.z;
        tFar = (box.max.z - orig.z) * inv.z;
        if (tNear > tFar) std.mem.swap(f32, &tNear, &tFar);
        t0 = @max(t0, tNear);
        t1 = @min(t1, tFar);
        return t0 <= t1;
    }

    /// Return axis (0,1,2) with largest extent.
    pub inline fn largestExtentAxis(box: AABB) u8 {
        const e = box.max.subtract(box.min);
        return if (e.x > e.y and e.x > e.z) 0 else if (e.y > e.z) 1 else 2;
    }
};

///////////////////////////////////////////////////////////////////////////////
// Triangle primitive
///////////////////////////////////////////////////////////////////////////////

pub const Triangle = struct {
    v0: rl.Vector3,
    v1: rl.Vector3,
    v2: rl.Vector3,

    pub inline fn centroid(self: Triangle) rl.Vector3 {
        return self.v0.add(self.v1).add(self.v2).scale(1.0 / 3.0);
    }

    pub inline fn bounds(self: Triangle) AABB {
        var aabb = AABB.empty();
        aabb.expand(self.v0);
        aabb.expand(self.v1);
        aabb.expand(self.v2);
        return aabb;
    }
};

///////////////////////////////////////////////////////////////////////////////
// BVH node
///////////////////////////////////////////////////////////////////////////////

const Node = struct {
    aabb: AABB,
    // If leaf: start/count into triangles. If internal: child indices.
    start_or_left: u32,
    count_or_right: u32,
    is_leaf: bool,
};

///////////////////////////////////////////////////////////////////////////////
// Hit record
///////////////////////////////////////////////////////////////////////////////

pub const Hit = struct {
    t: f32,
    prim_index: u32,
    u: f32,
    v: f32,
};

///////////////////////////////////////////////////////////////////////////////
// BVH container
///////////////////////////////////////////////////////////////////////////////

pub const BVH = struct {
    allocator: std.mem.Allocator,
    nodes: []Node,
    node_count: u32,
    triangles: []Triangle,

    /// Build from vertex/index arrays (3 indices per triangle).
    pub fn build(alloc: std.mem.Allocator, vertices: []const rl.Vector3, indices: []const u32) !BVH {
        std.debug.assert(indices.len % 3 == 0);
        // Copy triangles
        var tris = try alloc.alloc(Triangle, indices.len / 3);
        var i: usize = 0;
        while (i < indices.len) : (i += 3) {
            tris[i / 3] = Triangle{
                .v0 = vertices[indices[i]],
                .v1 = vertices[indices[i + 1]],
                .v2 = vertices[indices[i + 2]],
            };
        }
        // worst‑case 2N nodes
        const nodes = try alloc.alloc(Node, tris.len * 2);
        var bvh = BVH{
            .allocator = alloc,
            .nodes = nodes,
            .node_count = 0,
            .triangles = tris,
        };
        var next_free: u32 = 1; // root = 0, children start at 1
        bvh.buildRecursive(0, 0, @intCast(tris.len), &next_free);
        bvh.node_count = next_free;
        return bvh;
    }

    /// Recursive median‑split builder.
    fn buildRecursive(self: *BVH, node_idx: u32, start: u32, count: u32, next_free: *u32) void {
        var node_ptr = &self.nodes[node_idx];
        // Compute bounds
        node_ptr.aabb = self.triangles[start].bounds();
        var j: u32 = 1;
        while (j < count) : (j += 1) {
            node_ptr.aabb = AABB.unite(node_ptr.aabb, self.triangles[start + j].bounds());
        }
        if (count <= 4) { // leaf
            node_ptr.* = .{ .aabb = node_ptr.aabb, .start_or_left = start, .count_or_right = count, .is_leaf = true };
            return;
        }
        // Split axis
        const axis: u8 = AABB.largestExtentAxis(node_ptr.aabb);
        // Sort by centroid along axis
        const Ctx = struct { axis: u8 };
        var ctx = Ctx{ .axis = axis };
        const lessFn = struct {
            fn cmp(_ctx: *Ctx, a: Triangle, b: Triangle) bool {
                const ca = a.centroid();
                const cb = b.centroid();
                return switch (_ctx.axis) {
                    0 => ca.x < cb.x,
                    1 => ca.y < cb.y,
                    else => ca.z < cb.z,
                };
            }
        }.cmp;
        std.sort.block(Triangle, self.triangles[start .. start + count], &ctx, lessFn);
        const mid = start + count / 2;
        // Allocate children
        const left_idx: u32 = next_free.*;
        next_free.* += 1;
        const right_idx: u32 = next_free.*;
        next_free.* += 1;
        node_ptr.* = .{ .aabb = node_ptr.aabb, .start_or_left = left_idx, .count_or_right = right_idx, .is_leaf = false };
        self.buildRecursive(left_idx, start, mid - start, next_free);
        self.buildRecursive(right_idx, mid, start + count - mid, next_free);
    }

    ///////////////////////////////////////////////////////////////////////////
    // Intersection
    ///////////////////////////////////////////////////////////////////////////

    /// Return first hit (smallest positive t) or null.
    pub fn intersect(self: *const BVH, ray: rl.Ray, t_min: f32, t_max: f32) ?Hit {
        var stack: [64]u32 = undefined;
        var sp: usize = 0;
        stack[sp] = 0;
        sp += 1; // push root
        var closest = t_max;
        var best: ?Hit = null;
        while (sp > 0) {
            sp -= 1;
            const idx = stack[sp];
            const node = self.nodes[idx];
            if (!AABB.hit(node.aabb, ray, t_min, closest)) continue;
            if (node.is_leaf) {
                var k: u32 = 0;
                while (k < node.count_or_right) : (k += 1) {
                    const prim_idx: u32 = node.start_or_left + k;
                    if (rayTriangleIntersect(self.triangles[prim_idx], ray, t_min, closest)) |t_hit| {
                        closest = t_hit;
                        best = Hit{ .t = t_hit, .prim_index = prim_idx, .u = 0, .v = 0 }; // u,v filled later if needed
                    }
                }
            } else {
                // push right then left so left is processed first
                stack[sp] = node.count_or_right;
                sp += 1;
                stack[sp] = node.start_or_left;
                sp += 1;
            }
        }
        return best;
    }

    /// Free memory (triangles + nodes)
    pub fn deinit(self: BVH) void {
        self.allocator.free(self.nodes);
        self.allocator.free(self.triangles);
    }
};

///////////////////////////////////////////////////////////////////////////////
// Ray–triangle intersection (Möller‑Trumbore) — returns t, or null
///////////////////////////////////////////////////////////////////////////////

fn rayTriangleIntersect(tri: Triangle, ray: rl.Ray, t_min: f32, t_max: f32) ?f32 {
    const eps: f32 = 1e-6;
    const e1 = tri.v1.subtract(tri.v0);
    const e2 = tri.v2.subtract(tri.v0);
    const pvec = ray.direction.crossProduct(e2);
    const det = e1.dotProduct(pvec);
    if (det > -eps and det < eps) return null;
    const inv_det = 1.0 / det;
    const tvec = ray.position.subtract(tri.v0);
    const u = tvec.dotProduct(pvec) * inv_det;
    if (u < 0.0 or u > 1.0) return null;
    const qvec = tvec.crossProduct(e1);
    const v = ray.direction.dotProduct(qvec) * inv_det;
    if (v < 0.0 or u + v > 1.0) return null;
    const t_hit = e2.dotProduct(qvec) * inv_det;
    if (t_hit < t_min or t_hit > t_max) return null;
    return t_hit;
}

///////////////////////////////////////////////////////////////////////////////
// Example (zig run bvh.zig)
///////////////////////////////////////////////////////////////////////////////
// pub fn main() !void {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     defer std.debug.assert(gpa.deinit() == 0);
//     const alloc = gpa.allocator();
//
//     // Quad in XY plane (two triangles)
//     const verts = &[_]rl.Vector3{
//         rl.Vector3.init(-1, -1, 0),
//         rl.Vector3.init(1, -1, 0),
//         rl.Vector3.init(1, 1, 0),
//         rl.Vector3.init(-1, 1, 0),
//     };
//     const idx = &[_]u32{ 0, 1, 2, 0, 2, 3 };
//
//     var bvh = try BVH.build(alloc, verts, idx);
//     defer bvh.deinit();
//
//     const ray = Ray{ .origin = rl.Vector3.init(0, 0, -2), .dir = rl.Vector3.init(0, 0, 1) };
//     if (bvh.intersect(ray, 0.0, std.math.inf(f32))) |hit| {
//         std.debug.print("Hit tri {} at t = {}\n", .{ hit.prim_index, hit.t });
//     } else {
//         std.debug.print("No hit\n", .{});
//     }
// }
