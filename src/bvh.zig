const std = @import("std");
const simd = @import("raylib_simd.zig");

/// SIMD triangle (w is ignored – keep it 1 for vertices, 0 for edges)
pub const Triangle = struct {
    v0: simd.Vec4f,
    v1: simd.Vec4f,
    v2: simd.Vec4f,

    pub inline fn centroid(self: Triangle) simd.Vec4f {
        const splat: simd.Vec4f = @splat(1.0 / 3.0);
        return (self.v0 + self.v1 + self.v2) * splat;
    }

    pub inline fn bounds(self: Triangle) simd.AABB_SIMD {
        var box = simd.AABB_SIMD.empty();
        box.expand(self.v0);
        box.expand(self.v1);
        box.expand(self.v2);
        return box;
    }
};

const Node = struct {
    aabb: simd.AABB_SIMD,
    start_or_left: u32,
    count_or_right: u32,
    is_leaf: bool,
};

pub const Hit = struct { t: f32, prim_index: u32, u: f32, v: f32 };

pub const BVH = struct {
    allocator: std.mem.Allocator,
    nodes: []Node,
    node_count: u32,
    triangles: []Triangle,

    /// Build from a **flat SIMD vertex list** and an index array.
    pub fn build(alloc: std.mem.Allocator, vertices: []const simd.Vec4f, indices: []const u32) !BVH {
        std.debug.assert(indices.len % 3 == 0);

        var tris = try alloc.alloc(Triangle, indices.len / 3);
        var i: usize = 0;
        while (i < indices.len) : (i += 3) {
            tris[i / 3] = .{
                .v0 = vertices[indices[i]],
                .v1 = vertices[indices[i + 1]],
                .v2 = vertices[indices[i + 2]],
            };
        }

        const nodes = try alloc.alloc(Node, tris.len * 2);
        var bvh = BVH{ .allocator = alloc, .nodes = nodes, .node_count = 0, .triangles = tris };
        var next_free: u32 = 1;
        bvh.buildRecursive(0, 0, @intCast(tris.len), &next_free);
        bvh.node_count = next_free;
        return bvh;
    }

    fn buildRecursive(self: *BVH, idx: u32, start: u32, count: u32, next_free: *u32) void {
        var node = &self.nodes[idx];

        // ----- bounds of the range -----
        node.aabb = self.triangles[start].bounds();
        var k: u32 = 1;
        while (k < count) : (k += 1)
            node.aabb = simd.AABB_SIMD.unite(node.aabb, self.triangles[start + k].bounds());

        if (count <= 4) { // ---- leaf ----
            node.* = .{ .aabb = node.aabb, .start_or_left = start, .count_or_right = count, .is_leaf = true };
            return;
        }

        // ----- split axis -----
        const axis = simd.AABB_SIMD.largestExtentAxis(node.aabb);

        const Ctx = struct { axis: u8 };
        var ctx = Ctx{ .axis = axis };
        const less = struct {
            fn lt(c: *Ctx, a: Triangle, b: Triangle) bool {
                const ca = a.centroid();
                const cb = b.centroid();
                return switch (c.axis) {
                    0 => ca[0] < cb[0],
                    1 => ca[1] < cb[1],
                    else => ca[2] < cb[2],
                };
            }
        }.lt;
        std.sort.block(Triangle, self.triangles[start .. start + count], &ctx, less);

        const mid = start + count / 2;
        const l = next_free.*;
        next_free.* += 1;
        const r = next_free.*;
        next_free.* += 1;

        node.* = .{ .aabb = node.aabb, .start_or_left = l, .count_or_right = r, .is_leaf = false };
        self.buildRecursive(l, start, mid - start, next_free);
        self.buildRecursive(r, mid, start + count - mid, next_free);
    }

    // ------------------------------------------------------------------------
    // Intersection – identical external API, but now uses RaySIMD
    // ------------------------------------------------------------------------
    pub fn intersect(self: *const BVH, ray: simd.RaySIMD, tMin: f32, tMax: f32) ?Hit {
        var stack: [64]u32 = undefined;
        var sp: usize = 0;
        stack[sp] = 0;
        sp += 1;

        var closest = tMax;
        var best: ?Hit = null;

        while (sp > 0) {
            sp -= 1;
            const nidx = stack[sp];
            const node = self.nodes[nidx];

            if (!simd.AABB_SIMD.hit(node.aabb, ray, tMin, closest)) continue;

            if (node.is_leaf) {
                var j: u32 = 0;
                while (j < node.count_or_right) : (j += 1) {
                    const pidx = node.start_or_left + j;
                    if (rayTriangleIntersectSIMD(self.triangles[pidx], ray, tMin, closest)) |tHit| {
                        closest = tHit;
                        best = .{ .t = tHit, .prim_index = pidx, .u = 0, .v = 0 };
                    }
                }
            } else {
                // depth‑first: push right then left
                stack[sp] = node.count_or_right;
                sp += 1;
                stack[sp] = node.start_or_left;
                sp += 1;
            }
        }
        return best;
    }

    pub fn deinit(self: BVH) void {
        self.allocator.free(self.nodes);
        self.allocator.free(self.triangles);
    }
};

/// Möller–Trumbore on SIMD data (fully branch‑free except for final tests)
fn rayTriangleIntersectSIMD(tri: Triangle, ray: simd.RaySIMD, tMin: f32, tMax: f32) ?f32 {
    const eps: f32 = 1e-6;

    const e1 = tri.v1 - tri.v0;
    const e2 = tri.v2 - tri.v0;

    const pvec = simd.crossSIMD(ray.dir, e2);
    const det = simd.dot3SIMD(e1, pvec);
    if (@abs(det) < eps) return null;

    const invDet = 1.0 / det;
    const tvec = ray.origin - tri.v0;

    const u = simd.dot3SIMD(tvec, pvec) * invDet;
    if (u < 0.0 or u > 1.0) return null;

    const qvec = simd.crossSIMD(tvec, e1);
    const v = simd.dot3SIMD(ray.dir, qvec) * invDet;
    if (v < 0.0 or u + v > 1.0) return null;

    const tHit = simd.dot3SIMD(e2, qvec) * invDet;
    if (tHit < tMin or tHit > tMax) return null;

    return tHit;
}
