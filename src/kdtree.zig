const std = @import("std");
const rl = @import("raylib");
const s = @import("structs.zig");

pub const HitResult = struct {
    hit: bool = false,
    distance: f32 = 0,
    point: rl.Vector3 = .{ .x = 0, .y = 0, .z = 0 },
    hit_class: u32 = 0,
};

/// Simple axis-aligned KD-tree for rigid objects that already have
/// world-space bounding boxes (`bbox_ws`).
pub const KDTree = struct {
    const LEAF_SIZE = 4;

    pub const Node = struct {
        bbox: rl.BoundingBox,
        first: u32, // first index in prim_idx
        count: u16, // >0 ⇒ leaf (# primitives), 0 ⇒ interior
        left: u32 = 0,
        right: u32 = 0,
    };

    nodes: []Node,
    prim_idx: []u32,

    // ─────────────────────────────────────────────────────────────
    //  build
    // ─────────────────────────────────────────────────────────────
    pub fn build(
        alloc: std.mem.Allocator,
        models: []const s.Object,
    ) !KDTree {
        const prim_idx = try alloc.alloc(u32, models.len);
        for (prim_idx, 0..) |*dst, i| dst.* = @intCast(i);

        var list = std.ArrayList(Node).init(alloc);
        _ = try buildRec(alloc, &list, models, prim_idx, 0, prim_idx.len);

        return KDTree{
            .nodes = try list.toOwnedSlice(),
            .prim_idx = prim_idx,
        };
    }

    fn buildRec(
        alloc: std.mem.Allocator,
        out: *std.ArrayList(Node),
        models: []const s.Object,
        prim_idx: []u32,
        first: usize,
        count: usize,
    ) !u32 {
        // 1) bounding box of this range
        var bb = models[prim_idx[first]].bbox_ws;
        for (prim_idx[first + 1 .. first + count]) |i|
            bb = mergeBB(bb, models[i].bbox_ws);

        // 2) append the (still incomplete) node, remember index
        const idx: u32 = @intCast(out.items.len);
        try out.append(.{
            .bbox = bb,
            .first = @intCast(first),
            .count = if (count <= LEAF_SIZE) @intCast(count) else 0,
        });

        // leaf finished
        if (count <= LEAF_SIZE) return idx;

        // 3) choose longest axis & median-split
        const ex = bb.max.x - bb.min.x;
        const ey = bb.max.y - bb.min.y;
        const ez = bb.max.z - bb.min.z;

        var axis: usize = 0; // 0-x, 1-y, 2-z
        if (ey > ex and ey >= ez) axis = 1 else if (ez > ex and ez > ey) axis = 2;

        const Ctx = struct {
            models: []const s.Object,
            axis: usize,
        };
        const sort_ctx = Ctx{ .models = models, .axis = axis };

        std.sort.block(u32, prim_idx[first .. first + count], sort_ctx, struct {
            fn lessThan(ctx: Ctx, a: u32, b: u32) bool {
                return center(ctx.models[a].bbox_ws, ctx.axis) <
                    center(ctx.models[b].bbox_ws, ctx.axis);
            }
        }.lessThan);

        const mid = first + count / 2;

        // 4) recurse
        const l = try buildRec(alloc, out, models, prim_idx, first, mid - first);
        const r = try buildRec(alloc, out, models, prim_idx, mid, first + count - mid);

        out.items[idx].left = l;
        out.items[idx].right = r;
        return idx;
    }

    pub fn deinit(self: *KDTree, alloc: std.mem.Allocator) void {
        alloc.free(self.nodes);
        alloc.free(self.prim_idx);
    }

    // ─────────────────────────────────────────────────────────────
    //  traversal
    // ─────────────────────────────────────────────────────────────
    fn toLocalRay(ray_ws: rl.Ray, inv: rl.Matrix) rl.Ray {
        // position: full 4×4 transform (w = 1)
        const pos = rl.Vector3.transform(ray_ws.position, inv);

        // direction: rotate & scale only (w = 0), ignore translation
        const dir = rl.Vector3{
            .x = ray_ws.direction.x * inv.m0 + ray_ws.direction.y * inv.m4 + ray_ws.direction.z * inv.m8,
            .y = ray_ws.direction.x * inv.m1 + ray_ws.direction.y * inv.m5 + ray_ws.direction.z * inv.m9,
            .z = ray_ws.direction.x * inv.m2 + ray_ws.direction.y * inv.m6 + ray_ws.direction.z * inv.m10,
        };
        return rl.Ray{ .position = pos, .direction = dir };
    }

    pub fn closestHit(
        self: *const KDTree,
        ray_ws: rl.Ray,
        models: []const s.Object,
        max_range: f32,
    ) HitResult {
        var best = HitResult{ .distance = max_range };

        var stack: [64]u32 = undefined;
        var sp: usize = 1;
        stack[0] = 0; // root

        while (sp > 0) {
            sp -= 1;
            const node = self.nodes[stack[sp]];
            if (!rayBoxHit(ray_ws, node.bbox, best.distance)) continue;

            if (node.count > 0) {
                for (self.prim_idx[node.first .. node.first + node.count]) |pi| {
                    const obj = models[pi];

                    // 1. quick reject with world-space AABB
                    const bc = rl.getRayCollisionBox(ray_ws, obj.bbox_ws);
                    if (!bc.hit or bc.distance >= best.distance) continue;

                    // 2. transform ray to the object’s space
                    const ray_ms = toLocalRay(ray_ws, obj.inv_transform);

                    // 3. precise test inside the mesh BVH
                    if (obj.bvh.intersect(ray_ms, 0.0, std.math.inf(f32))) |hit| {
                        // local → world
                        const hit_ms = ray_ms.position.add(ray_ms.direction.scale(hit.t));
                        const hit_ws = rl.Vector3.transform(hit_ms, obj.model.transform);

                        const dist_ws = hit_ws.subtract(ray_ws.position).length();
                        if (dist_ws < best.distance) {
                            best = .{
                                .hit = true,
                                .distance = dist_ws, // world units
                                .point = hit_ws,
                                .hit_class = obj.class,
                            };
                        }
                    }
                }
            } else { // push children (right first)
                stack[sp] = node.right;
                sp += 1;
                stack[sp] = node.left;
                sp += 1;
            }
        }
        return best;
    }

    // ─────────────────────────────────────────────────────────────
    //  helpers
    // ─────────────────────────────────────────────────────────────
    fn center(bb: rl.BoundingBox, axis: usize) f32 {
        const cx = (bb.min.x + bb.max.x) * 0.5;
        const cy = (bb.min.y + bb.max.y) * 0.5;
        const cz = (bb.min.z + bb.max.z) * 0.5;
        return switch (axis) {
            0 => cx,
            1 => cy,
            else => cz,
        };
    }

    fn mergeBB(a: rl.BoundingBox, b: rl.BoundingBox) rl.BoundingBox {
        return .{
            .min = .{
                .x = @min(a.min.x, b.min.x),
                .y = @min(a.min.y, b.min.y),
                .z = @min(a.min.z, b.min.z),
            },
            .max = .{
                .x = @max(a.max.x, b.max.x),
                .y = @max(a.max.y, b.max.y),
                .z = @max(a.max.z, b.max.z),
            },
        };
    }

    fn rayBoxHit(ray: rl.Ray, bb: rl.BoundingBox, limit: f32) bool {
        const rc = rl.getRayCollisionBox(ray, bb);
        return rc.hit and rc.distance < limit;
    }
};
