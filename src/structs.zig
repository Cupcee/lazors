const rl = @import("raylib");
const rlsimd = @import("raylib_simd.zig");
const std = @import("std");
const bvh = @import("bvh.zig");

//------------------------------------------------------------------
// BASIC TYPES
//------------------------------------------------------------------
pub const RayPoint = struct {
    xyz: rlsimd.Vec4f = .{ 0, 0, 0, 1 },
    hit: bool = false,
    hit_class: u32 = 0,
};

pub const Object = struct {
    model: rl.Model,
    class: u32,
    color: rl.Color,
    /// world-space axis-aligned bounding box (step 1)
    bbox_ws: rl.BoundingBox,
    bvh: bvh.BVH,
    // inv_transform: rl.Matrix, // cached
    transform_simd: rlsimd.Mat4x4_SIMD,
    inv_transform_simd: rlsimd.Mat4x4_SIMD,
};

pub const Simulation = struct {
    debug: bool = true,
    num_objects: u32 = 20,
    target_fps: i32 = 60,
    class_count: u32 = 6,
    plane_half_size: f32 = 20.0,
    collect: bool = true,
    collect_wait_seconds: f32 = 0.1, // how long collector waits in seconds until next collection
};

//------------------------------------------------------------------
// SENSOR (runtime resolution)
//------------------------------------------------------------------
pub const Sensor = struct {
    // –– pose and optics ––
    pos: rl.Vector3 = .{ .x = 0, .y = 1, .z = 0 },
    fwd: rlsimd.Vec4f = .{ 0, 0, 1, 1 },
    up: rlsimd.Vec4f = .{ 0, 1, 0, 1 },
    yaw: f32 = 0,
    pitch: f32 = 0,
    velocity: f32 = 5,
    turn_speed: f32 = std.math.pi,
    fov_h_deg: f32 = 120,
    fov_v_deg: f32 = 70,
    max_range: f32 = 70,

    // –– runtime resolution ––
    res_h: usize,
    res_v: usize,

    // –– working buffers ––
    dirs: []rlsimd.Vec4f, // pre-computed directions (sensor space)
    points: []RayPoint, // collision results (refilled every frame)
    // transforms: []rl.Matrix, // used for instanced-draw spheres

    allocator: std.mem.Allocator,
    // local_to_world: rl.Matrix,
    local_to_world_simd: rlsimd.Mat4x4_SIMD,

    //--------------------------------------------------------------
    pub fn init(allocator: std.mem.Allocator, res_h: usize, res_v: usize, fov_h_deg: f32, fov_v_deg: f32) !Sensor {
        var self = Sensor{
            .allocator = allocator,
            .res_h = res_h,
            .res_v = res_v,
            .fov_h_deg = fov_h_deg,
            .fov_v_deg = fov_v_deg,
            .dirs = &.{},
            .points = &.{},
            // .local_to_world = rl.Matrix.identity(),
            .local_to_world_simd = rlsimd.Mat4x4_SIMD.fromRlMatrix(rl.Matrix.identity()),
        };
        try self.allocateBuffers();
        self.precomputeDirs();
        return self;
    }

    pub fn deinit(self: *Sensor) void {
        self.allocator.free(self.dirs);
        self.allocator.free(self.points);
        // self.allocator.free(self.transforms);
    }

    /// Update `fwd`/`up` and build the 3×3 rotation matrix
    pub fn updateLocalAxes(self: *Sensor, fwd: rlsimd.Vec4f, up: rlsimd.Vec4f) void {
        const fwd_simd = rlsimd.normalizeSIMD(fwd);
        const up_simd = rlsimd.normalizeSIMD(up);
        self.fwd = fwd_simd;
        self.up = up_simd;
        const right_simd = rlsimd.normalizeSIMD(rlsimd.crossSIMD(up_simd, fwd_simd));
        const local_to_world: rl.Matrix = .{
            .m0 = right_simd[0],
            .m1 = right_simd[1],
            .m2 = right_simd[2],
            .m3 = 0,
            .m4 = self.up[0],
            .m5 = self.up[1],
            .m6 = self.up[2],
            .m7 = 0,
            .m8 = self.fwd[0],
            .m9 = self.fwd[1],
            .m10 = self.fwd[2],
            .m11 = 0,
            .m12 = 0,
            .m13 = 0,
            .m14 = 0,
            .m15 = 1,
        };
        self.local_to_world_simd = rlsimd.Mat4x4_SIMD.fromRlMatrix(local_to_world);
    }

    //–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    fn allocateBuffers(self: *Sensor) !void {
        const n = self.res_h * self.res_v;
        self.dirs = try self.allocator.alloc(rlsimd.Vec4f, n);
        self.points = try self.allocator.alloc(RayPoint, n);
        // self.transforms = try self.allocator.alloc(rl.Matrix, n);
    }

    /// Pay the sin/cos cost only once, when the grid size changes.
    fn precomputeDirs(self: *Sensor) void {
        const deg2rad: f32 = 0.0174532925;
        const res_hf: f32 = @floatFromInt(self.res_h - 1);
        const res_vf: f32 = @floatFromInt(self.res_v - 1);
        const h_step: f32 = (self.fov_h_deg * deg2rad) / res_hf;
        const v_step: f32 = (self.fov_v_deg * deg2rad) / res_vf;

        var idx: usize = 0;
        for (0..self.res_v) |v| {
            const fv: f32 = @floatFromInt(v);
            const el = (fv - (res_vf * 0.5)) * v_step;
            for (0..self.res_h) |h| {
                const fh: f32 = @floatFromInt(h);
                const az = (fh - (res_hf * 0.5)) * h_step;
                self.dirs[idx] = .{
                    @cos(el) * @sin(az),
                    @sin(el),
                    @cos(el) * @cos(az),
                    1,
                };
                idx += 1;
            }
        }
    }
};
