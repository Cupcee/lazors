const rl = @import("raylib");
const std = @import("std");
const bvh = @import("bvh.zig");

//------------------------------------------------------------------
// BASIC TYPES
//------------------------------------------------------------------
pub const RayPoint = struct {
    xyz: rl.Vector3 = .{ .x = 0, .y = 0, .z = 0 },
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
    inv_transform: rl.Matrix, // cached
};

pub const Simulation = struct {
    debug: bool = true,
};

//------------------------------------------------------------------
// SENSOR (runtime resolution)
//------------------------------------------------------------------
pub const Sensor = struct {
    // –– pose and optics ––
    pos: rl.Vector3 = .{ .x = 0, .y = 1, .z = 0 },
    fwd: rl.Vector3 = .{ .x = 0, .y = 0, .z = 1 },
    up: rl.Vector3 = .{ .x = 0, .y = 1, .z = 0 },
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
    dirs: []rl.Vector3, // pre-computed directions (sensor space)
    points: []RayPoint, // collision results (refilled every frame)
    transforms: []rl.Matrix, // used for instanced-draw spheres

    allocator: std.mem.Allocator,
    local_to_world: rl.Matrix,

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
            .transforms = &.{},
            .local_to_world = rl.Matrix.identity(),
        };
        try self.allocateBuffers();
        self.precomputeDirs();
        return self;
    }

    pub fn deinit(self: *Sensor) void {
        self.allocator.free(self.dirs);
        self.allocator.free(self.points);
        self.allocator.free(self.transforms);
    }

    /// Change the grid size on the fly (buffers are re-created).
    pub fn setResolution(self: *Sensor, w: usize, h: usize) !void {
        if (w == self.res_h and h == self.res_v) return;
        self.allocator.free(self.dirs);
        self.allocator.free(self.points);
        self.allocator.free(self.transforms);
        self.res_h = w;
        self.res_v = h;
        try self.allocateBuffers();
        self.precomputeDirs();
    }

    /// Update `fwd`/`up` and build the 3×3 rotation matrix
    pub fn updateLocalAxes(self: *Sensor, fwd: rl.Vector3, up: rl.Vector3) void {
        self.fwd = rl.Vector3.normalize(fwd);
        self.up = rl.Vector3.normalize(up);
        const right = rl.Vector3.normalize(rl.Vector3.crossProduct(self.up, self.fwd));
        self.local_to_world = .{
            .m0 = right.x,
            .m1 = right.y,
            .m2 = right.z,
            .m3 = 0,
            .m4 = self.up.x,
            .m5 = self.up.y,
            .m6 = self.up.z,
            .m7 = 0,
            .m8 = self.fwd.x,
            .m9 = self.fwd.y,
            .m10 = self.fwd.z,
            .m11 = 0,
            .m12 = 0,
            .m13 = 0,
            .m14 = 0,
            .m15 = 1,
        };
    }

    //–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    fn allocateBuffers(self: *Sensor) !void {
        const n = self.res_h * self.res_v;
        self.dirs = try self.allocator.alloc(rl.Vector3, n);
        self.points = try self.allocator.alloc(RayPoint, n);
        self.transforms = try self.allocator.alloc(rl.Matrix, n);
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
                    .x = std.math.cos(el) * std.math.sin(az),
                    .y = std.math.sin(el),
                    .z = std.math.cos(el) * std.math.cos(az),
                };
                idx += 1;
            }
        }
    }
};
