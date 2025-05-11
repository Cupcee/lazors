const std = @import("std");
const rl = @import("raylib");
const structs = @import("structs.zig");
const rc = @import("raycasting.zig");
const scene = @import("scene.zig");
const sim = @import("simulation.zig");
const pcd = @import("pcd_exporter.zig");
const tp = @import("thread_pool.zig");
const rlsimd = @import("raylib_simd.zig");
const fn_biome = @import("biome.zig");

const RayPool = tp.ThreadPool(rc.RaycastContext, rc.raycastWorker);
pub const WIN_W = 1240;
pub const WIN_H = 800;
pub const JITTER = 0.002;

pub const ModelPlacerItem = enum {
    cube,
    cylinder,
    sphere,
    grandpa,
    dinosaur,
};

pub const State = struct {
    alloc: std.mem.Allocator,
    //‑‑ persistent
    simulation: *structs.Simulation,

    //‑‑ graphics / IO
    camera: rl.Camera,
    camera_mode: rl.CameraMode,
    models: std.ArrayList(structs.Object),
    biome: fn_biome.Biome,

    //‑‑ sensor & drawing helpers
    sensor: structs.Sensor,
    collision: rl.Mesh,
    inst_mats: []rl.Material,
    class_tx: []std.ArrayList(rl.Matrix),
    class_counter: []usize,
    max_points: usize,

    //‑‑ PCD export
    exporter: pcd.Exporter,

    //‑‑ multithreading
    thread_res: rc.ThreadResources,
    thread_ctx: []rc.RaycastContext,
    pool: RayPool,

    //-- editor state
    selected_model: ModelPlacerItem = ModelPlacerItem.cube,
    show_editor: bool = false,
    editor_tx: rl.Matrix,
    editor_pivot: rl.Vector3,
    editor_yaw: f32 = 0.0,
    editor_scale: f32 = 1.0,
    preview_mat: rl.Material,
    preview_meshes: struct {
        cube: rl.Mesh,
        cylinder: rl.Mesh,
        sphere: rl.Mesh,
        grandpa: rl.Model,
        // godzilla: rl.Model,
    },

    pub fn init(alloc: std.mem.Allocator, sim_cfg: *structs.Simulation) !State {
        //------------------------------------------------------------------
        //  Window & camera
        //------------------------------------------------------------------
        rl.initWindow(WIN_W, WIN_H, "lazors");
        rl.disableCursor();
        rl.setTargetFPS(sim_cfg.target_fps);

        const cam, const cam_mode = sim.initCamera();

        //------------------------------------------------------------------
        //  Scene
        //------------------------------------------------------------------
        const models = try scene.buildScene(
            alloc,
            sim_cfg.num_objects,
            sim_cfg.class_count,
            sim_cfg.terrain_width,
        );
        const biome = try fn_biome.Biome.init(alloc, sim_cfg.map_size, sim_cfg.terrain_width, sim_cfg.terrain_height, 1337);

        //------------------------------------------------------------------
        //  Sensor
        //------------------------------------------------------------------
        var sensor = try structs.Sensor.init(alloc, 800, 192, 360, 70);
        sensor.updateLocalAxes(sensor.fwd, sensor.up);
        const max_points = sensor.res_h * sensor.res_v;

        //------------------------------------------------------------------
        //  Collided‑point rendering helpers
        //------------------------------------------------------------------
        var cube = rl.genMeshCube(0.02, 0.02, 0.02);
        rl.uploadMesh(&cube, false);

        const class_counter = try alloc.alloc(usize, sim_cfg.class_count);
        for (class_counter) |*c| c.* = 0;

        const inst_mats = try sim.initInstanceMats(alloc, @intCast(sim_cfg.class_count));
        const class_tx = try sim.initClassTxLists(
            alloc,
            @intCast(sim_cfg.class_count),
            max_points / sim_cfg.class_count + 32,
        );

        //------------------------------------------------------------------
        //  PCD exporter  (frames/ dir is created if missing)
        //------------------------------------------------------------------
        std.fs.cwd().makeDir("frames") catch |e| switch (e) {
            error.PathAlreadyExists => {},
            else => return e,
        };
        const exporter = try pcd.Exporter.create(alloc);

        //------------------------------------------------------------------
        //  Ray‑casting thread pool
        //------------------------------------------------------------------
        const n_threads = rc.getNumThreads();
        const thread_res = try rc.ThreadResources.init(alloc, n_threads, max_points);
        const thread_ctx = try alloc.alloc(rc.RaycastContext, n_threads);
        const pool = try RayPool.init(alloc, n_threads, thread_ctx);

        //------------------------------------------------------------------
        //  Model editor
        //------------------------------------------------------------------
        var pmat = try rl.loadMaterialDefault();
        pmat.maps[@intFromEnum(rl.MATERIAL_MAP_DIFFUSE)].color = rl.Color.dark_gray;
        // preview meshes -----------------------------------------------------------
        var cube_mesh = rl.genMeshCube(1, 1, 1);
        var cylinder_mesh = rl.genMeshCylinder(2, 4, 12);
        var sphere_mesh = rl.genMeshSphere(1.5, 12, 12);
        rl.uploadMesh(&cube_mesh, false);
        rl.uploadMesh(&cylinder_mesh, false);
        rl.uploadMesh(&sphere_mesh, false);
        const grandpa_mdl = try rl.loadModel("resources/objects/grandpa/scene.gltf");

        return State{
            .alloc = alloc,
            .simulation = sim_cfg,
            .camera = cam,
            .camera_mode = cam_mode,
            .models = models,
            .biome = biome,
            .sensor = sensor,
            .collision = cube,
            .inst_mats = inst_mats,
            .class_tx = class_tx,
            .class_counter = class_counter,
            .max_points = max_points,
            .exporter = exporter,
            .thread_res = thread_res,
            .thread_ctx = thread_ctx,
            .pool = pool,
            .editor_tx = rl.Matrix.identity(),
            .editor_pivot = rl.Vector3.zero(),
            .preview_mat = pmat,
            .preview_meshes = .{
                .cube = cube_mesh,
                .cylinder = cylinder_mesh,
                .sphere = sphere_mesh,
                .grandpa = grandpa_mdl,
            },
        };
    }

    pub fn deinit(self: *State, a: std.mem.Allocator) void {
        // workers → threads
        self.pool.deinit(a);
        a.free(self.thread_ctx);
        self.thread_res.deinit(a);

        // exporter
        self.exporter.destroy();

        // class‑tx lists
        for (self.class_tx) |*list| list.deinit();
        a.free(self.class_tx);

        // instanced materials & counters
        a.free(self.inst_mats);
        a.free(self.class_counter);

        // sensor
        self.sensor.deinit();

        // mesh & models
        rl.unloadMesh(self.collision);
        for (self.models.items) |*o| {
            rl.unloadModel(o.model);
            o.bvh.deinit();
        }
        self.models.deinit();
        self.biome.deinit();

        rl.unloadMaterial(self.preview_mat);
        rl.unloadMesh(self.preview_meshes.cube);
        rl.unloadMesh(self.preview_meshes.cylinder);
        rl.unloadMesh(self.preview_meshes.sphere);

        // window *last*
        rl.closeWindow();
    }
};
