const std = @import("std");
const rl = @import("raylib");
const fastnoise = @import("fastnoise.zig");

pub const Biome = struct {
    model: rl.Model,
    texture: rl.Texture2D,
    height_pixels: []u8,
    colour_pixels: []rl.Color,
    allocator: std.mem.Allocator,

    /// Build a height-field mesh + colour texture from FastNoise.
    pub fn init(
        alloc: std.mem.Allocator,
        map_size: i32,
        terrain_width: f32,
        terrain_height: f32,
        seed: i32,
    ) !Biome {
        const Noise = fastnoise.Noise(f32);

        var height_noise = Noise{
            .seed = 1337,
            .noise_type = .cellular,
            .frequency = 0.025,
            .gain = 0.40,
            .fractal_type = .fbm,
            .lacunarity = 0.40,
            .cellular_distance = .euclidean,
            .cellular_return = .distance2,
            .cellular_jitter_mod = 0.88,
        };

        var biome_noise = Noise{
            .seed = seed,
            .noise_type = .perlin,
            .frequency = 0.04,
        };

        const side: usize = @intCast(map_size);
        const px_cnt: usize = side * side;

        var height_buf = try alloc.alloc(u8, px_cnt); // grey heights
        errdefer alloc.free(height_buf);

        var colour_buf = try alloc.alloc(rl.Color, px_cnt); // RGBA biomes
        errdefer alloc.free(colour_buf);

        // fill the buffers with noise
        for (0..side) |y| {
            for (0..side) |x| {
                const fx: f32 = @floatFromInt(x);
                const fy: f32 = @floatFromInt(y);

                var h = height_noise.genNoise2D(fx, fy); // [-1,1]
                h = (h + 1.0) * 0.5; // → [0,1]
                const h_byte: u8 = @intFromFloat(h * 255);

                const idx = y * side + x;
                height_buf[idx] = h_byte;

                const m = biome_noise.genNoise2D(fx, fy);

                const col = blk: {
                    if (h < 0.30) break :blk rl.Color.beige; // beach
                    if (h < 0.60 and m > 0.0) break :blk rl.Color.dark_green; // forest
                    if (h < 0.60) break :blk rl.Color.green; // plains
                    if (h < 0.85) break :blk rl.Color.brown; // rocks
                    break :blk rl.Color.ray_white; // snow
                };
                colour_buf[idx] = col;
            }
        }

        // wrap raw data in Raylib images
        const img_height = rl.Image{
            .data = @ptrCast(height_buf.ptr),
            .width = map_size,
            .height = map_size,
            .mipmaps = 1,
            .format = rl.PixelFormat.uncompressed_r8g8b8a8,
        };

        const img_colour = rl.Image{
            .data = @ptrCast(colour_buf.ptr),
            .width = map_size,
            .height = map_size,
            .mipmaps = 1,
            .format = rl.PixelFormat.uncompressed_r8g8b8a8,
        };

        // build mesh, model & GPU texture
        const mesh = rl.genMeshHeightmap(
            img_height,
            .{ .x = terrain_width, .y = terrain_height, .z = terrain_width },
        );

        var model = try rl.loadModelFromMesh(mesh);

        const texture = try rl.loadTextureFromImage(img_colour);
        model.materials[0].maps[@intFromEnum(rl.MATERIAL_MAP_DIFFUSE)].texture = texture;

        return Biome{
            .model = model,
            .texture = texture,
            .height_pixels = height_buf,
            .colour_pixels = colour_buf,
            .allocator = alloc,
        };
    }

    /// Free every resource that *init* created.
    pub fn deinit(self: *Biome) void {
        // GPU objects
        rl.unloadTexture(self.texture);
        rl.unloadModel(self.model);
        // heap buffers
        self.allocator.free(self.height_pixels);
        self.allocator.free(self.colour_pixels);
    }
};
