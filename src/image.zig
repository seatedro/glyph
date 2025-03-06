const std = @import("std");
const stb = @import("stb");
const core = @import("libglyph");
const bitmap = core.bitmap;
const term = @import("libglyphterm");

// -----------------------
// IMAGE PROCESSING FUNCTIONS
// -----------------------

pub fn downloadImage(allocator: std.mem.Allocator, url: []const u8) ![]u8 {
    var client = std.http.Client{ .allocator = allocator };
    defer client.deinit();

    const uri = try std.Uri.parse(url);

    var buf: [4096]u8 = undefined;
    var req = try client.open(.GET, uri, .{ .server_header_buffer = &buf });
    defer req.deinit();

    // Sending HTTP req headers
    try req.send();
    try req.finish();

    // Wait for response
    try req.wait();

    if (req.response.status != .ok) {
        return error.HttpRequestFailed;
    }

    const content_len = req.response.content_length orelse return error.NoContentLength;
    const body = try allocator.alloc(u8, content_len);
    errdefer allocator.free(body);

    const bytes_read = try req.readAll(body);
    if (bytes_read != content_len) {
        return error.IncompleteRead;
    }

    return body;
}

pub fn loadImage(allocator: std.mem.Allocator, path: []const u8) !core.Image {
    const is_url = std.mem.startsWith(u8, path, "http://") or std.mem.startsWith(u8, path, "https://");

    var image_data: []u8 = undefined;
    defer if (is_url) allocator.free(image_data);

    if (is_url) {
        image_data = try downloadImage(allocator, path);
    }

    var w: c_int = undefined;
    var h: c_int = undefined;
    var chan: c_int = undefined;
    const data = if (is_url)
        stb.stbi_load_from_memory(image_data.ptr, @intCast(image_data.len), &w, &h, &chan, 0)
    else
        stb.stbi_load(path.ptr, &w, &h, &chan, 0);

    if (@intFromPtr(data) == 0) {
        std.debug.print("Error loading image: {s}\n", .{path});
        return error.ImageLoadFailed;
    }

    defer stb.stbi_image_free(data);

    // Make sure w, h, and chan are valid to prevent integer overflow
    if (w <= 0 or h <= 0 or chan <= 0) {
        std.debug.print("Invalid image dimensions: w={d}, h={d}, chan={d}\n", .{ w, h, chan });
        return error.InvalidImageDimensions;
    }

    // Validate buffer size to prevent overflow
    const total_pixels = @as(usize, @intCast(w)) * @as(usize, @intCast(h));
    const pixel_size = @as(usize, @intCast(chan));
    const buffer_size = total_pixels * (if (chan == 4) @as(usize, 3) else pixel_size);

    var rgb_data = try allocator.alloc(u8, buffer_size);
    errdefer allocator.free(rgb_data);

    const ext = std.fs.path.extension(path);
    if (std.mem.eql(u8, ext, ".png")) {
        // If image has 4 channels (RGBA), strip the alpha channel
        if (chan == 4) {
            var i: usize = 0;
            var j: usize = 0;
            while (i < total_pixels * 4) : (i += 4) {
                rgb_data[j] = data[i]; // R
                rgb_data[j + 1] = data[i + 1]; // G
                rgb_data[j + 2] = data[i + 2]; // B
                j += 3;
            }

            return core.Image{
                .data = rgb_data,
                .width = @intCast(w),
                .height = @intCast(h),
                .channels = 3,
            };
        }

        @memcpy(rgb_data, data[0 .. total_pixels * @as(usize, @intCast(chan))]);

        return core.Image{
            .data = rgb_data,
            .width = @intCast(w),
            .height = @intCast(h),
            .channels = @intCast(chan),
        };
    }

    @memcpy(rgb_data, data[0 .. total_pixels * @as(usize, @intCast(chan))]);

    return core.Image{
        .data = rgb_data,
        .width = @intCast(w),
        .height = @intCast(h),
        .channels = @intCast(chan),
    };
}

pub fn loadAndScaleImage(allocator: std.mem.Allocator, args: core.CoreParams) !core.Image {
    const original_img = loadImage(allocator, args.input) catch |err| {
        std.debug.print("Error loading image: {}\n", .{err});
        return err;
    };

    if (args.scale != 1.0 and args.scale > 0.0) {
        // Need to free the original image data after scaling
        defer allocator.free(original_img.data);
        return scaleImage(allocator, original_img, args.scale);
    } else {
        return original_img;
    }
}

pub fn scaleImage(allocator: std.mem.Allocator, img: core.Image, scale: f32) !core.Image {
    var img_w = @as(usize, @intFromFloat(@round(@as(f32, @floatFromInt(img.width)) / scale)));
    var img_h = @as(usize, @intFromFloat(@round(@as(f32, @floatFromInt(img.height)) / scale)));

    img_w = @max(img_w, 1);
    img_h = @max(img_h, 1);

    const total_pixels = img_w * img_h;
    const buffer_size = total_pixels * (if (img.channels == 4) @as(usize, 3) else img.channels);

    const scaled_data = try allocator.alloc(u8, buffer_size);
    errdefer allocator.free(scaled_data);

    const scaled_img = stb.stbir_resize_uint8_linear(
        img.data.ptr,
        @intCast(img.width),
        @intCast(img.height),
        0,
        0,
        @intCast(img_w),
        @intCast(img_h),
        0,
        @intCast(img.channels),
    );
    if (scaled_img == null) {
        std.debug.print("Error downscaling image\n", .{});
        return error.ImageScaleFailed;
    }

    defer stb.stbi_image_free(scaled_img);

    @memcpy(scaled_data, scaled_img[0..buffer_size]);

    return core.Image{
        .data = scaled_data,
        .width = img_w,
        .height = img_h,
        .channels = img.channels,
    };
}

pub fn generateAsciiTxt(
    allocator: std.mem.Allocator,
    img: core.Image,
    edge_result: ?core.EdgeData,
    args: core.CoreParams,
) ![]u8 {
    var out_w = (img.width / args.block_size) * args.block_size;
    var out_h = (img.height / args.block_size) * args.block_size;

    out_w = @max(out_w, 1);
    out_h = @max(out_h, 1);

    var ascii_text = std.ArrayList(u8).init(allocator);

    var y: usize = 0;
    while (y < out_h) : (y += args.block_size) {
        var x: usize = 0;
        while (x < out_w) : (x += args.block_size) {
            const block_info = core.calculateBlockInfo(img, edge_result, x, y, out_w, out_h, args);
            const ascii_char = core.selectAsciiChar(block_info, args);
            try ascii_text.appendSlice(ascii_char);
        }
        try ascii_text.append('\n');
    }

    return ascii_text.toOwnedSlice();
}

fn saveOutputTxt(ascii_text: []const u8, args: core.CoreParams) !void {
    const file = try std.fs.cwd().createFile(args.output.?, .{});
    defer file.close();

    try file.writeAll(ascii_text);
}

fn saveOutputImage(ascii_img: []u8, img: core.Image, args: core.CoreParams) !void {
    var out_w = (img.width / args.block_size) * args.block_size;
    var out_h = (img.height / args.block_size) * args.block_size;

    out_w = @max(out_w, 1);
    out_h = @max(out_h, 1);
    const channels: u8 = if (args.transparent_bg) 4 else 3;
    const save_result = stb.stbi_write_png(
        @ptrCast(args.output.?.ptr),
        @intCast(out_w),
        @intCast(out_h),
        channels,
        @ptrCast(ascii_img.ptr),
        @intCast(out_w * channels),
    );
    if (save_result == 0) {
        std.debug.print("Error writing output image\n", .{});
        return error.ImageWriteFailed;
    }
}
pub fn processImage(allocator: std.mem.Allocator, args: core.CoreParams) !void {
    const original_img = try loadAndScaleImage(allocator, args);

    // Safety check for image dimensions
    if (original_img.width == 0 or original_img.height == 0) {
        std.debug.print("Error: Invalid image dimensions\n", .{});
        return error.InvalidImageDimensions;
    }

    const expected_size = original_img.width * original_img.height * original_img.channels;
    const adjusted_data = if (args.auto_adjust)
        try core.autoBrightnessContrast(allocator, original_img, 1.0)
    else
        try allocator.dupe(u8, original_img.data[0..expected_size]);

    const adjusted_img = core.Image{
        .data = adjusted_data,
        .width = original_img.width,
        .height = original_img.height,
        .channels = original_img.channels,
    };

    const edge_result = try core.detectEdges(allocator, adjusted_img, args.detect_edges, args.sigma1, args.sigma2);

    switch (args.output_type) {
        core.OutputType.Image => {
            const ascii_img = try core.generateAsciiArt(
                allocator,
                adjusted_img,
                edge_result,
                args,
            );
            try saveOutputImage(ascii_img, adjusted_img, args);
        },
        core.OutputType.Stdout => {
            var t = try term.init(allocator, args.ascii_chars);

            var new_w: usize = 0;
            var new_h: usize = 0;
            if (args.stretched) {
                new_w = t.size.w - 2;
                new_h = t.size.h - 4;
            } else {
                const rw = adjusted_img.width / (t.size.w - 2);
                const rh = adjusted_img.height / (t.size.h - 4);
                if (rw > rh) {
                    new_h = adjusted_img.height / (rw * 2);
                    new_w = t.size.w - 2;
                } else {
                    new_h = (t.size.h - 4) / 2;
                    new_w = adjusted_img.width / rh;
                }
            }
            new_w = @max(new_w, 1);
            new_h = @max(new_h, 1);

            const img = try core.resizeImage(allocator, adjusted_img, new_w, new_h);

            t.stats = .{
                .original_w = adjusted_img.width,
                .original_h = adjusted_img.height,
                .new_w = img.width,
                .new_h = img.height,
            };

            const img_len = img.height * img.width * img.channels;

            try t.enableAsciiMode();
            const params = term.RenderParams{
                .img = img.data[0..img_len],
                .width = img.width,
                .height = img.height,
                .channels = img.channels,
                .color = args.color,
                .invert = args.invert_color,
            };
            try t.renderAsciiArt(params);

            // Wait for user input before exiting
            _ = try t.stdin.readByte();
            try t.disableAsciiMode();
        },
        core.OutputType.Text => {
            // 1 : og
            // dr : grid
            var img: core.Image = adjusted_img;
            var h = adjusted_img.height;
            const w = adjusted_img.width;
            if (adjusted_img.height >= 2) {
                h = adjusted_img.height / 2;
                img = try core.resizeImage(allocator, adjusted_img, w, h);
            }
            const ascii_txt = try generateAsciiTxt(
                allocator,
                img,
                edge_result,
                args,
            );
            try saveOutputTxt(ascii_txt, args);
        },
        else => {},
    }
}

// TESTS
