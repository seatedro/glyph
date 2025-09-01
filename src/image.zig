const std = @import("std");
const stb = @import("stb");
const core = @import("libglyph");
const bitmap = core.bitmap;
const term = @import("libglyphterm");
const rescale = @import("libglyphrescale");

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

    try req.send();
    try req.finish();

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

    if (w <= 0 or h <= 0 or chan <= 0) {
        std.debug.print("Invalid image dimensions: w={d}, h={d}, chan={d}\n", .{ w, h, chan });
        return error.InvalidImageDimensions;
    }

    const total_pixels = @as(usize, @intCast(w)) * @as(usize, @intCast(h));
    const pixel_size = @as(usize, @intCast(chan));
    const buffer_size = total_pixels * (if (chan == 4) @as(usize, 3) else pixel_size);

    var rgb_data = try allocator.alloc(u8, buffer_size);
    errdefer allocator.free(rgb_data);

    // If image has 4 channels (RGBA), strip the alpha channel for consistency
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

pub fn loadAndScaleImage(allocator: std.mem.Allocator, args: core.CoreParams) !core.Image {
    const original_img = loadImage(allocator, args.input) catch |err| {
        std.debug.print("Error loading image: {}\n", .{err});
        return err;
    };

    if (args.scale != 1.0 and args.scale > 0.0) {
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

    if (img.channels == 4) {
        const total_pixels = img.width * img.height;
        var rgb_data = try allocator.alloc(u8, total_pixels * 3);
        errdefer allocator.free(rgb_data);

        var i: usize = 0;
        var j: usize = 0;
        while (i < total_pixels * 4) : (i += 4) {
            rgb_data[j] = img.data[i]; // R
            rgb_data[j + 1] = img.data[i + 1]; // G
            rgb_data[j + 2] = img.data[i + 2]; // B
            j += 3;
        }

        const rgb_img = core.Image{
            .data = rgb_data,
            .width = img.width,
            .height = img.height,
            .channels = 3,
        };

        const result = try rescale.resizeImage(
            core.Image,
            allocator,
            rgb_img,
            img_w,
            img_h,
            rescale.FilterType.Lanczos3,
        );

        // Free the temporary RGB buffer
        allocator.free(rgb_data);

        return result;
    }

    return rescale.resizeImage(
        core.Image,
        allocator,
        img,
        img_w,
        img_h,
        rescale.FilterType.Lanczos3,
    );
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
    var out_w: usize = undefined;
    var out_h: usize = undefined;
    if (args.render == .Pixels) {
        out_w = img.width;
        out_h = img.height;
    } else {
        out_w = (img.width / args.block_size) * args.block_size;
        out_h = (img.height / args.block_size) * args.block_size;
    }

    out_w = @max(out_w, 1);
    out_h = @max(out_h, 1);

    const save_result = stb.stbi_write_png(
        @ptrCast(args.output.?.ptr),
        @intCast(out_w),
        @intCast(out_h),
        3, // ascii_img is RGB
        @ptrCast(ascii_img.ptr),
        @intCast(out_w * 3),
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
            const out_img = if (args.render == .Pixels)
                try core.generatePixelDither(allocator, adjusted_img, args)
            else
                try core.generateAsciiArt(
                    allocator,
                    adjusted_img,
                    edge_result,
                    args,
                );
            try saveOutputImage(out_img, adjusted_img, args);
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
