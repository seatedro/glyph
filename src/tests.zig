const std = @import("std");
pub const core = @import("libglyph");
pub const image = @import("libglyphimg");
pub const term = @import("libglyphterm");
const build_options = @import("build_options");
pub const video = if (build_options.av) @import("libglyphav") else struct {};
const stb = core.stb;

test "loadImage - png" {
    const allocator = std.testing.allocator;
    const test_path = "test_img.png";

    const img = try image.loadImage(allocator, test_path);
    defer allocator.free(img.data);

    try std.testing.expect(img.width > 0);
    try std.testing.expect(img.height > 0);
    try std.testing.expect(img.channels == 3 or img.channels == 4);
}

test "scaleImage - downscaling" {
    const allocator = std.testing.allocator;
    const test_data = try allocator.alloc(u8, 100 * 100 * 3);
    defer allocator.free(test_data);
    @memset(test_data, 128); // fill with mid-gray

    const orig_img = core.Image{
        .data = test_data,
        .width = 100,
        .height = 100,
        .channels = 3,
    };

    const scaled = try image.scaleImage(allocator, orig_img, 2.0);
    defer allocator.free(scaled.data);

    try std.testing.expectEqual(@as(usize, 50), scaled.width);
    try std.testing.expectEqual(@as(usize, 50), scaled.height);
}

test "scaleImage - upscaling" {
    const allocator = std.testing.allocator;
    const test_data = try allocator.alloc(u8, 100 * 100 * 3);
    defer allocator.free(test_data);
    @memset(test_data, 128); // fill with mid-gray

    const orig_img = core.Image{
        .data = test_data,
        .width = 100,
        .height = 100,
        .channels = 3,
    };

    const scaled = try image.scaleImage(allocator, orig_img, 0.5);
    defer allocator.free(scaled.data);

    try std.testing.expectEqual(@as(usize, 200), scaled.width);
    try std.testing.expectEqual(@as(usize, 200), scaled.height);
}

test "text" {
    const allocator = std.testing.allocator;
    var test_data = try allocator.alloc(u8, 16 * 16 * 3);
    defer allocator.free(test_data);

    // Create gradient
    for (0..16) |y| {
        for (0..16) |x| {
            const idx = (y * 16 + x) * 3;
            test_data[idx] = @intCast(x * 16); // R
            test_data[idx + 1] = @intCast(y * 16); // G
            test_data[idx + 2] = 128; // B
        }
    }

    const img = core.Image{
        .data = test_data,
        .width = 16,
        .height = 16,
        .channels = 3,
    };

    const ascii_info = try core.initAsciiChars(allocator, " .:-=+*#@");
    defer allocator.free(ascii_info);

    const args = core.CoreParams{
        .input = "test_img.png",
        .output = null,
        .color = false,
        .invert_color = false,
        .auto_adjust = false,
        .scale = 1.0,
        .detect_edges = false,
        .sigma1 = 0.5,
        .sigma2 = 1.0,
        .brightness_boost = 1.0,
        .ascii_chars = " .:-=+*#@",
        .ascii_info = ascii_info,
        .stretched = false,
        .block_size = 8,
        .output_type = .Image,
        .frame_rate = null,
        .ffmpeg_options = std.StringHashMap([]const u8).init(allocator),
        .keep_audio = false,
        .codec = null,
        .dither = .None,
        .bg_color = null,
        .fg_color = null,
        .threshold_disabled = false,
    };

    const ascii_txt = try image.generateAsciiTxt(allocator, img, null, args);
    defer allocator.free(ascii_txt);

    try std.testing.expect(ascii_txt.len > 0);
    try std.testing.expect(std.mem.indexOf(u8, ascii_txt, "\n") != null);
}
