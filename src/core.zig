const std = @import("std");
pub const bitmap = @import("bitmap.zig");
pub const stb = @import("stb");
const rescale = @import("libglyphrescale");

pub const OutputType = enum {
    Stdout,
    Text,
    Image,
    Video,
};

pub const SymbolType = enum {
    Ascii,
    Block,
};

pub const Image = struct {
    data: []u8,
    width: usize,
    height: usize,
    channels: usize,
};

const SobelFilter = struct {
    magnitude: []f32,
    direction: []f32,
};

pub const EdgeData = struct {
    grayscale: []u8,
    magnitude: []f32,
    direction: []f32,
};

pub const DitherType = enum { FloydSteinberg, Bayer4, Bayer8, Bayer16, None };

pub const RenderType = enum { Ascii, Pixels };

pub const CoreParams = struct {
    input: []const u8,
    output: ?[]const u8,
    color: bool,
    invert_color: bool,
    scale: f32,
    brightness_boost: f32,
    auto_adjust: bool,
    ascii_chars: []const u8,
    ascii_info: []AsciiCharInfo,
    block_size: u8,
    stretched: bool,
    output_type: OutputType,
    detect_edges: bool,
    threshold_disabled: bool,
    sigma1: f32,
    sigma2: f32,
    frame_rate: ?f32,
    ffmpeg_options: std.StringHashMap([]const u8),
    keep_audio: bool,
    codec: ?[]const u8,
    dither: ?DitherType,
    render: RenderType = .Ascii,
    dither_levels: u8 = 2,
    bg_color: ?[3]u8,
    fg_color: ?[3]u8,

    pub fn deinit(self: *CoreParams) void {
        var it = self.ffmpeg_options.iterator();
        while (it.next()) |entry| {
            self.ffmpeg_options.allocator.free(entry.key_ptr.*);
            self.ffmpeg_options.allocator.free(entry.value_ptr.*);
        }
        self.ffmpeg_options.deinit();
    }
};

pub const AsciiCharInfo = struct { start: usize, len: u8 };

// -----------------------
// CORE GLYPH FUNCTIONS
// -----------------------

pub fn initAsciiChars(allocator: std.mem.Allocator, ascii_chars: []const u8) ![]AsciiCharInfo {
    var char_info = std.ArrayList(AsciiCharInfo).init(allocator);
    defer char_info.deinit();

    var i: usize = 0;
    while (i < ascii_chars.len) {
        const len = try std.unicode.utf8ByteSequenceLength(ascii_chars[i]);
        try char_info.append(.{ .start = i, .len = @intCast(len) });
        i += len;
    }

    return char_info.toOwnedSlice();
}

pub fn selectAsciiChar(block_info: BlockInfo, args: CoreParams) []const u8 {
    const avg_brightness: usize = @intCast(block_info.sum_brightness / block_info.pixel_count);
    const boosted_brightness: usize = @intFromFloat(@as(f32, @floatFromInt(avg_brightness)) * args.brightness_boost);
    const clamped_brightness = std.math.clamp(boosted_brightness, 0, 255);

    if (args.detect_edges) {
        const avg_mag: f32 = block_info.sum_mag / @as(f32, @floatFromInt(block_info.pixel_count));
        const avg_dir: f32 = block_info.sum_dir / @as(f32, @floatFromInt(block_info.pixel_count));
        if (getEdgeChar(avg_mag, avg_dir, args.threshold_disabled)) |ec| {
            return &[_]u8{ec};
        }
    }

    if (clamped_brightness == 0) return " ";

    const char_index = (clamped_brightness * args.ascii_info.len) / 256;
    const selected_char = args.ascii_info[@min(char_index, args.ascii_info.len - 1)];
    return args.ascii_chars[selected_char.start .. selected_char.start + selected_char.len];
}

fn brightnessToAsciiIndex(brightness: u8, ascii_count: usize) usize {
    return (@as(usize, brightness) * ascii_count) / 256;
}

fn buildBrightnessLUT(allocator: std.mem.Allocator, ascii_count: usize) !struct {
    map: []u16,
    quant: []u8,
} {
    var map = try allocator.alloc(u16, 256);
    var quant = try allocator.alloc(u8, 256);
    for (0..256) |b| {
        const idx = brightnessToAsciiIndex(@intCast(b), ascii_count);
        map[b] = @intCast(@min(idx, if (ascii_count == 0) 0 else ascii_count - 1));
        const q: u32 = @as(u32, @intCast(map[b])) * (256 / @as(u32, @intCast(@max(ascii_count, 1))));
        quant[b] = @intCast(@min(q, 255));
    }
    return .{ .map = map, .quant = quant };
}

pub fn rgbToGrayScale(allocator: std.mem.Allocator, img: Image) ![]u8 {
    const grayscale_img = try allocator.alloc(u8, img.width * img.height);
    errdefer allocator.free(grayscale_img);

    // Integer luma approximation: Y ≈ (77*R + 150*G + 29*B) >> 8
    const total = img.width * img.height;
    var p: usize = 0;
    var o: usize = 0;
    while (o < total and p + 2 < img.data.len) : (o += 1) {
        const r: u32 = img.data[p + 0];
        const g: u32 = img.data[p + 1];
        const b: u32 = img.data[p + 2];
        const y8: u8 = @intCast(((77 * r + 150 * g + 29 * b) >> 8) & 0xFF);
        grayscale_img[o] = y8;
        p += img.channels;
    }

    return grayscale_img;
}

pub fn resizeImage(allocator: std.mem.Allocator, img: Image, new_width: usize, new_height: usize) !Image {
    if (img.width == 0 or img.height == 0 or new_width == 0 or new_height == 0) {
        return error.InvalidDimensions;
    }

    return rescale.resizeImage(
        Image,
        allocator,
        img,
        new_width,
        new_height,
        rescale.FilterType.Lanczos3,
    );
}

pub fn autoBrightnessContrast(
    allocator: std.mem.Allocator,
    img: Image,
    clip_hist_percent: f32,
) ![]u8 {
    const gray = try rgbToGrayScale(allocator, img);
    defer allocator.free(gray);

    var hist = [_]usize{0} ** 256;
    for (gray) |px| {
        hist[px] += 1;
    }

    var accumulator = [_]usize{0} ** 256;
    accumulator[0] = hist[0];
    for (1..256) |i| {
        accumulator[i] = accumulator[i - 1] + hist[i];
    }

    const max = accumulator[255];
    const clip_hist_count = @as(usize, @intFromFloat(@as(f32, @floatFromInt(max)) * clip_hist_percent / 100.0 / 2.0));

    var min_gray: usize = 0;
    while (accumulator[min_gray] < clip_hist_count) : (min_gray += 1) {}

    var max_gray: usize = 255;
    while (accumulator[max_gray] >= (max - clip_hist_count)) : (max_gray -= 1) {}

    const alpha = 255.0 / @as(f32, @floatFromInt(max_gray - min_gray));
    const beta = -@as(f32, @floatFromInt(min_gray)) * alpha;

    const len = img.width * img.height * img.channels;
    var res = try allocator.alloc(u8, len);
    for (0..len) |i| {
        const adjusted = @as(f32, @floatFromInt(img.data[i])) * alpha + beta;
        res[i] = @intFromFloat(std.math.clamp(adjusted, 0, 255));
    }

    return res;
}

pub fn gaussianKernel(allocator: std.mem.Allocator, sigma: f32) ![]f32 {
    const size: usize = @intFromFloat(6 * sigma);
    const kernel_size = if (size % 2 == 0) size + 1 else size;
    const half: f32 = @floatFromInt(kernel_size / 2);

    var kernel = try allocator.alloc(f32, kernel_size);
    var sum: f32 = 0;

    for (0..kernel_size) |i| {
        const x = @as(f32, @floatFromInt(i)) - half;
        kernel[i] = @exp(-(x * x) / (2 * sigma * sigma));
        sum += kernel[i];
    }

    // Normalize the kernel
    for (0..kernel_size) |i| {
        kernel[i] /= sum;
    }

    return kernel;
}

pub fn applyGaussianBlur(allocator: std.mem.Allocator, img: Image, sigma: f32) ![]u8 {
    const kernel = try gaussianKernel(allocator, sigma);
    defer allocator.free(kernel);

    var temp = try allocator.alloc(u8, img.width * img.height);
    defer allocator.free(temp);
    var res = try allocator.alloc(u8, img.width * img.height);

    // Horizontal pass
    for (0..img.height) |y| {
        for (0..img.width) |x| {
            var sum: f32 = 0;
            for (0..kernel.len) |i| {
                const ix: i32 = @as(i32, @intCast(x)) + @as(i32, @intCast(i)) - @as(i32, @intCast(kernel.len / 2));
                if (ix >= 0 and ix < img.width) {
                    sum += @as(f32, @floatFromInt(img.data[y * img.width + @as(usize, @intCast(ix))])) * kernel[i];
                }
            }
            temp[y * img.width + x] = @intFromFloat(sum);
        }
    }

    // Vertical pass
    for (0..img.height) |y| {
        for (0..img.width) |x| {
            var sum: f32 = 0;
            for (0..kernel.len) |i| {
                const iy: i32 = @as(i32, @intCast(y)) + @as(i32, @intCast(i)) - @as(i32, @intCast(kernel.len / 2));
                if (iy >= 0 and iy < img.height) {
                    sum += @as(f32, @floatFromInt(temp[@as(usize, @intCast(iy)) * img.width + x])) * kernel[i];
                }
            }
            res[y * img.width + x] = @intFromFloat(sum);
        }
    }

    return res;
}

pub fn differenceOfGaussians(allocator: std.mem.Allocator, img: Image, sigma1: f32, sigma2: f32) ![]u8 {
    const blur1 = try applyGaussianBlur(allocator, img, sigma1);
    defer allocator.free(blur1);
    const blur2 = try applyGaussianBlur(allocator, img, sigma2);
    defer allocator.free(blur2);

    var res = try allocator.alloc(u8, img.width * img.height);
    for (0..img.width * img.height) |i| {
        const diff = @as(i16, blur1[i]) - @as(i16, blur2[i]);
        res[i] = @as(u8, @intCast(std.math.clamp(diff + 128, 0, 255)));
    }

    return res;
}

pub fn applySobelFilter(allocator: std.mem.Allocator, img: Image) !SobelFilter {
    const Gx = [_][3]i32{ .{ -1, 0, 1 }, .{ -2, 0, 2 }, .{ -1, 0, 1 } };
    const Gy = [_][3]i32{ .{ -1, -2, -1 }, .{ 0, 0, 0 }, .{ 1, 2, 1 } };

    var mag = try allocator.alloc(f32, img.width * img.height);
    errdefer allocator.free(mag);

    var dir = try allocator.alloc(f32, img.width * img.height);
    errdefer allocator.free(dir);

    @memset(mag, 0);
    @memset(dir, 0);

    if (img.width < 3 or img.height < 3) {
        return SobelFilter{
            .magnitude = mag,
            .direction = dir,
        };
    }

    const height_max = if (img.height > 0) img.height - 1 else 0;
    const width_max = if (img.width > 0) img.width - 1 else 0;

    var y: usize = 1;
    while (y < height_max) : (y += 1) {
        var x: usize = 1;
        while (x < width_max) : (x += 1) {
            var gx: f32 = 0;
            var gy: f32 = 0;

            for (0..3) |i| {
                for (0..3) |j| {
                    const pixel_idx = (y + i - 1) * img.width + (x + j - 1);
                    if (pixel_idx < img.width * img.height) {
                        const pixel = img.data[pixel_idx];
                        gx += @as(f32, @floatFromInt(Gx[i][j])) * @as(f32, @floatFromInt(pixel));
                        gy += @as(f32, @floatFromInt(Gy[i][j])) * @as(f32, @floatFromInt(pixel));
                    }
                }
            }

            const idx = y * img.width + x;
            if (idx < img.width * img.height) {
                mag[idx] = @sqrt(gx * gx + gy * gy);
                dir[idx] = std.math.atan2(gy, gx);
            }
        }
    }

    return SobelFilter{
        .magnitude = mag,
        .direction = dir,
    };
}

pub fn getEdgeChar(mag: f32, dir: f32, threshold_disabled: bool) ?u8 {
    const threshold: f32 = 50;
    if (mag < threshold and !threshold_disabled) {
        return null;
    }

    const angle = (dir + std.math.pi) * (@as(f32, 180) / std.math.pi);
    return switch (@as(u8, @intFromFloat(@mod(angle + 22.5, 180) / 45))) {
        0, 4 => '-',
        1, 5 => '\\',
        2, 6 => '|',
        3, 7 => '/',
        else => unreachable,
    };
}

pub fn detectEdges(allocator: std.mem.Allocator, img: Image, detect_edges: bool, sigma1: f32, sigma2: f32) !?EdgeData {
    if (!detect_edges) {
        return null;
    }

    if (img.width == 0 or img.height == 0) {
        const empty_u8 = try allocator.alloc(u8, 0);
        const empty_f32_1 = try allocator.alloc(f32, 0);
        const empty_f32_2 = try allocator.alloc(f32, 0);

        return .{
            .grayscale = empty_u8,
            .magnitude = empty_f32_1,
            .direction = empty_f32_2,
        };
    }

    const grayscale_img = try rgbToGrayScale(allocator, img);
    errdefer allocator.free(grayscale_img);

    if (grayscale_img.len == 0) {
        const empty_f32_1 = try allocator.alloc(f32, 0);
        const empty_f32_2 = try allocator.alloc(f32, 0);

        return .{
            .grayscale = grayscale_img,
            .magnitude = empty_f32_1,
            .direction = empty_f32_2,
        };
    }

    const dog_img = try differenceOfGaussians(allocator, .{
        .data = grayscale_img,
        .width = img.width,
        .height = img.height,
        .channels = 1,
    }, sigma1, sigma2);
    defer allocator.free(dog_img);

    const edge_result = try applySobelFilter(allocator, .{
        .data = dog_img,
        .width = img.width,
        .height = img.height,
        .channels = 1,
    });

    return .{
        .grayscale = grayscale_img,
        .magnitude = edge_result.magnitude,
        .direction = edge_result.direction,
    };
}

const BlockInfo = struct {
    sum_brightness: u64,
    sum_color: [3]u64,
    pixel_count: u64,
    sum_mag: f32,
    sum_dir: f32,
};
pub fn calculateBlockInfo(
    img: Image,
    edge_result: ?EdgeData,
    x: usize,
    y: usize,
    out_w: usize,
    out_h: usize,
    args: CoreParams,
) BlockInfo {
    var info = BlockInfo{ .sum_brightness = 0, .sum_color = .{ 0, 0, 0 }, .pixel_count = 0, .sum_mag = 0, .sum_dir = 0 };

    const block_w = @min(args.block_size, out_w - x);
    const block_h = @min(args.block_size, out_h - y);

    // Fast integer luma accumulation (fallback when integrals are not available in this scope)
    // Y ≈ (77*R + 150*G + 29*B) >> 8
    for (0..block_h) |dy| {
        const iy = y + dy;
        if (iy >= img.height) break;
        var ix: usize = x;
        var dx: usize = 0;
        while (dx < block_w) : (dx += 1) {
            if (ix >= img.width) break;
            const pixel_index = (iy * img.width + ix) * img.channels;
            if (pixel_index + 2 >= img.width * img.height * img.channels) break;
            const r: u32 = img.data[pixel_index];
            const g: u32 = img.data[pixel_index + 1];
            const b: u32 = img.data[pixel_index + 2];
            const y8: u64 = @intCast(((77 * r + 150 * g + 29 * b) >> 8) & 0xFF);
            info.sum_brightness += y8;
            if (args.color) {
                info.sum_color[0] += r;
                info.sum_color[1] += g;
                info.sum_color[2] += b;
            }
            if (edge_result != null) {
                const edge_index = iy * img.width + ix;
                info.sum_mag += edge_result.?.magnitude[edge_index];
                info.sum_dir += edge_result.?.direction[edge_index];
            }
            info.pixel_count += 1;
            ix += 1;
        }
    }

    return info;
}

const IntegralPlanes = struct {
    w: usize,
    h: usize,
    // (w+1)*(h+1) summed area for easy border handling
    y: []u64,
    r: ?[]u64,
    g: ?[]u64,
    b: ?[]u64,
};

fn integralIndex(width: usize, x: usize, y: usize) usize {
    // integral array width is (w+1)
    return (y * (width + 1)) + x;
}

fn buildIntegralPlanes(allocator: std.mem.Allocator, img: Image, need_color: bool) !IntegralPlanes {
    const w = img.width;
    const h = img.height;
    const int_size = (w + 1) * (h + 1);
    var Iy = try allocator.alloc(u64, int_size);
    @memset(Iy, 0);
    var Ir: ?[]u64 = null;
    var Ig: ?[]u64 = null;
    var Ib: ?[]u64 = null;
    if (need_color) {
        Ir = try allocator.alloc(u64, int_size);
        Ig = try allocator.alloc(u64, int_size);
        Ib = try allocator.alloc(u64, int_size);
        @memset(Ir.?, 0);
        @memset(Ig.?, 0);
        @memset(Ib.?, 0);
    }

    var y_row: usize = 1;
    while (y_row <= h) : (y_row += 1) {
        var row_sum_y: u64 = 0;
        var row_sum_r: u64 = 0;
        var row_sum_g: u64 = 0;
        var row_sum_b: u64 = 0;
        var x_col: usize = 1;
        while (x_col <= w) : (x_col += 1) {
            const px = x_col - 1;
            const py = y_row - 1;
            const pidx = (py * w + px) * img.channels;
            const r: u32 = img.data[pidx + 0];
            const g: u32 = img.data[pidx + 1];
            const b: u32 = img.data[pidx + 2];
            const y8: u64 = @intCast(((77 * r + 150 * g + 29 * b) >> 8) & 0xFF);
            row_sum_y += y8;
            const above_y = Iy[integralIndex(w, x_col, y_row - 1)];
            Iy[integralIndex(w, x_col, y_row)] = row_sum_y + above_y;
            if (need_color) {
                row_sum_r += r;
                row_sum_g += g;
                row_sum_b += b;
                const idx = integralIndex(w, x_col, y_row);
                Ir.?[idx] = row_sum_r + Ir.?[integralIndex(w, x_col, y_row - 1)];
                Ig.?[idx] = row_sum_g + Ig.?[integralIndex(w, x_col, y_row - 1)];
                Ib.?[idx] = row_sum_b + Ib.?[integralIndex(w, x_col, y_row - 1)];
            }
        }
    }

    return .{ .w = w, .h = h, .y = Iy, .r = Ir, .g = Ig, .b = Ib };
}

fn freeIntegralPlanes(allocator: std.mem.Allocator, ip: *IntegralPlanes) void {
    allocator.free(ip.y);
    if (ip.r) |buf| allocator.free(buf);
    if (ip.g) |buf| allocator.free(buf);
    if (ip.b) |buf| allocator.free(buf);
}

fn rectSum(intw: usize, I: []const u64, x0: usize, y0: usize, x1: usize, y1: usize) u64 {
    // rectangle [x0,x1) x [y0,y1), coordinates in pixel space
    const ax = x0;
    const ay = y0;
    const bx = x1;
    const by = y1;
    const A = I[integralIndex(intw, ax, ay)];
    const B = I[integralIndex(intw, bx, ay)];
    const C = I[integralIndex(intw, ax, by)];
    const D = I[integralIndex(intw, bx, by)];
    return D + A - B - C; // inclusion-exclusion
}

fn calculateBlockInfoFast(
    ip: *const IntegralPlanes,
    edge_result: ?EdgeData,
    x: usize,
    y: usize,
    block_w: usize,
    block_h: usize,
    use_color: bool,
) BlockInfo {
    var info = BlockInfo{ .sum_brightness = 0, .sum_color = .{ 0, 0, 0 }, .pixel_count = 0, .sum_mag = 0, .sum_dir = 0 };
    const x1 = @min(x + block_w, ip.w);
    const y1 = @min(y + block_h, ip.h);
    if (x1 <= x or y1 <= y) return info;
    const area: u64 = @intCast((x1 - x) * (y1 - y));
    const sumY = rectSum(ip.w, ip.y, x, y, x1, y1);
    info.sum_brightness = sumY;
    info.pixel_count = area;
    if (use_color) {
        info.sum_color[0] = rectSum(ip.w, ip.r.?, x, y, x1, y1);
        info.sum_color[1] = rectSum(ip.w, ip.g.?, x, y, x1, y1);
        info.sum_color[2] = rectSum(ip.w, ip.b.?, x, y, x1, y1);
    }
    if (edge_result != null) {
        var yy: usize = y;
        while (yy < y1) : (yy += 1) {
            var xx: usize = x;
            while (xx < x1) : (xx += 1) {
                const ei = yy * ip.w + xx;
                info.sum_mag += edge_result.?.magnitude[ei];
                info.sum_dir += edge_result.?.direction[ei];
            }
        }
    }
    return info;
}

fn calculateAverageColor(block_info: BlockInfo, args: CoreParams) [3]u8 {
    if (args.color) {
        var color = [3]u8{
            @intCast(block_info.sum_color[0] / block_info.pixel_count),
            @intCast(block_info.sum_color[1] / block_info.pixel_count),
            @intCast(block_info.sum_color[2] / block_info.pixel_count),
        };

        if (args.invert_color) {
            color[0] = 255 - color[0];
            color[1] = 255 - color[1];
            color[2] = 255 - color[2];
        }

        return color;
    } else {
        return .{ 255, 255, 255 };
    }
}

fn convertToAscii(
    img: []u8,
    w: usize,
    h: usize,
    x: usize,
    y: usize,
    glyph_mask: *const [8]u8,
    color: [3]u8,
    block_size: u8,
    color_enabled: bool,
    args: CoreParams,
) !void {
    const bm = glyph_mask;
    const block_w = @min(block_size, w - x);
    const block_h = @min(block_size, img.len / (w * 3) - y);

    // Determine foreground/background triplets
    const background_color = if (args.bg_color != null) args.bg_color.? else [3]u8{ 21, 9, 27 }; // Blackcurrant
    const text_color = if (args.fg_color != null) args.fg_color.? else [3]u8{ 211, 106, 111 }; // Indian Red
    const fg: [3]u8 = if (color_enabled) color else text_color;
    const bg: [3]u8 = if (color_enabled) .{ 0, 0, 0 } else background_color;

    // Precompute a mask to limit to visible columns (top bits are leftmost)
    const top_mask: u8 = if (block_w >= 8) 0xFF else @as(u8, 0xFF) << @intCast(8 - block_w);

    var dy: usize = 0;
    while (dy < block_h) : (dy += 1) {
        const img_y = y + dy;
        if (img_y >= h) break;

        const row_start = (img_y * w + x) * 3;
        const row_end = row_start + @as(usize, block_w) * 3;
        var row_slice = img[row_start..row_end];

        if (bg[0] == 0 and bg[1] == 0 and bg[2] == 0) {
            @memset(row_slice, 0);
        } else {
            var ofs: usize = 0;
            while (ofs < row_slice.len) : (ofs += 3) {
                row_slice[ofs + 0] = bg[0];
                row_slice[ofs + 1] = bg[1];
                row_slice[ofs + 2] = bg[2];
            }
        }

        const m: u8 = bm[dy] & top_mask;
        if (m == 0) continue;

        var col: usize = 0;
        while (col < block_w) : (col += 1) {
            const bit: u8 = @as(u8, 1) << @intCast(7 - col);
            if ((m & bit) == 0) continue;
            const start = col;
            while (col + 1 < block_w and ((m & (@as(u8, 1) << @intCast(7 - (col + 1)))) != 0)) : (col += 1) {}
            const end = col + 1;
            var j: usize = start * 3;
            const j_end = end * 3;
            while (j < j_end) : (j += 3) {
                row_slice[j + 0] = fg[0];
                row_slice[j + 1] = fg[1];
                row_slice[j + 2] = fg[2];
            }
        }
    }
}

pub fn generateAsciiArt(
    allocator: std.mem.Allocator,
    img: Image,
    edge_result: ?EdgeData,
    args: CoreParams,
) ![]u8 {
    var out_w = (img.width / args.block_size) * args.block_size;
    var out_h = (img.height / args.block_size) * args.block_size;

    out_w = @max(out_w, 1);
    out_h = @max(out_h, 1);

    var curr_ditherr = if (args.dither != .None)
        try allocator.alloc(u32, out_w)
    else
        null;
    var next_ditherr = if (args.dither != .None)
        try allocator.alloc(u32, out_w)
    else
        null;
    defer if (curr_ditherr) |buf| allocator.free(buf);
    defer if (next_ditherr) |buf| allocator.free(buf);

    if (curr_ditherr) |buf| @memset(buf, 0);
    if (next_ditherr) |buf| @memset(buf, 0);

    const ascii_img = try allocator.alloc(u8, out_w * out_h * 3);

    var glyph_masks = try allocator.alloc([8]u8, args.ascii_info.len);
    defer allocator.free(glyph_masks);
    var i_mask: usize = 0;
    while (i_mask < args.ascii_info.len) : (i_mask += 1) {
        const info = args.ascii_info[i_mask];
        const slice = args.ascii_chars[info.start .. info.start + info.len];
        glyph_masks[i_mask] = try bitmap.getCharSet(slice);
    }

    const lut = try buildBrightnessLUT(allocator, args.ascii_info.len);
    defer allocator.free(lut.map);
    defer allocator.free(lut.quant);

    var ip = try buildIntegralPlanes(allocator, img, args.color);
    defer freeIntegralPlanes(allocator, &ip);

    const use_parallel = args.dither == .None and (out_h / args.block_size) >= 2;
    if (use_parallel) {
        const blocks_per_col: usize = out_h / args.block_size;
        var pool: std.Thread.Pool = undefined;
        try pool.init(.{ .allocator = allocator, .track_ids = false });
        defer pool.deinit();

        var wg: std.Thread.WaitGroup = .{};

        const n_jobs = @max(1, std.Thread.getCpuCount() catch 1);
        const stripes = @min(n_jobs, blocks_per_col);
        const blocks_per_stripe = (blocks_per_col + stripes - 1) / stripes;

        const Worker = struct {
            fn run(
                ascii_img_out: []u8,
                out_w_loc: usize,
                out_h_loc: usize,
                ip_ptr: *const IntegralPlanes,
                edge_res: ?EdgeData,
                args_loc: CoreParams,
                glyph_masks_loc: []const [8]u8,
                lut_map: []const u16,
                y0_blocks: usize,
                y_blocks: usize,
            ) void {
                const y_start: usize = y0_blocks * args_loc.block_size;
                const y_end: usize = @min(out_h_loc, y_start + y_blocks * args_loc.block_size);
                var yb: usize = y_start;
                while (yb < y_end) : (yb += args_loc.block_size) {
                    var x: usize = 0;
                    while (x < out_w_loc) : (x += args_loc.block_size) {
                        const bw = @min(args_loc.block_size, out_w_loc - x);
                        const bh = @min(args_loc.block_size, y_end - yb);
                        const block_info = calculateBlockInfoFast(ip_ptr, edge_res, x, yb, bw, bh, args_loc.color);
                        const avg_brightness: usize = @intCast(block_info.sum_brightness / block_info.pixel_count);
                        const boosted: usize = @intFromFloat(@as(f32, @floatFromInt(avg_brightness)) * args_loc.brightness_boost);
                        const clamped: u8 = @intCast(std.math.clamp(boosted, 0, 255));
                        const idx = @min(@as(usize, lut_map[clamped]), glyph_masks_loc.len - 1);
                        const avg_color = calculateAverageColor(block_info, args_loc);
                        convertToAscii(ascii_img_out, out_w_loc, out_h_loc, x, yb, &glyph_masks_loc[idx], avg_color, args_loc.block_size, args_loc.color, args_loc) catch {};
                    }
                }
            }
        };

        var s: usize = 0;
        while (s < stripes) : (s += 1) {
            const y0 = s * blocks_per_stripe;
            if (y0 >= blocks_per_col) break;
            const rem = blocks_per_col - y0;
            const take = @min(blocks_per_stripe, rem);
            pool.spawnWg(&wg, Worker.run, .{ ascii_img, out_w, out_h, &ip, edge_result, args, glyph_masks, lut.map, y0, take });
        }
        pool.waitAndWork(&wg);
    } else {
        var y: usize = 0;
        while (y < out_h) : (y += args.block_size) {
            if (args.dither != .None) {
                @memset(next_ditherr.?, 0);
            }
            var x: usize = 0;
            while (x < out_w) : (x += args.block_size) {
                const bw = @min(args.block_size, out_w - x);
                const bh = @min(args.block_size, out_h - y);
                var block_info = calculateBlockInfoFast(&ip, edge_result, x, y, bw, bh, args.color);

                if (args.dither != .None) {
                    const avg_brightness: u8 = @as(u8, @intCast(block_info.sum_brightness / block_info.pixel_count));
                    const boosted_u32: u32 = @intCast(@min(@as(usize, @intFromFloat(@as(f32, @floatFromInt(avg_brightness)) * args.brightness_boost)), 255));
                    const adjusted_brightness = boosted_u32 +
                        (if (curr_ditherr) |buf| buf[x / args.block_size] else 0);

                    const clamped_brightness = @as(u8, @intCast(std.math.clamp(adjusted_brightness, 0, 255)));
                    const q = lut.quant[clamped_brightness];
                    const closest = .{ q, @as(u32, @intCast(@as(u32, clamped_brightness) - q)) };
                    switch (args.dither.?) {
                        .FloydSteinberg => floydSteinberg(
                            curr_ditherr.?,
                            next_ditherr.?,
                            @as(u8, @intCast(x)) / args.block_size,
                            @as(u8, @intCast(out_w)) / args.block_size,
                            closest[1],
                        ),
                        .None, .Bayer4, .Bayer8, .Bayer16 => {},
                    }
                    block_info.sum_brightness = @as(u64, closest[0]) * block_info.pixel_count;
                }
                // Non-dither or post-dither: choose glyph index
                const avg_brightness2: usize = @intCast(block_info.sum_brightness / block_info.pixel_count);
                const boosted2: usize = @intFromFloat(@as(f32, @floatFromInt(avg_brightness2)) * args.brightness_boost);
                const clamped2: u8 = @intCast(std.math.clamp(boosted2, 0, 255));
                const idx2 = @min(@as(usize, lut.map[clamped2]), glyph_masks.len - 1);
                const avg_color = calculateAverageColor(block_info, args);
                try convertToAscii(ascii_img, out_w, out_h, x, y, &glyph_masks[idx2], avg_color, args.block_size, args.color, args);
            }

            if (curr_ditherr != null and next_ditherr != null) {
                const t = curr_ditherr;
                curr_ditherr = next_ditherr;
                next_ditherr = t;
                if (next_ditherr) |buf| @memset(buf, 0);
            }
        }
    }

    return ascii_img;
}

//-----------------Dithering Functions-----------------------
fn findClosestBrightness(
    desired: u8,
    ascii_chars: []const u8,
    ascii_info: []const AsciiCharInfo,
) struct { u8, u32 } {
    const brightness = @as(u32, @intCast(desired));

    const char_index = (desired * ascii_chars.len) / 256;
    const selected_char = @min(char_index, ascii_info.len - 1);

    const quantized: u32 = @as(u32, @intCast(selected_char)) * (256 / @as(u32, @intCast(ascii_info.len)));

    return .{
        @as(u8, @intCast(quantized)),
        brightness - quantized,
    };
}

/// Original Floyd Steinberg dithering algorithm
/// _ X 7
/// 3 5 1
///
/// (/16)
fn floydSteinberg(
    curr: []u32,
    next: []u32,
    x: u8,
    w: u8,
    quant_error: u32,
) void {
    if (x + 1 < w) {
        curr[x + 1] += (quant_error * 7) >> 4;
        next[x + 1] += (quant_error) >> 4;
    }
    if (x > 0) {
        next[x - 1] += (quant_error * 3) >> 4;
    }
    next[x] += (quant_error * 5) >> 4;
}

const bayer4: [16]u8 = .{
    0,  8,  2,  10,
    12, 4,  14, 6,
    3,  11, 1,  9,
    15, 7,  13, 5,
};

const bayer8: [64]u8 = .{
    0,  48, 12, 60, 3,  51, 15, 63,
    32, 16, 44, 28, 35, 19, 47, 31,
    8,  56, 4,  52, 11, 59, 7,  55,
    40, 24, 36, 20, 43, 27, 39, 23,
    2,  50, 14, 62, 1,  49, 13, 61,
    34, 18, 46, 30, 33, 17, 45, 29,
    10, 58, 6,  54, 9,  57, 5,  53,
    42, 26, 38, 22, 41, 25, 37, 21,
};

fn bayerThreshold(x: usize, y: usize, kind: DitherType) f32 {
    return switch (kind) {
        .Bayer4 => blk: {
            const n: usize = 4;
            const v: u8 = bayer4[(y % n) * n + (x % n)];
            break :blk @as(f32, @floatFromInt(v)) / @as(f32, @floatFromInt(n * n));
        },
        .Bayer8 => blk: {
            const n: usize = 8;
            const v: u8 = bayer8[(y % n) * n + (x % n)];
            break :blk @as(f32, @floatFromInt(v)) / @as(f32, @floatFromInt(n * n));
        },
        .Bayer16 => blk: {
            //TODO: proper 16x16 can be added later.
            const n: usize = 8;
            const v: u8 = bayer8[(y % n) * n + (x % n)];
            break :blk @as(f32, @floatFromInt(v)) / 64.0;
        },
        else => 0.0,
    };
}

pub fn generatePixelDither(
    allocator: std.mem.Allocator,
    img: Image,
    args: CoreParams,
) ![]u8 {
    const out_w = img.width;
    const out_h = img.height;
    const out = try allocator.alloc(u8, out_w * out_h * 3);

    const levels_u8: u8 = if (args.dither_levels < 2) 2 else args.dither_levels;
    const levels_u16: u16 = levels_u8;
    const denom_u16: u16 = if (levels_u8 <= 1) 1 else (levels_u8 - 1);

    const boost_q8_8: u16 = @intFromFloat(@as(f32, args.brightness_boost) * 256.0);

    var trow = try allocator.alloc(u8, out_w);
    defer allocator.free(trow);

    var r_row = try allocator.alloc(u16, out_w);
    defer allocator.free(r_row);
    var g_row = try allocator.alloc(u16, out_w);
    defer allocator.free(g_row);
    var b_row = try allocator.alloc(u16, out_w);
    defer allocator.free(b_row);

    const LANES: usize = 16; // tune per target

    var y: usize = 0;
    while (y < out_h) : (y += 1) {
        var n: usize = 0;
        var scale: u8 = 0;
        var use_bayer8 = false;
        switch (args.dither orelse .None) {
            .Bayer4 => {
                n = 4;
                scale = 16;
            },
            .Bayer8 => {
                n = 8;
                scale = 4;
            },
            .Bayer16 => {
                n = 8;
                scale = 4;
                use_bayer8 = true;
            },
            else => {
                n = 0;
            },
        }
        if (n == 0) {
            @memset(trow, 0);
        } else {
            const ymod = y & (n - 1);
            var xi: usize = 0;
            var base_row: [16]u8 = undefined; // enough for 8 or 4
            while (xi < n) : (xi += 1) {
                const v: u8 = if (use_bayer8 or n == 8)
                    bayer8[ymod * 8 + xi]
                else
                    bayer4[ymod * 4 + xi];
                base_row[xi] = @as(u8, v * scale);
            }
            var xrep: usize = 0;
            while (xrep < out_w) : (xrep += n) {
                const take = @min(n, out_w - xrep);
                @memcpy(trow[xrep .. xrep + take], base_row[0..take]);
            }
        }

        var x_build: usize = 0;
        while (x_build < out_w) : (x_build += 1) {
            const in_idx = (y * out_w + x_build) * img.channels;
            var r8: u16 = img.data[in_idx + 0];
            var g8: u16 = img.data[in_idx + 1];
            var b8: u16 = img.data[in_idx + 2];
            if (args.invert_color) {
                r8 = 255 - r8;
                g8 = 255 - g8;
                b8 = 255 - b8;
            }
            r_row[x_build] = @intCast(@min((@as(u32, r8) * boost_q8_8) >> 8, 255));
            g_row[x_build] = @intCast(@min((@as(u32, g8) * boost_q8_8) >> 8, 255));
            b_row[x_build] = @intCast(@min((@as(u32, b8) * boost_q8_8) >> 8, 255));
        }

        var x: usize = 0;
        const x_end = out_w & ~@as(usize, LANES - 1);
        const v_levels: @Vector(LANES, u16) = @splat(levels_u16);
        const v_maxq: @Vector(LANES, u16) = @splat(levels_u16 - 1);
        const v_255_u16: @Vector(LANES, u16) = @splat(255);
        const v_denom: @Vector(LANES, u16) = @splat(denom_u16);
        const v_half: @Vector(LANES, u16) = @splat(denom_u16 / 2);

        while (x < x_end) : (x += LANES) {
            var rv: @Vector(LANES, u16) = undefined;
            var gv: @Vector(LANES, u16) = undefined;
            var bv: @Vector(LANES, u16) = undefined;
            var tv: @Vector(LANES, u16) = undefined;
            inline for (0..LANES) |i| {
                rv[i] = r_row[x + i];
                gv[i] = g_row[x + i];
                bv[i] = b_row[x + i];
                tv[i] = trow[x + i];
            }

            if (args.color) {
                var qr = @as(@Vector(LANES, u16), @intCast((rv * v_levels + tv) >> @as(@Vector(LANES, u4), @splat(8))));
                var qg = @as(@Vector(LANES, u16), @intCast((gv * v_levels + tv) >> @as(@Vector(LANES, u4), @splat(8))));
                var qb = @as(@Vector(LANES, u16), @intCast((bv * v_levels + tv) >> @as(@Vector(LANES, u4), @splat(8))));
                // clamp to levels-1
                qr = @min(qr, v_maxq);
                qg = @min(qg, v_maxq);
                qb = @min(qb, v_maxq);
                // map to 0..255: (q*255 + denom/2)/denom
                const rr = @divTrunc(qr * v_255_u16 + v_half, v_denom);
                const gg = @divTrunc(qg * v_255_u16 + v_half, v_denom);
                const bb = @divTrunc(qb * v_255_u16 + v_half, v_denom);
                inline for (0..LANES) |i| {
                    const out_idx = (y * out_w + (x + i)) * 3;
                    out[out_idx + 0] = @intCast(rr[i]);
                    out[out_idx + 1] = @intCast(gg[i]);
                    out[out_idx + 2] = @intCast(bb[i]);
                }
            } else {
                // luminance: (77*r + 150*g + 29*b) >> 8
                const rv32: @Vector(LANES, u32) = @intCast(rv);
                const gv32: @Vector(LANES, u32) = @intCast(gv);
                const bv32: @Vector(LANES, u32) = @intCast(bv);
                const y32 = (rv32 * @as(@Vector(LANES, u32), @splat(77))) +
                    (gv32 * @as(@Vector(LANES, u32), @splat(150))) +
                    (bv32 * @as(@Vector(LANES, u32), @splat(29)));
                const yv: @Vector(LANES, u16) = @intCast(@as(@Vector(LANES, u16), @intCast(y32 >> @as(@Vector(LANES, u5), @splat(8)))));
                var q = @as(@Vector(LANES, u16), @intCast((yv * v_levels + tv) >> @as(@Vector(LANES, u4), @splat(8))));
                q = @min(q, v_maxq);
                const vv = @divTrunc(q * v_255_u16 + v_half, v_denom);
                inline for (0..LANES) |i| {
                    const out_idx = (y * out_w + (x + i)) * 3;
                    const v: u8 = @intCast(vv[i]);
                    out[out_idx + 0] = v;
                    out[out_idx + 1] = v;
                    out[out_idx + 2] = v;
                }
            }
        }

        // Tail scalar for leftover pixels
        while (x < out_w) : (x += 1) {
            const t_scaled: u16 = trow[x];
            var or_: u8 = 0;
            var og: u8 = 0;
            var ob: u8 = 0;
            if (args.color) {
                var qr: u16 = @intCast((@as(u32, r_row[x]) * @as(u32, levels_u16) + t_scaled) >> 8);
                var qg: u16 = @intCast((@as(u32, g_row[x]) * @as(u32, levels_u16) + t_scaled) >> 8);
                var qb: u16 = @intCast((@as(u32, b_row[x]) * @as(u32, levels_u16) + t_scaled) >> 8);
                if (qr >= levels_u16) qr = levels_u16 - 1;
                if (qg >= levels_u16) qg = levels_u16 - 1;
                if (qb >= levels_u16) qb = levels_u16 - 1;
                or_ = @intCast((@as(u32, qr) * 255 + (denom_u16 / 2)) / denom_u16);
                og = @intCast((@as(u32, qg) * 255 + (denom_u16 / 2)) / denom_u16);
                ob = @intCast((@as(u32, qb) * 255 + (denom_u16 / 2)) / denom_u16);
            } else {
                const y8s: u16 = @intCast(((77 * r_row[x] + 150 * g_row[x] + 29 * b_row[x]) >> 8) & 0xFF);
                var q: u16 = @intCast((@as(u32, y8s) * @as(u32, levels_u16) + t_scaled) >> 8);
                if (q >= levels_u16) q = levels_u16 - 1;
                const v: u8 = @intCast((@as(u32, q) * 255 + (denom_u16 / 2)) / denom_u16);
                or_ = v;
                og = v;
                ob = v;
            }
            const out_idx = (y * out_w + x) * 3;
            out[out_idx + 0] = or_;
            out[out_idx + 1] = og;
            out[out_idx + 2] = ob;
        }
    }

    return out;
}
