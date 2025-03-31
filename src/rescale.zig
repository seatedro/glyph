const std = @import("std");

pub const FilterType = enum {
    Box,
    Triangle,
    Bell,
    BSpline,
    Sinc,
    Lanczos3,
    CatmullRom,
    Mitchell,
};

fn resizeFilterBox(t: f32) f32 {
    if (t > -0.5 and t <= 0.5) {
        return 1.0;
    }
    return 0.0;
}

fn resizeFilterTriangle(t: f32) f32 {
    const f = @abs(t);
    if (f < 1) return 1 - f;
    return 0;
}

fn resizeFilterBell(t: f32) f32 {
    var f = @abs(t);
    if (f < 0.5) {
        return 0.75 - (f * f);
    }
    if (f < 1.5) {
        f = f - 1.5;
        return 0.5 * (f * f);
    }
    return 0.0;
}

fn resizeFilterBSpline(t: f32) f32 {
    var f = @abs(t);
    if (f < 1) {
        const tt = f * f;
        return ((0.5 * tt * f) - tt + (2.0 / 3.0));
    }
    if (f < 2) {
        f = 2 - f;
        return ((1.0 / 6.0) * (f * f * f));
    }
    return 0.0;
}

fn resizeFilterSinc(t: f32) f32 {
    const f = t * std.math.pi;
    if (f != 0) return @sin(f) / f;
    return 1.0;
}

fn resizeFilterLanczos3(t: f32) f32 {
    const f = @abs(t);
    if (f < 3) {
        return resizeFilterSinc(f) * resizeFilterSinc(f / 3);
    }
    return 0.0;
}

fn resizeFilterCatmullRom(t: f32) f32 {
    const f = @abs(t);
    if (f < 1) {
        return 1 - f * f * (2.5 - (1.5 * f));
    }
    if (f < 2) {
        return 2 - f * (4 + f * (0.5 * f - 2.5));
    }
    return 0.0;
}

fn resizeFilterMitchell(t: f32) f32 {
    const f = @abs(t);
    if (f < 1) {
        return (16 + f * f * (21 * f - 36)) / 18;
    }
    if (f < 2) {
        return (32 + f * (-60 + f * (36 - 7 * f))) / 18;
    }
    return 0.0;
}

const FilterSupport = struct {
    comptime filter: fn (f32) f32 = resizeFilterLanczos3,
    support: f32,
};

fn getFilterInfo(filter_type: FilterType) FilterSupport {
    return switch (filter_type) {
        .Box => .{ .filter = resizeFilterBox, .support = 0.5 },
        .Triangle => .{ .filter = resizeFilterTriangle, .support = 1.0 },
        .Bell => .{ .filter = resizeFilterBell, .support = 1.5 },
        .BSpline => .{ .filter = resizeFilterBSpline, .support = 2.0 },
        .Mitchell => .{ .filter = resizeFilterMitchell, .support = 2.0 },
        .CatmullRom => .{ .filter = resizeFilterCatmullRom, .support = 2.0 },
        .Sinc => .{ .filter = resizeFilterSinc, .support = 4.0 },
        .Lanczos3 => .{ .filter = resizeFilterLanczos3, .support = 3.0 },
    };
}

const ContribInfo = struct {
    first: u16,
    n: u16,
};

const ResizeWeights = struct {
    weights: []f32,
    contrib: []ContribInfo,
    width: f32,
    res: i32,
    allocator: std.mem.Allocator,

    fn deinit(self: *ResizeWeights) void {
        self.allocator.free(self.weights);
        self.allocator.free(self.contrib);
    }
};

fn makeWeights(
    allocator: std.mem.Allocator,
    from_res: i32,
    to_res: i32,
    comptime filter_fn: fn (f32) f32,
    comptime support: f32,
) !ResizeWeights {
    var result = ResizeWeights{
        .res = to_res,
        .width = undefined,
        .weights = undefined,
        .contrib = undefined,
        .allocator = allocator,
    };

    const scale = @as(f32, @floatFromInt(to_res)) / @as(f32, @floatFromInt(from_res));
    const delta = (0.5 / scale) - 0.5;
    result.width = support;
    result.width = if (scale < 1) result.width / scale else result.width;

    const weights_step = @as(usize, @intFromFloat(result.width * 2 + 1 + 1));
    result.weights = try allocator.alloc(f32, weights_step * @as(usize, @intCast(to_res)));
    result.contrib = try allocator.alloc(ContribInfo, @as(usize, @intCast(to_res)));

    const min_weight: f32 = @as(f32, 1.0) / (1 << 10);

    if (scale < 1.0) {
        var w_idx: usize = 0;
        for (0..@as(usize, @intCast(to_res))) |i| {
            const center = @as(f32, @floatFromInt(i)) / scale + delta;
            var left = @as(i32, @intFromFloat(center - result.width + 1));
            if (left < 0) left = 0;
            var right = @as(i32, @intFromFloat(center + result.width));
            if (right < 0) right = 0;

            var n: u16 = 0;
            var first: i32 = std.math.maxInt(i32);
            var total_weight: f32 = 0;

            for (@as(usize, @intCast(left))..@as(usize, @intCast(right)) + 1) |j_usize| {
                const j = @as(i32, @intCast(j_usize));
                if (j >= 0 and j < from_res) {
                    var w = center - @as(f32, @floatFromInt(j));
                    w = filter_fn(w * scale) * scale;

                    if (j < first) first = j;
                    if (j == first and (w > -min_weight and w < min_weight)) {
                        first = std.math.maxInt(i32); // not first!
                    } else {
                        result.weights[w_idx] = w;
                        w_idx += 1;
                        n += 1;
                        total_weight += w;
                    }
                }
            }

            // Trim trailing low weights
            while (n > 1 and @abs(result.weights[w_idx - 1]) < min_weight) {
                n -= 1;
                total_weight -= result.weights[w_idx - 1];
                w_idx -= 1;
            }

            // Normalize weights
            if (total_weight < (1 - min_weight)) {
                const weight_scale = 1.0 / total_weight;
                for (0..n) |j| {
                    result.weights[w_idx - n + j] *= weight_scale;
                }
            }

            result.contrib[i] = .{ .n = n, .first = @intCast(first) };
        }
    } else {
        var w_idx: usize = 0;
        for (0..@as(usize, @intCast(to_res))) |i| {
            const center = @as(f32, @floatFromInt(i)) / scale + delta;
            var left = @as(i32, @intFromFloat(center - result.width + 1));
            if (left < 0) left = 0;
            var right = @as(i32, @intFromFloat(center + result.width));
            if (right < 0) right = 0;

            var n: u16 = 0;
            var first: i32 = std.math.maxInt(i32);
            var total_weight: f32 = 0;

            for (@as(usize, @intCast(left))..@as(usize, @intCast(right)) + 1) |j_i32| {
                const j = @as(i32, @intCast(j_i32));
                if (j >= 0 and j < from_res) {
                    var w = center - @as(f32, @floatFromInt(j));
                    w = filter_fn(w);

                    if (j < first) first = j;
                    if (j == first and (w > -min_weight and w < min_weight)) {
                        first = std.math.maxInt(i32); // not first!
                    } else {
                        result.weights[w_idx] = w;
                        w_idx += 1;
                        n += 1;
                        total_weight += w;
                    }
                }
            }

            // Trim trailing low weights
            while (n > 1 and @abs(result.weights[w_idx - 1]) < min_weight) {
                n -= 1;
                total_weight -= result.weights[w_idx - 1];
                w_idx -= 1;
            }

            // Normalize weights
            if (total_weight < (1 - min_weight)) {
                const weight_scale = 1.0 / total_weight;
                for (0..n) |j| {
                    result.weights[w_idx - n + j] *= weight_scale;
                }
            }

            result.contrib[i] = .{ .n = n, .first = @intCast(first) };
        }
    }

    return result;
}

/// Generic resize function that works with any image type with
/// width, height, channels and data fields
pub fn resizeImage(
    comptime Image: type,
    allocator: std.mem.Allocator,
    img: Image,
    new_width: usize,
    new_height: usize,
    comptime filter_type: FilterType,
) !Image {
    // Safety checks
    if (img.width == 0 or img.height == 0 or new_width == 0 or new_height == 0) {
        return error.InvalidDimensions;
    }

    // Get filter information
    const filter_info = comptime getFilterInfo(filter_type);
    var wx = try makeWeights(
        allocator,
        @intCast(img.width),
        @intCast(new_width),
        filter_info.filter,
        filter_info.support,
    );
    defer wx.deinit();

    var wy = try makeWeights(
        allocator,
        @intCast(img.height),
        @intCast(new_height),
        filter_info.filter,
        filter_info.support,
    );
    defer wy.deinit();

    const from_max = @max(img.width, img.height);
    const to_max = @max(new_width, new_height);
    const all_max = @max(from_max, to_max);

    // Allocate temporary buffers
    var ftmp = try allocator.alloc(f32, all_max * img.channels);
    defer allocator.free(ftmp);

    const ring_size = @as(usize, @intFromFloat(wy.width * 2 + 1));
    var ring_buf = try allocator.alloc(f32, new_width * ring_size * img.channels);
    defer allocator.free(ring_buf);

    // Allocate output buffer
    const total_pixels = new_width * new_height;
    const buffer_size = total_pixels * img.channels;
    const output = try allocator.alloc(u8, buffer_size);
    errdefer allocator.free(output);

    var scanline_h: usize = 0;
    var scanline_v: usize = 0;

    while (scanline_v < new_height) {
        if (scanline_h < img.height) {
            // Process horizontal scanline
            for (0..img.width) |x| {
                for (0..img.channels) |c| {
                    ftmp[x * img.channels + c] = @floatFromInt(img.data[(scanline_h * img.width + x) * img.channels + c]);
                }
            }

            var h = ring_buf[(scanline_h % ring_size) * new_width * img.channels ..];
            var w_idx: usize = 0;

            for (0..new_width) |i| {
                const first = wx.contrib[i].first;
                const n = wx.contrib[i].n;

                for (0..img.channels) |c| {
                    var val: f32 = 0;

                    for (0..n) |j| {
                        const src_idx = first + j;
                        val += ftmp[src_idx * img.channels + c] * wx.weights[w_idx + j];
                    }

                    h[i * img.channels + c] = val;
                }

                w_idx += n;
            }

            scanline_h += 1;
        }

        while (scanline_v < new_height and scanline_h >= (wy.contrib[scanline_v].first + wy.contrib[scanline_v].n)) {
            // Process vertical scanline
            const first = wy.contrib[scanline_v].first;
            const n = wy.contrib[scanline_v].n;
            var w_idx: usize = 0;

            // Initialize temporary buffer with bias for integer rounding
            for (0..new_width * img.channels) |i| {
                ftmp[i] = 0.5; // bias for rounding
            }

            for (0..n) |j| {
                const src_idx = (first + j) % ring_size;
                const weight = wy.weights[w_idx + j];
                const h = ring_buf[src_idx * new_width * img.channels ..];

                for (0..new_width * img.channels) |i| {
                    ftmp[i] += h[i] * weight;
                }
            }

            // Write to output
            for (0..new_width) |x| {
                for (0..img.channels) |c| {
                    const val = std.math.clamp(ftmp[x * img.channels + c], 0, 255);
                    output[(scanline_v * new_width + x) * img.channels + c] = @intFromFloat(val);
                }
            }

            w_idx += n;
            scanline_v += 1;
        }
    }

    return Image{
        .data = output,
        .width = new_width,
        .height = new_height,
        .channels = img.channels,
    };
}
