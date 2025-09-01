const std = @import("std");
const clap = @import("clap");
const termmod = @import("libglyphterm");

const CSI = "\x1b[";
const ALT_BUF_ENABLE = CSI ++ "?1049h";
const ALT_BUF_DISABLE = CSI ++ "?1049l";
const HIDE_CURSOR = CSI ++ "?25l";
const SHOW_CURSOR = CSI ++ "?25h";
const CLEAR_SCREEN = CSI ++ "2J";
const HOME_CURSOR = CSI ++ "H";

const UPDATE_INTERVAL_NS: u64 = 15_000_000;

const Raindrop = struct {
    x: f32,
    y: f32,
    speed: f32,
    char: u8,
    wind_x: f32, // Horizontal wind velocity
};

const BoltSegment = struct { y: usize, x: usize, created_ns: u64 };

const LightningBolt = struct {
    segments: std.ArrayList(BoltSegment),
    last_grow_ns: u64,
    target_len: usize,
    growing: bool,
    max_y: usize,
    max_x: usize,

    pub fn init(alloc: std.mem.Allocator, start_row: usize, start_col: usize, max_y: usize, max_x: usize) LightningBolt {
        var segs = std.ArrayList(BoltSegment).init(alloc);
        const now = std.time.nanoTimestamp();
        _ = segs.append(.{ .y = start_row, .x = start_col, .created_ns = @intCast(now) }) catch {};
        return .{
            .segments = segs,
            .last_grow_ns = @intCast(now),
            .target_len = std.crypto.random.intRangeAtMost(usize, max_y / 2, if (max_y > 2) max_y - 2 else 2),
            .growing = true,
            .max_y = max_y,
            .max_x = max_x,
        };
    }

    pub fn deinit(self: *LightningBolt) void {
        self.segments.deinit();
    }

    pub fn update(self: *LightningBolt) bool {
        const now: u64 = @intCast(std.time.nanoTimestamp());
        const growth_delay_ns: u64 = 2_000_000;
        if (self.growing and (now - self.last_grow_ns >= growth_delay_ns)) {
            self.last_grow_ns = now;
            var added = false;
            const last = self.segments.items[self.segments.items.len - 1];
            if (self.segments.items.len < self.target_len and last.y < self.max_y - 1) {
                var branches: usize = 1;
                if (std.crypto.random.float(f32) < 0.30) {
                    branches = 1 + std.crypto.random.intRangeAtMost(usize, 1, 2);
                }
                var current_x = last.x;
                var next_primary_x = current_x;
                var i: usize = 0;
                while (i < branches) : (i += 1) {
                    const off = std.crypto.random.intRangeAtMost(i32, -2, 2);
                    const nx_u = @as(i32, @intCast(current_x)) + off;
                    const nx = @as(
                        usize,
                        @intCast(@max(0, @min(@as(i32, @intCast(self.max_x - 1)), nx_u))),
                    );
                    const ny = @min(self.max_y - 1, last.y + 1);
                    _ = self.segments.append(.{
                        .y = ny,
                        .x = nx,
                        .created_ns = @intCast(now),
                    }) catch {};
                    if (i == 0) next_primary_x = nx;
                    current_x = nx;
                    added = true;
                }
                if (std.crypto.random.float(f32) < 0.15) {
                    const fork_off = blk: {
                        var o: i32 = 0;
                        while (o == 0) o = std.crypto.random.intRangeAtMost(i32, -3, 3);
                        break :blk o;
                    };
                    const fork_xu = @as(i32, @intCast(last.x)) + fork_off;
                    const fork_x = @as(
                        usize,
                        @intCast(@max(0, @min(@as(i32, @intCast(self.max_x - 1)), fork_xu))),
                    );
                    const fork_y = @min(self.max_y - 1, last.y + 1);
                    if (fork_x != next_primary_x) {
                        _ = self.segments.append(.{ .y = fork_y, .x = fork_x, .created_ns = @intCast(now) }) catch {};
                        added = true;
                    }
                }
            }
            if (!added or self.segments.items.len >= self.target_len or last.y >= self.max_y - 1) {
                self.growing = false;
            }
        }

        const lifespan_ns: u64 = 800_000_000; // 0.8s
        var any_alive = false;
        for (self.segments.items) |seg| {
            if (now - seg.created_ns <= lifespan_ns) {
                any_alive = true;
                break;
            }
        }
        return any_alive;
    }
};

fn colorSgrCode(name: []const u8) []const u8 {
    if (std.mem.eql(u8, name, "black")) return CSI ++ "30m";
    if (std.mem.eql(u8, name, "red")) return CSI ++ "31m";
    if (std.mem.eql(u8, name, "green")) return CSI ++ "32m";
    if (std.mem.eql(u8, name, "yellow")) return CSI ++ "33m";
    if (std.mem.eql(u8, name, "blue")) return CSI ++ "34m";
    if (std.mem.eql(u8, name, "magenta")) return CSI ++ "35m";
    if (std.mem.eql(u8, name, "cyan")) return CSI ++ "36m";
    return CSI ++ "37m"; // white default
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const params = comptime clap.parseParamsComptime(
        \\-h, --help                   Show help and exit
        \\    --rain-color <str>       Rain color name (default: cyan)
        \\    --lightning-color <str>  Lightning color name (default: yellow)
        \\    --mode <str>             Weather mode: rain, snow, storm (default: rain)
    );
    var diag = clap.Diagnostic{};
    const res = clap.parse(clap.Help, &params, clap.parsers.default, .{
        .allocator = allocator,
        .diagnostic = &diag,
    }) catch |err| {
        diag.report(std.io.getStdErr().writer(), err) catch {};
        return err;
    };
    defer res.deinit();
    if (res.args.help != 0) {
        try clap.help(std.io.getStdOut().writer(), clap.Help, &params, .{});
        return;
    }

    const rain_color = if (res.args.@"rain-color") |c| c else "cyan";
    const bolt_color = if (res.args.@"lightning-color") |c| c else "yellow";
    const mode = if (res.args.mode) |m| m else "rain";

    const stdout = std.io.getStdOut().writer();
    try stdout.writeAll(ALT_BUF_ENABLE ++ HIDE_CURSOR ++ CLEAR_SCREEN ++ HOME_CURSOR);
    defer stdout.writeAll(ALT_BUF_DISABLE ++ SHOW_CURSOR) catch {};

    var size = try termmod.getTermSize(std.io.getStdOut().handle);

    var drops = std.ArrayList(Raindrop).init(allocator);
    defer drops.deinit();
    var bolts = std.ArrayList(LightningBolt).init(allocator);
    defer {
        for (bolts.items) |*b| b.deinit();
        bolts.deinit();
    }

    const precip_chars = if (std.mem.eql(u8, mode, "snow")) "*." else if (std.mem.eql(u8, mode, "storm")) "|.`" else "|.`";
    const is_thunderstorm = std.mem.eql(u8, mode, "storm");
    const is_snow = std.mem.eql(u8, mode, "snow");

    const rng = std.crypto.random;
    var last_frame_ns: u64 = @intCast(std.time.nanoTimestamp());

    var global_wind: f32 = 0.0;
    var wind_change_time: u64 = @intCast(std.time.nanoTimestamp());

    while (true) {
        const now: u64 = @intCast(std.time.nanoTimestamp());
        const elapsed = now - last_frame_ns;
        if (elapsed < UPDATE_INTERVAL_NS) std.time.sleep(UPDATE_INTERVAL_NS - elapsed);
        last_frame_ns = @intCast(std.time.nanoTimestamp());

        size = termmod.getTermSize(std.io.getStdOut().handle) catch size;

        if (now - wind_change_time > 2_000_000_000 + rng.intRangeAtMost(u64, 0, 3_000_000_000)) {
            wind_change_time = now;
            global_wind = (rng.float(f32) - 0.5) * 0.6;
            if (is_thunderstorm) global_wind *= 2.0;
        }

        if (is_thunderstorm and bolts.items.len < 3 and rng.float(f32) < 0.005) {
            const start_col = rng.intRangeAtMost(usize, size.w / 4, if (size.w > 4) (3 * size.w) / 4 else size.w - 1);
            const start_row = rng.intRangeAtMost(usize, 0, if (size.h > 5) size.h / 5 else 0);
            try bolts.append(LightningBolt.init(allocator, start_row, start_col, size.h, size.w));
        }

        var i: usize = 0;
        while (i < bolts.items.len) {
            if (!bolts.items[i].update()) {
                bolts.items[i].deinit();
                _ = bolts.orderedRemove(i);
            } else i += 1;
        }

        const gen_chance: f32 = if (is_thunderstorm) 0.5 else 0.3;
        const max_new = if (is_thunderstorm) (size.w / 8) else (size.w / 15);
        if (rng.float(f32) < gen_chance) {
            const n = rng.intRangeAtMost(usize, 1, @max(@as(usize, 1), max_new));
            var k: usize = 0;
            while (k < n) : (k += 1) {
                const x = @as(f32, @floatFromInt(rng.intRangeAtMost(usize, 0, if (size.w > 0) size.w - 1 else 0)));
                const speed = if (is_snow) (0.1 + rng.float(f32) * 0.3) else if (is_thunderstorm) (0.3 + rng.float(f32) * 0.7) else (0.3 + rng.float(f32) * 0.3);
                const ch = precip_chars[rng.intRangeAtMost(usize, 0, precip_chars.len - 1)];
                const wind_variation = global_wind + (rng.float(f32) - 0.5) * 0.1;
                try drops.append(.{ .x = x, .y = 0, .speed = speed, .char = ch, .wind_x = wind_variation });
            }
        }

        var next = std.ArrayList(Raindrop).init(allocator);
        for (drops.items) |d| {
            const ny = d.y + d.speed;
            const nx = d.x + d.wind_x;
            const wrapped_x = if (nx < 0)
                @as(f32, @floatFromInt(size.w - 1))
            else if (nx >= @as(f32, @floatFromInt(size.w)))
                0
            else
                nx;

            if (@as(usize, @intFromFloat(ny)) < size.h) {
                try next.append(.{
                    .x = wrapped_x,
                    .y = ny,
                    .speed = d.speed,
                    .char = d.char,
                    .wind_x = d.wind_x,
                });
            }
        }
        drops.deinit();
        drops = next;

        try stdout.writeAll(CLEAR_SCREEN ++ HOME_CURSOR);

        var render_buffer = std.ArrayList(u8).init(allocator);
        defer render_buffer.deinit();

        const rain_sgr = colorSgrCode(rain_color);
        const bolt_sgr = colorSgrCode(bolt_color);
        const reset = CSI ++ "0m";

        for (bolts.items) |b| {
            const now_ns: u64 = @intCast(std.time.nanoTimestamp());
            for (b.segments.items) |seg| {
                const age = now_ns - seg.created_ns;
                const lifespan_ns: u64 = 800_000_000;
                if (age > lifespan_ns) continue;
                const norm = @as(f32, @floatFromInt(age)) / @as(f32, @floatFromInt(lifespan_ns));
                const ch: u8 = if (norm < 0.33) '#' else if (norm < 0.66) '+' else '*';
                const cursor_str = try std.fmt.allocPrint(allocator, CSI ++ "{d};{d}H{s}{c}{s}", .{
                    seg.y + 1,
                    seg.x + 1,
                    bolt_sgr,
                    ch,
                    reset,
                });
                defer allocator.free(cursor_str);
                try render_buffer.appendSlice(cursor_str);
            }
        }

        for (drops.items) |d| {
            const cursor_str = try std.fmt.allocPrint(allocator, CSI ++ "{d};{d}H{s}{c}{s}", .{
                @as(usize, @intFromFloat(d.y)) + 1,
                @as(usize, @intFromFloat(d.x)) + 1,
                rain_sgr,
                d.char,
                reset,
            });
            defer allocator.free(cursor_str);
            try render_buffer.appendSlice(cursor_str);
        }

        try stdout.writeAll(render_buffer.items);
    }
}
