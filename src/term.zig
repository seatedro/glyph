const std = @import("std");
const builtin = @import("builtin");
const core = @import("libglyph");

pub const TermSize = struct {
    h: usize,
    w: usize,
};

pub const Stats = struct {
    original_w: usize,
    original_h: usize,
    new_w: usize,
    new_h: usize,
    fps: ?f32 = null,
    frame_delay: ?i64 = null,
    avg_frame_time: ?u128 = null,
    frame_count: ?u64 = null,
    total_time: ?u128 = 0,
};

const MAX_COLOR = 256;
const LAST_COLOR = MAX_COLOR - 1;

// ANSI escape codes
const ESC = "\x1B";
const CSI = ESC ++ "[";

const SHOW_CURSOR = CSI ++ "?25h";
const HIDE_CURSOR = CSI ++ "?25l";
const HOME_CURSOR = CSI ++ "1;1H";
const SAVE_CURSOR = ESC ++ "7";
const LOAD_CURSOR = ESC ++ "8";

const CLEAR_SCREEN = CSI ++ "2J";
const ALT_BUF_ENABLE = CSI ++ "?1049h";
const ALT_BUF_DISABLE = CSI ++ "?1049l";

const CLEAR_TO_EOL = CSI ++ "0K";

const RESET_COLOR = CSI ++ "0m";
const SET_FG_COLOR = "38;5";
const SET_BG_COLOR = "48;5";

const WHITE_FG = CSI ++ SET_FG_COLOR ++ ";15m";
const BLACK_BG = CSI ++ SET_BG_COLOR ++ ";0m";
const BLACK_FG = CSI ++ SET_FG_COLOR ++ ";0m";
const OG_COLOR = BLACK_BG ++ WHITE_FG;

const ASCII_TERM_ON = ALT_BUF_ENABLE ++ HIDE_CURSOR ++ HOME_CURSOR ++ CLEAR_SCREEN;
const ASCII_TERM_OFF = ALT_BUF_DISABLE ++ SHOW_CURSOR ++ "\n";

// RGB ANSI escape codes
const RGB_FG = CSI ++ "38;2;";
const RGB_BG = CSI ++ "48;2;";

const TIOCGWINSZ = std.c.T.IOCGWINSZ;

allocator: std.mem.Allocator,
stdout: std.fs.File.Writer,
stdin: std.fs.File.Reader,
size: TermSize,
ascii_chars: []const u8,
ascii_info: []core.AsciiCharInfo,
stats: Stats,
buf: []u8,
buf_index: usize,
buf_len: usize,
init_frame: []u8,
// fg: [MAX_COLOR][]u8,
// bg: [MAX_COLOR][]u8,

const Self = @This();

pub fn init(allocator: std.mem.Allocator, ascii_chars: []const u8) !Self {
    const stdout = std.io.getStdOut().writer();
    const stdin = std.io.getStdIn().reader();
    const size = try getTermSize(std.io.getStdOut().handle);

    const ascii_info = try core.initAsciiChars(allocator, ascii_chars);

    const char_size = ascii_chars.len;
    const color_size = RGB_FG.len + 12;
    const ascii_size = char_size + color_size;
    const screen_size: u64 = @intCast(ascii_size * size.w * size.h);
    const overflow_size: u64 = char_size * 100;
    const buf_size = screen_size + overflow_size;
    const buf = try allocator.alloc(u8, buf_size);

    const init_frame = std.fmt.allocPrint(
        allocator,
        "{s}{s}{s}",
        .{ HOME_CURSOR, BLACK_BG, BLACK_FG },
    ) catch unreachable;

    const self = Self{
        .allocator = allocator,
        .stdout = stdout,
        .stdin = stdin,
        .size = size,
        .ascii_chars = ascii_chars,
        .ascii_info = ascii_info,
        .stats = undefined,
        .buf = buf,
        .buf_index = 0,
        .buf_len = 0,
        .init_frame = init_frame,
        // .fg = undefined,
        // .bg = undefined,
    };

    // try self.initColor();
    return self;
}

pub fn deinit(self: *Self) void {
    self.allocator.free(self.buf);
    self.allocator.free(self.init_frame);
    self.allocator.free(self.ascii_info);
    // for (0..MAX_COLOR) |i| {
    //     self.allocator.free(self.fg[i]);
    //     self.allocator.free(self.bg[i]);
    // }
}

pub fn enableAsciiMode(self: *Self) !void {
    try self.stdout.writeAll(ASCII_TERM_ON);
}

pub fn disableAsciiMode(self: *Self) !void {
    try self.stdout.writeAll(ASCII_TERM_OFF);
}

pub fn clear(self: *Self) !void {
    try self.stdout.writeAll(CLEAR_SCREEN);
    try self.stdout.writeAll(HOME_CURSOR);
}

fn resetBuffer(self: *Self) void {
    @memset(self.buf, 0);
    self.buf_index = 0;
    self.buf_len = 0;
}

fn writeToBuffer(self: *Self, s: []const u8) void {
    @memcpy(self.buf[self.buf_index..][0..s.len], s);
    self.buf_index += s.len;
    self.buf_len += s.len;
}

pub const RenderParams = struct {
    img: []const u8,
    width: usize,
    height: usize,
    channels: usize,
    color: bool,
    invert: bool,
};
pub fn renderAsciiArt(
    self: *Self,
    params: RenderParams,
) !void {
    const v_padding: usize = (self.size.h - params.height - 1) / 2; // Account for top and bottom borders

    var i: usize = 0;
    self.writeToBuffer(self.init_frame);
    if (!params.color) {
        self.writeToBuffer(WHITE_FG);
    }
    while (i < v_padding) : (i += 1) {
        self.writeToBuffer("\n");
    }

    // Print top border
    // for (0..h_padding) |_| self.writeToBuffer(" ");
    // self.writeToBuffer("┌");
    // for (0..width) |_| self.writeToBuffer("-");
    // self.writeToBuffer("┐\n");

    var timer = try std.time.Timer.start();
    var y: usize = 0;
    while (y < params.height) : (y += 1) {
        // for (0..h_padding) |_| self.writeToBuffer(" ");
        // self.writeToBuffer("│");

        var x: usize = 0;
        while (x < params.width) : (x += 1) {
            const idx = (y * params.width + x) * params.channels;

            const ascii_char = getAsciiChar(self, params, idx);

            if (params.color) {
                var r = params.img[idx];
                var g = params.img[idx + 1];
                var b = params.img[idx + 2];
                if (params.invert) {
                    r = 255 - r;
                    g = 255 - g;
                    b = 255 - b;
                }
                var color_code_buf: [32]u8 = undefined;
                const color_code = std.fmt.bufPrint(
                    &color_code_buf,
                    RGB_FG ++ "{d};{d};{d}m{s}" ++ RESET_COLOR,
                    .{ r, g, b, ascii_char },
                ) catch unreachable;
                self.writeToBuffer(color_code);
            } else {
                self.writeToBuffer(ascii_char);
            }
        }

        // self.writeToBuffer("│\n");
        self.writeToBuffer("\n");
    }

    const elapsed = timer.read();
    self.stats.total_time.? += @as(u128, @intCast(elapsed));

    try self.flushBuffer();
    // Print bottom border
    // for (0..h_padding) |_| self.writeToBuffer(" ");
    // self.writeToBuffer("└");
    // for (0..width) |_| self.writeToBuffer("-");
    // self.writeToBuffer("┘\n");

    // Print bottom padding
    i = 0;
    while (i < v_padding) : (i += 1) {
        self.writeToBuffer("\n");
    }

    try self.flushBuffer();
    try self.printStats();
    self.resetBuffer();
}

fn getAsciiChar(
    self: *Self,
    params: RenderParams,
    idx: usize,
) []const u8 {
    const brightness = if (params.invert) 255 - params.img[idx] else params.img[idx];
    const ascii_idx = (brightness * self.ascii_info.len) / 256;
    const selected_char = self.ascii_info[@min(ascii_idx, self.ascii_info.len - 1)];
    return self.ascii_chars[selected_char.start .. selected_char.start + selected_char.len];
}

fn flushBuffer(self: *Self) !void {
    _ = try self.stdout.write(self.buf[0 .. self.buf_len - 1]);
}

pub fn printStats(self: *Self) !void {
    const original_aspect_ratio = @as(f32, @floatFromInt(self.stats.original_w)) / @as(f32, @floatFromInt(self.stats.original_h));
    const new_aspect_ratio = @as(f32, @floatFromInt(self.stats.new_w)) / @as(f32, @floatFromInt(self.stats.new_h));

    const stats_str = if (self.stats.fps) |fps| blk: {
        const s = try std.fmt.allocPrint(self.allocator, "\nOriginal: {}x{} (AR: {d:.2}) | New: {}x{} (AR: {d:.2}) | FPS: {d:.2} | Frame Delay: {d}", .{
            self.stats.original_w,
            self.stats.original_h,
            original_aspect_ratio,
            self.stats.new_w,
            self.stats.new_h,
            new_aspect_ratio,
            fps,
            self.stats.frame_delay.?,
        });
        break :blk s;
    } else try std.fmt.allocPrint(self.allocator, "\nOriginal: {}x{} (AR: {d:.2}) | New: {}x{} (AR: {d:.2}) | Term: {}x{} |", .{
        self.stats.original_w,
        self.stats.original_h,
        original_aspect_ratio,
        self.stats.new_w,
        self.stats.new_h,
        new_aspect_ratio,
        self.size.w,
        self.size.h,
    });
    defer self.allocator.free(stats_str);

    self.writeToBuffer(stats_str);
    try self.flushBuffer();
}

pub fn getTermSize(tty: std.posix.fd_t) !TermSize {
    switch (builtin.os.tag) {
        .windows => {
            const win32 = std.os.windows.kernel32;
            var info: std.os.windows.CONSOLE_SCREEN_BUFFER_INFO = undefined;
            if (win32.GetConsoleScreenBufferInfo(tty, &info) == 0) switch (win32.GetLastError()) {
                else => |e| return std.os.windows.unexpectedError(e),
            };
            return .{
                .h = @intCast(info.srWindow.Bottom - info.srWindow.Top + 1),
                .w = @intCast(info.srWindow.Right - info.srWindow.Left + 1),
            };
        },
        else => {
            var winsize = std.c.winsize{
                .col = 0,
                .row = 0,
                .xpixel = 0,
                .ypixel = 0,
            };
            const ret_val = std.c.ioctl(tty, TIOCGWINSZ, @intFromPtr(&winsize));
            const err = std.posix.errno(ret_val);

            if (ret_val >= 0) {
                return .{
                    .h = winsize.row,
                    .w = winsize.col,
                };
            } else {
                return std.posix.unexpectedErrno(err);
            }
        },
    }
}
