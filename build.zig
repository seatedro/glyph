const std = @import("std");
const Build = std.Build;
const Module = Build.Module;
const Version = std.SemanticVersion;

fn getEnvOrDefault(b: *std.Build, name: []const u8, default_value: []const u8) []const u8 {
    const v = std.process.getEnvVarOwned(b.allocator, name) catch {
        return default_value;
    };
    return v;
}

const BuildOptions = struct {
    libglyph: *Module,
    stb: *Module,
    term: *Module,
    libav: ?*Module,
    video: ?*Module,
    img: *Module,
    version: Version,
    av_enabled: bool,
};

pub fn build(b: *std.Build) !void {
    const optimize = b.standardOptimizeOption(.{});
    const strip = b.option(bool, "strip", "Omit debug information") orelse false;
    const target = b.standardTargetOptions(.{});
    const dep_stb = b.dependency("stb", .{});
    const av_enabled = b.option(bool, "av", "Enable FFmpeg/AV support") orelse true;

    const version = try Version.parse("1.1.0");

    const stb_module = b.addModule("stb", .{
        .root_source_file = b.path("vendor/stb.zig"),
        .target = target,
        .optimize = optimize,
    });
    stb_module.addIncludePath(dep_stb.path(""));
    stb_module.addCSourceFile(.{ .file = b.path("vendor/stb.c") });

    var av_module: ?*Module = null;
    if (av_enabled) {
        if (b.lazyDependency("ffmpeg", .{})) |ffmpeg_dep| {
            av_module = ffmpeg_dep.module("av");
        }
    }

    const libglyph = b.addModule("libglyph", .{
        .root_source_file = b.path("src/core.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "stb", .module = stb_module },
        },
    });

    const term_module = b.addModule("libglyphterm", .{
        .root_source_file = b.path("src/term.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "libglyph", .module = libglyph },
        },
    });

    var video_module: ?*Module = null;
    if (av_enabled) {
        video_module = b.addModule("libglyphav", .{
            .root_source_file = b.path("src/video.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "stb", .module = stb_module },
                .{ .name = "av", .module = av_module.? },
                .{ .name = "libglyph", .module = libglyph },
                .{ .name = "libglyphterm", .module = term_module },
            },
        });
    }

    const image_module = b.addModule("libglyphimg", .{
        .root_source_file = b.path("src/image.zig"),
        .imports = &.{
            .{ .name = "stb", .module = stb_module },
            .{ .name = "libglyph", .module = libglyph },
            .{ .name = "libglyphterm", .module = term_module },
        },
    });
    if (av_enabled) {
        image_module.addImport("av", av_module.?);
    }

    const buildOpts = &BuildOptions{
        .libglyph = libglyph,
        .stb = stb_module,
        .term = term_module,
        .img = image_module,
        .libav = av_module,
        .video = video_module,
        .version = version,
        .av_enabled = av_enabled,
    };

    try runZig(
        buildOpts,
        b,
        target,
        optimize,
        strip,
    );
}

fn setupExecutable(
    self: *const BuildOptions,
    b: *std.Build,
    name: []const u8,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    strip: bool,
    link_libc: bool,
) !*std.Build.Step.Compile {
    const exe = b.addExecutable(.{
        .name = name,
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .strip = strip,
        .link_libc = link_libc,
    });
    exe.root_module.addImport("build_options", buildOptionsModule(self, b));

    const clap = b.dependency("clap", .{});
    exe.root_module.addImport("clap", clap.module("clap"));
    exe.root_module.addImport("libglyph", self.libglyph);
    exe.root_module.addImport("libglyphimg", self.img);
    if (self.av_enabled) {
        exe.root_module.addImport("libglyphav", self.video.?);
        switch (target.result.os.tag) {
            .macos => {
                exe.linkFramework("CoreFoundation");
                exe.linkFramework("CoreMedia");
                exe.linkFramework("CoreVideo");
                exe.linkFramework("VideoToolbox");
            },
            .linux => {
                exe.linkSystemLibrary("va");
                exe.linkSystemLibrary("cuda");
                exe.linkSystemLibrary("cudart");
            },
            .windows => {
                // nvenc uses dynamic loading on Windows, no need to link CUDA libraries
                exe.linkSystemLibrary("ws2_32"); // Windows sockets
            },
            else => {},
        }
    }
    exe.root_module.addImport("libglyphterm", self.term);

    return exe;
}

fn setupTest(
    self: *const BuildOptions,
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    strip: bool,
) !*std.Build.Step.Compile {
    const unit_test = b.addTest(.{
        .root_source_file = b.path("src/tests.zig"),
        .target = target,
        .optimize = optimize,
        .strip = strip,
        .link_libc = true,
    });
    unit_test.root_module.addImport("build_options", buildOptionsModule(self, b));

    const clap = b.dependency("clap", .{});
    unit_test.root_module.addImport("clap", clap.module("clap"));
    unit_test.root_module.addImport("libglyph", self.libglyph);
    unit_test.root_module.addImport("libglyphimg", self.img);
    if (self.av_enabled) {
        unit_test.root_module.addImport("libglyphav", self.video.?);
    }
    unit_test.root_module.addImport("libglyphterm", self.term);

    return unit_test;
}

fn runZig(
    self: *const BuildOptions,
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    strip: bool,
) !void {
    const exe = try setupExecutable(
        self,
        b,
        "glyph",
        target,
        optimize,
        strip,
        true,
    );

    const exe_check = try setupExecutable(
        self,
        b,
        "glyph-check",
        target,
        optimize,
        strip,
        false,
    );
    const check_step = b.step("check", "Run the check");
    check_step.dependOn(&exe_check.step);

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| run_cmd.addArgs(args);

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    const test_step = b.step("test", "Run unit tests");
    const unit_tests = try setupTest(
        self,
        b,
        target,
        optimize,
        strip,
    );
    const run_unit_tests = b.addRunArtifact(unit_tests);
    test_step.dependOn(&run_unit_tests.step);
}

fn buildOptionsModule(self: *const BuildOptions, b: *std.Build) *Module {
    var opts = b.addOptions();

    opts.addOption(std.SemanticVersion, "version", self.version);
    opts.addOption(bool, "av", self.av_enabled);

    const mod = opts.createModule();
    return mod;
}
