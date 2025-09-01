const std = @import("std");
const builtin = @import("builtin");
const core = @import("libglyph");
const term = @import("libglyphterm");
const ffmpeg = @import("av");
const av = ffmpeg.c;
const stb = @import("stb");
const Thread = std.Thread;
const Mutex = Thread.Mutex;
const Condition = Thread.Condition;
const log = std.log.scoped(.video);

pub fn FrameBuffer(comptime T: type) type {
    return struct {
        const Self = @This();
        frames: std.ArrayList(T),
        w: usize = 0,
        h: usize = 0,
        channels: usize = 3,
        mutex: Mutex,
        cond: Condition,
        max_size: usize,
        is_finished: bool,
        allocator: std.mem.Allocator,
        ready: bool,

        pub fn init(allocator: std.mem.Allocator, max_size: usize) Self {
            return .{
                .frames = std.ArrayList(T).init(allocator),
                .mutex = Mutex{},
                .cond = Condition{},
                .max_size = max_size,
                .is_finished = false,
                .allocator = allocator,
                .ready = false,
            };
        }

        pub fn deinit(self: *Self) void {
            self.frames.deinit();
        }

        pub fn push(self: *Self, frame: T) !void {
            self.mutex.lock();
            defer self.mutex.unlock();

            while (self.frames.items.len >= self.max_size) {
                self.cond.wait(&self.mutex);
            }

            try self.frames.append(frame);
            self.cond.signal();
        }

        pub fn pop(self: *Self) ?T {
            self.mutex.lock();
            defer self.mutex.unlock();

            while (self.frames.items.len == 0 and !self.is_finished) {
                self.cond.wait(&self.mutex);
            }

            if (self.frames.items.len == 0) {
                return null;
            }

            const frame = self.frames.orderedRemove(0);
            self.cond.signal();
            return frame;
        }

        pub fn setFinished(self: *Self) void {
            self.mutex.lock();
            defer self.mutex.unlock();
            self.is_finished = true;
            self.cond.broadcast();
        }

        pub fn setReady(self: *Self) void {
            self.mutex.lock();
            defer self.mutex.unlock();
            self.ready = true;
            self.cond.broadcast();
        }

        pub fn waitUntilReady(self: *Self) void {
            self.mutex.lock();
            defer self.mutex.unlock();
            while (!self.ready) {
                self.cond.wait(&self.mutex);
            }
        }
    };
}

fn isGifOutput(args: core.CoreParams) bool {
    if (args.output) |output_path| {
        return std.mem.eql(u8, std.fs.path.extension(output_path), ".gif");
    }
    return false;
}

const ScalerSpec = struct {
    src_w: c_int,
    src_h: c_int,
    src_fmt: c_int,
    dst_w: c_int,
    dst_h: c_int,
    dst_fmt: c_int,
};

fn makeScaler(spec: ScalerSpec) ?*av.struct_SwsContext {
    return av.sws_getContext(
        spec.src_w,
        spec.src_h,
        spec.src_fmt,
        spec.dst_w,
        spec.dst_h,
        spec.dst_fmt,
        av.SWS_BILINEAR,
        null,
        null,
        null,
    );
}

const GifFilter = struct {
    graph: *av.AVFilterGraph,
    src: *av.AVFilterContext,
    sink: *av.AVFilterContext,

    fn init(src_w: c_int, src_h: c_int, dst_w: c_int, dst_h: c_int) !GifFilter {
        const graph = av.avfilter_graph_alloc();
        if (graph == null) return error.FailedToCreateOutputCtx;

        const buffersrc = av.avfilter_get_by_name("buffer");
        const buffersink = av.avfilter_get_by_name("buffersink");
        if (buffersrc == null or buffersink == null) return error.FailedToCreateOutputCtx;

        var args_buf: [128]u8 = undefined;
        const args_str = try std.fmt.bufPrintZ(
            &args_buf,
            "video_size={d}x{d}:pix_fmt={d}:time_base=1/100:pixel_aspect=1/1",
            .{ src_w, src_h, @as(c_int, av.AV_PIX_FMT_RGB24) },
        );

        var src_ctx_ptr: ?*av.AVFilterContext = null;
        if (av.avfilter_graph_create_filter(&src_ctx_ptr, buffersrc, "in", args_str.ptr, null, graph) < 0) {
            av.avfilter_graph_free(@constCast(@ptrCast(&graph)));
            return error.FailedToCreateOutputCtx;
        }

        var sink_ctx_ptr: ?*av.AVFilterContext = null;
        if (av.avfilter_graph_create_filter(&sink_ctx_ptr, buffersink, "out", null, null, graph) < 0) {
            av.avfilter_graph_free(@constCast(@ptrCast(&graph)));
            return error.FailedToCreateOutputCtx;
        }

        var graph_desc_buf: [256]u8 = undefined;
        const graph_desc = try std.fmt.bufPrintZ(
            &graph_desc_buf,
            "[in]scale={d}:{d}:flags=lanczos,split[s0][s1];[s0]palettegen=stats_mode=full[p];[s1][p]paletteuse=dither=sierra2_4a:new=1[out]",
            .{ dst_w, dst_h },
        );

        const inputs = av.avfilter_inout_alloc();
        const outputs = av.avfilter_inout_alloc();
        if (inputs == null or outputs == null) {
            av.avfilter_inout_free(@constCast(@ptrCast(&inputs)));
            av.avfilter_inout_free(@constCast(@ptrCast(&outputs)));
            av.avfilter_graph_free(@constCast(@ptrCast(&graph)));
            return error.FailedToCreateOutputCtx;
        }
        defer av.avfilter_inout_free(@constCast(@ptrCast(&inputs)));
        defer av.avfilter_inout_free(@constCast(@ptrCast(&outputs)));

        outputs.*.name = av.av_strdup("in");
        outputs.*.filter_ctx = src_ctx_ptr;
        outputs.*.pad_idx = 0;
        outputs.*.next = null;

        inputs.*.name = av.av_strdup("out");
        inputs.*.filter_ctx = sink_ctx_ptr;
        inputs.*.pad_idx = 0;
        inputs.*.next = null;

        if (av.avfilter_graph_parse_ptr(graph, graph_desc.ptr, @constCast(@ptrCast(&inputs)), @constCast(@ptrCast(&outputs)), null) < 0) {
            av.avfilter_graph_free(@constCast(@ptrCast(&graph)));
            return error.FailedToCreateOutputCtx;
        }
        if (av.avfilter_graph_config(graph, null) < 0) {
            av.avfilter_graph_free(@constCast(@ptrCast(&graph)));
            return error.FailedToCreateOutputCtx;
        }

        return .{ .graph = graph, .src = src_ctx_ptr.?, .sink = sink_ctx_ptr.? };
    }

    fn deinit(self: *GifFilter) void {
        var g: ?*av.AVFilterGraph = self.graph;
        av.avfilter_graph_free(@constCast(@ptrCast(&g)));
        self.* = undefined;
    }
};

fn openInputVideo(path: []const u8) !*av.AVFormatContext {
    var fmt_ctx: ?*av.AVFormatContext = null;
    if (av.avformat_open_input(
        &fmt_ctx,
        path.ptr,
        null,
        null,
    ) < 0) {
        return error.FailedToOpenInputVideo;
    }
    if (av.avformat_find_stream_info(fmt_ctx, null) < 0) {
        return error.FailedToFindStreamInfo;
    }
    return fmt_ctx.?;
}

const AVStream = struct {
    stream: *av.AVStream,
    index: c_int,
};
fn openVideoStream(fmt_ctx: *av.AVFormatContext) !AVStream {
    const index = av.av_find_best_stream(
        fmt_ctx,
        av.AVMEDIA_TYPE_VIDEO,
        -1,
        -1,
        null,
        0,
    );
    if (index < 0) {
        return error.VideoStreamNotFound;
    }

    return .{
        .stream = fmt_ctx.streams[@intCast(index)],
        .index = index,
    };
}

fn createDecoder(stream: *av.AVStream) !*av.AVCodecContext {
    const decoder = av.avcodec_find_decoder(
        stream.codecpar.*.codec_id,
    ) orelse {
        return error.DecoderNotFound;
    };
    const codex_ctx = av.avcodec_alloc_context3(decoder);
    if (codex_ctx == null) {
        return error.FailedToAllocCodecCtx;
    }
    if (av.avcodec_parameters_to_context(
        codex_ctx,
        stream.codecpar,
    ) < 0) {
        return error.FailedToSetCodecParams;
    }
    if (av.avcodec_open2(
        codex_ctx,
        decoder,
        null,
    ) < 0) {
        return error.FailedToOpenEncoder;
    }

    return codex_ctx;
}

fn setEncoderOption(enc_ctx: *av.AVCodecContext, key: []const u8, value: []const u8) bool {
    var opt: ?*const av.AVOption = null;

    opt = av.av_opt_find(@ptrCast(enc_ctx), key.ptr, null, 0, 0);
    if (opt != null) {
        if (av.av_opt_set(enc_ctx, key.ptr, value.ptr, 0) >= 0) {
            return true;
        }
    }

    if (enc_ctx.*.priv_data != null) {
        opt = av.av_opt_find(enc_ctx.*.priv_data, key.ptr, null, 0, 0);
        if (opt != null) {
            if (av.av_opt_set(enc_ctx.*.priv_data, key.ptr, value.ptr, 0) >= 0) {
                return true;
            }
        }
    }

    return false;
}

fn createEncoder(
    codec_ctx: *av.AVCodecContext,
    stream: *av.AVStream,
    args: core.CoreParams,
) !*av.AVCodecContext {
    const encoder = if (args.codec) |codec| av.avcodec_find_encoder_by_name(codec.ptr) else blk: {
        if (args.output) |output_path| {
            const ext = std.fs.path.extension(output_path);
            if (std.mem.eql(u8, ext, ".gif")) {
                break :blk av.avcodec_find_encoder_by_name("gif") orelse return error.EncoderNotFound;
            }
        }

        break :blk av.avcodec_find_encoder_by_name("libx265") orelse
            av.avcodec_find_encoder_by_name("libx264") orelse
            av.avcodec_find_encoder_by_name("h264_nvenc") orelse
            av.avcodec_find_encoder_by_name("h264_vaapi") orelse
            av.avcodec_find_encoder_by_name("h264_videotoolbox") orelse
            av.avcodec_find_encoder_by_name("mpeg4") orelse
            return error.EncoderNotFound;
    };

    const enc_ctx = av.avcodec_alloc_context3(encoder);
    if (enc_ctx == null) {
        return error.FailedToAllocCodecCtx;
    }

    enc_ctx.*.width = codec_ctx.width;
    enc_ctx.*.height = codec_ctx.height;

    const is_gif_encoder = if (args.output) |output_path|
        std.mem.eql(u8, std.fs.path.extension(output_path), ".gif")
    else
        false;

    if (is_gif_encoder) {
        // Use PAL8 (palettized) for gif encoder together with paletteuse filter output.
        // Set a stable time_base for GIF in centiseconds (1/100), so each increment of PTS is 1 frame delay unit.
        enc_ctx.*.pix_fmt = av.AV_PIX_FMT_PAL8;
        enc_ctx.*.time_base = .{ .num = 1, .den = 100 };
        enc_ctx.*.framerate = .{
            .num = codec_ctx.framerate.num,
            .den = codec_ctx.framerate.den,
        };
    } else {
        enc_ctx.*.pix_fmt = av.AV_PIX_FMT_YUV420P;

        const encoder_name = if (encoder.*.name) |name| std.mem.span(name) else "";
        if (std.mem.eql(u8, encoder_name, "mpeg4")) {
            enc_ctx.*.time_base = .{ .num = 1, .den = 25000 };
        } else {
            enc_ctx.*.time_base = stream.time_base;
        }

        enc_ctx.*.framerate = .{
            .num = codec_ctx.framerate.num,
            .den = 1,
        };
        enc_ctx.*.gop_size = 10;
        enc_ctx.*.max_b_frames = 1;
        enc_ctx.*.flags |= av.AV_CODEC_FLAG_GLOBAL_HEADER;
    }

    // Ensure the stride is aligned to 32 bytes (not needed for GIF)
    if (!is_gif_encoder) {
        const stride = (enc_ctx.*.width + 31) & ~@as(c_int, 31);
        _ = av.av_opt_set(enc_ctx, "stride", stride, 0);
    }

    var it = args.ffmpeg_options.iterator();
    while (it.next()) |entry| {
        const k = entry.key_ptr.*;
        const v = entry.value_ptr.*;
        if (!setEncoderOption(enc_ctx, k, v)) {
            log.warn("Failed to set FFmpeg option: {s}={s}", .{ k, v });
        }
    }

    if (av.avcodec_open2(enc_ctx, encoder, null) < 0) {
        return error.FailedToOpenEncoder;
    }

    return enc_ctx;
}

const OutputContext = struct {
    ctx: *av.AVFormatContext,
    video_stream: *av.AVStream,
    audio_stream: ?*av.AVStream,
};
fn createOutputCtx(output_path: []const u8, enc_ctx: *av.AVCodecContext, audio_stream: ?*av.AVStream) !OutputContext {
    var fmt_ctx: ?*av.AVFormatContext = null;
    if (av.avformat_alloc_output_context2(&fmt_ctx, null, null, output_path.ptr) < 0) {
        return error.FailedToCreateOutputCtx;
    }

    const video_stream = av.avformat_new_stream(fmt_ctx, null);
    if (video_stream == null) {
        return error.FailedToCreateNewStream;
    }

    if (av.avcodec_parameters_from_context(video_stream.*.codecpar, enc_ctx) < 0) {
        return error.FailedToSetCodecParams;
    }

    video_stream.*.time_base = enc_ctx.*.time_base;

    var audio_out_stream: ?*av.AVStream = null;
    if (audio_stream) |as| {
        audio_out_stream = av.avformat_new_stream(fmt_ctx, null);
        if (audio_out_stream == null) {
            return error.FailedToCreateAudioStream;
        }

        if (av.avcodec_parameters_copy(audio_out_stream.?.*.codecpar, as.*.codecpar) < 0) {
            return error.FailedToCopyAudioCodecParams;
        }
    }

    if (av.avio_open(&fmt_ctx.?.pb, output_path.ptr, av.AVIO_FLAG_WRITE) < 0) {
        return error.FailedToOpenOutputFile;
    }

    if (av.avformat_write_header(fmt_ctx, null) < 0) {
        return error.FailedToWriteHeader;
    }

    return .{ .ctx = fmt_ctx.?, .video_stream = video_stream, .audio_stream = audio_out_stream };
}

fn openAudioStream(fmt_ctx: *av.AVFormatContext) !AVStream {
    const index = av.av_find_best_stream(
        fmt_ctx,
        av.AVMEDIA_TYPE_AUDIO,
        -1,
        -1,
        null,
        0,
    );
    if (index < 0) {
        return error.AudioStreamNotFound;
    }

    return .{
        .stream = fmt_ctx.streams[@intCast(index)],
        .index = index,
    };
}

pub fn processVideo(allocator: std.mem.Allocator, args: core.CoreParams) !void {
    var input_ctx = try openInputVideo(args.input);
    defer av.avformat_close_input(@ptrCast(&input_ctx));

    const stream_info = try openVideoStream(input_ctx);
    var dec_ctx = try createDecoder(stream_info.stream);
    defer av.avcodec_free_context(@ptrCast(&dec_ctx));

    var enc_ctx = try createEncoder(dec_ctx, stream_info.stream, args);
    defer av.avcodec_free_context(@ptrCast(&enc_ctx));

    const input_frame_rate = @as(f64, @floatFromInt(stream_info.stream.*.r_frame_rate.num)) /
        @as(f64, @floatFromInt(stream_info.stream.*.r_frame_rate.den));
    const target_frame_rate = args.frame_rate orelse input_frame_rate;
    const frame_time_ns = @as(u64, @intFromFloat(1e9 / target_frame_rate));
    log.info("fps_in={d:.2} fps_out={d:.2} frame_ns={d}", .{ input_frame_rate, target_frame_rate, frame_time_ns });

    var audio_stream_info: ?AVStream = null;
    if (args.keep_audio) {
        audio_stream_info = openAudioStream(input_ctx) catch |err| blk: {
            if (err == error.AudioStreamNotFound) {
                log.info("No audio stream; continue without audio", .{});
                break :blk null;
            } else {
                return err;
            }
        };
    }

    var op: ?OutputContext = null;
    var t: term = undefined;
    var frames = std.ArrayList(core.Image).init(allocator);
    if (args.output) |output| {
        op = try createOutputCtx(output, enc_ctx, if (audio_stream_info) |asi| asi.stream else null);
        // Set up progress bar
    } else {
        t = try term.init(allocator, args.ascii_chars);
    }
    defer {
        if (op) |output| {
            _ = av.av_write_trailer(output.ctx);
            if ((output.ctx.oformat.*.flags & av.AVFMT_NOFILE) == 0) {
                _ = av.avio_closep(&output.ctx.pb);
            }
            av.avformat_free_context(output.ctx);
        } else {
            t.deinit();
        }
        for (frames.items) |f| {
            const img_len = f.height * f.width * f.channels;
            allocator.free(f.data[0..img_len]);
        }
        frames.deinit();
    }

    // Creates a FrameBuffer that holds enough frames for a 2 second buffer
    var frame_buf = FrameBuffer(core.Image).init(allocator, @as(usize, @intFromFloat(target_frame_rate * 2)));
    defer frame_buf.deinit();

    const producer_thread = try std.Thread.spawn(
        .{},
        producerTask,
        .{ allocator, &frame_buf, input_ctx, stream_info, audio_stream_info, dec_ctx, enc_ctx, op, t, args },
    );
    defer producer_thread.join();

    var processed_frames: usize = 0;
    const start_time = std.time.nanoTimestamp();
    var last_frame_time = std.time.nanoTimestamp();
    if (args.output_type != core.OutputType.Stdout) return;

    // Consume the frames and render if we are targeting stdout
    frame_buf.waitUntilReady();
    t.stats = .{
        .original_w = @intCast(dec_ctx.width),
        .original_h = @intCast(dec_ctx.height),
        .new_w = t.size.w,
        .new_h = t.size.h - 4,
    };
    try t.enableAsciiMode();
    defer t.disableAsciiMode() catch {};

    while (true) {
        const f = frame_buf.pop() orelse break;
        defer stb.stbi_image_free(f.data.ptr);

        const target_time: i128 = start_time + (@as(i128, processed_frames) * @as(i128, frame_time_ns));
        const curr_time = std.time.nanoTimestamp();
        const sleep_duration: i128 = target_time - curr_time;

        if (sleep_duration > 0) {
            std.time.sleep(@as(u64, @intCast(sleep_duration)));
        } else {
            // If we are lagging behind, we should probably log that we're not able to
            // match the target fps.
            // We will not sleep in this case.
        }

        const post_sleep_time = std.time.nanoTimestamp();
        const elapsed_seconds = @as(f32, @floatFromInt(post_sleep_time - start_time)) / 1e9;

        processed_frames += 1;
        t.stats.frame_count = processed_frames;
        t.stats.fps = @as(f32, @floatFromInt(processed_frames)) / elapsed_seconds;
        t.stats.frame_delay = @as(i64, @intCast(post_sleep_time - (start_time + ((processed_frames - 1) * frame_time_ns))));

        const adjusted_data = if (args.auto_adjust)
            try core.autoBrightnessContrast(allocator, f, 1.0)
        else
            f.data[0 .. f.width * f.height * f.channels];

        const adjusted_img = core.Image{
            .data = adjusted_data,
            .width = f.width,
            .height = f.height,
            .channels = f.channels,
        };
        defer if (args.auto_adjust) allocator.free(adjusted_data);

        const img_len = adjusted_img.height * adjusted_img.width * adjusted_img.channels;
        const params = term.RenderParams{
            .img = adjusted_img.data[0..img_len],
            .width = adjusted_img.width,
            .height = adjusted_img.height,
            .channels = adjusted_img.channels,
            .color = args.color,
            .invert = args.invert_color,
        };
        try t.renderAsciiArt(params);
        last_frame_time = curr_time;
    }

    for (frames.items) |f| {
        stb.stbi_image_free(f.data.ptr);
    }

    const avg_time = t.stats.total_time.? / @as(u128, t.stats.frame_count.?);
    t.stats.avg_frame_time = avg_time;
    std.debug.print("Average time for loop: {d}ms\n", .{t.stats.avg_frame_time.? / 1_000_000});
    std.debug.print("Total Time rendering: {d}ms\n", .{@divFloor((std.time.nanoTimestamp() - start_time), 1_000_000)});
}

fn producerTask(
    allocator: std.mem.Allocator,
    frame_buf: *FrameBuffer(core.Image),
    input_ctx: *av.AVFormatContext,
    stream_info: AVStream,
    audio_stream_info: ?AVStream,
    dec_ctx: *av.AVCodecContext,
    enc_ctx: *av.AVCodecContext,
    op: ?OutputContext,
    t: term,
    args: core.CoreParams,
) !void {
    var total_frames: usize = undefined;
    var progress: std.Progress.Node = undefined;
    var root_node: std.Progress.Node = undefined;
    var eta_node: std.Progress.Node = undefined;
    if (args.output_type == core.OutputType.Video) {
        total_frames = @intCast(getTotalFrames(input_ctx, stream_info));
        progress = std.Progress.start(.{});
        root_node = progress.start("Processing video", total_frames);
        eta_node = progress.start("(time elapsed (s)/time remaining(s))", 100);
    }

    var packet = av.av_packet_alloc();
    defer av.av_packet_free(&packet);

    var frame = av.av_frame_alloc();
    defer av.av_frame_free(&frame);

    var rgb_frame = av.av_frame_alloc();
    defer av.av_frame_free(&rgb_frame);

    const input_pix_fmt = dec_ctx.*.pix_fmt;
    if (builtin.mode == .Debug) {
        log.debug("input_pix_fmt={s}", .{av.av_get_pix_fmt_name(input_pix_fmt)});
    }

    const output_pix_fmt = av.AV_PIX_FMT_RGB24;

    rgb_frame.*.format = output_pix_fmt;
    rgb_frame.*.width = @max(@divFloor(dec_ctx.*.width, args.block_size) * args.block_size, 1);
    rgb_frame.*.height = @max(@divFloor(dec_ctx.*.height, args.block_size) * args.block_size, 1);
    if (av.av_frame_get_buffer(rgb_frame, 0) < 0) {
        return error.FailedToAllocFrameBuf;
    }

    const is_gif_output = isGifOutput(args);

    var output_frame: ?*av.AVFrame = null;
    if (!is_gif_output) {
        output_frame = av.av_frame_alloc();
        output_frame.?.*.format = enc_ctx.*.pix_fmt; // Use encoder's pixel format
        output_frame.?.*.width = enc_ctx.*.width;
        output_frame.?.*.height = enc_ctx.*.height;
        if (av.av_frame_get_buffer(output_frame.?, 0) < 0) {
            return error.FailedToAllocFrameBuf;
        }
    }
    defer {
        if (output_frame) |of| {
            av.av_frame_free(@constCast(@ptrCast(&of)));
        }
    }

    const sws_ctx = makeScaler(.{
        .src_w = dec_ctx.width,
        .src_h = dec_ctx.height,
        .src_fmt = input_pix_fmt,
        .dst_w = rgb_frame.*.width,
        .dst_h = rgb_frame.*.height,
        .dst_fmt = output_pix_fmt,
    });
    defer av.sws_freeContext(sws_ctx);

    var term_ctx: ?*av.struct_SwsContext = undefined;
    var term_frame: *av.struct_AVFrame = undefined;
    if (op == null) {
        term_ctx = makeScaler(.{
            .src_w = dec_ctx.width,
            .src_h = dec_ctx.height,
            .src_fmt = output_pix_fmt,
            .dst_w = @intCast(t.size.w),
            .dst_h = @intCast(t.size.h),
            .dst_fmt = output_pix_fmt,
        });
        term_frame = av.av_frame_alloc();
    }
    defer {
        if (op == null) {
            av.sws_freeContext(term_ctx.?);
            av.av_frame_free(&rgb_frame);
        }
    }

    // Set color space and range
    _ = av.sws_setColorspaceDetails(
        sws_ctx,
        av.sws_getCoefficients(av.SWS_CS_DEFAULT),
        0,
        av.sws_getCoefficients(av.SWS_CS_DEFAULT),
        0,
        0,
        (1 << 16) - 1,
        (1 << 16) - 1,
    );

    // Only create output scaler for non-GIF formats
    var out_sws_ctx: ?*av.struct_SwsContext = null;
    if (!is_gif_output) {
        out_sws_ctx = makeScaler(.{
            .src_w = rgb_frame.*.width,
            .src_h = rgb_frame.*.height,
            .src_fmt = @intCast(rgb_frame.*.format),
            .dst_w = output_frame.?.*.width,
            .dst_h = output_frame.?.*.height,
            .dst_fmt = @intCast(output_frame.?.*.format),
        });
    }
    defer if (out_sws_ctx) |ctx| av.sws_freeContext(ctx);

    var gif_filter: ?GifFilter = null;
    var gif_pts: i64 = 0; // running PTS in 1/100s for GIF
    var gif_delay_ticks: i64 = 1; // default to 1 centisecond per frame
    defer if (gif_filter) |*gf| gf.deinit();
    if (op != null and is_gif_output) {
        gif_filter = try GifFilter.init(
            @intCast(rgb_frame.*.width),
            @intCast(rgb_frame.*.height),
            enc_ctx.width,
            enc_ctx.height,
        );
        // Compute frame duration in encoder time_base units (1/100s)
        // delay = max(floor(100 / fps), 1)
        const fps_num: i64 = @intCast(if (enc_ctx.*.framerate.num != 0) enc_ctx.*.framerate.num else stream_info.stream.*.r_frame_rate.num);
        const fps_den: i64 = @intCast(if (enc_ctx.*.framerate.den != 0) enc_ctx.*.framerate.den else stream_info.stream.*.r_frame_rate.den);
        if (fps_num > 0) {
            gif_delay_ticks = @max(@divFloor(100 * fps_den, fps_num), 1);
        } else {
            gif_delay_ticks = 5;
        }
    }

    var frame_count: usize = 0;
    const start_time = std.time.milliTimestamp();
    var last_update_time = start_time;
    const update_interval: i64 = 1000;
    while (av.av_read_frame(input_ctx, packet) >= 0) {
        defer av.av_packet_unref(packet);

        if (packet.*.stream_index == stream_info.index) {
            if (av.avcodec_send_packet(dec_ctx, packet) < 0) {
                continue;
            }

            while (av.avcodec_receive_frame(dec_ctx, frame) >= 0) {
                _ = av.sws_scale(
                    sws_ctx,
                    &frame.*.data,
                    &frame.*.linesize,
                    0,
                    frame.*.height,
                    &rgb_frame.*.data,
                    &rgb_frame.*.linesize,
                );
                frame_count += 1;
                if (op) |output| {
                    try convertFrameToAscii(allocator, rgb_frame, args);
                    if (is_gif_output and gif_filter != null) {
                        // Feed ASCII-modified RGB24 frame into filtergraph
                        // Note: format should already be RGB24 from above
                        std.debug.assert(rgb_frame.*.format == av.AV_PIX_FMT_RGB24);
                        // Set monotonically increasing PTS in centiseconds
                        rgb_frame.*.pts = gif_pts;
                        if (av.av_buffersrc_write_frame(gif_filter.?.src, rgb_frame) < 0) {
                            return error.FailedToFeedFilterGraph;
                        }
                        var filt_frame = av.av_frame_alloc();
                        defer av.av_frame_free(&filt_frame);
                        while (av.av_buffersink_get_frame(gif_filter.?.sink, filt_frame) >= 0) {
                            var enc_packet = av.av_packet_alloc();
                            defer av.av_packet_free(&enc_packet);
                            if (av.avcodec_send_frame(enc_ctx, filt_frame) >= 0) {
                                while (av.avcodec_receive_packet(enc_ctx, enc_packet) >= 0) {
                                    enc_packet.*.stream_index = 0;
                                    enc_packet.*.pts = av.av_rescale_q(enc_packet.*.pts, enc_ctx.*.time_base, output.video_stream.*.time_base);
                                    enc_packet.*.dts = av.av_rescale_q(enc_packet.*.dts, enc_ctx.*.time_base, output.video_stream.*.time_base);
                                    enc_packet.*.duration = av.av_rescale_q(enc_packet.*.duration, enc_ctx.*.time_base, output.video_stream.*.time_base);
                                    _ = av.av_interleaved_write_frame(output.ctx, enc_packet);
                                }
                            }
                            av.av_frame_unref(filt_frame);
                        }
                        gif_pts += gif_delay_ticks;
                    } else {
                        // Non-GIF: scale to encoder format and encode
                        const out_frame = output_frame.?;
                        _ = av.sws_scale(
                            out_sws_ctx.?,
                            &rgb_frame.*.data,
                            &rgb_frame.*.linesize,
                            0,
                            rgb_frame.*.height,
                            &out_frame.*.data,
                            &out_frame.*.linesize,
                        );
                        out_frame.*.pts = if (frame.*.pts != av.AV_NOPTS_VALUE) frame.*.pts else @intCast(frame_count - 1);
                        var enc_packet = av.av_packet_alloc();
                        defer av.av_packet_free(&enc_packet);
                        if (av.avcodec_send_frame(enc_ctx, out_frame) >= 0) {
                            while (av.avcodec_receive_packet(enc_ctx, enc_packet) >= 0) {
                                enc_packet.*.stream_index = 0;
                                enc_packet.*.pts = av.av_rescale_q(enc_packet.*.pts, enc_ctx.*.time_base, output.video_stream.*.time_base);
                                enc_packet.*.dts = av.av_rescale_q(enc_packet.*.dts, enc_ctx.*.time_base, output.video_stream.*.time_base);
                                enc_packet.*.duration = av.av_rescale_q(enc_packet.*.duration, enc_ctx.*.time_base, output.video_stream.*.time_base);
                                _ = av.av_interleaved_write_frame(output.ctx, enc_packet);
                            }
                        }
                    }
                } else {
                    const frame_size = @as(usize, @intCast(rgb_frame.*.width)) * @as(usize, @intCast(rgb_frame.*.height)) * 3;
                    const frame_data = try allocator.alloc(u8, frame_size);
                    defer allocator.free(frame_data);
                    @memcpy(frame_data, rgb_frame.*.data[0][0..frame_size]);
                    const f = core.Image{
                        .data = frame_data,
                        .width = @intCast(rgb_frame.*.width),
                        .height = @intCast(rgb_frame.*.height),
                        .channels = 3,
                    };
                    const resized_img = try core.resizeImage(allocator, f, t.size.w, t.size.h - 4);
                    try frame_buf.push(resized_img);
                    if (frame_count == frame_buf.max_size) {
                        frame_buf.setReady();
                    }
                }
                if (args.output_type == core.OutputType.Video) {
                    root_node.completeOne();

                    const current_time = std.time.milliTimestamp();
                    if (current_time - last_update_time >= update_interval) {
                        const elapsed_time = @as(f64, @floatFromInt(current_time - start_time)) / 1000.0;
                        const frames_per_second = @as(f64, @floatFromInt(frame_count)) / elapsed_time;
                        const estimated_total_time = @as(f64, @floatFromInt(total_frames)) / frames_per_second;
                        const estimated_remaining_time = estimated_total_time - elapsed_time;

                        eta_node.setCompletedItems(@as(usize, (@intFromFloat(elapsed_time))));
                        eta_node.setEstimatedTotalItems(@intFromFloat(estimated_remaining_time));

                        last_update_time = current_time;
                    }
                }
            }
        } else if (args.keep_audio and audio_stream_info != null and packet.*.stream_index == audio_stream_info.?.index) {
            // Audio packet processing
            const output = op.?;
            packet.*.stream_index = output.audio_stream.?.index;
            packet.*.pts = av.av_rescale_q(packet.*.pts, audio_stream_info.?.stream.time_base, output.audio_stream.?.time_base);
            packet.*.dts = av.av_rescale_q(packet.*.dts, audio_stream_info.?.stream.time_base, output.audio_stream.?.time_base);
            packet.*.duration = av.av_rescale_q(packet.*.duration, audio_stream_info.?.stream.time_base, output.audio_stream.?.time_base);

            if (av.av_interleaved_write_frame(output.ctx, packet) < 0) {
                return error.FailedToWriteAudioPacket;
            }
        }
    }

    // Flush filters (for GIF) then encoder to ensure all delayed frames are written
    if (op) |output| {
        if (is_gif_output and gif_filter != null) {
            _ = av.av_buffersrc_add_frame_flags(gif_filter.?.src, null, 0);
            var filt_frame = av.av_frame_alloc();
            defer av.av_frame_free(&filt_frame);
            while (av.av_buffersink_get_frame(gif_filter.?.sink, filt_frame) >= 0) {
                var pkt2 = av.av_packet_alloc();
                defer av.av_packet_free(&pkt2);
                if (av.avcodec_send_frame(enc_ctx, filt_frame) >= 0) {
                    while (av.avcodec_receive_packet(enc_ctx, pkt2) >= 0) {
                        pkt2.*.stream_index = 0;
                        pkt2.*.pts = av.av_rescale_q(pkt2.*.pts, enc_ctx.*.time_base, output.video_stream.*.time_base);
                        pkt2.*.dts = av.av_rescale_q(pkt2.*.dts, enc_ctx.*.time_base, output.video_stream.*.time_base);
                        pkt2.*.duration = av.av_rescale_q(pkt2.*.duration, enc_ctx.*.time_base, output.video_stream.*.time_base);
                        _ = av.av_interleaved_write_frame(output.ctx, pkt2);
                    }
                }
                av.av_frame_unref(filt_frame);
            }
        }

        var flush_pkt = av.av_packet_alloc();
        defer av.av_packet_free(&flush_pkt);
        _ = av.avcodec_send_frame(enc_ctx, null);
        while (av.avcodec_receive_packet(enc_ctx, flush_pkt) >= 0) {
            flush_pkt.*.stream_index = 0;
            flush_pkt.*.pts = av.av_rescale_q(
                flush_pkt.*.pts,
                enc_ctx.*.time_base,
                output.video_stream.*.time_base,
            );
            flush_pkt.*.dts = av.av_rescale_q(
                flush_pkt.*.dts,
                enc_ctx.*.time_base,
                output.video_stream.*.time_base,
            );
            flush_pkt.*.duration = av.av_rescale_q(
                flush_pkt.*.duration,
                enc_ctx.*.time_base,
                output.video_stream.*.time_base,
            );
            _ = av.av_interleaved_write_frame(output.ctx, flush_pkt);
        }
    }
}

fn convertFrameToAscii(allocator: std.mem.Allocator, frame: *av.AVFrame, args: core.CoreParams) !void {
    const img = core.Image{
        .data = frame.data[0][0 .. @as(usize, @intCast(frame.linesize[0])) * @as(usize, @intCast(frame.height))],
        .width = @intCast(frame.width),
        .height = @intCast(frame.height),
        .channels = 3,
    };

    const expected_size = img.width * img.height * img.channels;
    const adjusted_data = if (args.auto_adjust)
        try core.autoBrightnessContrast(allocator, img, 1.0)
    else
        try allocator.dupe(u8, @as([*]u8, @ptrCast(img.data))[0..expected_size]);

    const adjusted_img = core.Image{
        .data = adjusted_data,
        .width = img.width,
        .height = img.height,
        .channels = img.channels,
    };
    defer if (args.auto_adjust) allocator.free(adjusted_data);

    const edge_result = try core.detectEdges(allocator, adjusted_img, args.detect_edges, args.sigma1, args.sigma2);

    const ascii_img = try core.generateAsciiArt(allocator, adjusted_img, edge_result, args);

    // Copy ascii art back to frame
    const out_w = (adjusted_img.width / args.block_size) * args.block_size;
    const out_h = (adjusted_img.height / args.block_size) * args.block_size;
    const frame_linesize = @as(usize, @intCast(frame.linesize[0]));

    for (0..out_h) |y| {
        const src_start = y * out_w * 3;
        const dst_start = y * frame_linesize;
        const row_size = @min(out_w * 3, frame_linesize);
        @memcpy(frame.data[0][dst_start..][0..row_size], ascii_img[src_start..][0..row_size]);
    }
}

fn getTotalFrames(fmt_ctx: *av.AVFormatContext, stream_info: AVStream) i64 {
    if (stream_info.stream.nb_frames > 0) {
        return stream_info.stream.nb_frames;
    }

    var total_frames: i64 = 0;
    var pkt: av.AVPacket = undefined;
    while (av.av_read_frame(fmt_ctx, &pkt) >= 0) {
        defer av.av_packet_unref(&pkt);
        if (pkt.stream_index == stream_info.index) {
            total_frames += 1;
        }
    }

    // Reset the file position indicator
    _ = av.avio_seek(fmt_ctx.*.pb, 0, av.SEEK_SET);
    _ = av.avformat_seek_file(
        fmt_ctx,
        stream_info.index,
        std.math.minInt(i64),
        0,
        std.math.maxInt(i64),
        0,
    );

    return total_frames;
}
