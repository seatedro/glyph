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

// -----------------------
// VIDEO PROCESSING FUNCTIONS
// -----------------------

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

    // Try to find the option in AVCodecContext
    opt = av.av_opt_find(@ptrCast(enc_ctx), key.ptr, null, 0, 0);
    if (opt != null) {
        if (av.av_opt_set(enc_ctx, key.ptr, value.ptr, 0) >= 0) {
            return true;
        }
    }

    // If not found or setting failed, try in priv_data
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
    const encoder = if (args.codec) |codec| av.avcodec_find_encoder_by_name(codec.ptr) else av.avcodec_find_encoder_by_name("h264_nvenc") orelse
        av.avcodec_find_encoder_by_name("hevc_amf") orelse
        av.avcodec_find_encoder_by_name("hevc_qsv") orelse
        av.avcodec_find_encoder_by_name("hevc_videotoolbox") orelse
        av.avcodec_find_encoder_by_name("libx265") orelse
        av.avcodec_find_encoder_by_name("h264_amf") orelse
        av.avcodec_find_encoder_by_name("h264_qsv") orelse
        av.avcodec_find_encoder_by_name("libx264") orelse
        return error.EncoderNotFound;

    const enc_ctx = av.avcodec_alloc_context3(encoder);
    if (enc_ctx == null) {
        return error.FailedToAllocCodecCtx;
    }

    // Setting up encoding context
    enc_ctx.*.width = codec_ctx.width;
    enc_ctx.*.height = codec_ctx.height;
    enc_ctx.*.pix_fmt = av.AV_PIX_FMT_YUV420P;
    // enc_ctx.*.pix_fmt = encoder.*.pix_fmts[0];
    enc_ctx.*.time_base = stream.time_base;
    enc_ctx.*.framerate = .{
        .num = codec_ctx.framerate.num,
        .den = 1,
    };
    enc_ctx.*.gop_size = 10;
    enc_ctx.*.max_b_frames = 1;
    enc_ctx.*.flags |= av.AV_CODEC_FLAG_GLOBAL_HEADER;

    // Ensure the stride is aligned to 32 bytes
    const stride = (enc_ctx.*.width + 31) & ~@as(c_int, 31);
    _ = av.av_opt_set(enc_ctx, "stride", stride, 0);

    var it = args.ffmpeg_options.iterator();
    while (it.next()) |entry| {
        const k = entry.key_ptr.*;
        const v = entry.value_ptr.*;
        if (!setEncoderOption(enc_ctx, k, v)) {
            std.debug.print("Warning: Failed to set FFmpeg option: {s}={s}\n", .{ k, v });
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

    // Create audio stream
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

    // Extract frame rate
    const input_frame_rate = @as(f64, @floatFromInt(stream_info.stream.*.r_frame_rate.num)) /
        @as(f64, @floatFromInt(stream_info.stream.*.r_frame_rate.den));
    const target_frame_rate = args.frame_rate orelse input_frame_rate;
    const frame_time_ns = @as(u64, @intFromFloat(1e9 / target_frame_rate));
    std.debug.print("Input FPS: {d}, Target FPS: {d}, FrameTime: {d}\n", .{ input_frame_rate, target_frame_rate, frame_time_ns });

    var audio_stream_info: ?AVStream = null;
    if (args.keep_audio) {
        audio_stream_info = openAudioStream(input_ctx) catch |err| blk: {
            if (err == error.AudioStreamNotFound) {
                std.debug.print("No audio stream found in input video. Continuing without audio.\n", .{});
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
    // Get total frames
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
    std.debug.print("Input pixel format: {s}\n", .{av.av_get_pix_fmt_name(input_pix_fmt)});

    const output_pix_fmt = av.AV_PIX_FMT_RGB24;

    rgb_frame.*.format = output_pix_fmt;
    rgb_frame.*.width = @max(@divFloor(dec_ctx.*.width, args.block_size) * args.block_size, 1);
    rgb_frame.*.height = @max(@divFloor(dec_ctx.*.height, args.block_size) * args.block_size, 1);
    if (av.av_frame_get_buffer(rgb_frame, 0) < 0) {
        return error.FailedToAllocFrameBuf;
    }

    var yuv_frame = av.av_frame_alloc();
    defer av.av_frame_free(&yuv_frame);

    yuv_frame.*.format = av.AV_PIX_FMT_YUV420P;
    yuv_frame.*.width = enc_ctx.*.width;
    yuv_frame.*.height = enc_ctx.*.height;
    if (av.av_frame_get_buffer(yuv_frame, 0) < 0) {
        return error.FailedToAllocFrameBuf;
    }

    const sws_ctx = av.sws_getContext(
        dec_ctx.width,
        dec_ctx.height,
        input_pix_fmt,
        dec_ctx.width,
        dec_ctx.height,
        output_pix_fmt,
        av.SWS_BILINEAR,
        null,
        null,
        null,
    );
    defer av.sws_freeContext(sws_ctx);

    var term_ctx: ?*av.struct_SwsContext = undefined;
    var term_frame: *av.struct_AVFrame = undefined;
    if (op == null) {
        term_ctx = av.sws_getContext(
            dec_ctx.width,
            dec_ctx.height,
            output_pix_fmt,
            @intCast(t.size.w),
            @intCast(t.size.h),
            output_pix_fmt,
            av.SWS_BILINEAR,
            null,
            null,
            null,
        );
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

    const out_sws_ctx = av.sws_getContext(
        rgb_frame.*.width,
        rgb_frame.*.height,
        @intCast(rgb_frame.*.format),
        yuv_frame.*.width,
        yuv_frame.*.height,
        @intCast(yuv_frame.*.format),
        av.SWS_BILINEAR,
        null,
        null,
        null,
    );
    defer av.sws_freeContext(out_sws_ctx);

    var frame_count: usize = 0;
    const start_time = std.time.milliTimestamp();
    var last_update_time = start_time;
    const update_interval: i64 = 1000; // Update every 1 second
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
                    _ = av.sws_scale(
                        out_sws_ctx,
                        &rgb_frame.*.data,
                        &rgb_frame.*.linesize,
                        0,
                        rgb_frame.*.height,
                        &yuv_frame.*.data,
                        &yuv_frame.*.linesize,
                    );

                    yuv_frame.*.pts = frame.*.pts;

                    var enc_packet = av.av_packet_alloc();
                    defer av.av_packet_free(&enc_packet);

                    if (av.avcodec_send_frame(enc_ctx, yuv_frame) >= 0) {
                        while (av.avcodec_receive_packet(enc_ctx, enc_packet) >= 0) {
                            enc_packet.*.stream_index = 0;
                            enc_packet.*.pts = av.av_rescale_q(
                                enc_packet.*.pts,
                                enc_ctx.*.time_base,
                                output.video_stream.*.time_base,
                            );
                            enc_packet.*.dts = av.av_rescale_q(
                                enc_packet.*.dts,
                                enc_ctx.*.time_base,
                                output.video_stream.*.time_base,
                            );
                            enc_packet.*.duration = av.av_rescale_q(
                                enc_packet.*.duration,
                                enc_ctx.*.time_base,
                                output.video_stream.*.time_base,
                            );

                            _ = av.av_interleaved_write_frame(output.ctx, enc_packet);
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
