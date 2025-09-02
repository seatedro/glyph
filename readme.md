# glyph — a modern ascii renderer

Converts images and videos into ASCII art.

![shinto-ascii](https://github.com/user-attachments/assets/a6676a76-3bdf-4a63-8629-e121a5943b7d)


## Installing

### Homebrew
```bash
brew install glyph
```

### Build from source
```bash
zig build -Doptimize=ReleaseFast
```
The executable is placed at `./zig-out/bin`.

To disable compiling ffmpeg and remove the libglyphav module:
```bash
zig build -Doptimize=ReleaseFast -Dav=false
```

Run directly with:
```bash
zig build run -Doptimize=ReleaseFast -- [options]
```

## Usage

Basic usage (default Zig install path is `./zig-out/bin`):
```
/path/to/glyph [options]
```

- `-h, --help`: Print help and exit
- `-v, --version`: Print version and exit
- `-i, --input <str>`: Input media file (image or video)
- `-o, --output <str>`: Output file (png/gif/mp4/… or .txt). Omit to render in terminal
- `-c, --color`: Use color ASCII characters
- `-n, --invert_color`: Invert color values
- `-a, --auto_adjust`: Auto-adjust brightness/contrast
- `-s, --scale <f32>`: Scale factor (default 1.0). Values >1 downscale (e.g., 2 halves size)
- `    --symbols <str>`: Character set: `ascii` or `block` (default: ascii)
- `-e, --detect_edges`: Enable edge detection
- `    --sigma1 <f32>`: DoG sigma1 (default 0.5)
- `    --sigma2 <f32>`: DoG sigma2 (default 1.0)
- `-b, --brightness_boost <f32>`: Brightness multiplier (default 1.0)
- `    --full_characters`: Use full character spectrum
- `    --ascii_chars <str>`: Custom characters (default: " .:-=+*%#@")
- `    --disable_sort`: Don’t sort `--ascii_chars` by size
- `    --block_size <u8>`: ASCII block size (default 8)
- `    --threshold_disabled`: Disable threshold
- `    --codec <str>`: Encoder (e.g., `libx264`, `libx265`, `hevc_nvenc`, `h264_videotoolbox`)
- `    --keep_audio`: Keep input audio in video output
- `    --stretched`: Fit render to terminal window (stdout mode)
- `-f, --frame_rate <f32>`: Target fps for video output (default: input fps)
- `-m, --mode <str>`: `ascii` or `pixels` (default: ascii)
- `-d, --dither <str>`: `none`, `floydstein`, `bayer4`, `bayer8`, `bayer16` (default: none)
- `    --dither_levels <u8>`: Levels for ordered dither (default: 2)
- `    --fg <#rrggbb>`: Foreground color (pixel mode)
- `    --bg <#rrggbb>`: Background color (pixel mode)

> To render to the terminal, omit `--output`.
> To save ASCII text, set an output with `.txt` extension.

### FFmpeg Codec Support

Glyph includes FFmpeg with support for both software and hardware-accelerated codecs:

#### Software Codecs
- **x264** (`libx264`) - H.264 video encoder
- **x265** (`libx265`) - HEVC/H.265 video encoder (decoder only)
- **libmp3lame** - MP3 audio encoder
- **libvorbis** - Vorbis audio encoder

#### Hardware Codecs (Platform-dependent)
- **NVIDIA NVENC** (`h264_nvenc`, `hevc_nvenc`) - Available on Windows/Linux with NVIDIA GPUs
- **VideoToolbox** (`h264_videotoolbox`, `hevc_videotoolbox`) - Available on macOS

Use the `--codec` parameter to specify the encoder. For example:
```bash
# Software encoding
glyph -i input.mp4 -o output.mp4 --codec libx264

# Hardware encoding (NVIDIA)
glyph -i input.mp4 -o output.mp4 --codec h264_nvenc

# Hardware encoding (macOS)
glyph -i input.mp4 -o output.mp4 --codec h264_videotoolbox
```


## Examples

### Image

Basic usage:
```bash
glyph -i input.jpg -o output.png
```

Text output:
```bash
glyph -i input.jpg -o output.txt
```

Color:
```bash
glyph -i input.png -o output.png -c
```

Edge detection, color, downscale:
```bash
glyph -i input.jpeg -o output.png -s 4 -e -c
```

Dithering
```bash
glyph -i input.png -o output.png -a -s 2 --mode pixels --dither bayer8 --dither_levels 2
```


Terminal output:
```bash
glyph -i input.jpg -e -c -b 1.5
```

### Video

Encode MP4 with NVENC, keep audio:
```bash
glyph -i /path/to/input.mp4 -o ascii.mp4 --codec hevc_nvenc --keep_audio
```

Render in terminal (fit to terminal):
```bash
glyph -i /path/to/input.mp4 --stretched -c
```

Custom encoder options (after `--`):
```bash
glyph -i /path/to/input.mp4 -o ascii.mp4 -c --codec libx264 -- --preset fast --crf 20
```

GIF output:
```bash
glyph -i /path/to/input.mp4 -o out.gif -s 2 -f 12
```

