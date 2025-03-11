# glyph - ascii from media

converts images/video to ascii art


## Dependencies

these dependencies are only for the `av` library to output videos. This will be opt-in in the future.

#### Linux:
```bash
sudo apt-get install libavutil-dev libavformat-dev libavcodec-dev libswscale-dev
```

#### MacOS:
```bash
brew install ffmpeg pkgconf
```

#### Windows:
```bash
choco install ffmpeg-shared
```

## Installing

#### Homebrew
```bash
brew install glyph
```

### build from source

`zig build -Doptimize=ReleaseFast`

the above command builds an executable found at `./zig-out/bin`

if you want to just directly run the executable, run:

`zig build run -Doptimize=ReleaseFast -- [options]`

see below for explanations for available options

## Usage

run the program with the following options (the default zig install directory is `./zig-out/bin`):
   ```
   /path/to/glyph [options]
   ```
1. options:
   - `-h, --help`: print the help message and exit
   - `-i, --input <file>`: specify the input media file path (local path/URL) (required)
   - `-o, --output <file>`: specify the output media file (txt/img/vid) (required)
   - `-c, --color`: use color ascii characters (optional)
   - `-n, --invert_color`: Inverts the color values (optional)
   - `-s, --scale <float>`: set the downscale or upscale factor (optional, default: 1)
   - `-e, --detect_edges`: enable edge detection (optional)
   - `    --sigma1 <float>`: set the sigma1 value for DoG filter (optional, default: 0.3)
   - `    --sigma2 <float>`: set the sigma2 value for DoG filter (optional, default: 1.0)
   - `    --dither floydstein`: enable dithering (currently only supports floydstein algorithm)
   - `-b, --brightness_boost <float>`: increase/decrease perceived brightness (optional, default: 1.0)
   advanced options:
   - `    --full_characters`: Uses all ascii characters in generated output.
   - `    --ascii_chars <string>`: Use what characters you want to use in the generated output. (default: " .:-=+*%@#")
   - `    --disable_sort`: Prevents sorting of the ascii_chars by size.
   - `    --block_size <u8>`: Set the size of the blocks. (default: 8)
   - `    --threshold_disabled`: Disables the threshold.
   - `    --codec <string>`: Set the encoder codec like "libx264" or "hevc_videotoolbox". (default: searches for encoders on your machine)
   - `    --keep_audio`: Preserves audio from input video.
   - `    --stretched`: Resizes media to fit terminal window
   - `-f, --frame_rate`: Target frame rate for video output (default: matches input fps)

>To render on the terminal directly, just omit the output option.

>To output to a text file, use the .txt extension when setting the output option

2. examples:

   ### Image

   basic usage:
   ```bash
   glyph -i input.jpg -o output.png
   ```

   text file output:
   ```bash
   glyph -i input.jpg -o output.txt
   ```

   using color:
   ```bash
   glyph -i input.png -o output.png -c
   ```

   with edge detection, color, and custom downscale:
   ```bash
   glyph -i input.jpeg -o output.png -s 4 -e -c
   ```

   with brightness boost and url input:
   ```bash
   # bonus (this is a sweet wallpaper)
   glyph -i "https://w.wallhaven.cc/full/p9/wallhaven-p9gr2p.jpg" -o output.png -e -c -b 1.5
   ```

   terminal output (just omit the output option):
   ```bash
   glyph -i "https://w.wallhaven.cc/full/p9/wallhaven-p9gr2p.jpg" -e -c -b 1.5
   ```

   ### Video

   with an input video (no urls allowed):
   ```bash
   glyph -i /path/to/input/video.mp4 -o ascii.mp4 --codec hevc_nvenc --keep_audio
   ```

   with an input video and rendering on the terminal (stretched to fit terminal):
   ```bash
   glyph -i /path/to/input/video.mp4 --stretched -c
   ```

   with input video and custom ffmpeg encoder options:
   ```bash
   glyph -i /path/to/input/video.mp4 -o ascii.mp4 -c --codec libx264 --keep_audio-- -preset fast -crf 20
   ```

   with input video and custom ffmpeg encoder options:
   ```bash
   glyph -i /path/to/input/video.mp4 -o ascii.mp4 -c --codec libx264 --keep_audio-- -preset fast -crf 20
   ```

3. the program will generate an ascii art version of your input media and save it as a new media file.

for images: output file needs to be a `.png` since i saw some weird issues with jpegs.

4. using the long arguments on windows may or may not work. please use the short arguments for now.
