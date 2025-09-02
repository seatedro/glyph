# flake needed to make glyph run on nixOS
{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    zig-overlay.url = "github:mitchellh/zig-overlay";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      zig-overlay,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ zig-overlay.overlays.default ];
          config = {
            allowUnfree = true;
          };
        };

        zig = pkgs.zigpkgs."0.14.1";

        # FFmpeg dependencies - only on Linux
        ffmpegDeps = pkgs.lib.optionals pkgs.stdenv.isLinux [
          # VAAPI support
          pkgs.libva
          pkgs.libva-utils
          pkgs.libdrm

          # NVENC support
          pkgs.cudatoolkit
          pkgs.cudaPackages.cuda_cudart
          pkgs.nv-codec-headers
        ];

        glyph = pkgs.stdenv.mkDerivation {
          pname = "glyph";
          version = "1.1.0";

          src = self; # root of this flake

          nativeBuildInputs = [
            zig
            pkgs.pkg-config
            pkgs.makeWrapper
          ];

          buildInputs = ffmpegDeps;

          buildPhase = ''
            zig build
          '';

          installPhase = ''
            mkdir -p $out/bin
            cp zig-out/bin/glyph $out/bin/

            wrapProgram $out/bin/glyph \
              --prefix LD_LIBRARY_PATH : "${pkgs.lib.makeLibraryPath ffmpegDeps}"
          '';
        };
      in
      {
        packages.default = glyph;

        devShells.default = pkgs.mkShell {
          inputsFrom = [ glyph ];
          packages = ffmpegDeps;

          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath ffmpegDeps;

          # Additional environment variables from vendor/ffmpeg/shell.nix (Linux only)
          PKG_CONFIG_PATH = pkgs.lib.optionalString pkgs.stdenv.isLinux "${pkgs.libva.dev}/lib/pkgconfig:${pkgs.libdrm.dev}/lib/pkgconfig:${pkgs.nv-codec-headers}/lib/pkgconfig";
          C_INCLUDE_PATH = pkgs.lib.optionalString pkgs.stdenv.isLinux "${pkgs.libva.dev}/include:${pkgs.libdrm.dev}/include:${pkgs.cudatoolkit}/include:${pkgs.nv-codec-headers}/include";
          CUDA_PATH = pkgs.lib.optionalString pkgs.stdenv.isLinux "${pkgs.cudatoolkit}";
          CUDA_ROOT = pkgs.lib.optionalString pkgs.stdenv.isLinux "${pkgs.cudatoolkit}";
        };
      }
    );
}
