{
  description = "glyph - ascii from media";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        packages = {
          default = pkgs.stdenv.mkDerivation {
            pname = "glyph";
            version = "1.0.11";

            src = ./.;

            nativeBuildInputs = with pkgs; [
              zig
              pkg-config
            ];

            buildInputs = with pkgs; [
              ffmpeg
            ];

            dontConfigure = true;

            buildPhase = ''
              runHook preBuild
              export HOME=$TMPDIR
              zig build -Doptimize=ReleaseFast --cache-dir /tmp/zig-cache --global-cache-dir /tmp/zig-global-cache
              runHook postBuild
            '';

            installPhase = ''
              runHook preInstall
              mkdir -p $out/bin
              cp zig-out/bin/glyph $out/bin/
              runHook postInstall
            '';

            meta = with pkgs.lib; {
              description = "Converts images/video to ascii art";
              homepage = "https://github.com/felix-u/glyph";
              license = licenses.mit;
              maintainers = [ ];
              platforms = platforms.unix;
            };
          };

          glyph = self.packages.${system}.default;
        };

        apps = {
          default = flake-utils.lib.mkApp {
            drv = self.packages.${system}.default;
            name = "glyph";
          };
        };

        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            zig
            pkg-config
            ffmpeg
          ];

          shellHook = ''
            echo "glyph development environment"
            echo "Run 'zig build' to build the project"
            echo "Run 'zig build run -- [options]' to run directly"
          '';
        };
      });
}