{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  packages = with pkgs; [
    # Python + package management
    python314
    uv

    # Dev tools
    just
    prek
    ruff
    git

    # Needed by Python C extensions (e.g. torch, uvloop)
    stdenv.cc.cc.lib
  ];

  env = {
    # Let uv find the nix-provided Python
    UV_PYTHON = "${pkgs.python314}/bin/python3";

    # Ensure native libs (libstdc++ etc.) are visible to pip-installed packages
    LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
      pkgs.stdenv.cc.cc.lib
    ];
  };

  shellHook = ''
    # Create venv if it doesn't exist, sync dependencies
    if [ ! -d .venv ]; then
      uv sync
    fi
  '';
}
