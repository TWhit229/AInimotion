"""
Build AInimotion as a Windows executable.

Usage: python build.py

Output: dist/AInimotion/AInimotion.exe
        dist/AInimotion/model/ainimotion.pt  (copied automatically)

Requirements:
  - PyInstaller (pip install pyinstaller)
  - FFmpeg on PATH (https://www.gyan.dev/ffmpeg/builds/)
"""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
DIST_DIR = ROOT / "dist" / "AInimotion"
MODEL_SRC = ROOT / "model" / "ainimotion.pt"


def main():
    print("=" * 50)
    print("  Building AInimotion")
    print("=" * 50)

    # Check prerequisites
    if not MODEL_SRC.exists():
        print(f"\nError: Model not found at {MODEL_SRC}")
        sys.exit(1)

    # Run PyInstaller
    print("\n[1/3] Running PyInstaller...")
    result = subprocess.run(
        [sys.executable, "-m", "PyInstaller", "--noconfirm", "ainimotion.spec"],
        cwd=str(ROOT),
    )
    if result.returncode != 0:
        print("PyInstaller failed!")
        sys.exit(1)

    # Copy assets (icon for runtime)
    print("\n[2/4] Copying assets...")
    assets_dst = DIST_DIR / "assets"
    assets_dst.mkdir(exist_ok=True)
    for asset in (ROOT / "assets").glob("*"):
        dst = assets_dst / asset.name
        if not dst.exists():
            shutil.copy2(asset, dst)
    print(f"  Copied {len(list(assets_dst.iterdir()))} asset files")

    # Copy model
    print("\n[3/4] Copying model...")
    model_dst = DIST_DIR / "model"
    model_dst.mkdir(exist_ok=True)
    dst_file = model_dst / "ainimotion.pt"
    if not dst_file.exists():
        shutil.copy2(MODEL_SRC, dst_file)
        size_mb = dst_file.stat().st_size / 1024 / 1024
        print(f"  Copied ainimotion.pt ({size_mb:.0f} MB)")
    else:
        print("  Model already in dist (skipped)")

    # Check ffmpeg
    print("\n[4/4] Checking FFmpeg...")
    ffmpeg_in_dist = DIST_DIR / "ffmpeg.exe"
    if ffmpeg_in_dist.exists():
        print("  ffmpeg.exe found in dist folder")
    else:
        ffmpeg_on_path = shutil.which("ffmpeg")
        if ffmpeg_on_path:
            print(f"  ffmpeg found on PATH: {ffmpeg_on_path}")
            print("  (Optional: copy ffmpeg.exe and ffprobe.exe into dist/AInimotion/ for portability)")
        else:
            print("  WARNING: ffmpeg not found!")
            print("  Download from https://www.gyan.dev/ffmpeg/builds/")
            print("  Place ffmpeg.exe and ffprobe.exe in dist/AInimotion/")

    # Summary
    print("\n" + "=" * 50)
    print("  Build complete!")
    print(f"  Output: {DIST_DIR / 'AInimotion.exe'}")
    print("=" * 50)

    # List output
    total_size = sum(f.stat().st_size for f in DIST_DIR.rglob("*") if f.is_file())
    print(f"\n  Total size: {total_size / 1024 / 1024:.0f} MB")


if __name__ == "__main__":
    main()
