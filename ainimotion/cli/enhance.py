"""
AInimotion CLI entry point.

Main command-line interface for video enhancement.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

from ainimotion.pipeline import EnhancePipeline, get_video_info
from ainimotion.pipeline.pipeline import PipelineConfig


def format_time(seconds: float) -> str:
    """Format seconds to HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_size(bytes: int) -> str:
    """Format bytes to human readable."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024:
            return f"{bytes:.1f} {unit}"
        bytes /= 1024
    return f"{bytes:.1f} TB"


def progress_bar(current: int, total: int, width: int = 40) -> str:
    """Create a progress bar string."""
    if total == 0:
        return "=" * width
    filled = int(width * current / total)
    bar = "=" * filled + "-" * (width - filled)
    pct = current / total * 100
    return f"[{bar}] {pct:5.1f}%"


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(
        prog="ainimotion",
        description="AInimotion - GPU-accelerated anime video enhancer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ainimotion enhance input.mkv --out output.mkv
  ainimotion enhance input.mkv -o output.mkv --fps 48
  ainimotion enhance input.mkv -o output.mkv --preset quality
  ainimotion info input.mkv
        """,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # ========== enhance command ==========
    enhance_parser = subparsers.add_parser(
        "enhance",
        help="Enhance a video (interpolation + upscaling)",
    )
    enhance_parser.add_argument(
        "input",
        type=str,
        help="Input video file",
    )
    enhance_parser.add_argument(
        "--out", "-o",
        type=str,
        required=True,
        help="Output video file",
    )
    enhance_parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Target FPS (default: 2x source)",
    )
    enhance_parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Scale factor (e.g., 1.5 for 1.5x resolution)",
    )
    enhance_parser.add_argument(
        "--preset",
        type=str,
        choices=["fast", "balanced", "quality"],
        default="balanced",
        help="Quality preset (default: balanced)",
    )
    enhance_parser.add_argument(
        "--codec",
        type=str,
        choices=["h264", "h265"],
        default="h264",
        help="Output codec (default: h264)",
    )
    enhance_parser.add_argument(
        "--nvenc",
        action="store_true",
        help="Use NVIDIA hardware encoding (faster)",
    )
    enhance_parser.add_argument(
        "--temp-dir",
        type=str,
        default=None,
        help="Custom temp directory for frames",
    )
    enhance_parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Don't delete temp files after processing",
    )
    enhance_parser.add_argument(
        "--passthrough",
        action="store_true",
        help="Skip processing (decode + encode only, for testing)",
    )
    
    # ========== info command ==========
    info_parser = subparsers.add_parser(
        "info",
        help="Show video information",
    )
    info_parser.add_argument(
        "input",
        type=str,
        help="Input video file",
    )
    
    return parser


def cmd_info(args: argparse.Namespace) -> int:
    """Handle 'info' command."""
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"Error: File not found: {input_path}", file=sys.stderr)
        return 1
    
    try:
        info = get_video_info(input_path)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    print(f"\n{'='*50}")
    print(f"Video: {input_path.name}")
    print(f"{'='*50}")
    print(f"  Resolution:  {info.width}x{info.height}")
    print(f"  FPS:         {info.fps:.2f}")
    print(f"  Duration:    {format_time(info.duration)}")
    print(f"  Frames:      {info.frame_count:,}")
    print(f"  Codec:       {info.codec}")
    print(f"  Audio:       {'Yes' if info.has_audio else 'No'}")
    print(f"  File size:   {format_size(input_path.stat().st_size)}")
    print()
    
    return 0


def cmd_enhance(args: argparse.Namespace) -> int:
    """Handle 'enhance' command."""
    input_path = Path(args.input)
    output_path = Path(args.out)
    
    if not input_path.exists():
        print(f"Error: File not found: {input_path}", file=sys.stderr)
        return 1
    
    # Map preset to encoding settings
    preset_map = {
        "fast": ("fast", 23),
        "balanced": ("medium", 18),
        "quality": ("slow", 15),
    }
    enc_preset, crf = preset_map[args.preset]
    
    # Create pipeline config
    config = PipelineConfig(
        target_fps=args.fps,
        target_scale=args.scale,
        codec=args.codec,
        preset=enc_preset,
        crf=crf,
        use_nvenc=args.nvenc,
        keep_temp=args.keep_temp,
        temp_dir=Path(args.temp_dir) if args.temp_dir else None,
    )
    
    pipeline = EnhancePipeline(config)
    
    # Get input info
    try:
        info = get_video_info(input_path)
    except Exception as e:
        print(f"Error reading video: {e}", file=sys.stderr)
        return 1
    
    # Calculate target FPS
    target_fps = args.fps if args.fps else info.fps * 2
    
    print(f"\n{'='*60}")
    print(f"AInimotion - Anime Video Enhancer")
    print(f"{'='*60}")
    print(f"  Input:       {input_path.name}")
    print(f"  Output:      {output_path.name}")
    print(f"  Resolution:  {info.width}x{info.height}")
    print(f"  FPS:         {info.fps:.2f} -> {target_fps:.2f}")
    print(f"  Frames:      {info.frame_count:,}")
    print(f"  Preset:      {args.preset}")
    print(f"  Codec:       {args.codec}")
    if args.passthrough:
        print(f"  Mode:        PASSTHROUGH (no processing)")
    print(f"{'='*60}\n")
    
    start_time = datetime.now()
    current_stage = [None]  # Use list to allow mutation in closure
    
    def progress_callback(stage: str, current: int, total: int):
        """Print progress updates."""
        if stage != current_stage[0]:
            if current_stage[0] is not None:
                print()  # New line after previous stage
            current_stage[0] = stage
            print(f"\n{stage.upper()}:")
        
        bar = progress_bar(current, total)
        print(f"\r  {bar} {current:,}/{total:,}", end="", flush=True)
    
    try:
        if args.passthrough:
            pipeline.passthrough(input_path, output_path, progress_callback)
        else:
            # TODO: Add frame processor when models are ready
            pipeline.passthrough(input_path, output_path, progress_callback)
            print("\n\nNote: Processing not yet implemented, using passthrough mode.")
        
        elapsed = (datetime.now() - start_time).total_seconds()
        output_size = output_path.stat().st_size
        
        print(f"\n\n{'='*60}")
        print(f"COMPLETE!")
        print(f"  Time:        {format_time(elapsed)}")
        print(f"  Output:      {output_path}")
        print(f"  Size:        {format_size(output_size)}")
        print(f"{'='*60}\n")
        
        return 0
        
    except Exception as e:
        print(f"\n\nError: {e}", file=sys.stderr)
        return 1


def main(args: list[str] | None = None) -> int:
    """Main CLI entry point."""
    parser = create_parser()
    parsed = parser.parse_args(args)
    
    if parsed.command is None:
        parser.print_help()
        return 0
    
    if parsed.command == "info":
        return cmd_info(parsed)
    elif parsed.command == "enhance":
        return cmd_enhance(parsed)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
