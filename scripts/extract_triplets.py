#!/usr/bin/env python3
"""
Extract training triplets (F1, F2, F3) from anime video files.

Filters out static frames and duplicate-on-2s/3s frames to ensure
quality training data with actual motion.

Usage:
    python scripts/extract_triplets.py --input /path/to/videos --output /path/to/triplets
"""

import argparse
import subprocess
import tempfile
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm


@dataclass
class TripletConfig:
    """Configuration for triplet extraction."""
    # SSIM threshold: frames with SSIM > this are considered "same"
    ssim_threshold: float = 0.98
    # Minimum motion: F1-F3 SSIM must be below this to keep triplet
    min_motion_ssim: float = 0.90  # Lowered from 0.95 for stricter filtering
    # Output resolution (height, width) - None keeps original
    output_size: tuple[int, int] | None = (720, 1280)
    # Frame extraction step (1 = every frame, 2 = every other, etc.)
    frame_step: int = 1
    # Number of parallel workers for processing
    num_workers: int = 8
    # Pixels to crop from bottom (for hardsubs)
    crop_bottom: int = 0
    # Pixels to crop from top (for top-positioned hardsubs)
    crop_top: int = 0
    # Custom temp directory (uses system temp by default)
    temp_dir: Path | None = None
    # Seconds to skip at start of video (avoid logos/credits)
    skip_intro: float = 120.0
    # Output format: 'jpeg' or 'png'
    output_format: str = 'jpeg'
    # JPEG quality (1-100, higher = better quality, larger files)
    jpeg_quality: int = 95


def extract_frames_ffmpeg(
    video_path: Path,
    output_dir: Path,
    size: tuple[int, int] | None = None,
    crop_bottom: int = 0,
    crop_top: int = 0,
    start_time: float | None = None,
    duration: float | None = None,
) -> list[Path]:
    """
    Extract all frames from a video using FFmpeg.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save frames
        size: Optional (height, width) to resize frames
        start_time: Optional start time in seconds
        duration: Optional duration in seconds
        
    Returns:
        List of paths to extracted frame images
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    
    if start_time is not None:
        cmd.extend(["-ss", str(start_time)])
    
    cmd.extend(["-i", str(video_path)])
    
    if duration is not None:
        cmd.extend(["-t", str(duration)])
    
    # Video filters
    vf_parts = []
    
    # 1. Crop (applied before scaling to ensure clean cut)
    # crop=width:height:x:y
    if crop_bottom > 0 or crop_top > 0:
        total_crop = crop_bottom + crop_top
        # Start y position at crop_top to skip top pixels
        vf_parts.append(f"crop=in_w:in_h-{total_crop}:0:{crop_top}")
    
    # 2. Scale
    if size is not None:
        height, width = size
        # If cropping is used, we should NOT force exact height/width if it distorts aspect ratio.
        # But for simplicity, if user REQUESTED a specific WxH, we assume they know.
        # However, specifically for the crop_bottom case, we often want to keep width and let height adjust.
        # Let's check if the user passed -1 for height or width (handled by argparse as -1)
        
        target_w, target_h = width, height
        scale_str = f"scale={target_w}:{target_h}"
        
        # If cropping, and using default 720p (720x1280), squishing is bad.
        # Auto-adjust height to preserve AR if crop is active and height wasn't explicitly "forced" to a weird value
        # by the user (checking this inside the script is hard without more context, but let's be smart).
        if (crop_bottom > 0 or crop_top > 0) and height > 0 and width > 0:
            # Switch to width-based scaling (keep width, auto height) to avoid squash
            scale_str = f"scale={target_w}:-2"
            
        vf_parts.append(scale_str)
    
    if vf_parts:
        cmd.extend(["-vf", ",".join(vf_parts)])
    
    # Output pattern
    output_pattern = output_dir / "frame_%08d.png"
    cmd.extend(["-y", str(output_pattern)])
    
    subprocess.run(cmd, check=True)
    
    # Collect extracted frames
    frames = sorted(output_dir.glob("frame_*.png"))
    return frames


def compute_ssim_gray(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute SSIM between two images (converts to grayscale).
    
    Args:
        img1: First image as numpy array (H, W, C) or (H, W)
        img2: Second image as numpy array (H, W, C) or (H, W)
        
    Returns:
        SSIM score between 0 and 1
    """
    # Convert to grayscale if RGB
    if img1.ndim == 3:
        img1 = np.mean(img1, axis=2).astype(np.uint8)
    if img2.ndim == 3:
        img2 = np.mean(img2, axis=2).astype(np.uint8)
    
    # Downscale for faster computation
    from PIL import Image as PILImage
    h, w = img1.shape
    scale = min(256 / h, 256 / w, 1.0)
    if scale < 1.0:
        new_h, new_w = int(h * scale), int(w * scale)
        img1 = np.array(PILImage.fromarray(img1).resize((new_w, new_h)))
        img2 = np.array(PILImage.fromarray(img2).resize((new_w, new_h)))
    
    return ssim(img1, img2)


def is_valid_triplet(
    f1_path: Path,
    f2_path: Path,
    f3_path: Path,
    config: TripletConfig,
) -> bool:
    """
    Check if a triplet is valid for training.
    
    A valid triplet must have:
    1. Enough motion between F1 and F3 (SSIM < min_motion_ssim)
    2. F2 is not a duplicate of F1 (SSIM < ssim_threshold)
    3. F2 is not a duplicate of F3 (SSIM < ssim_threshold)
    
    Args:
        f1_path, f2_path, f3_path: Paths to the three frames
        config: Triplet extraction configuration
        
    Returns:
        True if triplet is valid for training
    """
    # Load images
    f1 = np.array(Image.open(f1_path))
    f2 = np.array(Image.open(f2_path))
    f3 = np.array(Image.open(f3_path))
    
    # Check for enough overall motion (F1 vs F3)
    ssim_13 = compute_ssim_gray(f1, f3)
    if ssim_13 > config.min_motion_ssim:
        return False  # Not enough motion
    
    # Check F2 is not duplicate of F1 (on-2s pattern)
    ssim_12 = compute_ssim_gray(f1, f2)
    if ssim_12 > config.ssim_threshold:
        return False  # F2 is duplicate of F1
    
    # Check F2 is not duplicate of F3 (on-2s pattern)
    ssim_23 = compute_ssim_gray(f2, f3)
    if ssim_23 > config.ssim_threshold:
        return False  # F2 is duplicate of F3
    
    return True


def save_triplet(
    f1_path: Path,
    f2_path: Path,
    f3_path: Path,
    output_dir: Path,
    triplet_idx: int,
    output_format: str = 'jpeg',
    jpeg_quality: int = 95,
) -> Path:
    """
    Save a triplet to the output directory.
    
    Creates a subdirectory for the triplet with F1, F2, F3 images.
    
    Args:
        f1_path, f2_path, f3_path: Source frame paths
        output_dir: Base output directory
        triplet_idx: Index for naming the triplet directory
        output_format: 'jpeg' or 'png'
        jpeg_quality: JPEG quality (1-100)
        
    Returns:
        Path to the created triplet directory
    """
    triplet_dir = output_dir / f"triplet_{triplet_idx:08d}"
    triplet_dir.mkdir(parents=True, exist_ok=True)
    
    if output_format == 'png':
        # Copy PNG directly
        shutil.copy(f1_path, triplet_dir / "f1.png")
        shutil.copy(f2_path, triplet_dir / "f2.png")
        shutil.copy(f3_path, triplet_dir / "f3.png")
    else:
        # Convert to JPEG for smaller file size
        for src, name in [(f1_path, "f1"), (f2_path, "f2"), (f3_path, "f3")]:
            img = Image.open(src)
            img.save(triplet_dir / f"{name}.jpg", "JPEG", quality=jpeg_quality)
    
    return triplet_dir


def get_video_duration(video_path: Path) -> float:
    """
    Get the duration of a video in seconds using ffprobe.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Duration in seconds
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def process_video(
    video_path: Path,
    output_dir: Path,
    config: TripletConfig,
    triplet_start_idx: int = 0,
    chunk_duration: float = 300.0,  # 5 minutes per chunk
) -> tuple[int, int]:
    """
    Extract valid triplets from a single video using chunked processing.
    
    Processes the video in chunks (default 5 minutes) to avoid excessive
    temp storage requirements for long or high-resolution videos.
    
    Args:
        video_path: Path to input video
        output_dir: Base output directory for triplets
        config: Extraction configuration
        triplet_start_idx: Starting index for triplet numbering
        chunk_duration: Duration of each chunk in seconds (default: 300 = 5 min)
        
    Returns:
        Tuple of (num_valid_triplets, num_total_candidates)
    """
    print(f"Processing: {video_path.name}")
    
    # Get video duration
    try:
        total_duration = get_video_duration(video_path)
    except Exception as e:
        print(f"  Error getting video duration: {e}")
        return 0, 0
    
    # Skip intro (logos, credits, etc.)
    skip_intro = config.skip_intro if hasattr(config, 'skip_intro') else 0
    effective_start = skip_intro
    effective_duration = total_duration - skip_intro
    
    if effective_duration <= 0:
        print(f"  Skipping: video too short ({total_duration:.1f}s) for skip_intro ({skip_intro}s)")
        return 0, 0
    
    num_chunks = int(effective_duration / chunk_duration) + 1
    print(f"  Duration: {total_duration:.1f}s (skipping first {skip_intro:.0f}s), processing in {num_chunks} chunk(s)")
    
    # Use custom temp_dir if provided, otherwise use system temp
    temp_base = config.temp_dir if config.temp_dir else None
    
    total_valid = 0
    total_candidates = 0
    triplet_idx = triplet_start_idx
    
    # Process each chunk
    for chunk_idx in range(num_chunks):
        start_time = effective_start + (chunk_idx * chunk_duration)
        # Add small overlap to avoid missing triplets at chunk boundaries
        overlap = 0.5  # 0.5 second overlap
        actual_start = max(effective_start, start_time - overlap) if chunk_idx > 0 else effective_start
        actual_duration = chunk_duration + overlap if chunk_idx < num_chunks - 1 else None
        
        end_time = min(start_time + chunk_duration, total_duration)
        print(f"  Chunk {chunk_idx + 1}/{num_chunks}: {start_time:.0f}s - {end_time:.0f}s")
        
        with tempfile.TemporaryDirectory(dir=temp_base) as temp_dir:
            temp_path = Path(temp_dir)
            
            frames = extract_frames_ffmpeg(
                video_path,
                temp_path / "frames",
                size=config.output_size,
                crop_bottom=config.crop_bottom,
                crop_top=config.crop_top,
                start_time=actual_start,
                duration=actual_duration,
            )
            
            if len(frames) < 3:
                print(f"    Skipping chunk: not enough frames ({len(frames)})")
                continue
            
            # Build list of triplet candidates
            step = config.frame_step
            triplet_candidates = []
            for i in range(0, len(frames) - 2 * step, step):
                triplet_candidates.append((
                    frames[i],
                    frames[i + step],
                    frames[i + 2 * step],
                    triplet_idx + len(triplet_candidates)
                ))
            
            chunk_candidates = len(triplet_candidates)
            chunk_valid = 0
            
            # Process triplets in parallel
            def validate_and_save(args):
                f1_path, f2_path, f3_path, idx = args
                if is_valid_triplet(f1_path, f2_path, f3_path, config):
                    save_triplet(f1_path, f2_path, f3_path, output_dir, idx,
                                 config.output_format, config.jpeg_quality)
                    return True
                return False
            
            with ThreadPoolExecutor(max_workers=config.num_workers) as executor:
                futures = {executor.submit(validate_and_save, args): args for args in triplet_candidates}
                for future in tqdm(as_completed(futures), total=len(futures), 
                                   desc=f"    Filtering", leave=False):
                    if future.result():
                        chunk_valid += 1
            
            triplet_idx += chunk_valid
            total_valid += chunk_valid
            total_candidates += chunk_candidates
            print(f"    Chunk valid: {chunk_valid}/{chunk_candidates}")
    
    print(f"  Total valid triplets: {total_valid}/{total_candidates}")
    return total_valid, total_candidates


def find_videos(input_path: Path, extensions: set[str] = None) -> list[Path]:
    """
    Find all video files in the input path.
    
    Args:
        input_path: File or directory to search
        extensions: Set of valid extensions (with dot)
        
    Returns:
        List of video file paths
    """
    if extensions is None:
        extensions = {".mkv", ".mp4", ".avi", ".webm", ".mov"}
    
    if input_path.is_file():
        if input_path.suffix.lower() in extensions:
            return [input_path]
        return []
    
    videos = []
    for ext in extensions:
        videos.extend(input_path.rglob(f"*{ext}"))
    
    return sorted(videos)


def main():
    parser = argparse.ArgumentParser(
        description="Extract training triplets from anime videos"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Input video file or directory containing videos"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output directory for triplets"
    )
    parser.add_argument(
        "--ssim-threshold",
        type=float,
        default=0.98,
        help="SSIM threshold for duplicate detection (default: 0.98)"
    )
    parser.add_argument(
        "--min-motion",
        type=float,
        default=0.90,
        help="Maximum SSIM between F1-F3 to consider as motion (default: 0.90, lower=stricter)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Output frame height (default: 720)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Output frame width (default: 1280)"
    )
    parser.add_argument(
        "--no-resize",
        action="store_true",
        help="Keep original resolution"
    )
    parser.add_argument(
        "--crop-bottom",
        type=int,
        default=0,
        help="Pixels to crop from bottom (removes hardsubs)"
    )
    parser.add_argument(
        "--crop-top",
        type=int,
        default=0,
        help="Pixels to crop from top (removes top-positioned hardsubs)"
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=None,
        help="Custom temp directory for frame extraction (use drive with more space)"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=8,
        help="Number of parallel workers for triplet validation (default: 8)"
    )
    parser.add_argument(
        "--skip-intro",
        type=float,
        default=120.0,
        help="Seconds to skip at start of each video to avoid logos/credits (default: 120)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=['jpeg', 'png'],
        default='jpeg',
        help="Output image format: jpeg (smaller, ~1.5MB/triplet) or png (lossless, ~14MB/triplet)"
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality 1-100 (default: 95, higher = better quality, larger files)"
    )
    
    args = parser.parse_args()
    
    # Build config
    config = TripletConfig(
        ssim_threshold=args.ssim_threshold,
        min_motion_ssim=args.min_motion,
        output_size=None if args.no_resize else (args.height, args.width),
        crop_bottom=args.crop_bottom,
        crop_top=args.crop_top,
        temp_dir=args.temp_dir,
        num_workers=args.workers,
        skip_intro=args.skip_intro,
        output_format=args.format,
        jpeg_quality=args.jpeg_quality,
    )
    
    # Find videos
    videos = find_videos(args.input)
    if not videos:
        print(f"No video files found in {args.input}")
        return
    
    print(f"Found {len(videos)} video(s)")
    
    # Process all videos
    args.output.mkdir(parents=True, exist_ok=True)
    
    total_valid = 0
    total_candidates = 0
    triplet_idx = 0
    
    # Progress bar for all videos
    video_pbar = tqdm(videos, desc="Overall Progress", unit="video")
    for video_num, video in enumerate(video_pbar, 1):
        video_pbar.set_postfix({
            "current": video.name[:30] + "..." if len(video.name) > 30 else video.name,
            "triplets": triplet_idx
        })
        
        valid, candidates = process_video(
            video, args.output, config, triplet_idx
        )
        triplet_idx += valid
        total_valid += valid
        total_candidates += candidates
    
    print(f"\n{'='*50}")
    print(f"Extraction complete!")
    print(f"Total valid triplets: {total_valid}/{total_candidates}")
    print(f"Output directory: {args.output}")
    
    if total_candidates > 0:
        keep_rate = total_valid / total_candidates * 100
        print(f"Keep rate: {keep_rate:.1f}%")


if __name__ == "__main__":
    main()
