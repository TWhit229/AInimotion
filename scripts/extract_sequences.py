"""
V5 Training Data Extraction Pipeline.

Extracts 9-frame sequences of UNIQUE drawings from anime episodes.

Key design: anime is typically animated "on 2s" or "on 3s", meaning each
drawing is held for 2-3 video frames. We DEDUPLICATE first to get a stream
of unique drawings, THEN extract 9-frame windows. This guarantees every
frame in a training sequence is a genuinely different drawing.

Pipeline (streaming — no massive temp files):
  1. Pipe raw frames from ffmpeg → deduplicate in memory (GPU-accelerated)
     → save only unique drawings as PNGs
  2. Detect scene cuts in the deduplicated stream
  3. Extract 9-frame windows from continuous segments
  4. Score each window by motion intensity
  5. Save as directories of PNGs ready for V5SequenceDataset

Usage:
  python scripts/extract_sequences.py --input anime_episodes/ --output training_data/v5/
  python scripts/extract_sequences.py --input "D:\\Anime\\DemonSlayer" --output training_data/v5 --stride 2
"""

import argparse
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


# ============================================================================
# Frame-Level Utilities
# ============================================================================

def _detect_codec(video_path: str) -> str | None:
    """Detect the video codec of a file using ffprobe."""
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_name',
        '-of', 'csv=p=0',
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip()
    return None


# Map of video codecs to their NVIDIA CUVID hardware decoders
_CUVID_DECODERS = {
    'hevc': 'hevc_cuvid',
    'h265': 'hevc_cuvid',
    'h264': 'h264_cuvid',
    'av1': 'av1_cuvid',
    'vp9': 'vp9_cuvid',
    'mpeg4': 'mpeg4_cuvid',
}


def _get_video_dimensions(video_path: str, target_height: int = 720) -> tuple[int, int] | None:
    """Get the scaled (width, height) for a video at target_height."""
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'csv=p=0',
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    try:
        w, h = result.stdout.strip().split(',')
        w, h = int(w), int(h)
    except (ValueError, IndexError):
        return None
    scale = target_height / h
    new_w = int(w * scale)
    # Ensure even dimensions
    new_w = new_w - (new_w % 2)
    new_h = target_height - (target_height % 2)
    return new_w, new_h


# ============================================================================
# Streaming Extraction + Deduplication
# ============================================================================

def _stream_and_deduplicate(
    video_path: str,
    output_dir: str,
    dup_threshold: float = 0.005,
    target_height: int = 720,
    gpu: bool = True,
) -> list[str]:
    """
    Stream frames from ffmpeg pipe, deduplicate in memory, save only unique.

    Instead of extracting ALL frames to disk (30-350 GB of temp PNGs), this
    pipes raw frames from ffmpeg and compares each one in memory. Only
    genuinely unique drawings are saved to disk as PNGs.

    When a CUDA GPU is available, frame diffs are computed on GPU via torch
    for ~100x faster deduplication.

    Peak temp disk = unique frames only (~30-50% of total, at 720p).
    Typical: 5-15 GB per episode instead of 30-350 GB.

    Returns:
        Sorted list of paths to saved unique frame PNGs
    """
    import torch

    dims = _get_video_dimensions(video_path, target_height)
    if dims is None:
        print("    Could not determine video dimensions")
        return []
    width, height = dims
    rgb_bytes = width * height * 3

    os.makedirs(output_dir, exist_ok=True)

    # Detect codec for GPU-accelerated decoding
    codec = _detect_codec(video_path)
    cuvid_dec = _CUVID_DECODERS.get(codec) if gpu else None

    if cuvid_dec:
        cmd = [
            'ffmpeg', '-hwaccel', 'cuda',
            '-c:v', cuvid_dec,
            '-i', video_path,
            '-vf', f'scale=-2:{target_height}',
            '-f', 'rawvideo', '-pix_fmt', 'rgb24',
            '-v', 'error',
            'pipe:1',
        ]
    else:
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vf', f'scale=-2:{target_height}',
            '-f', 'rawvideo', '-pix_fmt', 'rgb24',
            '-v', 'error',
            'pipe:1',
        ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Use GPU for frame diff if available
    use_cuda = gpu and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    # Grayscale weights (BT.601)
    gray_w = torch.tensor([0.299, 0.587, 0.114], device=device, dtype=torch.float32)

    unique_paths = []
    last_gray = None
    frame_idx = 0
    n_total = 0

    decode_label = f"GPU decode: {cuvid_dec}" if cuvid_dec else "CPU decode"
    diff_label = "GPU diff (torch)" if use_cuda else "CPU diff"
    print(f"    [{decode_label}] [{diff_label}] @ {target_height}p")

    try:
        while True:
            raw = proc.stdout.read(rgb_bytes)
            if len(raw) < rgb_bytes:
                break

            n_total += 1

            # Convert to torch tensor, compute grayscale
            rgb_t = torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(height, width, 3)
            rgb_t = rgb_t.to(device=device, dtype=torch.float32) / 255.0
            gray = (rgb_t * gray_w).sum(dim=2)

            # Deduplicate: compare to last kept frame
            if last_gray is not None:
                diff = (gray - last_gray).abs().mean().item()
                if diff <= dup_threshold:
                    continue  # Duplicate frame — skip

            # Unique frame — save as PNG
            frame_path = os.path.join(output_dir, f'frame_{frame_idx:06d}.png')
            if use_cuda:
                rgb_np = (rgb_t.cpu().numpy() * 255).astype(np.uint8)
            else:
                rgb_np = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
            Image.fromarray(rgb_np).save(frame_path)
            unique_paths.append(frame_path)
            last_gray = gray
            frame_idx += 1

            # Progress
            if n_total % 5000 == 0:
                print(f"      {n_total} frames streamed, {frame_idx} unique saved...")

    finally:
        proc.stdout.close()
        proc.terminate()
        proc.wait()

    dedup_ratio = frame_idx / max(n_total, 1)
    print(f"    Streamed {n_total} frames, saved {frame_idx} unique ({dedup_ratio:.1%})")
    return unique_paths


# ============================================================================
# GPU-Batched Scene Detection + Sequence Extraction
# ============================================================================

def _compute_all_diffs(frame_paths: list[str], device, chunk_size: int = 500) -> np.ndarray:
    """
    Compute all consecutive frame diffs using GPU, in chunks.

    Instead of loading ALL frames onto GPU at once (OOM for 45K+ frames),
    loads chunks of frames, computes diffs within each chunk and across
    chunk boundaries. Only ~2 chunks worth of VRAM used at any time.

    Returns:
        numpy array of shape (N-1,) with mean absolute diff between
        consecutive frames.
    """
    import torch

    all_diffs = []
    prev_last_gray = None  # Last frame of previous chunk (for boundary diff)
    n_loaded = 0

    for i in range(0, len(frame_paths), chunk_size):
        chunk_paths = frame_paths[i:i + chunk_size]
        frames = []
        for fp in chunk_paths:
            try:
                img = np.array(Image.open(fp).convert('L'), dtype=np.float32) / 255.0
                frames.append(img)
            except (OSError, SyntaxError, struct.error):
                # Corrupt frame — insert zeros (will show as scene cut)
                if frames:
                    frames.append(np.zeros_like(frames[-1]))
                continue

        if not frames:
            continue

        # Move chunk to GPU
        chunk_t = torch.from_numpy(np.stack(frames)).to(device)
        n_loaded += len(frames)

        # Diff between last frame of previous chunk and first frame of this chunk
        if prev_last_gray is not None:
            boundary_diff = (chunk_t[0] - prev_last_gray).abs().mean().item()
            all_diffs.append(boundary_diff)

        # Diffs within this chunk (vectorized on GPU)
        if chunk_t.shape[0] > 1:
            chunk_diffs = (chunk_t[1:] - chunk_t[:-1]).abs().mean(dim=(1, 2))
            all_diffs.extend(chunk_diffs.cpu().numpy().tolist())

        # Keep only the last frame for the next boundary
        prev_last_gray = chunk_t[-1]
        del chunk_t

        if n_loaded % 5000 < chunk_size:
            print(f"      Diff progress: {n_loaded}/{len(frame_paths)}...")

    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return np.array(all_diffs, dtype=np.float32)


def extract_sequences(
    unique_frames: list[str],
    output_dir: str,
    window_size: int = 9,
    stride: int = 3,
    series_name: str = "unknown",
    target_height: int = 720,
) -> int:
    """
    Extract 9-frame training windows from deduplicated frames.

    GPU-accelerated: computes all consecutive diffs in chunks on GPU,
    uses pre-computed diffs for both scene detection and motion scoring.

    Returns:
        Number of sequences extracted
    """
    import torch

    print(f"  Unique frames after dedup: {len(unique_frames)}")
    if len(unique_frames) < window_size:
        return 0

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f"  Computing diffs on {'GPU' if use_cuda else 'CPU'} (chunked)...")

    diff_array = _compute_all_diffs(unique_frames, device)
    if len(diff_array) == 0:
        return 0

    print(f"  Computed {len(diff_array)} frame diffs ({'GPU' if use_cuda else 'CPU'})")

    # Scene detection from pre-computed diffs
    cut_threshold = 0.08
    cuts = [0]
    for i, d in enumerate(diff_array):
        if d > cut_threshold:
            cuts.append(i + 1)
    cuts.append(len(unique_frames))

    segments = []
    for i in range(len(cuts) - 1):
        start, end = cuts[i], cuts[i + 1]
        if end - start >= window_size:
            segments.append((start, end))

    print(f"  Continuous segments (>= {window_size} frames): {len(segments)}")

    # Extract windows using pre-computed diffs for motion scoring
    os.makedirs(output_dir, exist_ok=True)
    seq_count = 0
    min_motion = 0.003

    for seg_idx, (start, end) in enumerate(segments):
        seg_diffs = diff_array[start:end - 1]

        for win_start in range(0, end - start - window_size + 1, stride):
            win_diffs = seg_diffs[win_start:win_start + window_size - 1]
            motion = float(win_diffs.mean()) if len(win_diffs) > 0 else 0.0

            if motion < min_motion:
                continue

            seq_dir = os.path.join(
                output_dir,
                f"{series_name}_seg{seg_idx:03d}_seq{seq_count:06d}_m{motion:.4f}"
            )
            os.makedirs(seq_dir, exist_ok=True)

            for j in range(window_size):
                src = unique_frames[start + win_start + j]
                dst = os.path.join(seq_dir, f"frame_{j:03d}.png")
                shutil.copy2(src, dst)

            seq_count += 1

        if (seg_idx + 1) % 50 == 0:
            print(f"    Segment {seg_idx + 1}/{len(segments)}, "
                  f"sequences so far: {seq_count}")

    return seq_count


# ============================================================================
# Full Episode Pipeline
# ============================================================================

def process_episode(
    video_path: str,
    output_dir: str,
    series_name: str,
    episode_num: int,
    stride: int = 3,
    dup_threshold: float = 0.005,
    tmp_dir: str | None = None,
    gpu: bool = True,
) -> int:
    """
    Full pipeline for one episode (streaming — minimal temp disk usage):
      1. Pipe frames from ffmpeg, deduplicate in memory (GPU-accelerated),
         save only unique PNGs to temp dir
      2. Extract 9-frame training windows from unique frames
      3. Clean up temp unique frames

    Peak temp disk = unique frames only (~5-15 GB at 720p per episode).

    Returns:
        Number of sequences extracted
    """
    import tempfile

    ep_name = f"{series_name}_ep{episode_num:02d}"

    if tmp_dir is None:
        tmp_dir = tempfile.mkdtemp(prefix=f"ainimotion_{ep_name}_")

    print(f"\n{'='*60}")
    print(f"Processing: {ep_name}")
    print(f"  Video: {video_path}")
    print(f"  Temp dir: {tmp_dir}")
    print(f"{'='*60}")

    # Step 1: Stream + deduplicate (ffmpeg pipe -> GPU diff -> save unique only)
    print(f"\n  Step 1: Streaming + deduplicating (threshold={dup_threshold})...")
    unique_frames = _stream_and_deduplicate(
        video_path, tmp_dir,
        dup_threshold=dup_threshold,
        gpu=gpu,
    )

    if len(unique_frames) < 9:
        print("    Not enough unique frames for even one sequence, skipping")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return 0

    # Step 2: Extract 9-frame windows
    print(f"\n  Step 2: Extracting 9-frame sequences (stride={stride})...")
    n_seqs = extract_sequences(
        unique_frames=unique_frames,
        output_dir=output_dir,
        series_name=ep_name,
        stride=stride,
    )
    print(f"    Extracted {n_seqs} sequences")

    # Step 3: Clean up temp unique frames
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"    Cleaned up temp dir")

    return n_seqs


# ============================================================================
# Video Discovery & Naming
# ============================================================================

def _sanitize_name(name: str) -> str:
    """Sanitize a name for use in directory paths. Strips special chars."""
    import re
    name = re.sub(r'[\[\]\(\)\{\}!@#$%^&\*\+\=\|<>?/\\\'\"~`]', '', name)
    name = name.replace(' ', '_').replace('.', '_')
    name = re.sub(r'_+', '_', name).strip('_')
    return name[:60]


def _find_all_videos(input_dir: str) -> list[tuple[str, str, int]]:
    """
    Recursively find all video files under input_dir.

    Returns list of (video_path, series_name, episode_number) tuples.
    Series name is derived from the top-level subfolder name, or the
    video filename stem for root-level files.
    """
    video_exts = {'.mkv', '.mp4', '.avi', '.webm'}
    results = []

    # Collect root-level videos (not inside any subfolder)
    root_videos = sorted([
        f for f in os.listdir(input_dir)
        if not os.path.isdir(os.path.join(input_dir, f))
        and Path(f).suffix.lower() in video_exts
    ])
    for ep_num, f in enumerate(root_videos, 1):
        stem = Path(f).stem
        series_name = _sanitize_name(stem)
        results.append((os.path.join(input_dir, f), series_name, ep_num))

    # Collect videos from subfolders (recursively)
    top_dirs = sorted([
        d for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d))
    ])
    for top_dir in top_dirs:
        top_path = os.path.join(input_dir, top_dir)
        series_name = _sanitize_name(top_dir)

        # Find all video files recursively under this top-level folder
        ep_num = 0
        for root, _dirs, files in os.walk(top_path):
            for f in sorted(files):
                if Path(f).suffix.lower() in video_exts:
                    ep_num += 1
                    results.append((os.path.join(root, f), series_name, ep_num))

    return results


# ============================================================================
# Main Orchestration
# ============================================================================

def process_all(
    input_dir: str,
    output_dir: str,
    stride: int = 3,
    val_output_dir: str | None = None,
    val_every_n: int = 5,
    gpu: bool = True,
):
    """
    Process all video files in a directory tree (any nesting depth).

    Recursively discovers .mkv/.mp4/.avi/.webm files. Series name is
    derived from the top-level subfolder each video lives under.

    Args:
        input_dir: Root directory containing videos (any nesting depth)
        output_dir: Output for training sequences
        stride: Window stride (default: 3)
        val_output_dir: If set, every val_every_n-th episode goes here
        val_every_n: Send every Nth episode to val set (default: 5 -> ~20%)
        gpu: Use GPU-accelerated ffmpeg decoding (default: True)
    """
    all_videos = _find_all_videos(input_dir)
    print(f"\nFound {len(all_videos)} videos in {input_dir}")

    if val_output_dir:
        os.makedirs(val_output_dir, exist_ok=True)
        print(f"   Val split: every {val_every_n}th episode -> {val_output_dir}")

    total_train = 0
    total_val = 0
    current_series = None
    ep_in_series = 0

    for video_path, series_name, ep_num in all_videos:
        # Track per-series episode counter for val splitting
        if series_name != current_series:
            current_series = series_name
            ep_in_series = 0
        ep_in_series += 1

        # Decide train vs val (episode-level split prevents data leakage)
        is_val = val_output_dir and (ep_in_series % val_every_n == 0)
        dest_dir = val_output_dir if is_val else output_dir
        split_label = "[VAL]" if is_val else "[TRAIN]"

        print(f"\n{split_label} {series_name} ep{ep_num:02d}")

        # Resume support: skip episodes that already have sequences
        ep_name = f"{series_name}_ep{ep_num:02d}"
        if os.path.exists(dest_dir):
            existing = [d for d in os.listdir(dest_dir)
                        if d.startswith(ep_name + "_seg") and os.path.isdir(os.path.join(dest_dir, d))]
            if existing:
                print(f"  >> Skipping (already have {len(existing)} sequences)")
                if is_val:
                    total_val += len(existing)
                else:
                    total_train += len(existing)
                continue

        n = process_episode(
            video_path, dest_dir, series_name, ep_num,
            stride=stride, gpu=gpu,
        )

        if is_val:
            total_val += n
        else:
            total_train += n

    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"   Train: {total_train} sequences -> {output_dir}")
    if val_output_dir:
        print(f"   Val:   {total_val} sequences -> {val_output_dir}")
    print(f"   Total: {total_train + total_val} sequences")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract V5 training sequences from anime episodes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all videos (recursive, GPU-accelerated)
  python scripts/extract_sequences.py --input raw_vids --output training_data/v5

  # With train/val split (~20%% val)
  python scripts/extract_sequences.py --input raw_vids --output training_data/v5 --val-output training_data/v5_val

  # Tighter stride for more sequences
  python scripts/extract_sequences.py --input raw_vids --output training_data/v5 --stride 2

  # CPU-only decoding
  python scripts/extract_sequences.py --input raw_vids --output training_data/v5 --no-gpu
""",
    )

    parser.add_argument('--input', type=str, required=True,
                        help='Directory with anime episodes (any nesting depth)')
    parser.add_argument('--output', type=str, default='training_data/v5',
                        help='Output directory for training sequences (default: training_data/v5)')
    parser.add_argument('--val-output', type=str, default=None,
                        help='Output directory for validation sequences (default: None = no split)')
    parser.add_argument('--val-every-n', type=int, default=5,
                        help='Send every Nth episode to val set (default: 5 = ~20%%)')
    parser.add_argument('--stride', type=int, default=3,
                        help='Window stride in unique frames (default: 3)')
    parser.add_argument('--dup-threshold', type=float, default=0.005,
                        help='Duplicate detection threshold (default: 0.005)')
    parser.add_argument('--no-gpu', action='store_true',
                        help='Disable GPU-accelerated ffmpeg decoding')

    args = parser.parse_args()
    process_all(
        args.input, args.output,
        stride=args.stride,
        val_output_dir=args.val_output,
        val_every_n=args.val_every_n,
        gpu=not args.no_gpu,
    )
