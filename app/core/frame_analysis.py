"""
Frame analysis for duplicate detection, scene cut identification, and credit detection.

Pre-scans a video to classify every consecutive frame pair as one of:
  DUPLICATE   - held/repeated frame (skip model inference)
  SCENE_CUT   - hard cut between scenes (skip interpolation, copy anchor)
  LOW_MOTION  - near-static (cheap CPU linear blend)
  CREDITS     - end credits (skip interpolation)
  INTERPOLATE - unique pair needing model inference

Supports parallel analysis (multiple ffmpeg decode threads per video).
"""

from __future__ import annotations

import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

import numpy as np

from .video_io import FrameDecoder, probe_video, _POPEN_FLAGS


class PairType(Enum):
    DUPLICATE = 'duplicate'
    SCENE_CUT = 'scene_cut'
    LOW_MOTION = 'low_motion'
    CREDITS = 'credits'
    CALM = 'calm'              # non-action: duplicate frames (action-only mode)
    INTERPOLATE = 'interpolate'


@dataclass
class PairInfo:
    """Analysis result for a single consecutive frame pair."""
    index: int
    pair_type: PairType
    mse: float


@dataclass
class AnalysisResult:
    """Full video analysis: per-pair classification and summary stats."""
    frame_count: int
    pairs: list[PairInfo]
    scene_cuts: list[int] = field(default_factory=list)
    credits_start: int | None = None  # frame index where credits begin

    @property
    def total_pairs(self) -> int:
        return len(self.pairs)

    @property
    def duplicate_count(self) -> int:
        return sum(1 for p in self.pairs if p.pair_type == PairType.DUPLICATE)

    @property
    def scene_cut_count(self) -> int:
        return sum(1 for p in self.pairs if p.pair_type == PairType.SCENE_CUT)

    @property
    def low_motion_count(self) -> int:
        return sum(1 for p in self.pairs if p.pair_type == PairType.LOW_MOTION)

    @property
    def credits_count(self) -> int:
        return sum(1 for p in self.pairs if p.pair_type == PairType.CREDITS)

    @property
    def calm_count(self) -> int:
        return sum(1 for p in self.pairs if p.pair_type == PairType.CALM)

    @property
    def interpolate_count(self) -> int:
        return sum(1 for p in self.pairs if p.pair_type == PairType.INTERPOLATE)

    def get_scene_for_frame(self, frame_idx: int) -> int:
        scene = 0
        for cut_idx in self.scene_cuts:
            if frame_idx >= cut_idx:
                scene += 1
            else:
                break
        return scene

    def same_scene(self, frame_a: int, frame_b: int) -> bool:
        return self.get_scene_for_frame(frame_a) == self.get_scene_for_frame(frame_b)

    def summary(self) -> str:
        tp = max(self.total_pairs, 1)
        lines = [
            f"Frames: {self.frame_count}",
            f"Pairs:  {self.total_pairs}",
            f"  Interpolate: {self.interpolate_count} ({self.interpolate_count/tp*100:.1f}%)",
            f"  Duplicate:   {self.duplicate_count} ({self.duplicate_count/tp*100:.1f}%)",
            f"  Low motion:  {self.low_motion_count} ({self.low_motion_count/tp*100:.1f}% - linear blend)",
            f"  Calm:        {self.calm_count} ({self.calm_count/tp*100:.1f}% - frame dup, action-only mode)",
            f"  Credits:     {self.credits_count} ({self.credits_count/tp*100:.1f}% - skipped)",
            f"  Scene cuts:  {self.scene_cut_count}",
        ]
        if self.credits_start is not None:
            lines.append(f"  Credits detected at frame {self.credits_start}")
        return '\n'.join(lines)


def compute_mse(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """Mean squared error between two uint8 RGB frames."""
    return float(np.mean((frame_a.astype(np.float32) - frame_b.astype(np.float32)) ** 2))


def _frame_complexity(frame: np.ndarray) -> float:
    """Estimate visual complexity (low = credits/solid, high = detailed scene)."""
    gray = frame.mean(axis=2)  # (H, W)
    return float(gray.std())


def _detect_credits(pairs: list[PairInfo], frame_count: int, fps: float) -> int | None:
    """
    Detect end credits by looking for sustained low-complexity pattern at the end.

    Credits have two signatures:
    1. Very low MSE (>80% dup/low_motion) — static/slow-scroll text
    2. Low MSE variance + moderate MSE — uniform scrolling motion (steady, not chaotic)

    Both patterns are checked. The credits must be in the last 30% of the video
    and last at least 30 seconds.
    """
    if len(pairs) < 100 or fps <= 0:
        return None

    window_size = min(int(fps * 30), len(pairs) // 3)
    if window_size < 20:
        return None

    max_credit_frames = min(int(fps * 180), int(frame_count * 0.3))
    max_credit_frames = max(max_credit_frames, window_size + 10)
    search_start = max(0, len(pairs) - max_credit_frames)

    best_start = None

    step = max(int(fps * 5), 1)
    for start in range(len(pairs) - window_size, search_start - 1, -step):
        if start < 0:
            break
        window = pairs[start:start + window_size]
        mse_vals = [p.mse for p in window]

        # Check signature 1: mostly static (>80% dup/low_motion)
        low_count = sum(1 for p in window if p.pair_type in (PairType.DUPLICATE, PairType.LOW_MOTION))
        if low_count / len(window) > 0.80:
            best_start = start
            continue

        # Check signature 2: uniform motion (low MSE std, consistent frame-to-frame)
        # Credits scroll at constant speed → MSE is nearly the same every frame
        mean_mse = np.mean(mse_vals)
        std_mse = np.std(mse_vals)
        if mean_mse < 500 and std_mse < mean_mse * 0.3 and mean_mse > 0:
            # Low variance relative to mean = uniform motion (credits scroll)
            # Anime action has chaotic MSE (high std relative to mean)
            best_start = start
            continue

        # Neither signature matched — this is real content, stop searching
        break

    return pairs[best_start].index if best_start is not None else None


def detect_action_regions(
    pairs: list[PairInfo],
    fps: float,
    action_threshold: float = 200.0,
    min_action_seconds: float = 3.0,
    link_seconds: float = 15.0,
    extend_seconds: float = 3.0,
) -> list[tuple[int, int]]:
    """
    Detect action/fight scenes from MSE timeline.

    Finds contiguous regions where motion is elevated, merges nearby bursts
    to avoid jarring fps transitions, and extends boundaries for smooth entry/exit.

    Args:
        pairs: Analyzed frame pairs with MSE values.
        fps: Video frame rate.
        action_threshold: MSE above this = action content (default 200).
        min_action_seconds: Minimum burst duration to count (default 3s).
        link_seconds: Merge bursts closer than this (the "sensitivity slider").
            5s = tight, only links very close fights.
            15s = bridges short dialogue between fight beats.
            30s = links entire extended sequences.
        extend_seconds: Extend each region by this much before/after (smooth entry/exit).

    Returns:
        List of (start_pair_index, end_pair_index) regions that are action.
    """
    if not pairs or fps <= 0:
        return []

    min_frames = int(fps * min_action_seconds)
    link_frames = int(fps * link_seconds)
    extend_frames = int(fps * extend_seconds)

    # Step 1: Find raw action frames using rolling average MSE
    # Rolling average smooths out single-frame spikes
    window = max(int(fps * 0.5), 3)  # 0.5 second rolling window
    mse_vals = np.array([p.mse for p in pairs])

    # Compute rolling mean MSE
    if len(mse_vals) >= window:
        kernel = np.ones(window) / window
        smoothed = np.convolve(mse_vals, kernel, mode='same')
    else:
        smoothed = mse_vals

    # Step 2: Mark frames above threshold
    is_action = smoothed > action_threshold

    # Step 3: Find contiguous runs of action
    raw_regions = []
    in_region = False
    start = 0
    for i in range(len(is_action)):
        if is_action[i] and not in_region:
            start = i
            in_region = True
        elif not is_action[i] and in_region:
            if i - start >= min_frames:
                raw_regions.append((start, i - 1))
            in_region = False
    if in_region and len(is_action) - start >= min_frames:
        raw_regions.append((start, len(is_action) - 1))

    if not raw_regions:
        return []

    # Step 4: Merge regions closer than link_frames (bridge short dialogue gaps)
    merged = [raw_regions[0]]
    for start, end in raw_regions[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= link_frames:
            # Close enough — merge
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))

    # Step 5: Extend boundaries for smooth entry/exit
    extended = []
    for start, end in merged:
        ext_start = max(0, start - extend_frames)
        ext_end = min(len(pairs) - 1, end + extend_frames)
        extended.append((ext_start, ext_end))

    # Step 6: Merge again after extension (in case extensions overlap)
    final = [extended[0]]
    for start, end in extended[1:]:
        prev_start, prev_end = final[-1]
        if start <= prev_end:
            final[-1] = (prev_start, max(prev_end, end))
        else:
            final.append((start, end))

    return final


def apply_action_only_mode(
    result: AnalysisResult,
    fps: float,
    action_threshold: float = 200.0,
    link_seconds: float = 15.0,
    extend_seconds: float = 3.0,
) -> list[tuple[int, int]]:
    """
    Reclassify non-action INTERPOLATE pairs as CALM (frame duplication instead of GPU).

    Returns the detected action regions for display/logging.
    """
    regions = detect_action_regions(
        result.pairs, fps,
        action_threshold=action_threshold,
        link_seconds=link_seconds,
        extend_seconds=extend_seconds,
    )

    if not regions:
        # No action detected — mark everything as CALM
        for p in result.pairs:
            if p.pair_type == PairType.INTERPOLATE:
                p.pair_type = PairType.CALM
        return []

    # Build a set of pair indices that are in action regions
    action_indices = set()
    for start, end in regions:
        for i in range(start, end + 1):
            if i < len(result.pairs):
                action_indices.add(i)

    # Reclassify: INTERPOLATE outside action → CALM
    for i, p in enumerate(result.pairs):
        if p.pair_type == PairType.INTERPOLATE and i not in action_indices:
            p.pair_type = PairType.CALM

    return regions


def _analyze_chunk(
    path: str,
    start_time: float,
    duration: float,
    chunk_start_idx: int,
    max_height: int | None,
    duplicate_threshold: float,
    low_motion_threshold: float,
    scene_cut_threshold: float,
    cancel_event: threading.Event | None,
    chunk_progress_callback: Callable[[int], None] | None = None,
) -> tuple[list[PairInfo], list[int], int, np.ndarray | None, np.ndarray | None]:
    """
    Analyze a chunk of video. Returns (pairs, scene_cuts, frame_count, first_frame, last_frame).
    first_frame and last_frame are needed to compute MSE at chunk boundaries.
    """
    from .video_io import align_resolution

    # Build ffmpeg command with seek
    info = probe_video(path)
    if max_height:
        w, h = align_resolution(info.width, info.height, max_height)
    else:
        w, h = align_resolution(info.width, info.height, info.height)

    cmd = ['ffmpeg', '-v', 'error']
    if start_time > 0:
        cmd += ['-ss', f'{start_time:.3f}']
    cmd += ['-i', str(path)]
    if duration > 0:
        cmd += ['-t', f'{duration:.3f}']
    if w != info.width or h != info.height:
        cmd += ['-vf', f'scale={w}:{h}:flags=lanczos']
    cmd += ['-f', 'rawvideo', '-pix_fmt', 'rgb24', '-vsync', 'passthrough', '-']

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, **_POPEN_FLAGS)

    frame_size = w * h * 3
    pairs = []
    scene_cuts = []
    prev_frame = None
    first_frame = None
    last_frame = None
    frame_count = 0

    try:
        while True:
            if cancel_event and cancel_event.is_set():
                break
            raw = proc.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3).copy()
            frame_idx = chunk_start_idx + frame_count

            if first_frame is None:
                first_frame = frame

            if prev_frame is not None:
                mse = compute_mse(prev_frame, frame)
                if mse < duplicate_threshold:
                    pair_type = PairType.DUPLICATE
                elif mse > scene_cut_threshold:
                    pair_type = PairType.SCENE_CUT
                    scene_cuts.append(frame_idx)
                elif mse < low_motion_threshold:
                    pair_type = PairType.LOW_MOTION
                else:
                    pair_type = PairType.INTERPOLATE
                pairs.append(PairInfo(index=frame_idx - 1, pair_type=pair_type, mse=mse))

            prev_frame = frame
            last_frame = frame
            frame_count += 1

            if chunk_progress_callback and frame_count % 50 == 0:
                chunk_progress_callback(frame_count)
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()

    return pairs, scene_cuts, frame_count, first_frame, last_frame


def analyze_video(
    path: str,
    max_height: int | None = 720,
    duplicate_threshold: float = 1.0,
    low_motion_threshold: float = 25.0,
    scene_cut_threshold: float = 5000.0,
    progress_callback: Callable[[int, int], None] | None = None,
    cancel_event: threading.Event | None = None,
    pause_event: threading.Event | None = None,
    n_workers: int = 4,
) -> AnalysisResult:
    """
    Analyze a video for duplicates, scene cuts, low motion, and credits.
    Uses parallel ffmpeg decoding for ~3-4x faster analysis on multi-core CPUs.
    """
    info = probe_video(path)
    total_frames = info.frame_count
    duration = info.duration
    fps = info.fps

    if total_frames <= 0 or duration <= 0:
        # Fallback to single-threaded
        return _analyze_single_thread(
            path, max_height, duplicate_threshold, low_motion_threshold,
            scene_cut_threshold, progress_callback, cancel_event, pause_event,
        )

    # For short videos (< 500 frames), single thread is fine
    if total_frames < 500:
        n_workers = 1

    if n_workers <= 1:
        return _analyze_single_thread(
            path, max_height, duplicate_threshold, low_motion_threshold,
            scene_cut_threshold, progress_callback, cancel_event, pause_event,
        )

    # Split into chunks by time
    chunk_duration = duration / n_workers
    frames_per_chunk = total_frames // n_workers

    if progress_callback:
        progress_callback(0, total_frames)

    # Launch parallel chunk analysis
    chunk_results = [None] * n_workers
    completed_frames = [0] * n_workers
    progress_lock = threading.Lock()

    def chunk_progress(chunk_idx, frames_done):
        """Called per-frame from each chunk to update overall progress."""
        with progress_lock:
            completed_frames[chunk_idx] = frames_done
            if progress_callback:
                total_done = sum(completed_frames)
                progress_callback(total_done, total_frames)

    def run_chunk(chunk_idx):
        start_time = chunk_idx * chunk_duration
        dur = chunk_duration if chunk_idx < n_workers - 1 else 0
        chunk_start_frame = chunk_idx * frames_per_chunk

        result = _analyze_chunk(
            path, start_time, dur, chunk_start_frame,
            max_height, duplicate_threshold, low_motion_threshold,
            scene_cut_threshold, cancel_event,
            chunk_progress_callback=lambda n: chunk_progress(chunk_idx, n),
        )
        chunk_results[chunk_idx] = result
        return chunk_idx

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(run_chunk, i) for i in range(n_workers)]
        for f in as_completed(futures):
            if pause_event:
                pause_event.wait()
            f.result()  # raise if errored

    if cancel_event and cancel_event.is_set():
        # Return partial result
        all_pairs = []
        for r in chunk_results:
            if r:
                all_pairs.extend(r[0])
        return AnalysisResult(frame_count=sum(r[2] for r in chunk_results if r), pairs=all_pairs)

    # Merge chunks: compute MSE at boundaries
    all_pairs = []
    all_scene_cuts = []
    total_frame_count = 0

    for i, result in enumerate(chunk_results):
        pairs_i, cuts_i, count_i, first_i, last_i = result
        all_pairs.extend(pairs_i)
        all_scene_cuts.extend(cuts_i)
        total_frame_count += count_i

        # Boundary: compute MSE between last frame of chunk i and first frame of chunk i+1
        if i < n_workers - 1 and chunk_results[i + 1] is not None:
            next_first = chunk_results[i + 1][3]  # first frame of next chunk
            if last_i is not None and next_first is not None:
                boundary_idx = (i + 1) * frames_per_chunk - 1
                mse = compute_mse(last_i, next_first)
                if mse < duplicate_threshold:
                    pt = PairType.DUPLICATE
                elif mse > scene_cut_threshold:
                    pt = PairType.SCENE_CUT
                    all_scene_cuts.append(boundary_idx + 1)
                elif mse < low_motion_threshold:
                    pt = PairType.LOW_MOTION
                else:
                    pt = PairType.INTERPOLATE
                all_pairs.append(PairInfo(index=boundary_idx, pair_type=pt, mse=mse))

    # Sort pairs by index (chunks may finish out of order, boundaries added at end)
    all_pairs.sort(key=lambda p: p.index)
    all_scene_cuts.sort()

    # Detect end credits
    credits_start = _detect_credits(all_pairs, total_frame_count, fps)
    if credits_start is not None:
        for p in all_pairs:
            if p.index >= credits_start and p.pair_type == PairType.INTERPOLATE:
                p.pair_type = PairType.CREDITS

    return AnalysisResult(
        frame_count=total_frame_count,
        pairs=all_pairs,
        scene_cuts=all_scene_cuts,
        credits_start=credits_start,
    )


def _analyze_single_thread(
    path, max_height, duplicate_threshold, low_motion_threshold,
    scene_cut_threshold, progress_callback, cancel_event, pause_event,
) -> AnalysisResult:
    """Original single-threaded analysis (fallback for short videos)."""
    pairs = []
    scene_cuts = []
    frame_count = 0

    with FrameDecoder(path, max_height=max_height) as decoder:
        total = decoder.frame_count
        prev_frame = None
        fps = decoder.fps

        for i, frame in enumerate(decoder):
            frame_count = i + 1
            if prev_frame is not None:
                mse = compute_mse(prev_frame, frame)
                if mse < duplicate_threshold:
                    pair_type = PairType.DUPLICATE
                elif mse > scene_cut_threshold:
                    pair_type = PairType.SCENE_CUT
                    scene_cuts.append(i)
                elif mse < low_motion_threshold:
                    pair_type = PairType.LOW_MOTION
                else:
                    pair_type = PairType.INTERPOLATE
                pairs.append(PairInfo(index=i - 1, pair_type=pair_type, mse=mse))

            prev_frame = frame
            if pause_event:
                pause_event.wait()
            if cancel_event and cancel_event.is_set():
                break
            if progress_callback and (i % 50 == 0 or i == total - 1):
                progress_callback(i + 1, total)

    # Detect credits
    credits_start = _detect_credits(pairs, frame_count, fps)
    if credits_start is not None:
        for p in pairs:
            if p.index >= credits_start and p.pair_type == PairType.INTERPOLATE:
                p.pair_type = PairType.CREDITS

    return AnalysisResult(
        frame_count=frame_count,
        pairs=pairs,
        scene_cuts=scene_cuts,
        credits_start=credits_start,
    )


def build_context_indices(
    pair_index: int,
    frame_count: int,
    analysis: AnalysisResult,
) -> list[int]:
    """
    Build 7 frame indices for the model's context window around a pair,
    handling video boundaries (mirror padding) and scene cuts (same-scene padding).
    """
    anchor_a = pair_index
    anchor_b = pair_index + 1

    raw = [
        anchor_a - 3, anchor_a - 2, anchor_a - 1,
        anchor_a, anchor_b,
        anchor_b + 1, anchor_b + 2,
    ]

    clamped = []
    for idx in raw:
        if idx < 0:
            clamped.append(-idx)
        elif idx >= frame_count:
            overshoot = idx - (frame_count - 1)
            clamped.append(frame_count - 1 - overshoot)
        else:
            clamped.append(idx)

    clamped = [max(0, min(c, frame_count - 1)) for c in clamped]

    anchor_scene = analysis.get_scene_for_frame(anchor_a)
    for pos in range(7):
        if analysis.get_scene_for_frame(clamped[pos]) != anchor_scene:
            if pos < 3:
                clamped[pos] = anchor_a
            else:
                clamped[pos] = anchor_b

    return clamped
