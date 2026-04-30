"""
Frame analysis for duplicate detection and scene cut identification.

Pre-scans a video to classify every consecutive frame pair as one of:
  DUPLICATE  - held/repeated frame (skip model inference)
  SCENE_CUT  - hard cut between scenes (skip interpolation, copy anchor)
  INTERPOLATE - unique pair needing model inference

This enables accurate progress estimation (we know exact work count upfront)
and efficient batching (skip duplicate pairs, pad context at scene cuts).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import numpy as np

from .video_io import FrameDecoder


class PairType(Enum):
    DUPLICATE = 'duplicate'
    SCENE_CUT = 'scene_cut'
    LOW_MOTION = 'low_motion'    # cheap linear blend (no GPU needed)
    INTERPOLATE = 'interpolate'  # full model inference


@dataclass
class PairInfo:
    """Analysis result for a single consecutive frame pair."""
    index: int          # pair index (frames[index] and frames[index + 1])
    pair_type: PairType
    mse: float


@dataclass
class AnalysisResult:
    """Full video analysis: per-pair classification and summary stats."""
    frame_count: int
    pairs: list[PairInfo]
    scene_cuts: list[int] = field(default_factory=list)

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
    def interpolate_count(self) -> int:
        return sum(1 for p in self.pairs if p.pair_type == PairType.INTERPOLATE)

    def get_scene_for_frame(self, frame_idx: int) -> int:
        """
        Return the scene number for a given frame index.

        Scene 0 runs from frame 0 to the first scene cut, scene 1 from
        the first cut to the second, etc. Used for context window padding.
        """
        scene = 0
        for cut_idx in self.scene_cuts:
            if frame_idx >= cut_idx:
                scene += 1
            else:
                break
        return scene

    def same_scene(self, frame_a: int, frame_b: int) -> bool:
        """Check if two frame indices are in the same scene."""
        return self.get_scene_for_frame(frame_a) == self.get_scene_for_frame(frame_b)

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Frames: {self.frame_count}",
            f"Pairs:  {self.total_pairs}",
            f"  Interpolate: {self.interpolate_count} ({self.interpolate_count/max(self.total_pairs,1)*100:.1f}%)",
            f"  Duplicate:   {self.duplicate_count} ({self.duplicate_count/max(self.total_pairs,1)*100:.1f}%)",
            f"  Low motion:  {self.low_motion_count} ({self.low_motion_count/max(self.total_pairs,1)*100:.1f}% - linear blend)",
            f"  Scene cuts:  {self.scene_cut_count}",
        ]
        if self.scene_cuts:
            lines.append(f"  Cut positions: {self.scene_cuts}")
        return '\n'.join(lines)


def compute_mse(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """Mean squared error between two uint8 RGB frames."""
    return float(np.mean((frame_a.astype(np.float32) - frame_b.astype(np.float32)) ** 2))


def analyze_video(
    path: str,
    max_height: int | None = 720,
    duplicate_threshold: float = 1.0,
    low_motion_threshold: float = 25.0,
    scene_cut_threshold: float = 5000.0,
    progress_callback: Callable[[int, int], None] | None = None,
    cancel_event=None,
    pause_event=None,
) -> AnalysisResult:
    """
    Pre-scan a video to classify all consecutive frame pairs.

    This is a fast CPU-only pass (no GPU needed). Decodes all frames,
    computes MSE between consecutive pairs, and classifies each as
    duplicate, scene cut, or interpolation target.

    Args:
        path: Path to input video file.
        max_height: Downscale target for analysis (should match inference resolution).
        duplicate_threshold: MSE below this = held/repeated frame (default 1.0).
            Anime codec artifacts on held frames produce MSE ~0.1-0.5.
            Very subtle motion (lip sync) starts at MSE ~2-5.
        scene_cut_threshold: MSE above this = hard scene cut (default 5000.0).
            High action motion peaks at MSE ~3000-4000.
            Scene cuts (completely different content) start at MSE ~5000+.
        progress_callback: Called with (frames_processed, total_frames).

    Returns:
        AnalysisResult with per-pair classification and scene cut positions.
    """
    pairs: list[PairInfo] = []
    scene_cuts: list[int] = []
    frame_count = 0

    with FrameDecoder(path, max_height=max_height) as decoder:
        total = decoder.frame_count
        prev_frame: np.ndarray | None = None

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

    return AnalysisResult(
        frame_count=frame_count,
        pairs=pairs,
        scene_cuts=scene_cuts,
    )


def build_context_indices(
    pair_index: int,
    frame_count: int,
    analysis: AnalysisResult,
) -> list[int]:
    """
    Build 7 frame indices for the model's context window around a pair,
    handling video boundaries (mirror padding) and scene cuts (same-scene padding).

    The model expects 7 frames: [i-3, i-2, i-1, i, i+1, i+2, i+3]
    where frames[3] and frames[4] are the anchor pair to interpolate between.

    Args:
        pair_index: Index of the pair (anchor_a = pair_index, anchor_b = pair_index + 1).
        frame_count: Total number of frames in the video.
        analysis: AnalysisResult with scene cut positions.

    Returns:
        List of 7 frame indices, safe for loading.
    """
    anchor_a = pair_index
    anchor_b = pair_index + 1

    # Raw indices: 3 before anchor_a, anchor pair, 2 after anchor_b
    raw = [
        anchor_a - 3, anchor_a - 2, anchor_a - 1,
        anchor_a, anchor_b,
        anchor_b + 1, anchor_b + 2,
    ]

    # Clamp to video boundaries (mirror pad)
    clamped = []
    for idx in raw:
        if idx < 0:
            clamped.append(-idx)  # mirror: -1 -> 1, -2 -> 2, -3 -> 3
        elif idx >= frame_count:
            overshoot = idx - (frame_count - 1)
            clamped.append(frame_count - 1 - overshoot)  # mirror from end
        else:
            clamped.append(idx)

    # Clamp mirrored indices to valid range (in case of very short videos)
    clamped = [max(0, min(c, frame_count - 1)) for c in clamped]

    # Scene cut safety: replace context frames from different scenes
    # with the nearest same-scene anchor
    anchor_scene = analysis.get_scene_for_frame(anchor_a)

    for pos in range(7):
        if analysis.get_scene_for_frame(clamped[pos]) != anchor_scene:
            # This context frame is from a different scene.
            # Replace with nearest anchor from same scene.
            if pos < 3:
                # Before anchor pair: pad with anchor_a
                clamped[pos] = anchor_a
            else:
                # After anchor pair: pad with anchor_b
                clamped[pos] = anchor_b

    return clamped


if __name__ == '__main__':
    import sys
    import time

    if len(sys.argv) < 2:
        print("Usage: python -m app.core.frame_analysis <video_file>")
        print("  Analyzes a video for duplicates and scene cuts.")
        sys.exit(1)

    path = sys.argv[1]
    print(f"Analyzing: {path}")
    t0 = time.time()

    def on_progress(done, total):
        pct = done / max(total, 1) * 100
        print(f"  {done}/{total} frames ({pct:.0f}%)", end='\r')

    result = analyze_video(path, progress_callback=on_progress)
    elapsed = time.time() - t0

    print(f"\n\nAnalysis complete in {elapsed:.1f}s")
    print(result.summary())

    # Show MSE distribution
    mse_values = [p.mse for p in result.pairs]
    if mse_values:
        print(f"\nMSE distribution:")
        print(f"  min:    {min(mse_values):.2f}")
        print(f"  median: {sorted(mse_values)[len(mse_values)//2]:.2f}")
        print(f"  mean:   {sum(mse_values)/len(mse_values):.2f}")
        print(f"  max:    {max(mse_values):.2f}")

    # Show first few pairs
    print(f"\nFirst 20 pairs:")
    for p in result.pairs[:20]:
        print(f"  pair {p.index:5d}: {p.pair_type.value:12s}  MSE={p.mse:8.2f}")
