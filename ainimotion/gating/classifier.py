"""
Frame pair classification for gating decisions.

Classifies consecutive frame pairs as CUT, HOLD, or MOTION
based on similarity metrics.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Sequence

import torch

from .similarity import compute_ssim, compute_mad


class FrameType(Enum):
    """Classification of frame transitions."""
    
    CUT = "cut"       # Scene cut - hard transition
    HOLD = "hold"     # Held frame - same or nearly same
    MOTION = "motion" # Normal motion - safe to interpolate


@dataclass
class GatingThresholds:
    """
    Thresholds for frame classification.
    
    These are tuned for anime content. Adjust based on:
    - Animation style (2s vs 3s vs full animation)
    - Encoding quality (compression artifacts)
    - Resolution
    
    Attributes:
        hold_ssim: SSIM above this = HOLD (default: 0.985)
        hold_mad: MAD below this = HOLD (default: 0.005)
        cut_ssim: SSIM below this = CUT (default: 0.4)
        cut_mad: MAD above this = CUT (default: 0.25)
    """
    
    # HOLD detection (near-identical frames)
    hold_ssim: float = 0.985  # Very high similarity
    hold_mad: float = 0.005   # Very low difference
    
    # CUT detection (scene transitions)
    cut_ssim: float = 0.4     # Low similarity
    cut_mad: float = 0.25     # High difference
    
    # Method to use
    method: str = "ssim"  # 'ssim' or 'mad'
    
    @classmethod
    def anime_default(cls) -> "GatingThresholds":
        """Default thresholds for anime content."""
        return cls()
    
    @classmethod
    def conservative(cls) -> "GatingThresholds":
        """
        Conservative thresholds - prefer duplication over interpolation.
        Use when artifacts are more noticeable than smoothness loss.
        """
        return cls(
            hold_ssim=0.98,
            hold_mad=0.01,
            cut_ssim=0.5,
            cut_mad=0.20,
        )
    
    @classmethod
    def aggressive(cls) -> "GatingThresholds":
        """
        Aggressive thresholds - prefer interpolation.
        Use for high-quality sources with clean motion.
        """
        return cls(
            hold_ssim=0.995,
            hold_mad=0.002,
            cut_ssim=0.3,
            cut_mad=0.35,
        )


@dataclass
class GatingDecision:
    """Result of classifying a frame pair."""
    
    frame_type: FrameType
    similarity: float  # SSIM or 1-MAD depending on method
    confidence: str    # 'high', 'medium', 'low'
    
    def should_interpolate(self) -> bool:
        """Return True if this pair should be interpolated."""
        return self.frame_type == FrameType.MOTION


def classify_pair(
    frame1: str | Path | torch.Tensor,
    frame2: str | Path | torch.Tensor,
    thresholds: GatingThresholds | None = None,
) -> GatingDecision:
    """
    Classify a frame pair as CUT, HOLD, or MOTION.
    
    Args:
        frame1: First frame
        frame2: Second frame
        thresholds: Classification thresholds (default: anime_default)
        
    Returns:
        GatingDecision with frame type, similarity, and confidence
    """
    if thresholds is None:
        thresholds = GatingThresholds.anime_default()
    
    # Compute similarity based on method
    if thresholds.method == "ssim":
        similarity = compute_ssim(frame1, frame2)
        
        if similarity >= thresholds.hold_ssim:
            confidence = "high" if similarity >= 0.995 else "medium"
            return GatingDecision(FrameType.HOLD, similarity, confidence)
        elif similarity <= thresholds.cut_ssim:
            confidence = "high" if similarity <= 0.2 else "medium"
            return GatingDecision(FrameType.CUT, similarity, confidence)
        else:
            # Motion zone
            confidence = "high" if 0.6 <= similarity <= 0.95 else "medium"
            return GatingDecision(FrameType.MOTION, similarity, confidence)
    
    else:  # MAD
        mad = compute_mad(frame1, frame2)
        similarity = 1.0 - mad  # Invert for consistency
        
        if mad <= thresholds.hold_mad:
            confidence = "high" if mad <= 0.002 else "medium"
            return GatingDecision(FrameType.HOLD, similarity, confidence)
        elif mad >= thresholds.cut_mad:
            confidence = "high" if mad >= 0.4 else "medium"
            return GatingDecision(FrameType.CUT, similarity, confidence)
        else:
            confidence = "high" if 0.02 <= mad <= 0.15 else "medium"
            return GatingDecision(FrameType.MOTION, similarity, confidence)


def classify_sequence(
    frames: Sequence[str | Path | torch.Tensor],
    thresholds: GatingThresholds | None = None,
    progress_callback=None,
) -> list[GatingDecision]:
    """
    Classify all consecutive frame pairs in a sequence.
    
    Args:
        frames: Sequence of frames (paths or tensors)
        thresholds: Classification thresholds
        progress_callback: Optional callback(current, total)
        
    Returns:
        List of GatingDecisions (length = len(frames) - 1)
    """
    if len(frames) < 2:
        return []
    
    decisions = []
    total = len(frames) - 1
    
    for i in range(total):
        decision = classify_pair(frames[i], frames[i + 1], thresholds)
        decisions.append(decision)
        
        if progress_callback is not None:
            progress_callback(i + 1, total)
    
    return decisions


def summarize_gating(decisions: list[GatingDecision]) -> dict:
    """
    Summarize gating decisions for a video.
    
    Args:
        decisions: List of gating decisions
        
    Returns:
        Dictionary with counts and percentages
    """
    total = len(decisions)
    if total == 0:
        return {"total": 0}
    
    counts = {
        FrameType.CUT: 0,
        FrameType.HOLD: 0,
        FrameType.MOTION: 0,
    }
    
    for d in decisions:
        counts[d.frame_type] += 1
    
    return {
        "total": total,
        "cuts": counts[FrameType.CUT],
        "holds": counts[FrameType.HOLD],
        "motion": counts[FrameType.MOTION],
        "cuts_pct": counts[FrameType.CUT] / total * 100,
        "holds_pct": counts[FrameType.HOLD] / total * 100,
        "motion_pct": counts[FrameType.MOTION] / total * 100,
        "interpolatable_pct": counts[FrameType.MOTION] / total * 100,
    }
