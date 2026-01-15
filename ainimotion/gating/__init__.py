"""
Gating module for anime-safe video processing.

Detects scene cuts and held frames to prevent interpolation artifacts.
"""

from .similarity import compute_ssim, compute_mad, compute_similarity
from .classifier import FrameType, classify_pair, classify_sequence, GatingThresholds
from .gating import GatingProcessor

__all__ = [
    "compute_ssim",
    "compute_mad", 
    "compute_similarity",
    "FrameType",
    "classify_pair",
    "classify_sequence",
    "GatingThresholds",
    "GatingProcessor",
]
