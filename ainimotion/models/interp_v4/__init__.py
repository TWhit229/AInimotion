"""
Interpolation model components (v4).

v4 improvements over v3:
  1. Motion Complexity Router (MCR) — per-pixel routing between
     AdaCoF warp path and deformable attention synthesis path
  2. Deformable Cross-Attention Synthesis Decoder — generates content
     for non-rigid motion regions using learned attention offsets
  3. Edge-Guided Synthesis — Sobel edge maps as structural guides
     for preserving anime line art during synthesis
  4. All v3 features: vectorized AdaCoF, swap-input flow,
     timestep conditioning, configurable correlation
"""

from .feature_extractor import FeaturePyramid
from .scene_gate import SceneGate
from .background_flow import BackgroundFlowNet
from .foreground_flow import AdaCoFNet
from .compositor import Compositor
from .edge_extractor import EdgeExtractor
from .motion_router import MotionComplexityRouter
from .synthesis_decoder import SynthesisDecoder
from .layered_interp import LayeredInterpolatorV4

__all__ = [
    "FeaturePyramid",
    "SceneGate",
    "BackgroundFlowNet",
    "AdaCoFNet",
    "Compositor",
    "EdgeExtractor",
    "MotionComplexityRouter",
    "SynthesisDecoder",
    "LayeredInterpolatorV4",
]
