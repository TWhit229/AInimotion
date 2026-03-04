"""
Interpolation model components (v3).

v3 improvements over v2:
  1. Vectorized AdaCoF sampler (single grid_sample, ~5x faster)
  2. Swap-input reverse flow (no naive -flow negation)
  3. Lightweight flow residual refinement
  4. Timestep conditioning (arbitrary-time interpolation)
  5. Configurable correlation displacement (default d=6, 169 channels)
"""

from .feature_extractor import FeaturePyramid
from .scene_gate import SceneGate
from .background_flow import BackgroundFlowNet
from .foreground_flow import AdaCoFNet
from .compositor import Compositor
from .layered_interp import LayeredInterpolatorV3

__all__ = [
    "FeaturePyramid",
    "SceneGate",
    "BackgroundFlowNet",
    "AdaCoFNet",
    "Compositor",
    "LayeredInterpolatorV3",
]
