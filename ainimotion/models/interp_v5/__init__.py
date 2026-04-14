"""
Interpolation model components (v5) - Multi-Frame Anime Interpolation.

V5 improvements over v4:
  1. Multi-frame context (7 frames: +-3 from anchor pair)
  2. Temporal Attention Fusion (windowed cross-frame attention)
  3. RIFE-style warp branch (replaces AdaCoF)
  4. Deformable cross-temporal synthesis (borrows pixels across time)
  5. FFT amplitude anti-blur loss (replaces Laplacian)
  6. Edge-guided refinement for anime line art
  7. 3-phase training: reconstruction -> anti-blur -> GAN
"""

from .spatial_encoder import SpatialEncoder
from .temporal_attention import TemporalFusion, compute_correlation
from .scene_gate import SceneGate
from .background_path import BackgroundPath
from .warp_branch import WarpBranch
from .synthesis_branch import SynthesisBranch, compute_sobel_edges
from .motion_router import MotionRouter
from .compositor import LayerCompositor, EdgeGuidedRefinement
from .layered_interp import LayeredInterpolatorV5, build_model

__all__ = [
    "SpatialEncoder",
    "TemporalFusion",
    "compute_correlation",
    "SceneGate",
    "BackgroundPath",
    "WarpBranch",
    "SynthesisBranch",
    "compute_sobel_edges",
    "MotionRouter",
    "LayerCompositor",
    "EdgeGuidedRefinement",
    "LayeredInterpolatorV5",
    "build_model",
]
