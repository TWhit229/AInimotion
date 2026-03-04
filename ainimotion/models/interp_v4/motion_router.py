"""
Motion Complexity Router — per-pixel routing between warp and synthesis paths.

Predicts a routing map R ∈ [0,1] where:
  R ≈ 0 → trust AdaCoF warp (simple translational motion)
  R ≈ 1 → trust synthesis decoder (complex non-rigid motion)

The router analyzes the correlation volume (which encodes motion patterns)
and FPN features to determine motion complexity at each pixel.

Key design: initialized with negative bias so R starts near 0 (trust AdaCoF).
The synthesis path must earn its way in during training by providing better
results on complex-motion pixels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MotionComplexityRouter(nn.Module):
    """
    Per-pixel motion complexity routing network.
    
    Architecture: 4-layer CNN operating on correlation + features.
    
    Args:
        corr_channels: Channels in correlation volume (default 169 for d=6)
        feat_channels: Channels per frame's FPN features at scale 0
    """
    
    def __init__(
        self,
        corr_channels: int = 169,
        feat_channels: int = 128,
    ):
        super().__init__()
        
        # Input: corr + feat1 + feat2
        in_channels = corr_channels + feat_channels * 2
        
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 1, 3, padding=1),
        )
        
        # Initialize final conv bias to -2.0 so sigmoid(output) ≈ 0.12
        # This biases the router toward AdaCoF by default
        self.net[-1].bias.data.fill_(-2.0)
        nn.init.zeros_(self.net[-1].weight.data)
    
    def forward(
        self,
        corr: torch.Tensor,
        feat1: torch.Tensor,
        feat2: torch.Tensor,
        target_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """
        Predict per-pixel routing map.
        
        Args:
            corr: (B, corr_channels, H', W') correlation volume at scale 0
            feat1: (B, C, H', W') FPN features for frame 1 at scale 0
            feat2: (B, C, H', W') FPN features for frame 2 at scale 0
            target_size: Optional (H, W) to upsample routing map to full res
            
        Returns:
            (B, 1, H, W) routing map in [0, 1]
        """
        combined = torch.cat([corr, feat1, feat2], dim=1)
        logits = self.net(combined)
        routing_map = torch.sigmoid(logits)
        
        # Upsample to full resolution if needed
        if target_size is not None and routing_map.shape[2:] != target_size:
            routing_map = F.interpolate(
                routing_map, size=target_size,
                mode='bilinear', align_corners=False,
            )
        
        return routing_map
