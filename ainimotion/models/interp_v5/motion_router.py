"""
V5 Motion Router - Per-pixel routing between warp and synthesis branches.

Generates a routing map R in [0, 1]:
  R = 0 -> use warp branch (simple trackable motion)
  R = 1 -> use synthesis branch (occlusions, non-rigid motion)

Initialized with bias toward warp (sigmoid bias = -1.0).
No auxiliary loss needed since synthesis branch has multi-frame context.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MotionRouter(nn.Module):
    """
    Per-pixel motion complexity router.
    
    Takes correlation between anchors + temporal features and predicts
    a routing map that blends between warp and synthesis outputs.
    
    Args:
        feat_channels: Feature channels at 1/4 resolution
    """
    
    def __init__(self, feat_channels: int = 256):
        super().__init__()
        
        # Input: anchor correlation (1ch) + feat_i + feat_ip1 at 1/4 res
        in_ch = 1 + feat_channels * 2
        
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 1, 3, padding=1),
        )
        
        # Initialize bias to 0.0 for balanced routing (R = sigmoid(0) = 0.5)
        # Previous -1.0 bias caused routing collapse → synthesis branch death
        self.net[-1].bias.data.fill_(0.0)
    
    def forward(
        self,
        corr: torch.Tensor,
        feat_i: torch.Tensor,
        feat_ip1: torch.Tensor,
        target_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """
        Args:
            corr: (B, 1, h, w) correlation between anchors at 1/4 res
            feat_i: (B, C, h, w) features for frame i at 1/4 res
            feat_ip1: (B, C, h, w) features for frame i+1 at 1/4 res
            target_size: Optional (H, W) to upsample routing map
            
        Returns:
            (B, 1, H, W) routing map in [0, 1]
        """
        x = torch.cat([corr, feat_i, feat_ip1], dim=1)
        routing = torch.sigmoid(self.net(x))  # (B, 1, h, w)
        
        if target_size is not None:
            routing = F.interpolate(
                routing, size=target_size, mode='bilinear', align_corners=False
            )
        
        return routing


if __name__ == '__main__':
    router = MotionRouter(feat_channels=256)
    corr = torch.randn(2, 1, 64, 64)
    f_i = torch.randn(2, 256, 64, 64)
    f_ip1 = torch.randn(2, 256, 64, 64)
    rm = router(corr, f_i, f_ip1, target_size=(256, 256))
    print(f"Routing map: {rm.shape}, range: [{rm.min():.3f}, {rm.max():.3f}]")
    print(f"Parameters: {sum(p.numel() for p in router.parameters()):,}")
