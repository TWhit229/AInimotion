"""
V5 Background Path - Multi-frame camera trajectory estimation.

Estimates affine transforms across the 7-frame window, fits a smooth
camera trajectory, and warps backgrounds with occlusion-aware blending.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BackgroundPath(nn.Module):
    """
    Multi-frame background motion estimation.
    
    Estimates per-frame affine transforms relative to the anchor pair,
    interpolates to the target timestep, and warps with occlusion handling.
    
    Args:
        feat_channels: Feature channels at 1/4 resolution
        grid_size: Affine grid resolution (default: 8)
    """
    
    def __init__(self, feat_channels: int = 256, grid_size: int = 8):
        super().__init__()
        self.grid_size = grid_size
        
        # Estimate 6 affine parameters from correlation + features
        self.affine_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(grid_size),
            nn.Flatten(),
            nn.Linear(feat_channels * grid_size * grid_size, 256),
            nn.GELU(),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 6),  # 2x3 affine matrix
        )
        # Initialize to identity
        self.affine_head[-1].weight.data.zero_()
        self.affine_head[-1].bias.data.copy_(
            torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        )
        
        # Occlusion-aware blending (3-layer CNN)
        self.blend_net = nn.Sequential(
            nn.Conv2d(9, 32, 3, padding=1),  # warped_f1 + warped_f2 + diff
            nn.GELU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid(),
        )
    
    def _estimate_affine(self, feat_diff: torch.Tensor) -> torch.Tensor:
        """Estimate affine params from feature difference."""
        params = self.affine_head(feat_diff)
        return params.view(-1, 2, 3)
    
    def _warp_affine(
        self,
        img: torch.Tensor,
        theta: torch.Tensor,
    ) -> torch.Tensor:
        """Apply affine warp to image."""
        grid = F.affine_grid(theta, img.shape, align_corners=False)
        return F.grid_sample(img, grid, mode='bilinear', align_corners=False)
    
    def forward(
        self,
        frame_i: torch.Tensor,
        frame_ip1: torch.Tensor,
        feat_i: torch.Tensor,
        feat_ip1: torch.Tensor,
        timestep: float = 0.5,
    ) -> dict[str, torch.Tensor]:
        """
        Estimate background at interpolation time t.
        
        Args:
            frame_i: (B, 3, H, W) anchor frame i
            frame_ip1: (B, 3, H, W) anchor frame i+1
            feat_i: (B, C, h, w) features at 1/4 res for frame i
            feat_ip1: (B, C, h, w) features at 1/4 res for frame i+1
            timestep: Target time between frames (0.0 to 1.0)
            
        Returns:
            Dict with:
              - 'background': (B, 3, H, W)
              - 'blend_weight': (B, 1, H, W)
        """
        # Estimate affine for frame_i -> target and frame_ip1 -> target
        feat_diff = feat_ip1 - feat_i
        theta_full = self._estimate_affine(feat_diff)
        
        # Interpolate affine: at t=0 -> identity, at t=1 -> full transform
        identity = torch.eye(2, 3, device=theta_full.device).unsqueeze(0).expand_as(theta_full)
        
        # Forward warp (from frame_i, scaled by t)
        theta_fwd = identity + timestep * (theta_full - identity)
        warped_i = self._warp_affine(frame_i, theta_fwd)
        
        # Backward warp (from frame_ip1, scaled by 1-t)
        theta_bwd = identity + (1 - timestep) * (identity - theta_full)
        warped_ip1 = self._warp_affine(frame_ip1, theta_bwd)
        
        # Occlusion-aware blending
        diff = (warped_i - warped_ip1).abs()
        blend_input = torch.cat([warped_i, warped_ip1, diff], dim=1)
        blend_weight = self.blend_net(blend_input)
        
        background = blend_weight * warped_i + (1 - blend_weight) * warped_ip1
        
        return {
            'background': background,
            'blend_weight': blend_weight,
        }


if __name__ == '__main__':
    bg = BackgroundPath(feat_channels=256)
    fi = torch.randn(2, 3, 256, 256)
    fi1 = torch.randn(2, 3, 256, 256)
    feat_i = torch.randn(2, 256, 64, 64)
    feat_i1 = torch.randn(2, 256, 64, 64)
    out = bg(fi, fi1, feat_i, feat_i1, timestep=0.5)
    print(f"Background: {out['background'].shape}")
    print(f"Blend weight: {out['blend_weight'].shape}")
    print(f"Parameters: {sum(p.numel() for p in bg.parameters()):,}")
