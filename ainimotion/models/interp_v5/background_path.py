"""
V5 Background Path - Multi-frame camera trajectory estimation.

Estimates affine transforms across the 7-frame window, fits a smooth
camera trajectory, and warps backgrounds with occlusion-aware blending.

ONNX-compatible: replaces AdaptiveAvgPool2d and F.affine_grid with
manual implementations that support dynamic shapes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BackgroundPath(nn.Module):
    """
    Multi-frame background motion estimation.

    Args:
        feat_channels: Feature channels at 1/4 resolution
        grid_size: Affine grid resolution (default: 8)
    """

    def __init__(self, feat_channels: int = 256, grid_size: int = 8):
        super().__init__()
        self.grid_size = grid_size

        # Estimate 6 affine parameters from pooled features
        # No AdaptiveAvgPool2d — we compute it manually for ONNX compatibility
        self.affine_head = nn.Sequential(
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
            nn.Conv2d(9, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def _pool_to_grid(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pool spatial dims to (grid_size, grid_size).
        Uses bilinear interpolate — ONNX-compatible (exports as Resize op).
        Close to avg pooling for downsampling, pretrained weights work fine.
        """
        return F.interpolate(
            x, size=(self.grid_size, self.grid_size),
            mode='bilinear', align_corners=False,
        )

    def _estimate_affine(self, feat_diff: torch.Tensor) -> torch.Tensor:
        """Estimate affine params from feature difference."""
        pooled = self._pool_to_grid(feat_diff)
        params = self.affine_head(pooled)
        return params.view(-1, 2, 3)

    @staticmethod
    def _make_affine_grid(theta: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Manual affine grid construction. Replaces F.affine_grid for ONNX compatibility.
        Matches F.affine_grid(theta, [B, C, H, W], align_corners=False).
        """
        # For align_corners=False: grid coords = (2*i + 1) / N - 1
        ys = (2.0 * torch.arange(H, device=theta.device, dtype=theta.dtype) + 1.0) / H - 1.0
        xs = (2.0 * torch.arange(W, device=theta.device, dtype=theta.dtype) + 1.0) / W - 1.0
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        ones = torch.ones_like(grid_x)
        # (H, W, 3) homogeneous coords
        grid = torch.stack([grid_x, grid_y, ones], dim=-1)  # (H, W, 3)
        # Apply affine: (B, 2, 3) @ (H*W, 3, 1) -> (B, H*W, 2)
        grid_flat = grid.view(-1, 3)  # (H*W, 3)
        # theta: (B, 2, 3), grid_flat.T: (3, H*W)
        transformed = torch.einsum('bij,kj->bki', theta, grid_flat)  # (B, H*W, 2)
        return transformed.view(-1, H, W, 2)

    def _warp_affine(self, img: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Apply affine warp to image using manual grid (ONNX-compatible)."""
        _, _, H, W = img.shape
        grid = self._make_affine_grid(theta, H, W)
        return F.grid_sample(img, grid, mode='bilinear', align_corners=False)

    def forward(
        self,
        frame_i: torch.Tensor,
        frame_ip1: torch.Tensor,
        feat_i: torch.Tensor,
        feat_ip1: torch.Tensor,
        timestep: float = 0.5,
    ) -> dict[str, torch.Tensor]:
        feat_diff = feat_ip1 - feat_i
        theta_full = self._estimate_affine(feat_diff)

        identity = torch.eye(2, 3, device=theta_full.device, dtype=theta_full.dtype).unsqueeze(0).expand_as(theta_full)

        theta_fwd = identity + timestep * (theta_full - identity)
        warped_i = self._warp_affine(frame_i, theta_fwd)

        theta_bwd = identity + (1 - timestep) * (identity - theta_full)
        warped_ip1 = self._warp_affine(frame_ip1, theta_bwd)

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
