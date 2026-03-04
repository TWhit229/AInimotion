"""
Background Flow Network with 8x8 Affine Tile Grid.

v3 improvements over v2:
  - Swap-input reverse flow: reuses same AffineGridPredictor with
    swapped [feat2, feat1] input instead of naive -flow negation.
  - Lightweight FlowRefiner: 3-layer conv residual correction after
    initial affine flow, using warped features to fix errors.
  - Configurable corr_channels (default 169 for displacement=6).
  - Timestep-aware warping in stitch_background.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AffineGridPredictor(nn.Module):
    """
    Predicts 8x8 grid of affine transformation parameters.
    
    Each tile gets 6 parameters: (a, b, tx, c, d, ty)
    representing the affine matrix:
        [a  b  tx]
        [c  d  ty]
    """
    
    def __init__(
        self,
        in_channels: int,
        grid_size: int = 8,
    ):
        super().__init__()
        self.grid_size = grid_size
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.AdaptiveAvgPool2d(grid_size),
            nn.Conv2d(64, 6, 1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize to identity transformation."""
        nn.init.zeros_(self.conv[-1].weight)
        nn.init.zeros_(self.conv[-1].bias)
        self.conv[-1].bias.data[0] = 1.0  # a
        self.conv[-1].bias.data[3] = 1.0  # d
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.conv(features)


class FlowRefiner(nn.Module):
    """
    Lightweight residual flow correction network.
    
    Takes initial flow + warped features + target features and predicts
    a small residual correction to the flow.
    
    Args:
        feat_channels: Feature channels from FPN finest scale
        flow_channels: Flow channels (2 for optical flow)
    """
    
    def __init__(self, feat_channels: int, flow_channels: int = 2):
        super().__init__()
        # Input: warped_feat (feat_channels) + target_feat (feat_channels) + flow (2)
        in_ch = feat_channels * 2 + flow_channels
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 2, 3, padding=1),
        )
        
        # Initialize to zero residual (start from initial flow)
        nn.init.zeros_(self.conv[-1].weight)
        nn.init.zeros_(self.conv[-1].bias)
    
    def forward(
        self,
        warped_feat: torch.Tensor,
        target_feat: torch.Tensor,
        flow: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict flow residual.
        
        Args:
            warped_feat: Features warped by initial flow
            target_feat: Target frame features
            flow: Initial flow estimate
            
        Returns:
            (B, 2, H, W) flow residual
        """
        x = torch.cat([warped_feat, target_feat, flow], dim=1)
        return self.conv(x)


class BackgroundFlowNet(nn.Module):
    """
    Background motion estimation using 8x8 affine grid.
    
    v3 improvements:
      - Swap-input reverse flow (no more -flow negation)
      - FlowRefiner for residual correction
      - Timestep-aware background stitching
    
    Args:
        feat_channels: List of feature channels at each FPN scale
        grid_size: Size of affine grid (default 8x8)
        corr_channels: Correlation volume channels
    """
    
    def __init__(
        self,
        feat_channels: list[int] = [32, 64, 128, 128],
        grid_size: int = 8,
        corr_channels: int = 169,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.corr_channels = corr_channels
        
        # Single affine predictor shared for both directions
        # (swap-input gives reverse flow for free)
        self.affine_predictor = AffineGridPredictor(
            feat_channels[-1] * 2,
            grid_size=grid_size,
        )
        
        # Correlation-based refinement
        self.refiner = nn.Sequential(
            nn.Conv2d(6 + self.corr_channels, 64, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 6, 3, padding=1),
        )
        nn.init.zeros_(self.refiner[-1].weight)
        nn.init.zeros_(self.refiner[-1].bias)
        
        # Lightweight flow residual correction
        self.flow_refiner = FlowRefiner(
            feat_channels=feat_channels[0],
            flow_channels=2,
        )
        
        # Occlusion-aware blending
        self.occ_predictor = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.occ_predictor[-2].weight)
        nn.init.zeros_(self.occ_predictor[-2].bias)
    
    def forward(
        self,
        feat1: list[torch.Tensor],
        feat2: list[torch.Tensor],
        corr: list[torch.Tensor],
        target_size: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        """
        Estimate bidirectional background flow.
        
        v3: Returns both forward and backward flow (no negation).
        """
        h, w = target_size
        
        # === Forward flow (frame1 → frame2) ===
        f1_coarse = feat1[-1]
        f2_coarse = feat2[-1]
        combined_fwd = torch.cat([f1_coarse, f2_coarse], dim=1)
        
        affine_fwd = self.affine_predictor(combined_fwd)
        
        # Refine with correlation
        corr_coarse = F.interpolate(
            corr[-1],
            size=(self.grid_size, self.grid_size),
            mode='bilinear',
            align_corners=False,
        )
        refine_input_fwd = torch.cat([affine_fwd, corr_coarse], dim=1)
        affine_fwd = affine_fwd + self.refiner(refine_input_fwd)
        
        flow_fwd = self._affine_to_flow(affine_fwd, target_size)
        
        # === Backward flow (frame2 → frame1) via swap-input ===
        combined_bwd = torch.cat([f2_coarse, f1_coarse], dim=1)
        affine_bwd = self.affine_predictor(combined_bwd)
        
        refine_input_bwd = torch.cat([affine_bwd, corr_coarse], dim=1)
        affine_bwd = affine_bwd + self.refiner(refine_input_bwd)
        
        flow_bwd = self._affine_to_flow(affine_bwd, target_size)
        
        # === Flow residual refinement ===
        # Get finest features at target resolution
        f1_fine = feat1[0]
        f2_fine = feat2[0]
        if f1_fine.shape[2] != h or f1_fine.shape[3] != w:
            f1_fine = F.interpolate(f1_fine, size=(h, w), mode='bilinear', align_corners=False)
            f2_fine = F.interpolate(f2_fine, size=(h, w), mode='bilinear', align_corners=False)
        
        # Refine forward flow: warp feat1 with flow_fwd, compare to feat2
        warped_f1 = self._warp_features(f1_fine, flow_fwd)
        flow_fwd = flow_fwd + self.flow_refiner(warped_f1, f2_fine, flow_fwd)
        
        # Refine backward flow: warp feat2 with flow_bwd, compare to feat1
        warped_f2 = self._warp_features(f2_fine, flow_bwd)
        flow_bwd = flow_bwd + self.flow_refiner(warped_f2, f1_fine, flow_bwd)
        
        return {
            'flow_fwd': flow_fwd,
            'flow_bwd': flow_bwd,
            'affine_fwd': affine_fwd,
            'affine_bwd': affine_bwd,
        }
    
    def _warp_features(
        self,
        features: torch.Tensor,
        flow: torch.Tensor,
    ) -> torch.Tensor:
        """Warp features using flow (for refinement)."""
        b, c, h, w = features.shape
        device = features.device
        
        y = torch.linspace(-1, 1, h, device=device)
        x = torch.linspace(-1, 1, w, device=device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1).expand(b, -1, -1, -1)
        
        flow_norm = torch.stack([
            flow[:, 0] * 2 / w,
            flow[:, 1] * 2 / h,
        ], dim=-1)
        
        return F.grid_sample(
            features,
            grid + flow_norm,
            mode='bilinear',
            padding_mode='border',
            align_corners=False,
        )
    
    def _affine_to_flow(
        self,
        affine_params: torch.Tensor,
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        """Convert 8x8 affine grid to dense optical flow."""
        b, _, gh, gw = affine_params.shape
        h, w = target_size
        device = affine_params.device
        
        y = torch.linspace(-1, 1, h, device=device)
        x = torch.linspace(-1, 1, w, device=device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        
        grid_x = grid_x.expand(b, 1, h, w)
        grid_y = grid_y.expand(b, 1, h, w)
        
        affine_up = F.interpolate(
            affine_params,
            size=target_size,
            mode='bilinear',
            align_corners=True,
        )
        
        a = affine_up[:, 0:1]
        b_ = affine_up[:, 1:2]
        tx = affine_up[:, 2:3]
        c = affine_up[:, 3:4]
        d = affine_up[:, 4:5]
        ty = affine_up[:, 5:6]
        
        new_x = a * grid_x + b_ * grid_y + tx
        new_y = c * grid_x + d * grid_y + ty
        
        flow_x = (new_x - grid_x) * (w / 2)
        flow_y = (new_y - grid_y) * (h / 2)
        
        return torch.cat([flow_x, flow_y], dim=1)
    
    def warp_with_flow(
        self,
        image: torch.Tensor,
        flow: torch.Tensor,
    ) -> torch.Tensor:
        """Warp an image using optical flow."""
        b, c, h, w = image.shape
        device = image.device
        
        y = torch.linspace(-1, 1, h, device=device)
        x = torch.linspace(-1, 1, w, device=device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1)
        grid = grid.expand(b, -1, -1, -1)
        
        flow_norm = torch.stack([
            flow[:, 0] * 2 / w,
            flow[:, 1] * 2 / h,
        ], dim=-1)
        
        return F.grid_sample(
            image,
            grid + flow_norm,
            mode='bilinear',
            padding_mode='border',
            align_corners=True,
        )
    
    def stitch_background(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor,
        flow_fwd: torch.Tensor,
        flow_bwd: torch.Tensor,
        timestep: float = 0.5,
    ) -> torch.Tensor:
        """
        Stitch background with occlusion-aware blending.
        
        v3: Uses separately-predicted flows and timestep-aware warping.
        """
        # Warp both frames to the target time
        warp1 = self.warp_with_flow(frame1, flow_fwd * timestep)
        warp2 = self.warp_with_flow(frame2, flow_bwd * (1 - timestep))
        
        # Compute occlusion cues
        warp_diff = torch.abs(warp1 - warp2)
        flow_mag = torch.cat([
            flow_fwd.norm(dim=1, keepdim=True),
            flow_bwd.norm(dim=1, keepdim=True),
        ], dim=1)
        
        occ_input = torch.cat([warp_diff, flow_mag], dim=1)
        Z = self.occ_predictor(occ_input)
        
        return Z * warp1 + (1 - Z) * warp2
