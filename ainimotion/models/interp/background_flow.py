"""
Background Flow Network with 8x8 Affine Tile Grid.

Estimates rigid camera motion (pans, zooms, rotations) using
a grid of local affine transformations.
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
        
        # Reduce spatial dimensions and predict affine params
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.AdaptiveAvgPool2d(grid_size),
            nn.Conv2d(64, 6, 1),  # 6 affine params per tile
        )
        
        # Initialize to identity affine (no transformation)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize to identity transformation."""
        # Set final conv to output identity affine params
        # Identity: a=1, b=0, tx=0, c=0, d=1, ty=0
        nn.init.zeros_(self.conv[-1].weight)
        nn.init.zeros_(self.conv[-1].bias)
        # Set a and d to 1 (indices 0 and 4)
        self.conv[-1].bias.data[0] = 1.0  # a
        self.conv[-1].bias.data[3] = 1.0  # d
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Predict affine grid.
        
        Args:
            features: (B, C, H, W) input features
            
        Returns:
            (B, 6, grid_size, grid_size) affine parameters
        """
        return self.conv(features)


class BackgroundFlowNet(nn.Module):
    """
    Background motion estimation using 8x8 affine grid.
    
    Estimates camera motion (pans, zooms, rotations) as a grid of
    local affine transformations, then interpolates to full resolution.
    
    Args:
        feat_channels: List of feature channels at each FPN scale
        grid_size: Size of affine grid (default 8x8)
    """
    
    def __init__(
        self,
        feat_channels: list[int] = [32, 64, 128, 128],
        grid_size: int = 8,
    ):
        super().__init__()
        self.grid_size = grid_size
        
        # Use coarsest features (most global view)
        self.affine_predictor = AffineGridPredictor(
            feat_channels[-1] * 2,  # Concatenate feat1 and feat2
            grid_size=grid_size,
        )
        
        # Optional: refinement based on correlation
        self.corr_channels = 81  # (2*4+1)^2
        self.refiner = nn.Sequential(
            nn.Conv2d(6 + self.corr_channels, 64, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 6, 3, padding=1),
        )
        nn.init.zeros_(self.refiner[-1].weight)
        nn.init.zeros_(self.refiner[-1].bias)
    
    def forward(
        self,
        feat1: list[torch.Tensor],
        feat2: list[torch.Tensor],
        corr: list[torch.Tensor],
        target_size: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        """
        Estimate background flow.
        
        Args:
            feat1: List of frame1 features at each scale
            feat2: List of frame2 features at each scale
            corr: List of correlation volumes
            target_size: (H, W) output resolution
            
        Returns:
            Dictionary with:
                - 'flow': (B, 2, H, W) optical flow for background
                - 'affine_grid': (B, 6, grid_size, grid_size) raw affine params
        """
        # Concatenate coarsest features
        f1 = feat1[-1]
        f2 = feat2[-1]
        combined = torch.cat([f1, f2], dim=1)
        
        # Predict initial affine grid
        affine_params = self.affine_predictor(combined)
        
        # Refine with correlation
        corr_coarse = F.interpolate(
            corr[-1], 
            size=(self.grid_size, self.grid_size),
            mode='bilinear',
            align_corners=False,
        )
        refine_input = torch.cat([affine_params, corr_coarse], dim=1)
        affine_residual = self.refiner(refine_input)
        affine_params = affine_params + affine_residual
        
        # Convert affine grid to dense flow
        flow = self._affine_to_flow(affine_params, target_size)
        
        return {
            'flow': flow,
            'affine_grid': affine_params,
        }
    
    def _affine_to_flow(
        self,
        affine_params: torch.Tensor,
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        """
        Convert 8x8 affine grid to dense optical flow.
        
        Args:
            affine_params: (B, 6, grid_size, grid_size)
            target_size: (H, W) output size
            
        Returns:
            (B, 2, H, W) optical flow
        """
        b, _, gh, gw = affine_params.shape
        h, w = target_size
        device = affine_params.device
        
        # Create meshgrid of target coordinates
        y = torch.linspace(-1, 1, h, device=device)
        x = torch.linspace(-1, 1, w, device=device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        
        # Expand for batch
        grid_x = grid_x.expand(b, 1, h, w)
        grid_y = grid_y.expand(b, 1, h, w)
        ones = torch.ones_like(grid_x)
        
        # Upsample affine params to target size
        affine_up = F.interpolate(
            affine_params,
            size=target_size,
            mode='bilinear',
            align_corners=False,
        )
        
        # Extract affine components: [a, b, tx, c, d, ty]
        a = affine_up[:, 0:1]
        b_ = affine_up[:, 1:2]
        tx = affine_up[:, 2:3]
        c = affine_up[:, 3:4]
        d = affine_up[:, 4:5]
        ty = affine_up[:, 5:6]
        
        # Apply affine transformation
        new_x = a * grid_x + b_ * grid_y + tx
        new_y = c * grid_x + d * grid_y + ty
        
        # Flow = displacement from original position
        flow_x = new_x - grid_x
        flow_y = new_y - grid_y
        
        # Scale from [-2, 2] to pixel coordinates
        flow_x = flow_x * (w / 2)
        flow_y = flow_y * (h / 2)
        
        return torch.cat([flow_x, flow_y], dim=1)
    
    def warp_with_flow(
        self,
        image: torch.Tensor,
        flow: torch.Tensor,
    ) -> torch.Tensor:
        """
        Warp an image using optical flow.
        
        Args:
            image: (B, C, H, W) input image
            flow: (B, 2, H, W) optical flow in pixels
            
        Returns:
            (B, C, H, W) warped image
        """
        b, c, h, w = image.shape
        device = image.device
        
        # Create base grid
        y = torch.linspace(-1, 1, h, device=device)
        x = torch.linspace(-1, 1, w, device=device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)
        grid = grid.expand(b, -1, -1, -1)  # (B, H, W, 2)
        
        # Convert flow to normalized coordinates
        flow_norm = torch.stack([
            flow[:, 0] * 2 / w,
            flow[:, 1] * 2 / h,
        ], dim=-1)  # (B, H, W, 2)
        
        # Apply flow
        grid_warped = grid + flow_norm
        
        # Sample with grid
        return F.grid_sample(
            image,
            grid_warped,
            mode='bilinear',
            padding_mode='border',
            align_corners=False,
        )
    
    def stitch_background(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor,
        flow_1to2: torch.Tensor,
        flow_2to1: torch.Tensor,
    ) -> torch.Tensor:
        """
        Stitch background from both frames to fill dis-occluded areas.
        
        Warps both frames to t=0.5 and blends based on which pixels
        are more reliable (less occluded).
        
        Args:
            frame1: (B, C, H, W) first frame
            frame2: (B, C, H, W) second frame
            flow_1to2: (B, 2, H, W) flow from frame1 to interpolated time
            flow_2to1: (B, 2, H, W) flow from frame2 to interpolated time
            
        Returns:
            (B, C, H, W) stitched background canvas
        """
        # Warp both frames to middle time
        warp1 = self.warp_with_flow(frame1, flow_1to2 * 0.5)
        warp2 = self.warp_with_flow(frame2, flow_2to1 * 0.5)
        
        # Simple 50/50 blend (can be improved with occlusion reasoning)
        return (warp1 + warp2) / 2
