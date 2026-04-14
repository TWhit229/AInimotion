"""
V5 Warp Branch - RIFE-style coarse-to-fine intermediate flow estimation.

Simple and fast. Directly estimates intermediate flow at 3 scales
(coarse to fine) without cost volumes or pyramid warping.
Inspired by RIFE IFNet architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IFBlock(nn.Module):
    """
    One stage of coarse-to-fine flow refinement.
    
    Takes current estimate (upsampled from previous stage) + features,
    predicts a flow residual + blending mask.
    """
    
    def __init__(self, in_channels: int, hidden: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 5, 3, padding=1),  # 2 (flow_fwd) + 2 (flow_bwd) + 1 (mask)
        )
    
    def forward(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor,
        flow_fwd: torch.Tensor | None = None,
        flow_bwd: torch.Tensor | None = None,
        scale: int = 1,
        extra_feat: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            frame1, frame2: (B, 3, H, W) input frames (at current scale)
            flow_fwd: (B, 2, H, W) current forward flow estimate or None
            flow_bwd: (B, 2, H, W) current backward flow estimate or None
            scale: downscale factor (for initial zero flow)
            extra_feat: (B, C, H, W) optional extra features to inject
            
        Returns:
            flow_fwd: updated forward flow
            flow_bwd: updated backward flow
            mask: (B, 1, H, W) blending mask
        """
        B, _, H, W = frame1.shape
        
        if flow_fwd is None:
            flow_fwd = torch.zeros(B, 2, H, W, device=frame1.device)
            flow_bwd = torch.zeros(B, 2, H, W, device=frame1.device)
        
        # Warp frames with current flow
        grid_fwd = self._flow_to_grid(flow_fwd)
        grid_bwd = self._flow_to_grid(flow_bwd)
        warped1 = F.grid_sample(frame1, grid_fwd, mode='bilinear', align_corners=True)
        warped2 = F.grid_sample(frame2, grid_bwd, mode='bilinear', align_corners=True)
        
        # Concatenate inputs
        inp = torch.cat([
            frame1, frame2, warped1, warped2, flow_fwd, flow_bwd
        ], dim=1)  # 3+3+3+3+2+2 = 16 channels
        if extra_feat is not None:
            inp = torch.cat([inp, extra_feat], dim=1)
        
        out = self.conv(inp)
        delta_fwd = out[:, 0:2]
        delta_bwd = out[:, 2:4]
        mask = torch.sigmoid(out[:, 4:5])
        
        flow_fwd = flow_fwd + delta_fwd
        flow_bwd = flow_bwd + delta_bwd
        
        return flow_fwd, flow_bwd, mask
    
    @staticmethod
    def _flow_to_grid(flow: torch.Tensor) -> torch.Tensor:
        """Convert optical flow to sampling grid."""
        B, _, H, W = flow.shape
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=flow.device),
            torch.linspace(-1, 1, W, device=flow.device),
            indexing='ij',
        )
        grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
        # Normalize flow to [-1, 1] range
        flow_norm = flow.clone()
        flow_norm[:, 0] = flow[:, 0] / (W / 2)
        flow_norm[:, 1] = flow[:, 1] / (H / 2)
        grid = grid + flow_norm
        return grid.permute(0, 2, 3, 1)


class WarpBranch(nn.Module):
    """
    RIFE-style coarse-to-fine warp branch.
    
    3 stages of flow refinement from 1/4 to full resolution.
    Each stage predicts forward + backward flow and a blending mask.
    Final output warps both frames and blends.
    
    Args:
        None (self-contained, simple architecture)
    """
    
    def __init__(self, feat_channels: int = 256):
        super().__init__()
        self.feat_proj = nn.Sequential(
            nn.Conv2d(feat_channels, 16, 1),
            nn.GELU()
        )
        # Stage 0: 1/4 resolution, 16 input channels + 32 (feat_proj)
        self.stage0 = IFBlock(in_channels=48, hidden=64)
        # Stage 1: 1/2 resolution
        self.stage1 = IFBlock(in_channels=16, hidden=64)
        # Stage 2: full resolution
        self.stage2 = IFBlock(in_channels=16, hidden=32)
    
    def forward(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor,
        feat1: torch.Tensor | None = None,
        feat2: torch.Tensor | None = None,
        timestep: float = 0.5,
    ) -> dict[str, torch.Tensor]:
        """
        Estimate intermediate frame via coarse-to-fine warping.
        
        Args:
            frame1: (B, 3, H, W) first anchor frame
            frame2: (B, 3, H, W) second anchor frame
            feat1: Optional temporal feature 1
            feat2: Optional temporal feature 2
            timestep: target interpolation time
            
        Returns:
            Dict with:
              - 'output': (B, 3, H, W) warped interpolation
              - 'flow_fwd': (B, 2, H, W) forward flow
              - 'flow_bwd': (B, 2, H, W) backward flow
        """
        B, C, H, W = frame1.shape
        
        # Downscale for coarse stages
        f1_4 = F.interpolate(frame1, scale_factor=0.25, mode='bilinear', align_corners=True)
        f2_4 = F.interpolate(frame2, scale_factor=0.25, mode='bilinear', align_corners=True)
        f1_2 = F.interpolate(frame1, scale_factor=0.5, mode='bilinear', align_corners=True)
        f2_2 = F.interpolate(frame2, scale_factor=0.5, mode='bilinear', align_corners=True)
        
        extra_feat = None
        if feat1 is not None and feat2 is not None:
            extra_feat = torch.cat([self.feat_proj(feat1), self.feat_proj(feat2)], dim=1)
        else:
            extra_feat = torch.zeros(B, 32, H//4, W//4, device=frame1.device)
        
        # Stage 0: 1/4 res
        flow_fwd, flow_bwd, mask = self.stage0(f1_4, f2_4, scale=4, extra_feat=extra_feat)
        
        # Upscale to 1/2
        flow_fwd = F.interpolate(flow_fwd, scale_factor=2, mode='bilinear', align_corners=True) * 2
        flow_bwd = F.interpolate(flow_bwd, scale_factor=2, mode='bilinear', align_corners=True) * 2
        
        # Stage 1: 1/2 res
        flow_fwd, flow_bwd, mask = self.stage1(f1_2, f2_2, flow_fwd, flow_bwd, scale=2)
        
        # Upscale to full
        flow_fwd = F.interpolate(flow_fwd, scale_factor=2, mode='bilinear', align_corners=True) * 2
        flow_bwd = F.interpolate(flow_bwd, scale_factor=2, mode='bilinear', align_corners=True) * 2
        
        # Stage 2: full res
        flow_fwd, flow_bwd, mask = self.stage2(frame1, frame2, flow_fwd, flow_bwd, scale=1)
        
        # Scale flows by timestep
        flow_t_fwd = flow_fwd * timestep
        flow_t_bwd = flow_bwd * (1 - timestep)
        
        # Warp both frames to target time
        grid_fwd = IFBlock._flow_to_grid(flow_t_fwd)
        grid_bwd = IFBlock._flow_to_grid(flow_t_bwd)
        warped1 = F.grid_sample(frame1, grid_fwd, mode='bilinear', align_corners=True)
        warped2 = F.grid_sample(frame2, grid_bwd, mode='bilinear', align_corners=True)
        
        # Blend with learned mask
        output = mask * warped1 + (1 - mask) * warped2
        
        return {
            'output': output,
            'flow_fwd': flow_fwd,
            'flow_bwd': flow_bwd,
            'mask': mask,
        }


if __name__ == '__main__':
    warp = WarpBranch()
    f1 = torch.randn(2, 3, 256, 256)
    f2 = torch.randn(2, 3, 256, 256)
    out = warp(f1, f2, timestep=0.5)
    print(f"Output: {out['output'].shape}")
    print(f"Flow fwd: {out['flow_fwd'].shape}")
    params = sum(p.numel() for p in warp.parameters())
    print(f"Parameters: {params:,}")
