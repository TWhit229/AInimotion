"""
Deformable Cross-Attention Synthesis Decoder.

Generates foreground content for regions with complex non-rigid motion
(face turns, body twists) where pixel warping (AdaCoF) fails.

Instead of moving pixels, this module:
1. Warps FPN features from both frames using background flow
2. Uses deformable cross-attention to query learned locations in both frames
3. Incorporates edge maps as structural guides for anime line preservation
4. Decodes attended features back to RGB

This is NOT the same as RIFE/IFRNet's approach — those warp features with
optical flow then decode. We use deformable attention (learned positional
queries per pixel) combined with edge-guided conditioning, which is unique.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeformableAttentionBlock(nn.Module):
    """
    Deformable cross-attention: each output pixel attends to a learned
    set of sampling locations in the source feature maps.
    
    Args:
        channels: Feature channels
        n_points: Number of sampling points per pixel (default 16 = 4x4 grid)
        n_heads: Number of attention heads
    """
    
    def __init__(
        self,
        channels: int,
        n_points: int = 16,
        n_heads: int = 4,
    ):
        super().__init__()
        self.channels = channels
        self.n_points = n_points
        self.n_heads = n_heads
        assert channels % n_heads == 0
        self.head_dim = channels // n_heads
        
        # Predict sampling offsets: (dx, dy) for each point
        self.offset_predictor = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=n_heads),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, n_heads * n_points * 2, 1),
        )
        
        # Predict attention weights for each point
        self.attn_weight_predictor = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=n_heads),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, n_heads * n_points, 1),
        )
        
        # Value projection
        self.value_proj = nn.Conv2d(channels, channels, 1)
        
        # Output projection
        self.out_proj = nn.Conv2d(channels, channels, 1)
        
        self._init_offsets()
    
    def _init_offsets(self):
        """Initialize offsets to a regular grid pattern."""
        # Create a grid of initial offsets
        grid_size = int(self.n_points ** 0.5)
        if grid_size * grid_size != self.n_points:
            grid_size = 4  # Default to 4x4
        
        offsets = []
        for dy in range(grid_size):
            for dx in range(grid_size):
                offsets.extend([
                    dx - grid_size // 2,
                    dy - grid_size // 2,
                ])
        
        # Pad if n_points != grid_size²
        while len(offsets) < self.n_heads * self.n_points * 2:
            offsets.extend([0.0, 0.0])
        offsets = offsets[:self.n_heads * self.n_points * 2]
        
        # Set as initial bias
        self.offset_predictor[-1].bias.data = torch.tensor(
            offsets, dtype=torch.float32
        )
        nn.init.zeros_(self.offset_predictor[-1].weight.data)
    
    def forward(
        self,
        query_features: torch.Tensor,
        source_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Deformable cross-attention.
        
        Args:
            query_features: (B, C, H, W) features at query positions
            source_features: (B, C, H, W) features to sample from
            
        Returns:
            (B, C, H, W) attended features
        """
        b, c, h, w = query_features.shape
        
        # Predict offsets and weights from query features
        offsets = self.offset_predictor(query_features)  # (B, heads*points*2, H, W)
        attn_weights = self.attn_weight_predictor(query_features)  # (B, heads*points, H, W)
        
        # Reshape offsets: (B, heads, points, 2, H, W)
        offsets = offsets.view(b, self.n_heads, self.n_points, 2, h, w)
        
        # Softmax weights per head: (B, heads, points, H, W)
        attn_weights = attn_weights.view(b, self.n_heads, self.n_points, h, w)
        attn_weights = F.softmax(attn_weights, dim=2)
        
        # Project values
        values = self.value_proj(source_features)  # (B, C, H, W)
        
        # Create base grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=query_features.device, dtype=torch.float32),
            torch.arange(w, device=query_features.device, dtype=torch.float32),
            indexing='ij',
        )
        grid_x = grid_x.view(1, 1, 1, h, w).expand(b, self.n_heads, self.n_points, -1, -1)
        grid_y = grid_y.view(1, 1, 1, h, w).expand(b, self.n_heads, self.n_points, -1, -1)
        
        # Apply offsets
        sample_x = grid_x + offsets[:, :, :, 0]  # (B, heads, points, H, W)
        sample_y = grid_y + offsets[:, :, :, 1]
        
        # Normalize to [-1, 1] for grid_sample
        sample_x = 2.0 * sample_x / max(w - 1, 1) - 1.0
        sample_y = 2.0 * sample_y / max(h - 1, 1) - 1.0
        
        # Sample values at deformed positions for each head
        # Reshape values: (B, heads, head_dim, H, W)
        values = values.view(b, self.n_heads, self.head_dim, h, w)
        
        output = torch.zeros_like(values)
        
        for p in range(self.n_points):
            # Grid for this point: (B*heads, H, W, 2)
            grid = torch.stack([
                sample_x[:, :, p],  # (B, heads, H, W)
                sample_y[:, :, p],
            ], dim=-1)  # (B, heads, H, W, 2)
            grid = grid.view(b * self.n_heads, h, w, 2)
            
            # Sample: (B*heads, head_dim, H, W)
            vals_flat = values.view(b * self.n_heads, self.head_dim, h, w)
            sampled = F.grid_sample(
                vals_flat, grid,
                mode='bilinear', padding_mode='border', align_corners=False,
            )  # (B*heads, head_dim, H, W)
            sampled = sampled.view(b, self.n_heads, self.head_dim, h, w)
            
            # Weight and accumulate
            w_p = attn_weights[:, :, p:p+1, :, :]  # (B, heads, 1, H, W)
            output = output + sampled * w_p
        
        # Reshape back: (B, C, H, W)
        output = output.view(b, c, h, w)
        output = self.out_proj(output)
        
        return output


class SynthesisDecoder(nn.Module):
    """
    Feature-space synthesis decoder for complex motion regions.
    
    Pipeline:
        1. Warp FPN features from both frames using provided flow
        2. Concatenate with edge maps and timestep
        3. Apply deformable cross-attention (1 layer for VRAM efficiency)
        4. Decode to RGB
    
    Operates at 1/4 input resolution (FPN scale 1) for VRAM efficiency,
    upsampled to full resolution by decode head.
    
    Args:
        feat_channels: FPN feature channels at scale 1
        n_attn_layers: Number of deformable attention layers
        n_points: Sampling points per pixel per attention layer
    """
    
    def __init__(
        self,
        feat_channels: int = 128,
        n_attn_layers: int = 1,
        n_points: int = 9,
    ):
        super().__init__()
        
        # Input fusion: feat1_warped + feat2_warped + edge1 + edge2 + timestep
        # = C + C + 1 + 1 + 1
        fuse_channels = feat_channels * 2 + 3
        
        self.input_fuse = nn.Sequential(
            nn.Conv2d(fuse_channels, feat_channels, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        
        # Deformable cross-attention layers
        self.attn_layers = nn.ModuleList()
        self.attn_norms = nn.ModuleList()
        self.attn_ffns = nn.ModuleList()
        
        for _ in range(n_attn_layers):
            self.attn_layers.append(
                DeformableAttentionBlock(feat_channels, n_points=n_points)
            )
            self.attn_norms.append(nn.GroupNorm(8, feat_channels))
            self.attn_ffns.append(nn.Sequential(
                nn.Conv2d(feat_channels, feat_channels * 2, 1),
                nn.GELU(),
                nn.Conv2d(feat_channels * 2, feat_channels, 1),
            ))
        
        # Decode head: features → RGB (with upsampling built-in)
        self.decode_head = nn.Sequential(
            nn.Conv2d(feat_channels, 64, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid(),
        )
    
    def _warp_features(
        self,
        features: torch.Tensor,
        flow: torch.Tensor,
    ) -> torch.Tensor:
        """Warp features using optical flow."""
        b, c, h, w = features.shape
        
        # Resize flow to feature resolution if needed
        if flow.shape[2:] != features.shape[2:]:
            scale_h = h / flow.shape[2]
            scale_w = w / flow.shape[3]
            flow = F.interpolate(flow, size=(h, w), mode='bilinear', align_corners=False)
            flow = flow * torch.tensor([scale_w, scale_h], device=flow.device).view(1, 2, 1, 1)
        
        # Create sampling grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=features.device, dtype=torch.float32),
            torch.arange(w, device=features.device, dtype=torch.float32),
            indexing='ij',
        )
        grid_x = grid_x.unsqueeze(0) + flow[:, 0]
        grid_y = grid_y.unsqueeze(0) + flow[:, 1]
        
        # Normalize to [-1, 1]
        grid_x = 2.0 * grid_x / max(w - 1, 1) - 1.0
        grid_y = 2.0 * grid_y / max(h - 1, 1) - 1.0
        
        grid = torch.stack([grid_x, grid_y], dim=-1)
        
        return F.grid_sample(
            features, grid,
            mode='bilinear', padding_mode='border', align_corners=True,
        )
    
    def forward(
        self,
        feat1: torch.Tensor,
        feat2: torch.Tensor,
        flow_fwd: torch.Tensor,
        flow_bwd: torch.Tensor,
        edge1: torch.Tensor,
        edge2: torch.Tensor,
        timestep: float = 0.5,
    ) -> torch.Tensor:
        """
        Synthesize content for complex-motion regions.
        
        Args:
            feat1: (B, C, H', W') FPN features for frame 1 at scale 0
            feat2: (B, C, H', W') FPN features for frame 2 at scale 0
            flow_fwd: (B, 2, H, W) forward background flow
            flow_bwd: (B, 2, H, W) backward background flow
            edge1: (B, 1, H, W) edge map for frame 1
            edge2: (B, 1, H, W) edge map for frame 2
            timestep: Interpolation timestep ∈ [0, 1]
            
        Returns:
            (B, 3, H, W) synthesized RGB output
        """
        b, c, fh, fw = feat1.shape
        h, w = edge1.shape[2:]
        
        # Warp features toward the target timestep
        feat1_warped = self._warp_features(feat1, flow_fwd * timestep)
        feat2_warped = self._warp_features(feat2, flow_bwd * (1 - timestep))
        
        # Resize edges and make timestep map at feature resolution
        edge1_feat = F.interpolate(edge1, size=(fh, fw), mode='bilinear', align_corners=False)
        edge2_feat = F.interpolate(edge2, size=(fh, fw), mode='bilinear', align_corners=False)
        t_map = torch.full((b, 1, fh, fw), timestep, device=feat1.device, dtype=feat1.dtype)
        
        # Fuse inputs
        fused = torch.cat([feat1_warped, feat2_warped, edge1_feat, edge2_feat, t_map], dim=1)
        query = self.input_fuse(fused)
        
        # Source features for cross-attention (blend of both frames' features)
        source = feat1_warped * (1 - timestep) + feat2_warped * timestep
        
        # Apply deformable attention layers with residual connections
        x = query
        for attn, norm, ffn in zip(self.attn_layers, self.attn_norms, self.attn_ffns):
            # Cross-attention + residual
            x = x + attn(norm(x), source)
            # FFN + residual  
            x = x + ffn(x)
        
        # Upsample to full resolution and decode
        if x.shape[2:] != (h, w):
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        
        output = self.decode_head(x)
        
        return output
