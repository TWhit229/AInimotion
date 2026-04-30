"""
V5 Synthesis Branch - Deformable cross-temporal attention for occlusions.

This is the key module that leverages multi-frame context. Each output pixel
can attend to ALL 7 context frames with learned deformable offsets, so the
model can borrow pixels from nearby frames where content is visible.

Key design choices (from risk analysis):
  - K=9 sampling points per head (RVRT default)
  - 4 attention heads
  - Offsets clamped with tanh * max_offset to prevent overflow
  - Edge map injection for line art guidance
  - Operates at 1/4 resolution, upsampled with pixel-shuffle
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeformableCrossTemporalAttention(nn.Module):
    """
    Deformable attention across temporal frames.
    
    For each query position in the target frame's feature map, this module
    predicts K sampling offsets into EACH of the T context frames and
    aggregates their features with learned attention weights.
    
    Args:
        channels: Feature dimension
        n_heads: Number of attention heads
        n_points: Sampling points per head per frame
        max_offset: Maximum offset range (clamped with tanh)
    """
    
    def __init__(
        self,
        channels: int,
        n_heads: int = 4,
        n_points: int = 9,
        max_offset: int = 32,
        num_frames: int = 7,
    ):
        super().__init__()
        self.channels = channels
        self.n_heads = n_heads
        self.n_points = n_points
        self.max_offset = max_offset
        self.num_frames = num_frames
        self.head_dim = channels // n_heads
        
        # Query projection
        self.q_proj = nn.Conv2d(channels, channels, 1)
        
        # Offset prediction: for each head, K points per frame, 2 coords (x, y)
        # Total offsets predicted per query = n_heads * n_points * 2 * num_frames
        self.offset_net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, n_heads * n_points * 2 * num_frames, 1),
        )
        
        # Attention weight prediction: frame-specific weights
        # Predicts per-head, per-frame, per-point weights for proper temporal differentiation
        self.attn_net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, n_heads * n_points * num_frames, 1),
        )
        
        # Value projection (applied to sampled features)
        self.v_proj = nn.Conv2d(channels, channels, 1)
        
        # Output projection
        self.out_proj = nn.Conv2d(channels, channels, 1)
        
        self.norm = nn.GroupNorm(8, channels)
    
    def forward(
        self,
        query_feat: torch.Tensor,
        context_feats: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            query_feat: (B, C, H, W) query features (target frame position)
            context_feats: List of T tensors (B, C, H, W) from context frames
                           (temporally-enriched features)
            
        Returns:
            (B, C, H, W) attended features
        """
        B, C, H, W = query_feat.shape
        T = len(context_feats)
        K = self.n_points
        nH = self.n_heads
        hd = self.head_dim
        
        # Predict offsets from query: (B, nH*T*K*2, H, W)
        offsets = self.offset_net(query_feat)
        # Clamp offsets to prevent overflow (risk analysis P1)
        offsets = torch.tanh(offsets) * self.max_offset
        # Reshape: (B, nH, T, K, 2, H, W)
        offsets = offsets.view(B, nH, self.num_frames, K, 2, H, W)
        
        # Predict attention weights: (B, nH*T*K, H, W) — frame-specific
        attn_logits = self.attn_net(query_feat)
        # Reshape to (B, nH, T*K, H, W) for softmax over all T*K sampling points
        attn_logits = attn_logits.view(B, nH, T * K, H, W)
        attn_weights = attn_logits.softmax(dim=2)  # (B, nH, T*K, H, W)
        
        # Base grid for sampling (fp32 for precision, cast to input dtype for grid_sample)
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=query_feat.device, dtype=torch.float32),
            torch.arange(W, device=query_feat.device, dtype=torch.float32),
            indexing='ij',
        )
        base_grid = torch.stack([grid_x, grid_y], dim=0)  # (2, H, W)
        
        # Sample from each context frame using frame-specific offsets
        sampled_per_frame = []
        for t, ctx_feat in enumerate(context_feats):
            # offsets for frame t: (B, nH, K, 2, H, W)
            offset_t = offsets[:, :, t]
            offset_flat = offset_t.reshape(B, nH * K, 2, H, W)
            
            sample_x = base_grid[0].unsqueeze(0).unsqueeze(0) + offset_flat[:, :, 0]
            sample_y = base_grid[1].unsqueeze(0).unsqueeze(0) + offset_flat[:, :, 1]
            # Normalize to [-1, 1] using align_corners=True convention:
            # pixel 0 → -1, pixel W-1 → +1
            sample_x = sample_x / (W - 1) * 2 - 1
            sample_y = sample_y / (H - 1) * 2 - 1
            grid_all = torch.stack([sample_x, sample_y], dim=-1).view(B * nH * K, H, W, 2)

            v = self.v_proj(ctx_feat)  # (B, C, H, W)
            v = v.view(B, nH, hd, H, W)
            # Repeat each head's values for K points: (B, nH, K, hd, H, W) -> (B*nH*K, hd, H, W)
            v_expanded = v.unsqueeze(2).expand(B, nH, K, hd, H, W).reshape(B * nH * K, hd, H, W)

            # Batched grid_sample for all heads × points of frame t
            sampled = F.grid_sample(
                v_expanded, grid_all, mode='bilinear',
                align_corners=True, padding_mode='zeros',
            )  # (B*nH*K, hd, H, W)
            sampled = sampled.view(B, nH, K, hd, H, W)
            sampled_per_frame.append(sampled)
        
        # Stack across frames: (B, nH, T, K, hd, H, W) -> (B, nH, T*K, hd, H, W)
        sampled = torch.stack(sampled_per_frame, dim=2).reshape(B, nH, T * K, hd, H, W)
        
        # Weighted sum: (B, nH, hd, H, W)
        attn_weights_expanded = attn_weights.unsqueeze(3)  # (B, nH, T*K, 1, H, W)
        out = (sampled * attn_weights_expanded).sum(dim=2)  # (B, nH, hd, H, W)
        
        # Reshape and project
        out = out.view(B, C, H, W)
        out = self.out_proj(out)
        
        # Residual
        return query_feat + out


class SynthesisBranch(nn.Module):
    """
    Synthesis branch for complex motion / occluded regions.
    
    Uses deformable cross-temporal attention to borrow pixels from
    all 7 context frames where the content might be visible.
    
    Operates at 1/4 resolution, upsamples with pixel-shuffle.
    
    Args:
        feat_channels: Feature channels at 1/4 res (e.g., 256)
        n_attn_layers: Number of deformable attention layers (default: 2)
        n_points: Sampling points per head (default: 9)
        max_offset: Maximum offset in pixels at 1/4 res (default: 32)
    """
    
    def __init__(
        self,
        feat_channels: int = 256,
        n_attn_layers: int = 2,
        n_points: int = 9,
        max_offset: int = 32,
    ):
        super().__init__()
        
        # Deformable cross-temporal attention layers
        self.attn_layers = nn.ModuleList([
            DeformableCrossTemporalAttention(
                channels=feat_channels,
                n_heads=4,
                n_points=n_points,
                max_offset=max_offset,
            )
            for _ in range(n_attn_layers)
        ])
        
        # Edge injection: concatenate Sobel edges as extra channel
        self.edge_fuse = nn.Conv2d(feat_channels + 2, feat_channels, 1)
        
        # Decoder: 4 conv blocks to go from features -> RGB * 16 (for pixel shuffle 4x)
        self.decoder = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
            nn.GroupNorm(8, feat_channels),
            nn.GELU(),
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
            nn.GroupNorm(8, feat_channels),
            nn.GELU(),
            nn.Conv2d(feat_channels, feat_channels // 2, 3, padding=1),
            nn.GroupNorm(8, feat_channels // 2),
            nn.GELU(),
            nn.Conv2d(feat_channels // 2, 3 * 4 * 4, 3, padding=1),  # 48 = 3 * 16 for pixel shuffle
        )
        
        self.pixel_shuffle = nn.PixelShuffle(4)
    
    def forward(
        self,
        anchor_feat: torch.Tensor,
        context_feats: list[torch.Tensor],
        edge_i: torch.Tensor,
        edge_ip1: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            anchor_feat: (B, C, H/4, W/4) - mean of anchor pair features
            context_feats: List of T tensors (B, C, H/4, W/4) from temporal fusion
            edge_i: (B, 1, H, W) Sobel edge map of frame i
            edge_ip1: (B, 1, H, W) Sobel edge map of frame i+1
            
        Returns:
            (B, 3, H, W) synthesized frame at full resolution
        """
        h4, w4 = anchor_feat.shape[2], anchor_feat.shape[3]
        
        # Edge injection at 1/4 res
        edge_i_small = F.interpolate(edge_i, size=(h4, w4), mode='bilinear', align_corners=False)
        edge_ip1_small = F.interpolate(edge_ip1, size=(h4, w4), mode='bilinear', align_corners=False)
        
        x = torch.cat([anchor_feat, edge_i_small, edge_ip1_small], dim=1)
        x = self.edge_fuse(x)  # (B, C, h4, w4)
        
        # Cross-temporal deformable attention
        for attn in self.attn_layers:
            x = attn(x, context_feats)
        
        # Decode to RGB at 1/4 res, then pixel-shuffle to full res
        x = self.decoder(x)  # (B, 48, h4, w4)
        x = self.pixel_shuffle(x)  # (B, 3, H, W)
        x = torch.sigmoid(x)  # clamp to [0, 1]
        
        return x


# Module-level cached Sobel kernels to avoid per-call GPU allocation
_SOBEL_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


def compute_sobel_edges(img: torch.Tensor) -> torch.Tensor:
    """
    Compute Sobel edge magnitude from an RGB image.
    
    Args:
        img: (B, 3, H, W) in [0, 1]
        
    Returns:
        (B, 1, H, W) edge magnitude
    """
    # Convert to grayscale
    gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
    
    # Get or create cached Sobel kernels
    cache_key = (img.dtype, str(img.device))
    if cache_key not in _SOBEL_CACHE:
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
        _SOBEL_CACHE[cache_key] = (sobel_x, sobel_y)
    sobel_x, sobel_y = _SOBEL_CACHE[cache_key]
    
    gx = F.conv2d(gray, sobel_x, padding=1)
    gy = F.conv2d(gray, sobel_y, padding=1)
    
    magnitude = torch.sqrt(gx ** 2 + gy ** 2 + 1e-3)
    # Normalize to [0, 1] per sample (eps=1e-3 safe for fp16, min normal ~6e-5)
    B = magnitude.shape[0]
    max_vals = magnitude.view(B, -1).max(dim=1).values.view(B, 1, 1, 1)
    magnitude = magnitude / (max_vals + 1e-3)
    
    return magnitude


if __name__ == '__main__':
    synth = SynthesisBranch(feat_channels=256, n_attn_layers=2, n_points=9)
    
    B = 2
    anchor = torch.randn(B, 256, 64, 64)
    ctx = [torch.randn(B, 256, 64, 64) for _ in range(7)]
    edge_i = torch.randn(B, 1, 256, 256)
    edge_ip1 = torch.randn(B, 1, 256, 256)
    
    out = synth(anchor, ctx, edge_i, edge_ip1)
    print(f"Synthesis output: {out.shape}")
    params = sum(p.numel() for p in synth.parameters())
    print(f"Parameters: {params:,}")
