"""
V5 Temporal Attention Fusion.

After spatial encoding, fuse information across 7 context frames using
windowed cross-frame attention at 1/4 resolution.

Each spatial position attends to the same spatial window across all
temporal frames. This is O(7 * window^2) per position, not O(7 * H * W).

Key design choices (from risk analysis):
  - Windowed attention (8x8 patches), not global
  - 2 layers only (RVRT shows 2 is sufficient)
  - 4 attention heads
  - Only at 1/4 resolution (scale 2) to save VRAM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TemporalWindowAttention(nn.Module):
    """
    Windowed cross-temporal attention layer.
    
    For each spatial window, tokens from all T frames attend to each other.
    Window size controls the spatial extent of attention (default: 8x8).
    
    Args:
        channels: Feature dimension
        n_heads: Number of attention heads
        window_size: Spatial window size for local attention
    """
    
    def __init__(
        self,
        channels: int,
        n_heads: int = 4,
        window_size: int = 8,
    ):
        super().__init__()
        self.channels = channels
        self.n_heads = n_heads
        self.window_size = window_size
        self.head_dim = channels // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)
    
    def _partition_windows(self, x, T, H, W):
        """Reshape (B*T, C, H, W) -> (B*nH*nW, T*wh*ww, C) for windowed attention."""
        ws = self.window_size
        BT, C = x.shape[0], x.shape[1]
        B = BT // T
        
        # Pad H, W to multiples of window_size
        pH = (ws - H % ws) % ws
        pW = (ws - W % ws) % ws
        if pH > 0 or pW > 0:
            x = F.pad(x, (0, pW, 0, pH))
        
        Hp, Wp = H + pH, W + pW
        nH, nW = Hp // ws, Wp // ws
        
        # (B*T, C, Hp, Wp) -> (B, T, C, nH, ws, nW, ws)
        x = x.view(B, T, C, nH, ws, nW, ws)
        # (B, nH, nW, T, ws, ws, C)
        x = x.permute(0, 3, 5, 1, 4, 6, 2).contiguous()
        # (B*nH*nW, T*ws*ws, C)
        x = x.view(B * nH * nW, T * ws * ws, C)
        
        return x, B, nH, nW, Hp, Wp, pH, pW
    
    def _unpartition_windows(self, x, B, T, nH, nW, Hp, Wp, H, W, pH, pW):
        """Reverse of _partition_windows."""
        ws = self.window_size
        C = self.channels
        
        # (B*nH*nW, T*ws*ws, C) -> (B, nH, nW, T, ws, ws, C)
        x = x.view(B, nH, nW, T, ws, ws, C)
        # (B, T, C, nH, ws, nW, ws)
        x = x.permute(0, 3, 6, 1, 4, 2, 5).contiguous()
        # (B*T, C, Hp, Wp)
        x = x.view(B * T, C, Hp, Wp)
        
        # Remove padding
        if pH > 0 or pW > 0:
            x = x[:, :, :H, :W]
        
        return x
    
    def forward(self, x: torch.Tensor, T: int) -> torch.Tensor:
        """
        Args:
            x: (B*T, C, H, W) features from T frames stacked along batch dim
            T: number of temporal frames
            
        Returns:
            (B*T, C, H, W) temporally-fused features
        """
        BT, C, H, W = x.shape
        
        # Partition into windows: (B*nH*nW, T*ws*ws, C)
        x_win, B, nH, nW, Hp, Wp, pH, pW = self._partition_windows(x, T, H, W)
        
        # Pre-norm
        x_norm = self.norm(x_win)
        
        # QKV
        qkv = self.qkv(x_norm).reshape(
            x_win.shape[0], x_win.shape[1], 3, self.n_heads, self.head_dim
        )
        q, k, v = qkv.unbind(2)  # Each: (B*nH*nW, T*ws*ws, heads, head_dim)
        
        # Attention — use SDPA for automatic FlashAttention/MemoryEfficient kernel
        q = q.transpose(1, 2)  # (B*nH*nW, heads, T*ws*ws, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(1, 2).reshape(x_win.shape[0], x_win.shape[1], C)
        out = self.proj(out)
        
        # Residual
        out = x_win + out
        
        # Unpartition
        out = self._unpartition_windows(out, B, T, nH, nW, Hp, Wp, H, W, pH, pW)
        
        return out


class TemporalFusion(nn.Module):
    """
    Temporal Attention Fusion module for V5.
    
    Takes features from N frames at 1/4 resolution and fuses temporal
    information using windowed cross-frame attention.
    
    Args:
        channels: Feature channels at 1/4 resolution (typically 4 * base_channels)
        n_layers: Number of attention layers (default: 2)
        n_heads: Number of attention heads (default: 4)
        window_size: Attention window size (default: 8)
    """
    
    def __init__(
        self,
        channels: int,
        n_layers: int = 2,
        n_heads: int = 4,
        window_size: int = 8,
    ):
        super().__init__()
        self.n_layers = n_layers
        
        self.layers = nn.ModuleList([
            TemporalWindowAttention(channels, n_heads, window_size)
            for _ in range(n_layers)
        ])
        
        # FFN after attention
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(channels),
                nn.Linear(channels, channels * 2),
                nn.GELU(),
                nn.Linear(channels * 2, channels),
            )
            for _ in range(n_layers)
        ])
    
    def forward(
        self,
        features: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """
        Args:
            features: List of T tensors, each (B, C, H, W) at 1/4 resolution
            
        Returns:
            List of T tensors (B, C, H, W) with temporal information fused
        """
        T = len(features)
        B, C, H, W = features[0].shape
        
        # Stack: (B, T, C, H, W) -> view as (B*T, C, H, W)
        x = torch.stack(features, dim=1).view(B * T, C, H, W)
        
        for attn_layer, ffn in zip(self.layers, self.ffns):
            # Attention
            x = attn_layer(x, T)
            
            # FFN (applied per-pixel)
            BT, C, H2, W2 = x.shape
            x_flat = x.permute(0, 2, 3, 1).reshape(BT * H2 * W2, C)
            x_flat = x_flat + ffn(x_flat)
            x = x_flat.reshape(BT, H2, W2, C).permute(0, 3, 1, 2)
        
        # Unstack: view as (B, T, C, H, W) -> unbind along T
        return list(x.view(B, T, C, H, W).unbind(dim=1))


# Pairwise correlation between two feature maps
def compute_correlation(feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
    """
    Compute normalized correlation between two feature maps.
    
    Args:
        feat1: (B, C, H, W)
        feat2: (B, C, H, W)
        
    Returns:
        (B, 1, H, W) correlation map (cosine similarity per pixel)
    """
    f1 = F.normalize(feat1, dim=1, eps=1e-4)  # fp16-safe eps
    f2 = F.normalize(feat2, dim=1, eps=1e-4)
    corr = (f1 * f2).sum(dim=1, keepdim=True)
    return corr


if __name__ == '__main__':
    # Test temporal fusion
    T = 7
    B = 2
    C = 256
    H, W = 64, 64
    
    fusion = TemporalFusion(channels=C, n_layers=2, n_heads=4, window_size=8)
    features = [torch.randn(B, C, H, W) for _ in range(T)]
    
    out = fusion(features)
    print(f"Input: {T} x ({B}, {C}, {H}, {W})")
    print(f"Output: {len(out)} x {out[0].shape}")
    
    params = sum(p.numel() for p in fusion.parameters())
    print(f"Parameters: {params:,}")
