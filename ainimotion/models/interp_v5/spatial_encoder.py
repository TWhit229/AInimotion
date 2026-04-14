"""
V5 Spatial Encoder - Shared Feature Pyramid Network.

Each input frame is encoded independently (shared weights) into
multi-scale spatial features at 3 scales: 1x, 1/2, 1/4.

Base channels: 64, keeps things simple with ResNet-style blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Simple residual block: conv-norm-relu-conv-norm + skip."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
    
    def forward(self, x):
        residual = x
        x = F.gelu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return F.gelu(x + residual)


class DownBlock(nn.Module):
    """Downsample by 2x with strided conv + residual block."""
    
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.down = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)
        self.norm = nn.GroupNorm(8, out_ch)
        self.block = ResBlock(out_ch)
    
    def forward(self, x):
        x = F.gelu(self.norm(self.down(x)))
        return self.block(x)


class SpatialEncoder(nn.Module):
    """
    Shared spatial encoder for V5.
    
    Takes a single frame (B, 3, H, W) and produces multi-scale features:
      - scale0: (B, C, H, W)        -- full resolution
      - scale1: (B, 2C, H/2, W/2)   -- half resolution  
      - scale2: (B, 4C, H/4, W/4)   -- quarter resolution
    
    Args:
        base_channels: Number of channels at scale 0 (default: 64)
    """
    
    def __init__(self, base_channels: int = 64):
        super().__init__()
        C = base_channels
        
        # Initial conv from RGB to feature space
        self.stem = nn.Sequential(
            nn.Conv2d(3, C, 3, padding=1),
            nn.GroupNorm(8, C),
            nn.GELU(),
            ResBlock(C),
        )
        
        # Scale 1: H/2
        self.down1 = DownBlock(C, C * 2)
        
        # Scale 2: H/4
        self.down2 = DownBlock(C * 2, C * 4)
        
        self.out_channels = [C, C * 2, C * 4]
    
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W) input frame, values in [0, 1]
            
        Returns:
            List of feature tensors at 3 scales [scale0, scale1, scale2]
        """
        s0 = self.stem(x)          # (B, C, H, W)
        s1 = self.down1(s0)        # (B, 2C, H/2, W/2)
        s2 = self.down2(s1)        # (B, 4C, H/4, W/4)
        return [s0, s1, s2]


if __name__ == '__main__':
    enc = SpatialEncoder(base_channels=64)
    x = torch.randn(2, 3, 256, 256)
    feats = enc(x)
    for i, f in enumerate(feats):
        print(f"Scale {i}: {f.shape}")
    params = sum(p.numel() for p in enc.parameters())
    print(f"Parameters: {params:,}")
