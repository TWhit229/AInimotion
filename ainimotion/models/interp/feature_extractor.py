"""
Feature Pyramid Network (FPN) for multi-scale feature extraction.

Extracts semantic features at multiple scales for motion estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Basic conv-bn-relu block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    """Residual block with two convolutions."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class Encoder(nn.Module):
    """
    Multi-scale encoder (bottom-up pathway).
    
    Produces features at 4 scales: 1/2, 1/4, 1/8, 1/16 of input resolution.
    """
    
    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        c = base_channels
        
        # Level 0: 1/2 scale
        self.conv0 = nn.Sequential(
            ConvBlock(in_channels, c, stride=2),
            ResBlock(c),
        )
        
        # Level 1: 1/4 scale
        self.conv1 = nn.Sequential(
            ConvBlock(c, c * 2, stride=2),
            ResBlock(c * 2),
        )
        
        # Level 2: 1/8 scale
        self.conv2 = nn.Sequential(
            ConvBlock(c * 2, c * 4, stride=2),
            ResBlock(c * 4),
        )
        
        # Level 3: 1/16 scale (bottleneck)
        self.conv3 = nn.Sequential(
            ConvBlock(c * 4, c * 8, stride=2),
            ResBlock(c * 8),
            ResBlock(c * 8),
        )
    
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            x: (B, C, H, W) input image
            
        Returns:
            List of features at scales [1/2, 1/4, 1/8, 1/16]
        """
        f0 = self.conv0(x)   # 1/2
        f1 = self.conv1(f0)  # 1/4
        f2 = self.conv2(f1)  # 1/8
        f3 = self.conv3(f2)  # 1/16
        return [f0, f1, f2, f3]


class FeaturePyramid(nn.Module):
    """
    Feature Pyramid Network for frame pair feature extraction.
    
    Extracts multi-scale features from both input frames and computes
    correlation volumes for motion estimation.
    
    Args:
        in_channels: Input image channels (default 3 for RGB)
        base_channels: Base feature channels (doubled at each level)
    """
    
    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        self.base_channels = base_channels
        
        # Shared encoder for both frames
        self.encoder = Encoder(in_channels, base_channels)
        
        # Lateral connections (1x1 conv to unify channels for FPN)
        # All project to c*4 for consistent top-down pathway
        c = base_channels
        self.lateral3 = nn.Conv2d(c * 8, c * 4, 1)
        self.lateral2 = nn.Conv2d(c * 4, c * 4, 1)
        self.lateral1 = nn.Conv2d(c * 2, c * 4, 1)  # Project up to c*4
        self.lateral0 = nn.Conv2d(c, c * 4, 1)      # Project up to c*4
        
        # Smooth layers after upsampling (all c*4)
        self.smooth2 = ConvBlock(c * 4, c * 4)
        self.smooth1 = ConvBlock(c * 4, c * 4)
        self.smooth0 = ConvBlock(c * 4, c * 4)
    
    def _upsample_add(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor
    ) -> torch.Tensor:
        """Upsample x and add to y."""
        return F.interpolate(
            x, size=y.shape[2:], mode='bilinear', align_corners=False
        ) + y
    
    def forward(
        self, 
        frame1: torch.Tensor, 
        frame2: torch.Tensor
    ) -> dict[str, list[torch.Tensor]]:
        """
        Extract FPN features from both frames.
        
        Args:
            frame1: (B, C, H, W) first frame
            frame2: (B, C, H, W) second frame
            
        Returns:
            Dictionary with:
                - 'feat1': List of frame1 features at 4 scales
                - 'feat2': List of frame2 features at 4 scales
                - 'corr': List of correlation volumes at 4 scales
        """
        # Bottom-up
        enc1 = self.encoder(frame1)  # [1/2, 1/4, 1/8, 1/16]
        enc2 = self.encoder(frame2)
        
        # Top-down for frame1
        p3_1 = self.lateral3(enc1[3])
        p2_1 = self.smooth2(self._upsample_add(p3_1, self.lateral2(enc1[2])))
        p1_1 = self.smooth1(self._upsample_add(p2_1, self.lateral1(enc1[1])))
        p0_1 = self.smooth0(self._upsample_add(p1_1, self.lateral0(enc1[0])))
        
        # Top-down for frame2
        p3_2 = self.lateral3(enc2[3])
        p2_2 = self.smooth2(self._upsample_add(p3_2, self.lateral2(enc2[2])))
        p1_2 = self.smooth1(self._upsample_add(p2_2, self.lateral1(enc2[1])))
        p0_2 = self.smooth0(self._upsample_add(p1_2, self.lateral0(enc2[0])))
        
        feat1 = [p0_1, p1_1, p2_1, p3_1]
        feat2 = [p0_2, p1_2, p2_2, p3_2]
        
        # Compute correlation volumes at each scale
        corr = []
        for f1, f2 in zip(feat1, feat2):
            c = self._compute_correlation(f1, f2)
            corr.append(c)
        
        return {
            'feat1': feat1,
            'feat2': feat2,
            'corr': corr,
        }
    
    def _compute_correlation(
        self, 
        feat1: torch.Tensor, 
        feat2: torch.Tensor,
        max_displacement: int = 4,
    ) -> torch.Tensor:
        """
        Compute local correlation volume between features.
        
        Uses a local window to avoid O(HW)^2 complexity.
        
        Args:
            feat1: (B, C, H, W) features from frame 1
            feat2: (B, C, H, W) features from frame 2
            max_displacement: Maximum displacement to search
            
        Returns:
            (B, (2d+1)^2, H, W) correlation volume
        """
        b, c, h, w = feat1.shape
        d = max_displacement
        
        # Normalize features
        feat1 = F.normalize(feat1, dim=1)
        feat2 = F.normalize(feat2, dim=1)
        
        # Pad feat2 for displacement search
        feat2_pad = F.pad(feat2, [d, d, d, d])
        
        # Compute correlations at each displacement
        corr_list = []
        for dy in range(-d, d + 1):
            for dx in range(-d, d + 1):
                feat2_shift = feat2_pad[:, :, d+dy:d+dy+h, d+dx:d+dx+w]
                corr = (feat1 * feat2_shift).sum(dim=1, keepdim=True)
                corr_list.append(corr)
        
        return torch.cat(corr_list, dim=1)  # (B, (2d+1)^2, H, W)
    
    @property
    def out_channels(self) -> list[int]:
        """Output channels at each scale (all levels are c*4 after FPN)."""
        c = self.base_channels
        return [c * 4, c * 4, c * 4, c * 4]
