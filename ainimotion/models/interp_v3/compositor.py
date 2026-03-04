"""
Compositor module for layer blending.

v3: Updated corr_channels (169) for wider correlation volume.
    All other logic identical to v2 (same refinement U-Net).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AlphaMaskPredictor(nn.Module):
    """
    Predicts soft alpha mask for layer compositing.
    
    Args:
        in_channels: Input feature channels
    """
    
    def __init__(self, in_channels: int):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid(),
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.conv(features)


class ResidualBlock(nn.Module):
    """Residual block with two conv layers and skip connection."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x)


class RefinementNet(nn.Module):
    """
    4-level U-Net with residual blocks for final refinement.
    
    Takes full context (composite + frames + alpha + bg + fg = 16ch)
    and learns to clean up blending artifacts.
    """
    
    def __init__(self, in_channels: int = 16):
        super().__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlock(32),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlock(64),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlock(128),
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlock(256),
        )
        
        # Decoder
        self.dec4 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.dec3 = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlock(64),
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 32, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlock(32),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlock(32),
            nn.Conv2d(32, 3, 3, padding=1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        composite = x[:, :3]
        
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        d4 = self.dec4(e4)
        d4_up = F.interpolate(d4, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d4_up, e3], dim=1))
        d3_up = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d3_up, e2], dim=1))
        d2_up = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        residual = self.dec1(torch.cat([d2_up, e1], dim=1))
        
        return composite + residual


class Compositor(nn.Module):
    """
    Layer compositor for combining background and foreground.
    
    v3: Updated corr_channels for wider correlation volume.
    
    Args:
        feat_channels: Feature channels at each FPN scale
        use_refinement: Whether to apply refinement U-Net
        corr_channels: Correlation volume channels
    """
    
    def __init__(
        self,
        feat_channels: list[int] = [32, 64, 128, 128],
        use_refinement: bool = True,
        corr_channels: int = 169,
    ):
        super().__init__()
        
        in_channels = feat_channels[0] * 2 + corr_channels
        
        self.alpha_predictor = AlphaMaskPredictor(in_channels)
        self.use_refinement = use_refinement
        
        if use_refinement:
            # composite(3) + frame1(3) + frame2(3) + alpha(1)
            # + background(3) + foreground(3) = 16
            self.refinement = RefinementNet(in_channels=16)
    
    def forward(
        self,
        background: torch.Tensor,
        foreground: torch.Tensor,
        feat1: list[torch.Tensor],
        feat2: list[torch.Tensor],
        corr: list[torch.Tensor],
        frame1: torch.Tensor | None = None,
        frame2: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Composite layers into final output."""
        h, w = background.shape[2:]
        
        f1 = feat1[0]
        f2 = feat2[0]
        c = corr[0]
        
        if f1.shape[2] != h:
            f1 = F.interpolate(f1, size=(h, w), mode='bilinear', align_corners=False)
            f2 = F.interpolate(f2, size=(h, w), mode='bilinear', align_corners=False)
            c = F.interpolate(c, size=(h, w), mode='bilinear', align_corners=False)
        
        combined = torch.cat([f1, f2, c], dim=1)
        alpha = self.alpha_predictor(combined)
        
        composite = alpha * foreground + (1 - alpha) * background
        
        if self.use_refinement and frame1 is not None and frame2 is not None:
            refine_input = torch.cat([
                composite, frame1, frame2,
                alpha, background, foreground,
            ], dim=1)
            output = self.refinement(refine_input)
        else:
            output = composite
        
        output = output.clamp(0, 1)
        
        return {
            'output': output,
            'alpha': alpha,
            'composite_raw': composite,
        }
