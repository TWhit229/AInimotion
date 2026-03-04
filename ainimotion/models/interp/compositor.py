"""
Compositor module for layer blending.

Combines background canvas and foreground character layers
using a learned soft alpha mask, with a strong refinement network.

v2: Deeper refinement U-Net with residual blocks and full intermediate context.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AlphaMaskPredictor(nn.Module):
    """
    Predicts soft alpha mask for layer compositing.
    
    Outputs values in [0, 1] where:
        - 1 = pure foreground (character)
        - 0 = pure background
    
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
        """
        Predict alpha mask.
        
        Args:
            features: (B, C, H, W) input features
            
        Returns:
            (B, 1, H, W) alpha mask in [0, 1]
        """
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
    Deeper U-Net with residual blocks for final refinement.
    
    Takes the full context (composite, both input frames, alpha mask,
    background, foreground) and learns to clean up blending artifacts,
    sharpen edges, and correct color.
    
    v2: 4-level U-Net (32→64→128→256) with residual blocks, 16ch input.
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
            nn.Conv2d(256, 64, 3, padding=1),  # 128 + 128 skip
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlock(64),
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 32, 3, padding=1),  # 64 + 64 skip
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlock(32),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),   # 32 + 32 skip
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlock(32),
            nn.Conv2d(32, 3, 3, padding=1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Refine composited image.
        
        Args:
            x: (B, 16, H, W) concatenated context
               [composite(3), frame1(3), frame3(3), alpha(1), background(3), foreground(3)]
            
        Returns:
            (B, 3, H, W) refined output (residual added to composite)
        """
        # Extract composite for residual connection
        composite = x[:, :3]
        
        # Encoder
        e1 = self.enc1(x)      # (B, 32, H, W)
        e2 = self.enc2(e1)     # (B, 64, H/2, W/2)
        e3 = self.enc3(e2)     # (B, 128, H/4, W/4)
        e4 = self.enc4(e3)     # (B, 256, H/8, W/8)
        
        # Decoder with skip connections
        d4 = self.dec4(e4)     # (B, 128, H/8, W/8)
        d4_up = F.interpolate(d4, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d4_up, e3], dim=1))  # (B, 64, H/4, W/4)
        d3_up = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d3_up, e2], dim=1))  # (B, 32, H/2, W/2)
        d2_up = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        residual = self.dec1(torch.cat([d2_up, e1], dim=1))  # (B, 3, H, W)
        
        # Add residual to composite
        return composite + residual


class Compositor(nn.Module):
    """
    Layer compositor for combining background and foreground.
    
    Uses soft alpha blending with strong refinement.
    
    v2: Refinement takes full context (16 channels) for better correction.
    
    Args:
        feat_channels: Feature channels at each FPN scale
        use_refinement: Whether to apply refinement U-Net
    """
    
    def __init__(
        self,
        feat_channels: list[int] = [32, 64, 128, 128],
        use_refinement: bool = True,
    ):
        super().__init__()
        
        # Input: finest features from both frames + correlation
        in_channels = feat_channels[0] * 2 + 81  # 81 = corr channels
        
        self.alpha_predictor = AlphaMaskPredictor(in_channels)
        self.use_refinement = use_refinement
        
        if use_refinement:
            # Input to refinement: composite(3) + frame1(3) + frame3(3)
            #                     + alpha(1) + background(3) + foreground(3) = 16
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
        """
        Composite layers into final output.
        
        Args:
            background: (B, 3, H, W) background canvas
            foreground: (B, 3, H, W) synthesized foreground
            feat1, feat2: FPN features
            corr: Correlation volumes
            frame1, frame2: Original input frames (for refinement)
            
        Returns:
            Dictionary with:
                - 'output': (B, 3, H, W) final composited frame
                - 'alpha': (B, 1, H, W) predicted alpha mask
        """
        h, w = background.shape[2:]
        
        # Get finest features
        f1 = feat1[0]
        f2 = feat2[0]
        c = corr[0]
        
        # Upsample to output resolution if needed
        if f1.shape[2] != h:
            f1 = F.interpolate(f1, size=(h, w), mode='bilinear', align_corners=False)
            f2 = F.interpolate(f2, size=(h, w), mode='bilinear', align_corners=False)
            c = F.interpolate(c, size=(h, w), mode='bilinear', align_corners=False)
        
        # Predict alpha mask
        combined = torch.cat([f1, f2, c], dim=1)
        alpha = self.alpha_predictor(combined)
        
        # Alpha blend layers
        composite = alpha * foreground + (1 - alpha) * background
        
        # Refinement with full context
        if self.use_refinement and frame1 is not None and frame2 is not None:
            refine_input = torch.cat([
                composite,   # 3 ch
                frame1,      # 3 ch
                frame2,      # 3 ch
                alpha,       # 1 ch
                background,  # 3 ch
                foreground,  # 3 ch
            ], dim=1)  # 16 ch total
            output = self.refinement(refine_input)
        else:
            output = composite
        
        # Clamp to valid range
        output = output.clamp(0, 1)
        
        return {
            'output': output,
            'alpha': alpha,
            'composite_raw': composite,
        }
    
    @staticmethod
    def visualize_alpha(alpha: torch.Tensor) -> torch.Tensor:
        """
        Convert alpha mask to RGB for visualization.
        
        Foreground areas shown in red, background in blue.
        
        Args:
            alpha: (B, 1, H, W) alpha mask
            
        Returns:
            (B, 3, H, W) RGB visualization
        """
        # Red = foreground, Blue = background
        r = alpha
        g = torch.zeros_like(alpha)
        b = 1 - alpha
        return torch.cat([r, g, b], dim=1)
