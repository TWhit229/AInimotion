"""
Compositor module for layer blending.

v4: Motion Complexity Routing (MCR)
  - Receives both AdaCoF output and synthesis output
  - Uses routing map to blend: fg = (1-R)*adacof + R*synthesis
  - Refinement U-Net takes 18 channels (added routing_map + edge_map)
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
    
    v4: 18 input channels:
      composite(3) + frame1(3) + frame2(3) + alpha(1)
      + background(3) + foreground(3) + routing_map(1) + edge_map(1) = 18
    """
    
    def __init__(self, in_channels: int = 18):
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
    Layer compositor with Motion Complexity Routing (MCR).
    
    v4 improvements:
      - Receives both AdaCoF and synthesis foreground outputs
      - Routes between them using per-pixel complexity map
      - Refinement U-Net gets routing map + edge map as context (18ch)
    
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
            # v4: 18 channels (added routing_map + edge_map)
            self.refinement = RefinementNet(in_channels=18)
    
    def forward(
        self,
        background: torch.Tensor,
        fg_warp: torch.Tensor,
        fg_synth: torch.Tensor,
        routing_map: torch.Tensor,
        edge_map: torch.Tensor,
        feat1: list[torch.Tensor],
        feat2: list[torch.Tensor],
        corr: list[torch.Tensor],
        frame1: torch.Tensor | None = None,
        frame2: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Composite layers with motion-routed foreground.
        
        Args:
            background: (B, 3, H, W) background canvas
            fg_warp: (B, 3, H, W) AdaCoF warped foreground
            fg_synth: (B, 3, H, W) synthesized foreground
            routing_map: (B, 1, H, W) complexity routing ∈ [0,1]
            edge_map: (B, 1, H, W) combined edge map
            feat1, feat2: FPN features
            corr: Correlation volumes
            frame1, frame2: Original input frames
        """
        h, w = background.shape[2:]
        
        # Route foreground: blend warp and synthesis based on motion complexity
        foreground = (1 - routing_map) * fg_warp + routing_map * fg_synth
        
        # Alpha prediction (same as v3)
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
                routing_map, edge_map,
            ], dim=1)  # 3+3+3+1+3+3+1+1 = 18
            output = self.refinement(refine_input)
        else:
            output = composite
        
        output = output.clamp(0, 1)
        
        return {
            'output': output,
            'alpha': alpha,
            'foreground': foreground,
            'composite_raw': composite,
        }
