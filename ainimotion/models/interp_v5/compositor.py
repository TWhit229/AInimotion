"""
V5 Layer Compositor + Edge-Guided Refinement.

Blends foreground (routed warp/synthesis) with background using a
learned alpha mask. Then applies a lightweight residual refinement
network with edge guidance for line art sharpening.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerCompositor(nn.Module):
    """
    Blend foreground and background layers with a learned alpha mask.
    
    The alpha mask is predicted from features + correlation + input frames.
    
    Args:
        feat_channels: Feature channels at 1/4 resolution
    """
    
    def __init__(self, feat_channels: int = 256):
        super().__init__()
        
        # Input: frame_i + frame_ip1 + fg + bg + edge_map = 3+3+3+3+1 = 13
        self.alpha_net = nn.Sequential(
            nn.Conv2d(13, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        foreground: torch.Tensor,
        background: torch.Tensor,
        frame_i: torch.Tensor,
        frame_ip1: torch.Tensor,
        edge_map: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            foreground: (B, 3, H, W) routed foreground
            background: (B, 3, H, W) warped background
            frame_i: (B, 3, H, W) anchor frame i
            frame_ip1: (B, 3, H, W) anchor frame i+1
            edge_map: (B, 1, H, W) combined edge map
            
        Returns:
            Dict with:
              - 'composite': (B, 3, H, W)
              - 'alpha': (B, 1, H, W)
        """
        alpha_input = torch.cat([
            frame_i, frame_ip1, foreground, background, edge_map
        ], dim=1)
        
        alpha = self.alpha_net(alpha_input)
        composite = alpha * foreground + (1 - alpha) * background
        
        return {
            'composite': composite,
            'alpha': alpha,
        }


class EdgeGuidedRefinement(nn.Module):
    """
    Lightweight residual refinement with edge guidance.
    
    Takes the composite + Sobel edge maps of both anchors
    and produces a residual correction focused on sharpening lines.
    
    3 residual blocks, very simple.
    """
    
    def __init__(self):
        super().__init__()
        
        # Input: composite (3) + edge_i (1) + edge_ip1 (1) = 5
        self.stem = nn.Conv2d(5, 32, 3, padding=1)
        
        self.blocks = nn.Sequential(
            self._res_block(32),
            self._res_block(32),
            self._res_block(32),
        )
        
        self.head = nn.Conv2d(32, 3, 3, padding=1)
        # Initialize near zero so refinement starts as identity
        self.head.weight.data.mul_(0.01)
        self.head.bias.data.zero_()
    
    @staticmethod
    def _res_block(channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
        )
    
    def forward(
        self,
        composite: torch.Tensor,
        edge_i: torch.Tensor,
        edge_ip1: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            composite: (B, 3, H, W) composited frame
            edge_i: (B, 1, H, W) Sobel edges from frame i
            edge_ip1: (B, 1, H, W) Sobel edges from frame i+1
            
        Returns:
            (B, 3, H, W) refined output
        """
        x = torch.cat([composite, edge_i, edge_ip1], dim=1)
        x = F.gelu(self.stem(x))
        
        # Residual blocks (with skip connections)
        for block in self.blocks:
            x = x + block(x)
        
        # Residual correction
        residual = self.head(x)
        output = torch.clamp(composite + residual, 0.0, 1.0)
        
        return output


if __name__ == '__main__':
    comp = LayerCompositor(feat_channels=256)
    ref = EdgeGuidedRefinement()
    
    fg = torch.randn(2, 3, 256, 256).sigmoid()
    bg = torch.randn(2, 3, 256, 256).sigmoid()
    fi = torch.randn(2, 3, 256, 256).sigmoid()
    fi1 = torch.randn(2, 3, 256, 256).sigmoid()
    ei = torch.randn(2, 1, 256, 256).abs()
    ei1 = torch.randn(2, 1, 256, 256).abs()
    em = (ei + ei1) / 2
    
    cout = comp(fg, bg, fi, fi1, em)
    print(f"Composite: {cout['composite'].shape}, Alpha: {cout['alpha'].shape}")
    
    refined = ref(cout['composite'], ei, ei1)
    print(f"Refined: {refined.shape}")
    
    print(f"Compositor params: {sum(p.numel() for p in comp.parameters()):,}")
    print(f"Refinement params: {sum(p.numel() for p in ref.parameters()):,}")
