"""
Differentiable Sobel edge extraction for anime line art.

Extracts edge maps from RGB images using fixed Sobel kernels.
Used to provide structural guidance to the synthesis decoder —
anime lines warp well even under rotation, giving the synthesis
path a "skeleton" to fill in.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeExtractor(nn.Module):
    """
    Fixed (non-learnable) Sobel edge detector.
    
    Converts RGB → grayscale → Sobel magnitude → normalized edge map.
    Kernels are registered as buffers so they move to GPU automatically.
    
    Args:
        normalize: Whether to normalize output to [0, 1] per-image
    """
    
    def __init__(self, normalize: bool = True):
        super().__init__()
        self.normalize = normalize
        
        # Sobel kernels (fixed, not learnable)
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1],
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        sobel_y = torch.tensor([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1],
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        
        # RGB to grayscale weights
        self.register_buffer(
            'rgb_weights',
            torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract edge map from RGB image.
        
        Args:
            x: (B, 3, H, W) RGB image in [0, 1]
            
        Returns:
            (B, 1, H, W) edge magnitude map
        """
        # RGB → grayscale
        gray = (x * self.rgb_weights).sum(dim=1, keepdim=True)
        
        # Sobel gradients
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        
        # Magnitude
        edges = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
        
        # Normalize per-image to [0, 1]
        if self.normalize:
            b = edges.shape[0]
            flat = edges.view(b, -1)
            min_val = flat.min(dim=1, keepdim=True).values.view(b, 1, 1, 1)
            max_val = flat.max(dim=1, keepdim=True).values.view(b, 1, 1, 1)
            edges = (edges - min_val) / (max_val - min_val + 1e-8)
        
        return edges
