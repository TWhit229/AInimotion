"""
Scene Gate for detecting hard cuts between frames.

v3: Configurable corr_channels (default 169 for displacement=6).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SceneGate(nn.Module):
    """
    Scene cut detector using correlation analysis.
    
    Args:
        corr_channels: Number of correlation volume channels
        threshold: Correlation threshold below which scene cut is detected
    """
    
    def __init__(
        self,
        corr_channels: int = 169,  # (2*6+1)^2 = 169 for displacement 6
        threshold: float = 0.15,
    ):
        super().__init__()
        self.threshold = threshold
        
        # Small network to analyze correlation statistics
        self.analyzer = nn.Sequential(
            nn.Conv2d(corr_channels, 64, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.AdaptiveAvgPool2d(8),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        corr_volumes: list[torch.Tensor],
        return_confidence: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Determine if each sample in batch is a scene cut.
        
        Args:
            corr_volumes: List of correlation volumes from FPN
            return_confidence: Whether to also return confidence scores
            
        Returns:
            (B,) boolean tensor where True = scene cut detected
            If return_confidence: also returns (B,) confidence scores
        """
        # Use the coarsest correlation volume (most global view)
        corr = corr_volumes[-1]
        
        # Get confidence from analyzer network
        confidence = self.analyzer(corr).squeeze(-1)
        
        # Scene cut if confidence is below threshold
        is_scene_cut = confidence < self.threshold
        
        if return_confidence:
            return is_scene_cut, confidence
        return is_scene_cut
