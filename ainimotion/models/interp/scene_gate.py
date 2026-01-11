"""
Scene Gate for detecting hard cuts between frames.

Uses correlation analysis to determine if frames are from different scenes,
in which case interpolation should be skipped.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SceneGate(nn.Module):
    """
    Scene cut detector using correlation analysis.
    
    Analyzes correlation volumes from the FPN to determine if two frames
    are from the same scene (should interpolate) or different scenes
    (should return frame 1 instead).
    
    Args:
        corr_channels: Number of correlation volume channels
        threshold: Correlation threshold below which scene cut is detected
    """
    
    def __init__(
        self,
        corr_channels: int = 81,  # (2*4+1)^2 = 81 for displacement 4
        threshold: float = 0.15,
    ):
        super().__init__()
        self.threshold = threshold
        
        # Small network to analyze correlation statistics
        self.analyzer = nn.Sequential(
            nn.Conv2d(corr_channels, 64, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.AdaptiveAvgPool2d(8),  # Reduce to 8x8
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
        corr = corr_volumes[-1]  # (B, D^2, H, W)
        
        # Get confidence from analyzer network
        confidence = self.analyzer(corr).squeeze(-1)  # (B,)
        
        # Scene cut if confidence is below threshold
        is_scene_cut = confidence < self.threshold
        
        if return_confidence:
            return is_scene_cut, confidence
        return is_scene_cut
    
    def get_statistics(
        self, 
        corr_volumes: list[torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Get correlation statistics for debugging/visualization.
        
        Args:
            corr_volumes: List of correlation volumes
            
        Returns:
            Dictionary with statistics per batch sample
        """
        # Analyze center correlation (identity displacement)
        center_idx = corr_volumes[0].shape[1] // 2
        
        stats = {}
        for i, corr in enumerate(corr_volumes):
            # Center correlation (pixel matches itself)
            center_corr = corr[:, center_idx].mean(dim=(1, 2))
            
            # Max correlation (best match anywhere)
            max_corr = corr.max(dim=1)[0].mean(dim=(1, 2))
            
            # Mean correlation overall
            mean_corr = corr.mean(dim=(1, 2, 3))
            
            stats[f'scale_{i}_center'] = center_corr
            stats[f'scale_{i}_max'] = max_corr
            stats[f'scale_{i}_mean'] = mean_corr
        
        return stats


def detect_scene_cut_simple(
    frame1: torch.Tensor,
    frame2: torch.Tensor,
    threshold: float = 0.85,
) -> torch.Tensor:
    """
    Simple scene cut detection using image-level SSIM-like metric.
    
    Fallback method that doesn't require the full FPN pipeline.
    
    Args:
        frame1: (B, C, H, W) first frame
        frame2: (B, C, H, W) second frame
        threshold: Similarity threshold (below = scene cut)
        
    Returns:
        (B,) boolean tensor where True = scene cut detected
    """
    # Downscale for efficiency
    f1 = F.interpolate(frame1, size=(64, 64), mode='bilinear', align_corners=False)
    f2 = F.interpolate(frame2, size=(64, 64), mode='bilinear', align_corners=False)
    
    # Convert to grayscale
    f1_gray = f1.mean(dim=1)
    f2_gray = f2.mean(dim=1)
    
    # Compute simple correlation
    f1_norm = f1_gray - f1_gray.mean(dim=(1, 2), keepdim=True)
    f2_norm = f2_gray - f2_gray.mean(dim=(1, 2), keepdim=True)
    
    f1_std = f1_norm.std(dim=(1, 2), keepdim=True).clamp(min=1e-6)
    f2_std = f2_norm.std(dim=(1, 2), keepdim=True).clamp(min=1e-6)
    
    ncc = (f1_norm * f2_norm).mean(dim=(1, 2)) / (f1_std * f2_std).squeeze()
    
    return ncc < threshold
