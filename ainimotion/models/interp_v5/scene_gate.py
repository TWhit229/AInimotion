"""
V5 Scene Gate - Detects hard cuts between consecutive frames.

Simple 3-layer MLP on global-average-pooled correlation between
the two anchor frames (F_i, F_{i+1}). Returns a binary flag.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SceneGate(nn.Module):
    """
    Detect scene cuts from pairwise correlation.
    
    Takes correlation between F_i and F_{i+1} at 1/4 resolution,
    global-average-pools it, and classifies as scene-cut or not.
    
    Args:
        threshold: Classification threshold (default: 0.5)
    """
    
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
        
        # Input: global-averaged correlation (1 value) + std + min
        self.classifier = nn.Sequential(
            nn.Linear(3, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
    
    def forward(
        self,
        corr: torch.Tensor,
        return_confidence: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """
        Args:
            corr: (B, 1, H, W) correlation map between two anchor frames
            return_confidence: Whether to also return raw logit
            
        Returns:
            is_scene_cut: (B,) boolean tensor
            confidence: (B,) raw sigmoid score (only if return_confidence=True)
        """
        B = corr.shape[0]
        
        # Global stats: mean, std, min correlation
        mean_corr = corr.view(B, -1).mean(dim=1)
        std_corr = corr.view(B, -1).std(dim=1)
        min_corr = corr.view(B, -1).min(dim=1).values
        
        stats = torch.stack([mean_corr, std_corr, min_corr], dim=1)  # (B, 3)
        
        logit = self.classifier(stats).squeeze(-1)  # (B,)
        confidence = torch.sigmoid(logit)
        is_scene_cut = confidence > self.threshold
        
        if return_confidence:
            return is_scene_cut, confidence
        return is_scene_cut


if __name__ == '__main__':
    gate = SceneGate()
    corr = torch.randn(2, 1, 64, 64)
    cut, conf = gate(corr, return_confidence=True)
    print(f"Scene cut: {cut}, Confidence: {conf}")
    print(f"Parameters: {sum(p.numel() for p in gate.parameters()):,}")
