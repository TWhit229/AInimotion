"""
Loss functions for VFI training.

Combines:
- L1 reconstruction loss
- VGG perceptual loss  
- Edge-weighted L1 loss (protects ink lines)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class PerceptualLoss(nn.Module):
    """
    VGG-based perceptual loss.
    
    Compares feature activations from VGG19 layers to preserve
    artistic style and texture consistency.
    
    Args:
        layer_weights: Dict of VGG layer name -> loss weight
    """
    
    def __init__(
        self,
        layer_weights: dict[str, float] | None = None,
    ):
        super().__init__()
        
        if layer_weights is None:
            # Default: use relu1_2, relu2_2, relu3_4, relu4_4
            layer_weights = {
                'relu1_2': 0.1,
                'relu2_2': 0.2,
                'relu3_4': 0.4,
                'relu4_4': 0.3,
            }
        
        self.layer_weights = layer_weights
        
        # Load pretrained VGG19
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        
        # Map layer names to indices
        self.layer_map = {
            'relu1_1': 1, 'relu1_2': 3,
            'relu2_1': 6, 'relu2_2': 8,
            'relu3_1': 11, 'relu3_2': 13, 'relu3_3': 15, 'relu3_4': 17,
            'relu4_1': 20, 'relu4_2': 22, 'relu4_3': 24, 'relu4_4': 26,
            'relu5_1': 29, 'relu5_2': 31, 'relu5_3': 33, 'relu5_4': 35,
        }
        
        # Build feature extractors for each layer we need
        max_idx = max(self.layer_map[name] for name in layer_weights.keys())
        self.features = nn.Sequential(*list(vgg.features.children())[:max_idx + 1])
        
        # Freeze VGG weights
        for param in self.features.parameters():
            param.requires_grad = False
        
        # ImageNet normalization
        self.register_buffer(
            'mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input to VGG expected range."""
        return (x - self.mean) / self.std
    
    def _extract_features(
        self, 
        x: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Extract features at specified layers."""
        x = self._normalize(x)
        features = {}
        
        for i, layer in enumerate(self.features):
            x = layer(x)
            # Check if this is a layer we want
            for name, idx in self.layer_map.items():
                if idx == i and name in self.layer_weights:
                    features[name] = x
        
        return features
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute perceptual loss.
        
        Args:
            pred: (B, 3, H, W) predicted image
            target: (B, 3, H, W) target image
            
        Returns:
            Scalar loss value
        """
        pred_features = self._extract_features(pred)
        target_features = self._extract_features(target)
        
        loss = 0.0
        for name, weight in self.layer_weights.items():
            loss += weight * F.l1_loss(
                pred_features[name],
                target_features[name],
            )
        
        return loss


class EdgeWeightedL1Loss(nn.Module):
    """
    L1 loss with higher weights on edge pixels.
    
    Uses Sobel filters to detect edges and applies higher
    loss weight to those pixels to protect ink lines.
    
    Args:
        edge_weight: Weight multiplier for edge pixels
        base_weight: Weight for non-edge pixels
    """
    
    def __init__(
        self,
        edge_weight: float = 10.0,
        base_weight: float = 1.0,
    ):
        super().__init__()
        self.edge_weight = edge_weight
        self.base_weight = base_weight
        
        # Sobel kernels for edge detection
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1],
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1],
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
    
    def _compute_edges(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute edge magnitude using Sobel filters.
        
        Args:
            x: (B, C, H, W) input image
            
        Returns:
            (B, 1, H, W) edge magnitude
        """
        # Convert to grayscale
        if x.shape[1] == 3:
            gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        else:
            gray = x
        
        # Apply Sobel filters
        grad_x = F.conv2d(gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(gray, self.sobel_y, padding=1)
        
        # Edge magnitude
        magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
        
        # Normalize to [0, 1]
        magnitude = magnitude / (magnitude.max() + 1e-6)
        
        return magnitude
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute edge-weighted L1 loss.
        
        Args:
            pred: (B, 3, H, W) predicted image
            target: (B, 3, H, W) target image
            
        Returns:
            Scalar loss value
        """
        # Compute edges from target
        edges = self._compute_edges(target)
        
        # Weight map: higher weight on edges
        weights = self.base_weight + edges * (self.edge_weight - self.base_weight)
        
        # Weighted L1
        l1_diff = torch.abs(pred - target)
        weighted_loss = (weights * l1_diff).mean()
        
        return weighted_loss


class VFILoss(nn.Module):
    """
    Combined loss for VFI training.
    
    Combines:
        - L1 reconstruction loss
        - VGG perceptual loss
        - Edge-weighted L1 loss
        - Optional GAN loss (added externally)
    
    Args:
        l1_weight: Weight for L1 loss
        perceptual_weight: Weight for VGG perceptual loss
        edge_weight: Weight for edge-weighted L1 loss
        edge_multiplier: Multiplier for edge pixels in edge loss
    """
    
    def __init__(
        self,
        l1_weight: float = 1.0,
        perceptual_weight: float = 0.1,
        edge_weight: float = 0.5,
        edge_multiplier: float = 10.0,
    ):
        super().__init__()
        
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.edge_weight = edge_weight
        
        # Component losses
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss()
        self.edge_loss = EdgeWeightedL1Loss(edge_weight=edge_multiplier)
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        return_components: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute combined loss.
        
        Args:
            pred: (B, 3, H, W) predicted image
            target: (B, 3, H, W) target image
            return_components: Whether to also return individual loss values
            
        Returns:
            Total loss, optionally with component dict
        """
        # Compute component losses
        l1 = self.l1_loss(pred, target)
        perceptual = self.perceptual_loss(pred, target)
        edge = self.edge_loss(pred, target)
        
        # Combine
        total = (
            self.l1_weight * l1 +
            self.perceptual_weight * perceptual +
            self.edge_weight * edge
        )
        
        if return_components:
            components = {
                'l1': l1,
                'perceptual': perceptual,
                'edge': edge,
                'total': total,
            }
            return total, components
        
        return total
