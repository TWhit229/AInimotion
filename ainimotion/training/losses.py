"""
Loss functions for VFI training.

Combines (RIFE/IFRNet-inspired recipe):
- Charbonnier reconstruction loss (replaces L1 — differentiable at zero)
- Census structural loss (illumination-invariant structural matching)
- VGG perceptual loss (optional — disabled by default for PSNR focus)
- Edge-weighted L1 loss (protects ink lines)

References:
    RIFE: Huang et al., ECCV 2022
    IFRNet: Kong et al., CVPR 2022
    Charbonnier: Charbonnier et al., ICIP 1994
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class CharbonnierLoss(nn.Module):
    """
    Charbonnier loss (pseudo-Huber loss).
    
    Acts like L2 for small errors (smooth gradients near optimum)
    and like L1 for large errors (robust to outliers). Differentiable
    everywhere, unlike L1 which has undefined gradient at zero.
    
    Used by both RIFE and IFRNet as their primary reconstruction loss.
    
    L(x) = sqrt(x^2 + eps^2)
    
    Args:
        eps: Smoothing parameter (default 1e-3, same as IFRNet)
    """
    
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps_sq = eps ** 2
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Charbonnier loss between pred and target."""
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps_sq)
        return loss.mean()


class CensusLoss(nn.Module):
    """
    Census Transform loss for structural similarity.
    
    Compares local structural patterns using the ternary census transform,
    which is invariant to global illumination changes and contrast shifts.
    Used by both RIFE and IFRNet as a complement to Charbonnier loss.
    
    The census transform encodes each pixel's relationship to its
    neighbors as a binary/ternary pattern, then compares patterns
    between predicted and target images using soft Hamming distance.
    
    Args:
        patch_size: Size of the local patch (default 7, must be odd)
        tau: Threshold for ternary transform (default 0.05)
    """
    
    def __init__(self, patch_size: int = 7, tau: float = 0.05):
        super().__init__()
        assert patch_size % 2 == 1, "Patch size must be odd"
        self.patch_size = patch_size
        self.tau = tau
        self.pad = patch_size // 2
    
    def _rgb_to_gray(self, x: torch.Tensor) -> torch.Tensor:
        """Convert RGB to grayscale."""
        return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    
    def _census_transform(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute ternary census transform.
        
        For each pixel, compare it with all neighbors in the patch.
        Output: (B, patch_size^2 - 1, H, W) ternary values in {-1, 0, 1}
        encoded as soft values via tanh.
        
        Args:
            x: (B, 1, H, W) grayscale image
            
        Returns:
            (B, P, H, W) where P = patch_size^2 - 1
        """
        B, C, H, W = x.shape
        
        # Pad input
        x_pad = F.pad(x, [self.pad] * 4, mode='reflect')
        
        # Extract all neighbor differences
        center = x  # (B, 1, H, W)
        patterns = []
        
        for i in range(self.patch_size):
            for j in range(self.patch_size):
                if i == self.pad and j == self.pad:
                    continue  # Skip center pixel
                neighbor = x_pad[:, :, i:i+H, j:j+W]
                # Soft ternary: tanh((neighbor - center) / tau)
                diff = (neighbor - center) / self.tau
                patterns.append(torch.tanh(diff))
        
        return torch.cat(patterns, dim=1)  # (B, P, H, W)
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute census loss.
        
        Args:
            pred: (B, 3, H, W) predicted image
            target: (B, 3, H, W) target image
            
        Returns:
            Scalar loss value
        """
        # Convert to grayscale
        pred_gray = self._rgb_to_gray(pred)
        target_gray = self._rgb_to_gray(target)
        
        # Compute census transforms
        pred_census = self._census_transform(pred_gray)
        target_census = self._census_transform(target_gray)
        
        # Soft Hamming distance
        diff = pred_census - target_census
        # Use Charbonnier-style distance for smoothness
        dist = torch.sqrt(diff * diff + 1e-6)
        
        return dist.mean()


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
    Combined loss for VFI training (RIFE/IFRNet-inspired).
    
    Combines:
        - Charbonnier reconstruction loss (replaces L1)
        - Census structural loss (illumination-invariant)
        - VGG perceptual loss (optional, disabled by default)
        - Edge-weighted L1 loss
        - GAN loss (added externally in train.py)
    
    Args:
        l1_weight: Weight for Charbonnier loss (named l1 for config compat)
        census_weight: Weight for Census loss (default 1.0, same as IFRNet)
        perceptual_weight: Weight for VGG perceptual loss (0 = disabled)
        edge_weight: Weight for edge-weighted L1 loss
        edge_multiplier: Multiplier for edge pixels in edge loss
    """
    
    def __init__(
        self,
        l1_weight: float = 1.0,
        census_weight: float = 1.0,
        perceptual_weight: float = 0.0,
        edge_weight: float = 0.5,
        edge_multiplier: float = 10.0,
    ):
        super().__init__()
        
        self.l1_weight = l1_weight
        self.census_weight = census_weight
        self.perceptual_weight = perceptual_weight
        self.edge_weight = edge_weight
        
        # Component losses
        self.reconstruction_loss = CharbonnierLoss(eps=1e-3)
        self.census_loss = CensusLoss(patch_size=7) if census_weight > 0 else None
        self.edge_loss = EdgeWeightedL1Loss(edge_weight=edge_multiplier)
        
        # Only load VGG if perceptual loss is actually used (saves ~500MB VRAM)
        if perceptual_weight > 0:
            self.perceptual_loss = PerceptualLoss()
        else:
            self.perceptual_loss = None
    
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
        # Charbonnier reconstruction (replaces L1)
        recon = self.reconstruction_loss(pred, target)
        total = self.l1_weight * recon
        
        # Census structural loss
        census = torch.tensor(0.0, device=pred.device)
        if self.census_loss is not None and self.census_weight > 0:
            census = self.census_loss(pred, target)
            total = total + self.census_weight * census
        
        # Perceptual loss (optional — only computed if weight > 0)
        perceptual = torch.tensor(0.0, device=pred.device)
        if self.perceptual_loss is not None and self.perceptual_weight > 0:
            perceptual = self.perceptual_loss(pred, target)
            total = total + self.perceptual_weight * perceptual
        
        # Edge-weighted L1
        edge = self.edge_loss(pred, target)
        total = total + self.edge_weight * edge
        
        if return_components:
            components = {
                'l1': recon,       # Named 'l1' for logging compatibility
                'census': census,
                'perceptual': perceptual,
                'edge': edge,
                'total': total,
            }
            return total, components
        
        return total
