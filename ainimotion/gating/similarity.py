"""
Frame similarity computation for gating decisions.

Provides SSIM and MAD (Mean Absolute Difference) metrics
for comparing consecutive frames.
"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from pathlib import Path


def _load_frame(frame: str | Path | torch.Tensor | np.ndarray) -> torch.Tensor:
    """
    Load a frame from various formats to torch tensor.
    
    Args:
        frame: Path, PIL Image, numpy array, or torch tensor
        
    Returns:
        Tensor of shape (C, H, W) in [0, 1] range
    """
    if isinstance(frame, torch.Tensor):
        if frame.dim() == 4:
            frame = frame.squeeze(0)
        return frame.float() / 255.0 if frame.max() > 1.0 else frame.float()
    
    if isinstance(frame, (str, Path)):
        frame = np.array(Image.open(frame).convert("RGB"))
    
    if isinstance(frame, np.ndarray):
        # HWC -> CHW and normalize
        frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
    
    return frame


def _to_grayscale(frame: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB tensor to grayscale using luminance weights.
    
    Args:
        frame: (C, H, W) or (B, C, H, W) tensor
        
    Returns:
        (1, H, W) or (B, 1, H, W) grayscale tensor
    """
    # ITU-R BT.601 luma coefficients
    weights = torch.tensor([0.299, 0.587, 0.114], device=frame.device)
    
    if frame.dim() == 3:
        return (frame * weights.view(3, 1, 1)).sum(dim=0, keepdim=True)
    else:
        return (frame * weights.view(1, 3, 1, 1)).sum(dim=1, keepdim=True)


def compute_mad(
    frame1: str | Path | torch.Tensor,
    frame2: str | Path | torch.Tensor,
    use_luma: bool = True,
    downscale: int = 4,
) -> float:
    """
    Compute Mean Absolute Difference between two frames.
    
    Lower MAD = more similar frames.
    MAD ≈ 0: Identical or near-identical (HOLD)
    MAD > 0.1: Significant motion
    MAD > 0.3: Likely scene cut
    
    Args:
        frame1: First frame
        frame2: Second frame
        use_luma: Convert to grayscale for speed
        downscale: Downscale factor for speed (4 = 1/4 resolution)
        
    Returns:
        MAD value in [0, 1] range
    """
    f1 = _load_frame(frame1)
    f2 = _load_frame(frame2)
    
    # Downscale for speed
    if downscale > 1:
        f1 = F.interpolate(f1.unsqueeze(0), scale_factor=1/downscale, mode="bilinear", align_corners=False).squeeze(0)
        f2 = F.interpolate(f2.unsqueeze(0), scale_factor=1/downscale, mode="bilinear", align_corners=False).squeeze(0)
    
    # Convert to grayscale if requested
    if use_luma:
        f1 = _to_grayscale(f1)
        f2 = _to_grayscale(f2)
    
    # Compute MAD
    mad = torch.abs(f1 - f2).mean().item()
    
    return mad


def compute_ssim(
    frame1: str | Path | torch.Tensor,
    frame2: str | Path | torch.Tensor,
    window_size: int = 11,
    downscale: int = 4,
) -> float:
    """
    Compute Structural Similarity Index between two frames.
    
    Higher SSIM = more similar frames.
    SSIM ≈ 1.0: Identical or near-identical (HOLD)
    SSIM < 0.95: Some motion
    SSIM < 0.5: Major change or scene cut
    
    Args:
        frame1: First frame
        frame2: Second frame
        window_size: Gaussian window size (odd number)
        downscale: Downscale factor for speed
        
    Returns:
        SSIM value in [-1, 1] range (typically 0-1 for images)
    """
    f1 = _load_frame(frame1)
    f2 = _load_frame(frame2)
    
    # Ensure same device
    if f1.device != f2.device:
        f2 = f2.to(f1.device)
    
    # Add batch dimension
    if f1.dim() == 3:
        f1 = f1.unsqueeze(0)
        f2 = f2.unsqueeze(0)
    
    # Downscale for speed
    if downscale > 1:
        f1 = F.interpolate(f1, scale_factor=1/downscale, mode="bilinear", align_corners=False)
        f2 = F.interpolate(f2, scale_factor=1/downscale, mode="bilinear", align_corners=False)
    
    # Convert to grayscale
    f1 = _to_grayscale(f1)
    f2 = _to_grayscale(f2)
    
    # Create Gaussian window
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32, device=f1.device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = g.unsqueeze(0) * g.unsqueeze(1)
    window = window.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    
    # SSIM constants
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # Compute local means
    mu1 = F.conv2d(f1, window, padding=window_size // 2)
    mu2 = F.conv2d(f2, window, padding=window_size // 2)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2
    
    # Compute local variances and covariance
    sigma1_sq = F.conv2d(f1 ** 2, window, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.conv2d(f2 ** 2, window, padding=window_size // 2) - mu2_sq
    sigma12 = F.conv2d(f1 * f2, window, padding=window_size // 2) - mu12
    
    # SSIM formula
    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean().item()


def compute_similarity(
    frame1: str | Path | torch.Tensor,
    frame2: str | Path | torch.Tensor,
    method: str = "ssim",
    **kwargs,
) -> float:
    """
    Compute frame similarity using specified method.
    
    Args:
        frame1: First frame
        frame2: Second frame
        method: 'ssim' or 'mad'
        **kwargs: Additional arguments for the method
        
    Returns:
        Similarity score (higher = more similar for both methods when normalized)
    """
    if method == "ssim":
        return compute_ssim(frame1, frame2, **kwargs)
    elif method == "mad":
        # Invert MAD so higher = more similar
        return 1.0 - compute_mad(frame1, frame2, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'ssim' or 'mad'")
