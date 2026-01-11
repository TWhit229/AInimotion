"""
AdaCoF (Adaptive Collaboration of Flows) for foreground/character motion.

Predicts per-pixel K×K sampling kernels for deformable motion estimation,
handling non-linear animation motions like squash and stretch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaCoFKernelPredictor(nn.Module):
    """
    Predicts adaptive kernel offsets and weights for each pixel.
    
    For each pixel, outputs:
        - K×K offset pairs (dx, dy) for sampling locations
        - K×K weights for combining samples
    
    Args:
        in_channels: Input feature channels
        kernel_size: Size of adaptive kernel (K×K)
    """
    
    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 7,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.k_squared = kernel_size * kernel_size
        
        # Network to predict offsets and weights
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        
        # Output heads
        # Offsets: 2 * K^2 (dx, dy for each kernel position)
        self.offset_head = nn.Conv2d(128, 2 * self.k_squared, 3, padding=1)
        
        # Weights: K^2 (one weight per kernel position)
        self.weight_head = nn.Conv2d(128, self.k_squared, 3, padding=1)
        
        # Initialize offsets to default grid pattern
        self._init_offsets()
    
    def _init_offsets(self):
        """Initialize offsets to regular grid pattern."""
        nn.init.zeros_(self.offset_head.weight)
        
        # Create default grid offsets
        k = self.kernel_size
        half = k // 2
        default_offsets = []
        for i in range(k):
            for j in range(k):
                dx = j - half
                dy = i - half
                default_offsets.extend([dx, dy])
        
        # Set bias to default grid
        self.offset_head.bias.data = torch.tensor(
            default_offsets, dtype=torch.float32
        )
        
        # Initialize weights to uniform
        nn.init.zeros_(self.weight_head.weight)
        nn.init.constant_(self.weight_head.bias, 1.0 / self.k_squared)
    
    def forward(
        self, 
        features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict kernel offsets and weights.
        
        Args:
            features: (B, C, H, W) input features
            
        Returns:
            offsets: (B, 2*K^2, H, W) sampling offsets
            weights: (B, K^2, H, W) combination weights (softmax normalized)
        """
        feat = self.conv(features)
        
        offsets = self.offset_head(feat)
        weights = self.weight_head(feat)
        
        # Normalize weights with softmax
        weights = F.softmax(weights, dim=1)
        
        return offsets, weights


class AdaCoFSampler(nn.Module):
    """
    Samples pixels using AdaCoF kernels.
    
    For each output pixel, samples K×K locations from input
    and combines them using predicted weights.
    """
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.kernel_size = kernel_size
        self.k_squared = kernel_size * kernel_size
    
    def forward(
        self,
        image: torch.Tensor,
        offsets: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample and combine pixels using adaptive kernels.
        
        Args:
            image: (B, C, H, W) input image to sample from
            offsets: (B, 2*K^2, H, W) sampling offsets
            weights: (B, K^2, H, W) combination weights
            
        Returns:
            (B, C, H, W) sampled output
        """
        b, c, h, w = image.shape
        device = image.device
        
        # Create base grid
        y = torch.arange(h, device=device, dtype=torch.float32)
        x = torch.arange(w, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        grid_x = grid_x.expand(b, 1, h, w)
        grid_y = grid_y.expand(b, 1, h, w)
        
        # Sample at each kernel position
        output = torch.zeros(b, c, h, w, device=device)
        
        for i in range(self.k_squared):
            # Get offset for this kernel position
            dx = offsets[:, 2*i:2*i+1]    # (B, 1, H, W)
            dy = offsets[:, 2*i+1:2*i+2]  # (B, 1, H, W)
            
            # Get weight for this kernel position
            w_i = weights[:, i:i+1]  # (B, 1, H, W)
            
            # Compute sampling locations
            sample_x = grid_x + dx
            sample_y = grid_y + dy
            
            # Normalize to [-1, 1] for grid_sample
            sample_x_norm = 2 * sample_x / (w - 1) - 1
            sample_y_norm = 2 * sample_y / (h - 1) - 1
            
            # Stack and reshape for grid_sample
            grid_sample = torch.stack(
                [sample_x_norm, sample_y_norm], dim=-1
            ).squeeze(1)  # (B, H, W, 2)
            
            # Sample
            sampled = F.grid_sample(
                image,
                grid_sample,
                mode='bilinear',
                padding_mode='border',
                align_corners=False,
            )  # (B, C, H, W)
            
            # Accumulate weighted sample
            output = output + w_i * sampled
        
        return output


class AdaCoFNet(nn.Module):
    """
    AdaCoF network for deformable foreground motion.
    
    Predicts per-pixel adaptive kernels for sampling from both
    input frames, handling squash/stretch and non-linear motion.
    
    Args:
        feat_channels: Feature channels at each scale
        kernel_size: AdaCoF kernel size (default K=7)
    """
    
    def __init__(
        self,
        feat_channels: list[int] = [32, 64, 128, 128],
        kernel_size: int = 7,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        
        # Use finest features for per-pixel prediction
        # Concatenate feat1, feat2, and correlation
        in_channels = feat_channels[0] * 2 + 81  # 81 = corr channels
        
        # Kernel predictor for sampling from frame1
        self.kernel_pred_1 = AdaCoFKernelPredictor(in_channels, kernel_size)
        
        # Kernel predictor for sampling from frame2 
        self.kernel_pred_2 = AdaCoFKernelPredictor(in_channels, kernel_size)
        
        # Sampler
        self.sampler = AdaCoFSampler(kernel_size)
        
        # Blend weight predictor
        self.blend_head = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor,
        feat1: list[torch.Tensor],
        feat2: list[torch.Tensor],
        corr: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Synthesize foreground using AdaCoF.
        
        Args:
            frame1: (B, C, H, W) first input frame
            frame2: (B, C, H, W) second input frame
            feat1: Features from frame1
            feat2: Features from frame2
            corr: Correlation volumes
            
        Returns:
            Dictionary with:
                - 'output': (B, C, H, W) synthesized foreground
                - 'blend': (B, 1, H, W) blend weight between frame1/frame2 samples
        """
        # Get finest scale features
        f1 = feat1[0]
        f2 = feat2[0]
        c = corr[0]
        
        # Upsample features to input resolution if needed
        h, w = frame1.shape[2:]
        if f1.shape[2] != h or f1.shape[3] != w:
            f1 = F.interpolate(f1, size=(h, w), mode='bilinear', align_corners=False)
            f2 = F.interpolate(f2, size=(h, w), mode='bilinear', align_corners=False)
            c = F.interpolate(c, size=(h, w), mode='bilinear', align_corners=False)
        
        # Concatenate features
        combined = torch.cat([f1, f2, c], dim=1)
        
        # Predict kernels for both frames
        offsets1, weights1 = self.kernel_pred_1(combined)
        offsets2, weights2 = self.kernel_pred_2(combined)
        
        # Sample from both frames
        sample1 = self.sampler(frame1, offsets1, weights1)
        sample2 = self.sampler(frame2, offsets2, weights2)
        
        # Predict blend weight
        blend = self.blend_head(combined)
        
        # Combine samples
        output = blend * sample1 + (1 - blend) * sample2
        
        return {
            'output': output,
            'blend': blend,
            'offsets1': offsets1,
            'offsets2': offsets2,
            'weights1': weights1,
            'weights2': weights2,
        }
