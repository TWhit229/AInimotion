"""
V5 Multi-Scale PatchGAN Discriminator.

Sees both full-res patches and 2x downscaled patches to catch
blur at multiple scales. Includes all stability measures from
risk analysis:
  - Spectral normalization on all conv layers
  - R1 gradient penalty
  - Minibatch standard deviation layer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class MinibatchStdDev(nn.Module):
    """
    Minibatch standard deviation layer (from ProGAN).
    
    Appends the mean stddev across features as an extra channel.
    Helps the discriminator detect mode collapse.
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Compute stddev across batch (population std to avoid NaN when B=1)
        std = x.std(dim=0, correction=0, keepdim=True)  # (1, C, H, W)
        # Average across channels and spatial
        mean_std = std.mean().expand(B, 1, H, W)
        return torch.cat([x, mean_std], dim=1)


class PatchDiscriminator(nn.Module):
    """
    PatchGAN discriminator with spectral normalization.
    
    Classifies 70x70 patches as real or fake.
    
    Args:
        in_channels: Input channels (default: 3 for RGB)
        base_channels: Base feature channels (default: 64)
    """
    
    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        C = base_channels
        
        self.layers = nn.Sequential(
            # (3, H, W) -> (C, H/2, W/2)
            spectral_norm(nn.Conv2d(in_channels, C, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (C, H/2, W/2) -> (2C, H/4, W/4)
            spectral_norm(nn.Conv2d(C, C * 2, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (2C, H/4, W/4) -> (4C, H/8, W/8)
            spectral_norm(nn.Conv2d(C * 2, C * 4, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Minibatch stddev
            MinibatchStdDev(),
            
            # (4C+1, H/8, W/8) -> (4C, H/8, W/8)
            spectral_norm(nn.Conv2d(C * 4 + 1, C * 4, 3, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Final prediction: (4C, H/8, W/8) -> (1, H/8, W/8)
            spectral_norm(nn.Conv2d(C * 4, 1, 3, padding=1)),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class MultiScalePatchGAN(nn.Module):
    """
    Multi-scale PatchGAN discriminator.
    
    Operates at two scales:
      1. Full resolution - catches fine-detail blur
      2. 2x downscaled - catches large-scale artifacts
    
    Args:
        base_channels: Base channels per scale (default: 64)
    """
    
    def __init__(self, base_channels: int = 64):
        super().__init__()
        self.d_full = PatchDiscriminator(base_channels=base_channels)
        self.d_half = PatchDiscriminator(base_channels=base_channels)
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W) real or fake image
            
        Returns:
            (full_pred, half_pred): predictions at both scales
        """
        full_pred = self.d_full(x)
        
        x_half = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        half_pred = self.d_half(x_half)
        
        return full_pred, half_pred
    
    def compute_loss(
        self,
        real_preds: tuple[torch.Tensor, torch.Tensor],
        fake_preds: tuple[torch.Tensor, torch.Tensor],
        for_generator: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Compute relativistic average GAN loss.
        
        Args:
            real_preds: Tuple of (full_pred, half_pred) for real images
            fake_preds: Tuple of (full_pred, half_pred) for generated images
            for_generator: If True, compute generator loss; else discriminator loss
            
        Returns:
            Dict with 'loss', 'full_loss', 'half_loss'
        """
        real_full, real_half = real_preds
        fake_full, fake_half = fake_preds
        
        if for_generator:
            # Symmetric RaGAN generator loss:
            # fake should look more real than real, AND real should look more fake
            loss_full = (
                F.binary_cross_entropy_with_logits(
                    fake_full - real_full.mean(), torch.ones_like(fake_full)
                ) +
                F.binary_cross_entropy_with_logits(
                    real_full - fake_full.mean(), torch.zeros_like(real_full)
                )
            ) / 2
            loss_half = (
                F.binary_cross_entropy_with_logits(
                    fake_half - real_half.mean(), torch.ones_like(fake_half)
                ) +
                F.binary_cross_entropy_with_logits(
                    real_half - fake_half.mean(), torch.zeros_like(real_half)
                )
            ) / 2
        else:
            # Discriminator: real > fake
            loss_full = (
                F.binary_cross_entropy_with_logits(
                    real_full - fake_full.mean(), torch.ones_like(real_full)
                ) +
                F.binary_cross_entropy_with_logits(
                    fake_full - real_full.mean(), torch.zeros_like(fake_full)
                )
            ) / 2
            loss_half = (
                F.binary_cross_entropy_with_logits(
                    real_half - fake_half.mean(), torch.ones_like(real_half)
                ) +
                F.binary_cross_entropy_with_logits(
                    fake_half - real_half.mean(), torch.zeros_like(fake_half)
                )
            ) / 2
        
        loss = (loss_full + loss_half) / 2
        
        return {
            'loss': loss,
            'full_loss': loss_full,
            'half_loss': loss_half,
        }


def r1_gradient_penalty(
    discriminator: MultiScalePatchGAN,
    real: torch.Tensor,
    weight: float = 10.0,
) -> torch.Tensor:
    """
    R1 gradient penalty for discriminator regularization.
    
    Penalizes the gradient norm of discriminator output w.r.t. real images.
    This prevents the discriminator from becoming too sharp
    and destabilizing training.
    
    Args:
        discriminator: The discriminator module
        real: (B, 3, H, W) real images (requires grad)
        weight: Penalty weight (default: 10.0)
        
    Returns:
        R1 penalty scalar
    """
    # Force fp32 for numerical stability — second-order gradients in bf16
    # can overflow when squared gradient norms are summed over all pixels
    with torch.amp.autocast('cuda', enabled=False):
        real = real.detach().float().requires_grad_(True)
        pred_full, pred_half = discriminator(real)

        # Gradient w.r.t. real for both scales
        grad_full = torch.autograd.grad(
            outputs=pred_full.sum(), inputs=real,
            create_graph=True, retain_graph=True,
        )[0]
        grad_half = torch.autograd.grad(
            outputs=pred_half.sum(), inputs=real,
            create_graph=True,
        )[0]

        penalty = (
            grad_full.view(grad_full.size(0), -1).pow(2).sum(1).mean()
            + grad_half.view(grad_half.size(0), -1).pow(2).sum(1).mean()
        ) / 2

    return weight * penalty


if __name__ == '__main__':
    disc = MultiScalePatchGAN(base_channels=64)
    
    real = torch.randn(2, 3, 256, 256).sigmoid()
    fake = torch.randn(2, 3, 256, 256).sigmoid()
    
    real_preds = disc(real)
    fake_preds = disc(fake)
    
    # Discriminator loss
    d_loss = disc.compute_loss(real_preds, fake_preds, for_generator=False)
    print(f"D loss: {d_loss['loss']:.4f} (full={d_loss['full_loss']:.4f}, half={d_loss['half_loss']:.4f})")
    
    # Generator loss
    g_loss = disc.compute_loss(real_preds, fake_preds, for_generator=True)
    print(f"G loss: {g_loss['loss']:.4f}")
    
    # R1 penalty
    r1 = r1_gradient_penalty(disc, real)
    print(f"R1 penalty: {r1:.4f}")
    
    params = sum(p.numel() for p in disc.parameters())
    print(f"Discriminator parameters: {params:,}")
