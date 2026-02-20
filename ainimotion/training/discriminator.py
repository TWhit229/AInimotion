"""
PatchGAN Discriminator for adversarial training.

Outputs a probability map indicating real/fake for local patches.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv-InstanceNorm-LeakyReLU block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        use_norm: bool = True,
    ):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        ]
        if use_norm:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PatchDiscriminator(nn.Module):
    """
    PatchGAN discriminator.
    
    Outputs a spatial map of real/fake probabilities, where each
    value corresponds to a receptive field (patch) of the input.
    
    Args:
        in_channels: Input image channels (default 3 for RGB)
        base_channels: Base number of filters (doubled each layer)
        num_layers: Number of downsampling layers
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_layers: int = 3,
    ):
        super().__init__()
        
        layers = []
        
        # First layer (no normalization)
        layers.append(ConvBlock(
            in_channels, base_channels,
            use_norm=False,
        ))
        
        # Intermediate layers
        nf = base_channels
        for i in range(1, num_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            layers.append(ConvBlock(nf_prev, nf))
        
        # Penultimate layer (stride 1)
        nf_prev = nf
        nf = min(nf * 2, 512)
        layers.append(ConvBlock(nf_prev, nf, stride=1))
        
        # Output layer
        layers.append(nn.Conv2d(nf, 1, kernel_size=4, stride=1, padding=1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute patch-wise real/fake predictions.
        
        Args:
            x: (B, C, H, W) input image
            
        Returns:
            (B, 1, H', W') probability map
        """
        return self.model(x)


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale PatchGAN discriminator.
    
    Uses multiple discriminators at different scales for better
    stability and coverage of different feature scales.
    
    Args:
        in_channels: Input channels
        base_channels: Base filters
        num_discriminators: Number of scales (default 2)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_discriminators: int = 2,
    ):
        super().__init__()
        
        self.discriminators = nn.ModuleList([
            PatchDiscriminator(in_channels, base_channels)
            for _ in range(num_discriminators)
        ])
        
        # Downsample for multi-scale
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1)
    
    def forward(
        self, 
        x: torch.Tensor
    ) -> list[torch.Tensor]:
        """
        Compute predictions at multiple scales.
        
        Args:
            x: (B, C, H, W) input image
            
        Returns:
            List of prediction maps at each scale
        """
        outputs = []
        for i, disc in enumerate(self.discriminators):
            if i > 0:
                x = self.downsample(x)
            outputs.append(disc(x))
        return outputs


# GAN loss functions

def gan_loss_vanilla(
    pred: torch.Tensor,
    is_real: bool,
) -> torch.Tensor:
    """
    Vanilla GAN loss (BCE).
    
    Args:
        pred: Discriminator output
        is_real: Whether target should be real (1) or fake (0)
        
    Returns:
        Scalar loss
    """
    target = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
    return nn.functional.binary_cross_entropy_with_logits(pred, target)


def gan_loss_lsgan(
    pred: torch.Tensor,
    is_real: bool,
) -> torch.Tensor:
    """
    LSGAN loss (MSE, more stable gradients).
    
    Args:
        pred: Discriminator output  
        is_real: Whether target should be real (1) or fake (0)
        
    Returns:
        Scalar loss
    """
    target = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
    return nn.functional.mse_loss(pred, target)


def gan_loss_hinge(
    pred: torch.Tensor,
    is_real: bool,
    for_discriminator: bool = True,
) -> torch.Tensor:
    """
    Hinge GAN loss (used in SAGAN, BigGAN).
    
    Args:
        pred: Discriminator output
        is_real: Whether input is real or fake
        for_discriminator: Whether computing D loss or G loss
        
    Returns:
        Scalar loss
    """
    if for_discriminator:
        if is_real:
            return torch.relu(1.0 - pred).mean()
        else:
            return torch.relu(1.0 + pred).mean()
    else:
        # Generator always wants discriminator to think it's real
        return -pred.mean()


# ── Zero-centered gradient penalties (Mescheder et al, 2018) ──

def compute_r1_penalty(
    discriminator: nn.Module,
    real_images: torch.Tensor,
) -> torch.Tensor:
    """
    R1 zero-centered gradient penalty on real data.
    
    Penalizes ||∇_x D(x_real)||² to keep D's gradients near zero
    on the real data manifold, preventing overconfident discrimination.
    
    Args:
        discriminator: The discriminator module
        real_images: (B, C, H, W) real images (will require grad)
        
    Returns:
        Scalar mean gradient penalty
    """
    real_images = real_images.detach().requires_grad_(True)
    pred_real = discriminator(real_images)
    
    # Handle multi-scale output
    if isinstance(pred_real, list):
        pred_sum = sum(p.sum() for p in pred_real)
    else:
        pred_sum = pred_real.sum()
    
    grad_real, = torch.autograd.grad(
        outputs=pred_sum,
        inputs=real_images,
        create_graph=True,
    )
    return grad_real.pow(2).reshape(grad_real.size(0), -1).sum(1).mean()


def compute_r2_penalty(
    discriminator: nn.Module,
    fake_images: torch.Tensor,
) -> torch.Tensor:
    """
    R2 zero-centered gradient penalty on fake (generated) data.
    
    Penalizes ||∇_x D(x_fake)||² to keep D's gradients near zero
    on the generated data manifold. Together with R1, this guarantees
    local convergence of GAN training.
    
    Args:
        discriminator: The discriminator module
        fake_images: (B, C, H, W) generated images (will require grad)
        
    Returns:
        Scalar mean gradient penalty
    """
    fake_images = fake_images.detach().requires_grad_(True)
    pred_fake = discriminator(fake_images)
    
    if isinstance(pred_fake, list):
        pred_sum = sum(p.sum() for p in pred_fake)
    else:
        pred_sum = pred_fake.sum()
    
    grad_fake, = torch.autograd.grad(
        outputs=pred_sum,
        inputs=fake_images,
        create_graph=True,
    )
    return grad_fake.pow(2).reshape(grad_fake.size(0), -1).sum(1).mean()


# ── Relativistic pairing GAN loss (RpGAN, Jolicoeur-Martineau 2019) ──

def gan_loss_relativistic(
    pred_real: torch.Tensor,
    pred_fake: torch.Tensor,
    for_discriminator: bool = True,
) -> torch.Tensor:
    """
    Relativistic pairing GAN loss (RpGAN).
    
    Instead of scoring real/fake independently, evaluates D(real) - D(fake)
    as a coupled pair. This prevents mode collapse by maintaining decision
    boundaries per real sample rather than a single global boundary.
    
    Args:
        pred_real: D output on real images
        pred_fake: D output on fake images
        for_discriminator: Whether computing D or G loss
        
    Returns:
        Scalar loss
    """
    if for_discriminator:
        # D wants D(real) - D(fake) to be positive (high)
        return nn.functional.binary_cross_entropy_with_logits(
            pred_real - pred_fake, torch.ones_like(pred_real)
        )
    else:
        # G wants D(fake) - D(real) to be positive (high)
        return nn.functional.binary_cross_entropy_with_logits(
            pred_fake - pred_real, torch.ones_like(pred_fake)
        )


class GANLoss(nn.Module):
    """
    Configurable GAN loss with label smoothing and gradient penalties.
    
    Supports four loss types:
        - 'vanilla': BCE with logits
        - 'lsgan': MSE (least-squares GAN)
        - 'hinge': Hinge loss (SAGAN/BigGAN style)
        - 'relativistic': RpGAN with BCE (R3GAN style)
    
    Args:
        loss_type: One of 'vanilla', 'lsgan', 'hinge', 'relativistic'
        label_smoothing: Amount to smooth labels (default 0.0, try 0.1)
            - Real labels become 1.0 - smoothing (e.g., 0.9)
            - Fake labels become 0.0 + smoothing (e.g., 0.1)
    """
    
    def __init__(self, loss_type: str = 'lsgan', label_smoothing: float = 0.0):
        super().__init__()
        self.loss_type = loss_type
        self.label_smoothing = label_smoothing
        
        if loss_type not in ('vanilla', 'lsgan', 'hinge', 'relativistic'):
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    @property
    def is_relativistic(self) -> bool:
        return self.loss_type == 'relativistic'
    
    def _get_target(self, pred: torch.Tensor, is_real: bool) -> torch.Tensor:
        """Get target tensor with optional label smoothing."""
        if is_real:
            target_val = 1.0 - self.label_smoothing
        else:
            target_val = self.label_smoothing
        return torch.full_like(pred, target_val)
    
    def _compute_loss(
        self, 
        pred: torch.Tensor, 
        is_real: bool, 
        for_discriminator: bool
    ) -> torch.Tensor:
        """Compute loss for a single prediction tensor (non-relativistic)."""
        if self.loss_type == 'vanilla':
            target = self._get_target(pred, is_real)
            return nn.functional.binary_cross_entropy_with_logits(pred, target)
        
        elif self.loss_type == 'lsgan':
            target = self._get_target(pred, is_real)
            return nn.functional.mse_loss(pred, target)
        
        else:  # hinge
            if for_discriminator:
                if is_real:
                    return torch.relu(1.0 - pred).mean()
                else:
                    return torch.relu(1.0 + pred).mean()
            else:
                return -pred.mean()
    
    def _compute_relativistic_loss(
        self,
        pred_real: torch.Tensor,
        pred_fake: torch.Tensor,
        for_discriminator: bool,
    ) -> torch.Tensor:
        """Compute relativistic pairing loss for a single scale."""
        return gan_loss_relativistic(pred_real, pred_fake, for_discriminator)
    
    def forward(
        self,
        pred: torch.Tensor,
        is_real: bool = True,
        for_discriminator: bool = True,
        pred_other: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute GAN loss.
        
        For non-relativistic losses:
            pred: D output, is_real: target label
        For relativistic loss:
            pred: D(real), pred_other: D(fake)  [or vice versa]
            is_real is ignored; for_discriminator determines sign.
        
        Args:
            pred: Discriminator output (or list for multi-scale)
            is_real: Target label (ignored for relativistic)
            for_discriminator: Whether this is D or G loss
            pred_other: Required for relativistic — the "other" prediction
            
        Returns:
            Scalar loss
        """
        if self.is_relativistic:
            if pred_other is None:
                raise ValueError(
                    "Relativistic loss requires pred_other "
                    "(pass both real and fake predictions)"
                )
            # For relativistic: pred = D(real), pred_other = D(fake) by convention
            if isinstance(pred, list):
                loss = 0.0
                for p_real, p_fake in zip(pred, pred_other):
                    loss += self._compute_relativistic_loss(
                        p_real, p_fake, for_discriminator
                    )
                return loss / len(pred)
            else:
                return self._compute_relativistic_loss(
                    pred, pred_other, for_discriminator
                )
        else:
            if isinstance(pred, list):
                loss = 0.0
                for p in pred:
                    loss += self._compute_loss(p, is_real, for_discriminator)
                return loss / len(pred)
            else:
                return self._compute_loss(pred, is_real, for_discriminator)

