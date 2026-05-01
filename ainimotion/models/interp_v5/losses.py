"""
V5 Loss Functions.

Simple, focused losses following the implementation plan:
  1. Charbonnier L1 (main reconstruction) - weight 1.0
  2. Edge-weighted L1 (line art focus) - weight 0.5
  3. FFT Amplitude Loss (anti-blur) - weight ramps 0.0 -> 1.0
  4. GAN loss (Phase 3 only) - weight ramps 0.01 -> 0.1

The FFT amplitude loss compares frequency spectra directly.
Missing high frequencies (blur) = high loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips


class CharbonnierLoss(nn.Module):
    """
    Charbonnier L1: sqrt((pred - gt)^2 + eps^2)
    
    More robust than L1 near zero, doesn't have the
    non-differentiable point at zero that L1 has.
    """
    
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
    
    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        diff = pred.float() - gt.float()
        # Cast to fp32 to avoid fp16 underflow where self.eps**2 is 0.0
        return torch.sqrt(diff ** 2 + self.eps ** 2).mean()


class EdgeWeightedL1Loss(nn.Module):
    """
    L1 error weighted by Sobel edge magnitude of ground truth.
    
    Focuses the loss on line art regions where sharpness matters most.
    """
    
    def __init__(self):
        super().__init__()
        # Sobel kernels (registered as buffers so they move to GPU with model)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
    
    def _compute_edges(self, img: torch.Tensor) -> torch.Tensor:
        """Compute Sobel edge magnitude from RGB image."""
        gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
    
    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        # Cast to fp32 to prevent division-by-near-zero inf under fp16
        gt_f = gt.float()
        pred_f = pred.float()
        edge_weight = self._compute_edges(gt_f)
        # Normalize edge weight so mean is ~1.0
        edge_weight = edge_weight / (edge_weight.mean() + 1e-6)
        l1 = (pred_f - gt_f).abs()
        return (l1 * edge_weight).mean()


class FFTAmplitudeLoss(nn.Module):
    """
    FFT Amplitude Loss - anti-blur penalty.
    
    Compares the amplitude spectrum of pred vs GT.
    If pred is blurry (missing high frequencies), the
    high-frequency amplitudes will be lower and loss will be high.
    
    Optionally weights high frequencies more heavily.
    """
    
    def __init__(self, high_freq_weight: float = 1.5):
        super().__init__()
        self.high_freq_weight = high_freq_weight
        self._weight_cache = {}
    
    def _get_frequency_weights(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Create frequency weights that emphasize high frequencies."""
        key = (H, W, str(device))
        if key not in self._weight_cache:
            # Create distance-from-center weights
            fy = torch.fft.fftfreq(H, device=device).view(-1, 1)
            fx = torch.fft.rfftfreq(W, device=device).view(1, -1)
            freq_dist = torch.sqrt(fy ** 2 + fx ** 2)
            
            # Linear weight: 1.0 at DC, high_freq_weight at Nyquist
            max_dist = freq_dist.max()
            weights = 1.0 + (self.high_freq_weight - 1.0) * (freq_dist / max_dist)
            self._weight_cache[key] = weights
        
        return self._weight_cache[key]
    
    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        B, C, H, W = pred.shape
        
        # Force fp32 — rfft2 intermediate values can lose precision under bf16
        pred = pred.float()
        gt = gt.float()
        fft_pred = torch.fft.rfft2(pred, norm='ortho')
        fft_gt = torch.fft.rfft2(gt, norm='ortho')
        
        # Compare amplitudes
        amp_pred = fft_pred.abs()
        amp_gt = fft_gt.abs()
        
        # Frequency weights
        weights = self._get_frequency_weights(H, W, pred.device)
        weights = weights.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W//2+1)
        
        # Weighted L1 on amplitudes
        loss = (weights * (amp_pred - amp_gt).abs()).mean()
        
        return loss


class LPIPSLoss(nn.Module):
    """
    Learned Perceptual Image Patch Similarity (LPIPS).
    """
    def __init__(self):
        super().__init__()
        # Use VGG backbone as it is standard for LPIPS in VFI
        self.lpips = lpips.LPIPS(net='vgg')
        # We don't want to train the VGG network
        for param in self.lpips.parameters():
            param.requires_grad = False
            
    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        # LPIPS expects inputs in [-1, 1]
        # Force FP32 to prevent NaN from VGG BatchNorm under AMP FP16
        with torch.amp.autocast('cuda', enabled=False):
            pred_scaled = pred.float() * 2.0 - 1.0
            gt_scaled = gt.float() * 2.0 - 1.0
            return self.lpips(pred_scaled, gt_scaled).mean()


class CensusLoss(nn.Module):
    """
    Census Loss for structural consistency.
    """
    def __init__(self, patch_size: int = 7):
        super().__init__()
        self.patch_size = patch_size
        
    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        # Cast to fp32 — F.unfold on 256x256 with patch=7 produces (B, 49, 65536)
        # tensors where squared differences easily overflow fp16 max (65504)
        pred_f = pred.float()
        gt_f = gt.float()
        pred_g = 0.299 * pred_f[:, 0:1] + 0.587 * pred_f[:, 1:2] + 0.114 * pred_f[:, 2:3]
        gt_g = 0.299 * gt_f[:, 0:1] + 0.587 * gt_f[:, 1:2] + 0.114 * gt_f[:, 2:3]
        
        pad = self.patch_size // 2
        p_pred = F.unfold(pred_g, kernel_size=self.patch_size, padding=pad)
        p_gt = F.unfold(gt_g, kernel_size=self.patch_size, padding=pad)
        
        center_idx = (self.patch_size * self.patch_size) // 2
        d_pred = p_pred - p_pred[:, center_idx:center_idx+1, :]
        d_gt = p_gt - p_gt[:, center_idx:center_idx+1, :]
        
        return torch.sqrt((d_pred - d_gt) ** 2 + 1e-6).mean()


class V5LossFunction(nn.Module):
    """
    Combined V5 loss with progressive scheduling.
    
    Manages all loss terms and their weights based on current epoch.
    
    Schedule:
      - Epochs 0-100: Charbonnier + Edge L1 only
      - Epochs 100-300: Add FFT amplitude (ramps 0 -> 1.0)
      - Epochs 300-500: FFT at full weight (anti-blur hardening)
      - Epochs 500-800: Add GAN (ramps 0.01 -> 0.1)
    """
    
    def __init__(
        self,
        charb_weight: float = 1.0,
        edge_weight: float = 0.5,
        fft_weight_max: float = 1.0,
        gan_weight_max: float = 0.1,
        high_freq_weight: float = 1.5,
        lpips_weight: float = 0.1,
        census_weight: float = 1.0,
        flow_weight: float = 0.5,
        # Routing auxiliary loss weights (prevent synthesis branch death)
        routing_balance_weight: float = 0.01,
        routing_entropy_weight: float = 0.1,
        branch_aux_weight: float = 0.5,
        # Temporal consistency (V5.1)
        temporal_weight_max: float = 0.0,
        temporal_ramp_epochs: int = 50,
    ):
        super().__init__()
        self.charb_weight = charb_weight
        self.edge_weight = edge_weight
        self.fft_weight_max = fft_weight_max
        self.gan_weight_max = gan_weight_max
        self.lpips_weight_max = lpips_weight
        self.census_weight = census_weight
        self.flow_weight = flow_weight
        self.routing_balance_weight = routing_balance_weight
        self.routing_entropy_weight = routing_entropy_weight
        self.branch_aux_weight = branch_aux_weight
        self.temporal_weight_max = temporal_weight_max
        self.temporal_ramp_epochs = temporal_ramp_epochs

        self.charbonnier = CharbonnierLoss()
        self.edge_l1 = EdgeWeightedL1Loss()
        self.fft_loss = FFTAmplitudeLoss(high_freq_weight=high_freq_weight)
        self.lpips = LPIPSLoss()
        self.census = CensusLoss()

    def get_temporal_weight(self, epoch: int) -> float:
        """Progressive temporal consistency weight ramp-up."""
        if self.temporal_weight_max <= 0:
            return 0.0
        ramp = min(1.0, epoch / max(self.temporal_ramp_epochs, 1))
        return self.temporal_weight_max * ramp
    
    def get_fft_weight(self, epoch: int) -> float:
        """Progressive FFT weight ramp-up."""
        if epoch < 100:
            return 0.0
        elif epoch < 300:
            return self.fft_weight_max * (epoch - 100) / 200.0
        else:
            return self.fft_weight_max
            
    def get_lpips_weight(self, epoch: int) -> float:
        """LPIPS ramps in during Phase 1→2 transition for earlier perceptual learning."""
        if epoch < 100:
            return 0.0
        elif epoch < 300:
            # Progressive ramp 0 → max over epochs 100-300
            return self.lpips_weight_max * (epoch - 100) / 200.0
        return self.lpips_weight_max
    
    def get_gan_weight(self, epoch: int) -> float:
        """Progressive GAN weight ramp-up."""
        if epoch < 500:
            return 0.0
        elif epoch < 550:
            return 0.01  # Warm-up
        elif epoch < 650:
            # Ramp 0.01 -> max
            t = (epoch - 550) / 100.0
            return 0.01 + (self.gan_weight_max - 0.01) * t
        else:
            return self.gan_weight_max
    
    def compute_routing_losses(
        self,
        routing_map: torch.Tensor,
        fg_warp: torch.Tensor,
        fg_synth: torch.Tensor,
        gt: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compute auxiliary losses to prevent routing collapse.
        
        Based on:
          - Switch Transformer (Fedus et al. 2021): load balancing loss
          - Gumbel-Softmax (Jang et al. 2017): entropy regularization
          - RIFE / EMA-VFI: per-branch reconstruction supervision
        
        Args:
            routing_map: (B, 1, H, W) in [0, 1], 0=warp, 1=synth
            fg_warp: (B, 3, H, W) warp branch output
            fg_synth: (B, 3, H, W) synthesis branch output
            gt: (B, 3, H, W) ground truth
            
        Returns:
            Dict with routing auxiliary loss terms
        """
        device = routing_map.device
        
        # 1. Routing balance loss (Switch Transformer style)
        # Penalizes deviation from uniform routing (mean=0.5)
        routing_mean = routing_map.mean()
        l_balance = 2.0 * (routing_mean - 0.5) ** 2
        
        # 2. Routing entropy loss (encourages exploration)
        # Maximized when R=0.5 (uncertain), minimized when R∈{0,1}
        eps = 1e-6
        entropy = -(routing_map * torch.log(routing_map + eps) +
                    (1 - routing_map) * torch.log(1 - routing_map + eps))
        l_entropy = entropy.mean()
        
        # 3. Per-branch reconstruction supervision (RIFE / EMA-VFI style)
        # Forces both branches to independently produce valid frames
        l_warp_aux = self.charbonnier(fg_warp, gt)
        l_synth_aux = self.charbonnier(fg_synth, gt)
        
        # Weighted total
        total_routing = (
            self.routing_balance_weight * l_balance
            - self.routing_entropy_weight * l_entropy  # negative because we MAXIMIZE entropy
            + self.branch_aux_weight * (l_warp_aux + l_synth_aux)
        )
        
        return {
            'routing_total': total_routing,
            'routing_balance': l_balance,
            'routing_entropy': l_entropy,
            'routing_mean': routing_mean,
            'branch_warp_aux': l_warp_aux,
            'branch_synth_aux': l_synth_aux,
        }
    
    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        epoch: int = 0,
        gan_loss: torch.Tensor | None = None,
        routing_map: torch.Tensor | None = None,
        fg_warp: torch.Tensor | None = None,
        fg_synth: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute all losses for current epoch.
        
        Args:
            pred: (B, 3, H, W) predicted frame
            gt: (B, 3, H, W) ground truth frame
            epoch: Current training epoch
            gan_loss: Optional GAN generator loss (from discriminator)
            routing_map: Optional (B, 1, H, W) routing map for auxiliary losses
            fg_warp: Optional (B, 3, H, W) warp branch output for aux supervision
            fg_synth: Optional (B, 3, H, W) synthesis branch output for aux supervision
            
        Returns:
            Dict with all loss terms
        """
        # Always-on losses
        l_charb = self.charbonnier(pred, gt)
        l_edge = self.edge_l1(pred, gt)
        l_census = self.census(pred, gt)
        
        total = self.charb_weight * l_charb + self.edge_weight * l_edge + self.census_weight * l_census
        
        # FFT amplitude loss (ramps in)
        fft_w = self.get_fft_weight(epoch)
        l_fft = self.fft_loss(pred, gt) if fft_w > 0 else torch.tensor(0.0, device=pred.device)
        total = total + fft_w * l_fft
        
        # LPIPS loss (Phase 2+)
        lpips_w = self.get_lpips_weight(epoch)
        l_lpips = self.lpips(pred, gt) if lpips_w > 0 else torch.tensor(0.0, device=pred.device)
        total = total + lpips_w * l_lpips
        
        # GAN loss (Phase 3)
        gan_w = self.get_gan_weight(epoch)
        l_gan = torch.tensor(0.0, device=pred.device)
        if gan_loss is not None and gan_w > 0:
            l_gan = gan_loss
            total = total + gan_w * l_gan
        
        # Routing auxiliary losses (always on — prevents synthesis branch death)
        zero = torch.tensor(0.0, device=pred.device)
        routing_losses = {
            'routing_total': zero, 'routing_balance': zero,
            'routing_entropy': zero, 'routing_mean': zero,
            'branch_warp_aux': zero, 'branch_synth_aux': zero,
        }
        if routing_map is not None and fg_warp is not None and fg_synth is not None:
            routing_losses = self.compute_routing_losses(routing_map, fg_warp, fg_synth, gt)
            total = total + routing_losses['routing_total']
            
        result = {
            'total': total,
            'charbonnier': l_charb,
            'edge_l1': l_edge,
            'census': l_census,
            'fft': l_fft,
            'lpips': l_lpips,
            'gan': l_gan,
            'fft_weight': fft_w,
            'gan_weight': gan_w,
            'lpips_weight': lpips_w,
        }
        result.update(routing_losses)
        return result

    def compute_temporal_consistency(
        self,
        pred_a: torch.Tensor,
        pred_b: torch.Tensor,
        gt_a: torch.Tensor,
        gt_b: torch.Tensor,
        epoch: int = 0,
    ) -> torch.Tensor:
        """
        Temporal consistency loss between consecutive interpolated frames.

        Penalizes the model when the difference between consecutive predictions
        doesn't match the difference between consecutive ground truths.
        This reduces flickering by encouraging smooth frame-to-frame transitions.

        Args:
            pred_a: Model output for pair A (B, 3, H, W)
            pred_b: Model output for pair B (consecutive) (B, 3, H, W)
            gt_a: Ground truth for pair A (B, 3, H, W)
            gt_b: Ground truth for pair B (B, 3, H, W)
            epoch: Current epoch (for weight scheduling)

        Returns:
            Weighted temporal consistency loss scalar.
        """
        tw = self.get_temporal_weight(epoch)
        if tw <= 0:
            return torch.tensor(0.0, device=pred_a.device)

        # The predicted temporal difference should match the ground truth temporal difference
        pred_diff = pred_b - pred_a
        gt_diff = gt_b - gt_a

        # Charbonnier on the difference-of-differences (smooth, robust)
        eps = 1e-6
        temporal_loss = torch.sqrt((pred_diff - gt_diff) ** 2 + eps).mean()

        return tw * temporal_loss


if __name__ == '__main__':
    loss_fn = V5LossFunction()
    
    pred = torch.randn(2, 3, 64, 64).sigmoid()
    gt = torch.randn(2, 3, 64, 64).sigmoid()
    
    # Test at different epochs
    for epoch in [0, 50, 150, 300, 500, 600, 800]:
        out = loss_fn(pred, gt, epoch=epoch)
        print(f"Epoch {epoch:>4d}: total={out['total']:.4f}, "
              f"charb={out['charbonnier']:.4f}, edge={out['edge_l1']:.4f}, "
              f"fft={out['fft']:.4f} (w={out['fft_weight']:.2f}), "
              f"gan={out['gan']:.4f} (w={out['gan_weight']:.3f})")
