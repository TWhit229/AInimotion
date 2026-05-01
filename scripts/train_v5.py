"""
V5 Training Script — Multi-Frame Anime Interpolation.

3-Phase Progressive Training:
  Phase 1 (epochs 0-300):   Charbonnier + Edge L1 + progressive FFT
  Phase 2 (epochs 300-500): FFT at full weight (anti-blur hardening)
  Phase 3 (epochs 500-800): GAN with progressive ramp-up

Usage:
  python scripts/train_v5.py --config configs/train_v5_5090.yaml
  python scripts/train_v5.py --config configs/train_v5_5090.yaml --resume checkpoints/v5/latest.pt
"""

import argparse
import os
import sys
import time
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import autocast
from torch.optim.swa_utils import AveragedModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ainimotion.models.interp_v5 import LayeredInterpolatorV5, build_model
from ainimotion.models.interp_v5.losses import V5LossFunction
from ainimotion.models.interp_v5.discriminator import MultiScalePatchGAN, r1_gradient_penalty
from ainimotion.data.dataset_v5 import V5SequenceDataset


def load_config(path: str) -> dict:
    """Load YAML config file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Compute PSNR in dB."""
    mse = F.mse_loss(pred.clamp(0, 1), gt.clamp(0, 1))
    if mse < 1e-10:
        return 50.0
    return -10.0 * torch.log10(mse).item()


def ssim(pred: torch.Tensor, gt: torch.Tensor, window_size: int = 11) -> float:
    """Compute structural similarity index (simplified)."""
    # SSIM approximation for logging
    C1 = (0.01 * 1.0) ** 2
    C2 = (0.03 * 1.0) ** 2
    mu1 = F.avg_pool2d(pred, window_size, 1, window_size//2)
    mu2 = F.avg_pool2d(gt, window_size, 1, window_size//2)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.avg_pool2d(pred * pred, window_size, 1, window_size//2) - mu1_sq
    sigma2_sq = F.avg_pool2d(gt * gt, window_size, 1, window_size//2) - mu2_sq
    sigma12 = F.avg_pool2d(pred * gt, window_size, 1, window_size//2) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()


class Trainer:
    """
    V5 Training Manager.
    
    Handles the full 3-phase training loop with:
      - Mixed precision (AMP)
      - Gradient accumulation
      - Checkpoint save/resume
      - W&B logging
      - OOM recovery
      - Phase-aware discriminator management
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Build model
        print("Building V5 model...")
        self.model = build_model(config.get('model')).to(self.device)
        params = sum(p.numel() for p in self.model.parameters())
        print(f"  Generator: {params:,} parameters")
        
        # Setup EMA BEFORE torch.compile to avoid state dict key mismatches
        def get_ema(averaged_model_parameter, model_parameter, num_averaged):
            return 0.999 * averaged_model_parameter + 0.001 * model_parameter
        self.ema_model = AveragedModel(self.model, avg_fn=get_ema)
        
        # Compile after EMA creation
        # Default mode uses Triton kernel fusion for throughput on Linux
        self.model = torch.compile(self.model)
        
        # Set total_epochs before discriminator (needed for scheduler T_max)
        train_cfg = config.get('training', {})
        self.total_epochs = train_cfg.get('total_epochs', 800)

        # Build discriminator (used in Phase 3)
        self.discriminator, self.optimizer_d, self.scheduler_d = self._create_discriminator()
        disc_params = sum(p.numel() for p in self.discriminator.parameters())
        print(f"  Discriminator: {disc_params:,} parameters")

        # Loss function
        loss_cfg = config.get('loss', {})
        self.loss_fn = V5LossFunction(
            charb_weight=loss_cfg.get('charbonnier_weight', 1.0),
            edge_weight=loss_cfg.get('edge_weight', 0.5),
            fft_weight_max=loss_cfg.get('fft_weight_max', 1.0),
            gan_weight_max=loss_cfg.get('gan_weight_max', 0.1),
            high_freq_weight=loss_cfg.get('high_freq_weight', 1.5),
            temporal_weight_max=loss_cfg.get('temporal_weight_max', 0.0),
            temporal_ramp_epochs=loss_cfg.get('temporal_ramp_epochs', 50),
            warp_loss_weight=loss_cfg.get('warp_loss_weight', 0.0),
        ).to(self.device)

        # Optimizers (train_cfg already set above)
        self.optimizer_g = torch.optim.AdamW(
            self.model.parameters(),
            lr=train_cfg.get('learning_rate', 3e-4),
            weight_decay=train_cfg.get('weight_decay', 1e-4),
            betas=tuple(train_cfg.get('betas', [0.9, 0.999])),
        )

        # LR scheduler
        total_epochs = self.total_epochs
        warmup = train_cfg.get('warmup_epochs', 10)
        min_lr = train_cfg.get('min_lr', 1e-6)

        self.scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_g, T_max=total_epochs - warmup, eta_min=min_lr,
        )
        
        # AMP — bfloat16 does NOT need GradScaler (same exponent range as fp32)
        # Leaving scaler enabled with bf16 causes it to double forever → overflow
        self.use_amp = train_cfg.get('amp', True)
        self.amp_dtype = torch.bfloat16  # bf16 is safer than fp16, no scaler needed
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_psnr = 0.0
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup
        self.grad_clip = train_cfg.get('grad_clip', 1.0)
        self.accum_steps = train_cfg.get('gradient_accumulation_steps', 1)
        self.save_every = train_cfg.get('save_every_n_epochs', 10)
        self.keep_last_n = train_cfg.get('keep_last_n_checkpoints', 5)
        self.val_every = train_cfg.get('val_every_n_epochs', 5)
        
        # GAN config
        disc_cfg = config.get('discriminator', {})
        self.r1_weight = disc_cfg.get('r1_weight', 10.0)
        self.r1_every = disc_cfg.get('r1_every_n_steps', 16)
        
        # Logging config
        log_cfg = config.get('logging', {})
        self.log_every = log_cfg.get('log_every_n_steps', 50)
        self.sample_every = log_cfg.get('sample_every_n_epochs', 5)
        self.use_wandb = log_cfg.get('wandb', False)
        self.wandb_run = None
        
        # Checkpoint dir
        self.ckpt_dir = os.path.join('checkpoints', 'v5')
        os.makedirs(self.ckpt_dir, exist_ok=True)
    
    @staticmethod
    def _clean_state_dict(state_dict):
        """Strip '_orig_mod.' prefix from torch.compile state dicts."""
        return {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    
    def _init_wandb(self):
        """Initialize W&B logging."""
        if not self.use_wandb:
            return
        try:
            import wandb
            log_cfg = self.config.get('logging', {})
            self.wandb_run = wandb.init(
                project=log_cfg.get('wandb_project', 'ainimotion-v5'),
                config=self.config,
                resume='allow',
            )
            print(f"  W&B initialized: {wandb.run.url}")
        except ImportError:
            print("  W&B not installed, skipping")
            self.use_wandb = False
    
    def _log(self, metrics: dict, step: int | None = None):
        """Log metrics to W&B and console."""
        if self.wandb_run:
            import wandb
            wandb.log(metrics, step=step or self.global_step)
    
    @property
    def current_phase(self) -> str:
        """Current training phase based on epoch."""
        if self.epoch < 300:
            return "Phase 1: Reconstruction"
        elif self.epoch < 500:
            return "Phase 2: Anti-Blur"
        else:
            return "Phase 3: GAN"
    
    @property
    def gan_active(self) -> bool:
        """Whether GAN training is active."""
        return self.loss_fn.get_gan_weight(self.epoch) > 0

    def _create_discriminator(self):
        """Create discriminator, optimizer, and scheduler from config."""
        disc_cfg = self.config.get('discriminator', {})
        min_lr = self.config.get('training', {}).get('min_lr', 1e-6)

        discriminator = MultiScalePatchGAN(
            base_channels=disc_cfg.get('base_channels', 64)
        ).to(self.device)
        optimizer = torch.optim.AdamW(
            discriminator.parameters(),
            lr=disc_cfg.get('lr', 1e-4),
            weight_decay=disc_cfg.get('weight_decay', 1e-4),
        )
        # D only trains during Phase 3 (epochs 500-800 = 300 epochs)
        gan_phase_epochs = self.total_epochs - 500
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=gan_phase_epochs, eta_min=min_lr / 10,
        )
        return discriminator, optimizer, scheduler

    def _check_disc_nan(self) -> bool:
        """Check if any discriminator parameter or buffer contains NaN/Inf.

        Must check buffers too: spectral norm stores weight_u/weight_v as
        buffers, and these diverge first when power iteration goes unstable.
        """
        for p in self.discriminator.parameters():
            if torch.isnan(p).any() or torch.isinf(p).any():
                return True
        for b in self.discriminator.buffers():
            if torch.isnan(b).any() or torch.isinf(b).any():
                return True
        return False

    def _reinit_discriminator(self):
        """Reinitialize discriminator from scratch after NaN corruption."""
        self.discriminator, self.optimizer_d, self.scheduler_d = self._create_discriminator()
        # Fast-forward scheduler to current epoch
        for _ in range(max(0, self.epoch - 500)):
            self.scheduler_d.step()
        print(f"  [!!] Discriminator reinitialized, LR={self.optimizer_d.param_groups[0]['lr']:.6f}")

    def build_dataloader(self, epoch: int = 0) -> DataLoader:
        """Build training dataloader."""
        data_cfg = self.config.get('data', {})
        dataset = V5SequenceDataset(
            root=data_cfg.get('train_dir', 'training_data/v5'),
            crop_size=data_cfg.get('crop_size', 256),
            augment=True,
            consecutive_pairs=data_cfg.get('consecutive_pairs', False),
        )
        
        # Weighted sampling (higher motion = sampled more)
        sampler = None
        if data_cfg.get('weighted_sampling', True):
            weights = dataset.get_sampler_weights(epoch=epoch)
            sampler = WeightedRandomSampler(weights, len(dataset), replacement=True)
        
        return DataLoader(
            dataset,
            batch_size=data_cfg.get('batch_size', 6),
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=data_cfg.get('num_workers', 0),
            persistent_workers=(data_cfg.get('num_workers', 0) > 0),
            pin_memory=True,
            drop_last=True,
        )
    
    def build_val_dataloader(self) -> DataLoader | None:
        """Build validation dataloader if val_dir is configured."""
        data_cfg = self.config.get('data', {})
        val_dir = data_cfg.get('val_dir')
        if not val_dir or not os.path.exists(val_dir):
            return None
        
        dataset = V5SequenceDataset(
            root=val_dir,
            crop_size=data_cfg.get('crop_size', 256),
            augment=False,
        )
        
        return DataLoader(
            dataset,
            batch_size=data_cfg.get('batch_size', 6),
            shuffle=False,
            num_workers=data_cfg.get('num_workers', 0),
            persistent_workers=(data_cfg.get('num_workers', 0) > 0),
            pin_memory=True,
            drop_last=False,
        )
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> dict:
        """Run validation and return metrics."""
        self.model.eval()
        
        total_psnr = 0.0
        total_loss = 0.0
        n_batches = 0
        
        for batch in dataloader:
            context = batch['context'].to(self.device)
            gt = batch['gt'].to(self.device)
            frames = [context[:, i] for i in range(context.shape[1])]

            with autocast('cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                output = self.model(frames)
                pred = output['output']
                losses = self.loss_fn(
                    pred, gt, epoch=self.epoch,
                    routing_map=output.get('routing_map'),
                    fg_warp=output.get('fg_warp'),
                    fg_synth=output.get('fg_synth'),
                )

            total_psnr += psnr(pred, gt)
            total_loss += losses['total'].item()
            n_batches += 1
        
        self.model.train()
        
        return {
            'psnr': total_psnr / max(n_batches, 1),
            'loss': total_loss / max(n_batches, 1),
        }
    
    def save_checkpoint(self, filename: str | None = None):
        """Save training checkpoint."""
        if filename is None:
            filename = f"checkpoint_epoch_{self.epoch:04d}.pt"
        
        path = os.path.join(self.ckpt_dir, filename)
        torch.save({
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_psnr': self.best_psnr,
            'generator_state_dict': self._clean_state_dict(self.model.state_dict()),
            'ema_state_dict': self._clean_state_dict(self.ema_model.state_dict()),
            'discriminator_state_dict': self._clean_state_dict(self.discriminator.state_dict()),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'scheduler_g_state_dict': self.scheduler_g.state_dict(),
            'scheduler_d_state_dict': self.scheduler_d.state_dict(),
            'config': self.config,
        }, path)
        
        # Also save as latest (atomic write to prevent corruption on crash)
        import shutil
        latest_path = os.path.join(self.ckpt_dir, 'latest.pt')
        tmp_path = latest_path + '.tmp'
        shutil.copy2(path, tmp_path)
        os.replace(tmp_path, latest_path)
        
        # Cleanup old checkpoints (keep last N + best + multiples of 10)
        ckpts = sorted([
            f for f in os.listdir(self.ckpt_dir)
            if f.startswith('checkpoint_epoch_') and f.endswith('.pt')
        ])
        
        # Keep the most recent `keep_last_n`
        keep_recent = set(ckpts[-self.keep_last_n:])
        
        for old in ckpts:
            if old in keep_recent or 'best' in old:
                continue
                
            # Keep multiples of 10 forever
            try:
                ep_str = old.replace('checkpoint_epoch_', '').replace('.pt', '')
                if int(ep_str) % 10 == 0:
                    continue
            except ValueError:
                pass
                
            # If it's not recent, not best, and not a multiple of 10, delete it
            os.remove(os.path.join(self.ckpt_dir, old))
    
    def load_checkpoint(self, path: str, fresh_optim: bool = False):
        """Resume from checkpoint. If fresh_optim=True, only loads model weights (skips optimizer/scaler)."""
        print(f"  Loading checkpoint: {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        
        def strip_compiled_prefix(state_dict):
            """Strip '_orig_mod.' prefix added by torch.compile()."""
            cleaned = {}
            had_prefix = False
            for k, v in state_dict.items():
                if k.startswith('_orig_mod.'):
                    cleaned[k[len('_orig_mod.'):]] = v
                    had_prefix = True
                else:
                    cleaned[k] = v
            if had_prefix:
                print(f"  [!] Stripped '_orig_mod.' prefix from {len(state_dict)} keys (torch.compile checkpoint)")
            return cleaned
        
        # Load into the UNDERLYING model, not the compiled wrapper.
        # torch.compile wraps the model and adds '_orig_mod.' prefix to state_dict keys.
        # The checkpoint has clean keys (prefix stripped during save), so loading directly
        # into the compiled model with strict=False silently drops ALL weights.
        gen_sd = strip_compiled_prefix(ckpt['generator_state_dict'])

        # Remap affine_head keys ONLY if checkpoint has old format (indices 2,4,6)
        # New format has indices 1,3,5 — skip remapping if already correct
        needs_remap = any('background_path.affine_head.2.' in k for k in gen_sd)
        if needs_remap:
            remapped = {}
            for k, v in gen_sd.items():
                if 'background_path.affine_head.' in k:
                    parts = k.split('.')
                    idx_pos = parts.index('affine_head') + 1
                    old_idx = int(parts[idx_pos])
                    if old_idx >= 2:
                        parts[idx_pos] = str(old_idx - 1)
                        remapped['.'.join(parts)] = v
                else:
                    remapped[k] = v
            gen_sd = remapped
            print(f"  [!] Remapped old affine_head keys to new format")

        if hasattr(self.model, '_orig_mod'):
            result = self.model._orig_mod.load_state_dict(gen_sd, strict=True)
        else:
            result = self.model.load_state_dict(gen_sd, strict=True)

        if 'ema_state_dict' in ckpt:
            ema_sd = strip_compiled_prefix(ckpt['ema_state_dict'])
            # Remap EMA affine_head keys only if old format
            if any('background_path.affine_head.2.' in k for k in ema_sd):
                ema_remapped = {}
                for k, v in ema_sd.items():
                    if 'background_path.affine_head.' in k:
                        parts = k.split('.')
                        idx_pos = parts.index('affine_head') + 1
                        old_idx = int(parts[idx_pos])
                        if old_idx >= 2:
                            parts[idx_pos] = str(old_idx - 1)
                            ema_remapped['.'.join(parts)] = v
                    else:
                        ema_remapped[k] = v
                ema_sd = ema_remapped
            self.ema_model.load_state_dict(ema_sd, strict=True)

        disc_sd = strip_compiled_prefix(ckpt['discriminator_state_dict'])
        if hasattr(self.discriminator, '_orig_mod'):
            self.discriminator._orig_mod.load_state_dict(disc_sd, strict=True)
        else:
            self.discriminator.load_state_dict(disc_sd, strict=True)
        
        if fresh_optim:
            print("  [!] --fresh-optim: fresh optimizer + scheduler (fine-tuning mode)")
            # Don't fast-forward schedulers — start fresh cosine from epoch 0
        else:
            self.optimizer_g.load_state_dict(ckpt['optimizer_g_state_dict'])
            self.optimizer_d.load_state_dict(ckpt['optimizer_d_state_dict'])
            try:
                self.scheduler_g.load_state_dict(ckpt['scheduler_g_state_dict'])
                self.scheduler_d.load_state_dict(ckpt['scheduler_d_state_dict'])
            except (ValueError, KeyError) as e:
                print(f"  [!] Could not restore scheduler state ({e}), using fresh scheduler")
        
        if fresh_optim:
            # Fine-tuning: reset epoch to 0 (new training run with old weights)
            self.epoch = 0
            self.global_step = 0
            self.best_psnr = 0.0
            print(f"  Fine-tuning from epoch 0 (loaded weights from epoch {ckpt['epoch']})")
        else:
            self.epoch = ckpt['epoch'] + 1
            self.global_step = ckpt['global_step']
            self.best_psnr = ckpt.get('best_psnr', 0.0)
            print(f"  Resumed from epoch {self.epoch - 1} (step {self.global_step})")
        print(f"  Best PSNR so far: {self.best_psnr:.2f} dB")
    
    def _warmup_lr(self, epoch: int, step_in_epoch: int, total_steps: int):
        """Linear warmup for first N epochs."""
        if epoch >= self.warmup_epochs:
            return
        warmup_factor = (epoch * total_steps + step_in_epoch) / (self.warmup_epochs * total_steps)
        warmup_factor = max(warmup_factor, 0.01)  # Minimum 1%
        for pg in self.optimizer_g.param_groups:
            pg['lr'] = self.config['training']['learning_rate'] * warmup_factor
    
    def train_one_epoch(self, dataloader: DataLoader) -> dict:
        """Train for one epoch. Returns metrics dict."""
        self.model.train()
        self.discriminator.train()
        
        epoch_losses = {
            'total': 0, 'charbonnier': 0, 'edge_l1': 0, 'census': 0, 'lpips': 0, 'flow': 0,
            'fft': 0, 'gan_g': 0, 'gan_d': 0, 'psnr': 0, 'ssim': 0,
            'routing_balance': 0, 'routing_entropy': 0, 'routing_mean': 0,
            'branch_warp_aux': 0, 'branch_synth_aux': 0,
            'temporal': 0,
        }
        n_batches = 0
        epoch_start_time = time.time()

        # Zero grads once before the epoch (accumulation-safe)
        self.optimizer_g.zero_grad()

        for batch_idx, batch in enumerate(dataloader):
            batch_start_time = time.time()
            context = batch['context'].to(self.device)  # (B, 7, 3, H, W)
            gt = batch['gt'].to(self.device)  # (B, 3, H, W)

            # Convert context to list of frames
            frames = [context[:, i] for i in range(context.shape[1])]

            # Warmup LR
            self._warmup_lr(self.epoch, batch_idx, len(dataloader))

            # =============== Generator Step ===============
            
            try:
                with autocast('cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                    output = self.model(frames)
                    pred = output['output']
                    
                    # Compute GAN loss if active
                    # D runs in fp32 to avoid spectral norm instability in bf16
                    gan_loss = None
                    if self.gan_active:
                        with autocast('cuda', enabled=False):
                            with torch.no_grad():
                                real_preds_for_g = self.discriminator(gt.float())
                            fake_preds = self.discriminator(pred.float())
                            gan_result = self.discriminator.compute_loss(
                                real_preds_for_g, fake_preds, for_generator=True
                            )
                            gan_loss = gan_result['loss']
                        # If D is corrupted, skip GAN loss but keep training on reconstruction
                        if torch.isnan(gan_loss) or torch.isinf(gan_loss):
                            gan_loss = None
                            if self._check_disc_nan():
                                print(f"  [!!] Discriminator corrupted (NaN in weights/buffers), reinitializing")
                                self._reinit_discriminator()
                    
                    # Full loss (includes routing auxiliary + warping losses)
                    losses = self.loss_fn(
                        pred, gt, epoch=self.epoch, gan_loss=gan_loss,
                        routing_map=output.get('routing_map'),
                        fg_warp=output.get('fg_warp'),
                        fg_synth=output.get('fg_synth'),
                        frame_i=frames[3], frame_ip1=frames[4],
                        flow_fwd=output.get('flow_fwd'),
                        flow_bwd=output.get('flow_bwd'),
                    )
                    total_loss = losses['total']
                    
                    # Flow consistency loss
                    flow_fwd = output.get('flow_fwd')
                    flow_bwd = output.get('flow_bwd')
                    l_flow = torch.tensor(0.0, device=gt.device)
                    if flow_fwd is not None and flow_bwd is not None:
                        l_flow = (flow_fwd + flow_bwd).abs().mean()
                        total_loss = total_loss + self.loss_fn.flow_weight * l_flow

                    # Temporal consistency loss (V5.1)
                    # Second forward pass uses no_grad to halve memory cost
                    l_temporal = torch.tensor(0.0, device=gt.device)
                    has_consec = batch.get('has_consecutive', False)
                    if isinstance(has_consec, torch.Tensor):
                        has_consec = has_consec.any().item()
                    if has_consec and 'context_next' in batch:
                        context_next = batch['context_next'].to(self.device)
                        gt_next = batch['gt_next'].to(self.device)
                        frames_next = [context_next[:, i] for i in range(context_next.shape[1])]
                        with torch.no_grad():
                            output_next = self.model(frames_next)
                            pred_next = output_next['output']
                        l_temporal = self.loss_fn.compute_temporal_consistency(
                            pred, pred_next, gt, gt_next, epoch=self.epoch
                        )
                        total_loss = total_loss + l_temporal

                    total_loss = total_loss / self.accum_steps
                
                # NaN guard — detect and skip before it poisons optimizer state
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f"  [!] NaN/Inf loss at batch {batch_idx}, skipping")
                    self.optimizer_g.zero_grad()
                    continue
                
                total_loss.backward()
                
                if (batch_idx + 1) % self.accum_steps == 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer_g.step()
                    self.optimizer_g.zero_grad()
                    # Step EMA
                    self.ema_model.update_parameters(self.model)
                
            except torch.cuda.OutOfMemoryError:
                print(f"  [!] OOM at batch {batch_idx}, skipping...")
                # Release autograd graph to free CUDA memory
                try:
                    del output, pred, losses, total_loss
                except NameError:
                    pass
                torch.cuda.empty_cache()
                self.optimizer_g.zero_grad()
                continue
            
            # =============== Discriminator Step (Phase 3 only) ===============
            d_loss_val = 0.0
            if self.gan_active:
                self.optimizer_d.zero_grad()

                try:
                    # D runs entirely in fp32 — spectral norm is unstable in bf16
                    real_preds_for_d = self.discriminator(gt.float())
                    fake_preds_detached = self.discriminator(pred.detach().float())
                    d_result = self.discriminator.compute_loss(
                        real_preds_for_d, fake_preds_detached, for_generator=False
                    )
                    d_loss = d_result['loss']

                    # NaN guard — skip D step only, G metrics still collected below
                    if not (torch.isnan(d_loss) or torch.isinf(d_loss)):
                        d_loss.backward()

                        # R1 penalty (every N steps) — separate backward pass
                        # Uses lazy R1: multiply weight by interval to maintain average strength
                        if self.global_step % self.r1_every == 0:
                            r1 = r1_gradient_penalty(
                                self.discriminator, gt,
                                weight=self.r1_weight * self.r1_every,
                            )
                            r1.backward()

                        # Clip and step
                        nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.grad_clip)
                        self.optimizer_d.step()
                        d_loss_val = d_result['loss'].item()

                        # Check for NaN in D weights after optimizer step
                        if self._check_disc_nan():
                            print(f"  [!!] Discriminator weights NaN after step at batch {batch_idx}")
                            self._reinit_discriminator()
                    else:
                        print(f"  [!] NaN/Inf D loss at batch {batch_idx}, skipping D step")
                        self.optimizer_d.zero_grad()

                except torch.cuda.OutOfMemoryError:
                    print(f"  [!] OOM in discriminator at batch {batch_idx}, skipping...")
                    torch.cuda.empty_cache()
                    self.optimizer_d.zero_grad()
            
            # =============== Metrics ===============
            with torch.no_grad():
                batch_psnr = psnr(pred, gt)
                batch_ssim = ssim(pred, gt)
            
            epoch_losses['total'] += total_loss.item() * self.accum_steps
            epoch_losses['charbonnier'] += losses['charbonnier'].item()
            epoch_losses['edge_l1'] += losses['edge_l1'].item()
            epoch_losses['census'] += losses.get('census', torch.tensor(0.0)).item()
            epoch_losses['fft'] += losses['fft'].item()
            epoch_losses['lpips'] += losses.get('lpips', torch.tensor(0.0)).item()
            epoch_losses['flow'] += l_flow.item()
            epoch_losses['gan_g'] += losses['gan'].item()
            epoch_losses['gan_d'] += d_loss_val
            epoch_losses['psnr'] += batch_psnr
            epoch_losses['ssim'] += batch_ssim
            epoch_losses['routing_balance'] += losses.get('routing_balance', torch.tensor(0.0)).item()
            epoch_losses['routing_entropy'] += losses.get('routing_entropy', torch.tensor(0.0)).item()
            epoch_losses['routing_mean'] += losses.get('routing_mean', torch.tensor(0.0)).item()
            epoch_losses['branch_warp_aux'] += losses.get('branch_warp_aux', torch.tensor(0.0)).item()
            epoch_losses['branch_synth_aux'] += losses.get('branch_synth_aux', torch.tensor(0.0)).item()
            epoch_losses['temporal'] += l_temporal.item()
            n_batches += 1
            self.global_step += 1
            
            # Log periodically
            if self.global_step % self.log_every == 0:
                avg_psnr = epoch_losses['psnr'] / n_batches
                lr = self.optimizer_g.param_groups[0]['lr']
                
                # Time estimates
                elapsed = time.time() - epoch_start_time
                batches_done = batch_idx + 1
                batches_left = len(dataloader) - batches_done
                secs_per_batch = elapsed / batches_done
                epoch_eta_secs = secs_per_batch * batches_left
                

                secs_per_epoch = elapsed / max(batches_done / len(dataloader), 1e-6)
                epochs_left = self.total_epochs - self.epoch - 1
                total_eta_secs = secs_per_epoch * epochs_left + epoch_eta_secs
                
                def fmt_time(s):
                    h, m = divmod(int(s), 3600)
                    m, sec = divmod(m, 60)
                    return f"{h}h{m:02d}m{sec:02d}s" if h else f"{m}m{sec:02d}s"
                
                print(
                    f"  [{self.current_phase}] Epoch {self.epoch} "
                    f"Batch {batch_idx + 1}/{len(dataloader)} "
                    f"Loss={epoch_losses['total'] / n_batches:.4f} "
                    f"PSNR={batch_psnr:.2f}dB (avg={avg_psnr:.2f}dB) "
                    f"SSIM={batch_ssim:.3f} "
                    f"LR={lr:.2e} "
                    f"ETA epoch={fmt_time(epoch_eta_secs)} "
                    f"total={fmt_time(total_eta_secs)}"
                )
                self._log({
                    'train/loss': epoch_losses['total'] / n_batches,
                    'train/charbonnier': losses['charbonnier'].item(),
                    'train/edge_l1': losses['edge_l1'].item(),
                    'train/census': losses.get('census', torch.tensor(0.0)).item(),
                    'train/fft': losses['fft'].item(),
                    'train/lpips': losses.get('lpips', torch.tensor(0.0)).item(),
                    'train/flow': l_flow.item(),
                    'train/fft_weight': losses['fft_weight'],
                    'train/lpips_weight': losses.get('lpips_weight', 0.0),
                    'train/gan_g': losses['gan'].item(),
                    'train/gan_d': d_loss_val,
                    'train/gan_weight': losses['gan_weight'],
                    'train/psnr': batch_psnr,
                    'train/ssim': batch_ssim,
                    'train/lr': lr,
                    'train/routing_mean': losses.get('routing_mean', torch.tensor(0.0)).item(),
                    'train/routing_balance': losses.get('routing_balance', torch.tensor(0.0)).item(),
                    'train/routing_entropy': losses.get('routing_entropy', torch.tensor(0.0)).item(),
                    'train/branch_warp_aux': losses.get('branch_warp_aux', torch.tensor(0.0)).item(),
                    'train/branch_synth_aux': losses.get('branch_synth_aux', torch.tensor(0.0)).item(),
                })
        
        # Average metrics
        for k in epoch_losses:
            epoch_losses[k] /= max(n_batches, 1)
        
        return epoch_losses
    
    def train(self, resume: str | None = None, fresh_optim: bool = False):
        """Full training loop."""
        print(f"\n{'='*60}")
        print(f"V5 Training — {self.total_epochs} epochs on {self.device}")
        print(f"  Model: {sum(p.numel() for p in self.model.parameters()):,} params")
        print(f"  AMP: {self.use_amp}")
        print(f"{'='*60}\n")
        
        self._init_wandb()
        
        if resume:
            self.load_checkpoint(resume, fresh_optim=fresh_optim)

        # Apply correct batch size / accum for the current phase on resume.
        # Without this, resuming from epoch 350+ would use Phase 1 batch size and OOM.
        if self.epoch >= 500:
            self.config['data']['batch_size'] = 4
            self.accum_steps = 2
            print(f"  => Resumed into Phase 3: BS=4, accum={self.accum_steps}")
        elif self.epoch >= 300:
            self.config['data']['batch_size'] = 5
            print(f"  => Resumed into Phase 2: BS=5")

        dataloader = self.build_dataloader(self.epoch)
        val_dataloader = self.build_val_dataloader()
        print(f"  Dataset: {len(dataloader.dataset)} sequences")
        print(f"  Batches/epoch: {len(dataloader)}")
        if val_dataloader:
            print(f"  Validation: {len(val_dataloader.dataset)} sequences")
        else:
            print(f"  Validation: disabled (no val_dir configured, using train PSNR)")
        print(f"  Effective batch size: {self.config['data']['batch_size'] * self.accum_steps}\n")

        for epoch in range(self.epoch, self.total_epochs):
            self.epoch = epoch

            # Rebuild dataloader at phase transitions and curriculum milestones
            if epoch == 500 and self.accum_steps != 2:
                print("  => Transitioning to Phase 3 (GAN), adjusting batch size to 4")
                self.config['data']['batch_size'] = 4
                self.accum_steps = 2
                dataloader = self.build_dataloader(epoch)
                print(f"  Effective batch size: {self.config['data']['batch_size'] * self.accum_steps}\n")
            elif epoch > 0 and (epoch % 50 == 0 and epoch <= 200):
                print(f"  => Rebuilding dataloader for curriculum (epoch {epoch})")
                dataloader = self.build_dataloader(epoch)
            elif epoch == 300 and self.config['data']['batch_size'] > 5:
                # LPIPS activates at epoch 300 — VGG adds ~1.5 GB, reduce BS to avoid OOM
                print("  => LPIPS activating (Phase 2), reducing batch size to 5")
                self.config['data']['batch_size'] = 5
                dataloader = self.build_dataloader(epoch)
                print(f"  Effective batch size: {self.config['data']['batch_size'] * self.accum_steps}\n")
            
            t0 = time.time()
            
            print(f"\nEpoch {epoch}/{self.total_epochs} [{self.current_phase}]")
            print(f"  FFT weight: {self.loss_fn.get_fft_weight(epoch):.3f}")
            print(f"  GAN weight: {self.loss_fn.get_gan_weight(epoch):.4f}")
            
            metrics = self.train_one_epoch(dataloader)
            
            elapsed = time.time() - t0
            
            # LR step (after warmup)
            if epoch >= self.warmup_epochs:
                self.scheduler_g.step()
                if self.gan_active:
                    self.scheduler_d.step()
            
            # Report
            print(f"\n  Epoch {epoch} Summary ({elapsed:.0f}s):")
            print(f"    Total Loss:  {metrics['total']:.4f}")
            print(f"    Charbonnier: {metrics['charbonnier']:.4f}")
            print(f"    Edge L1:     {metrics['edge_l1']:.4f}")
            print(f"    Census:      {metrics['census']:.4f}")
            print(f"    FFT:         {metrics['fft']:.4f}")
            print(f"    LPIPS:       {metrics['lpips']:.4f}")
            print(f"    PSNR:        {metrics['psnr']:.2f} dB")
            print(f"    SSIM:        {metrics['ssim']:.3f}")
            print(f"    Routing:     mean={metrics['routing_mean']:.3f} balance={metrics['routing_balance']:.4f} entropy={metrics['routing_entropy']:.4f}")
            print(f"    Branch Aux:  warp={metrics['branch_warp_aux']:.4f} synth={metrics['branch_synth_aux']:.4f}")
            if self.gan_active:
                print(f"    GAN (G/D):   {metrics['gan_g']:.4f} / {metrics['gan_d']:.4f}")
            
            log_metrics = {
                'epoch/loss': metrics['total'],
                'epoch/train_psnr': metrics['psnr'],
                'epoch/charbonnier': metrics['charbonnier'],
                'epoch/fft': metrics['fft'],
                'epoch/routing_mean': metrics['routing_mean'],
                'epoch/routing_balance': metrics['routing_balance'],
                'epoch/routing_entropy': metrics['routing_entropy'],
                'epoch/branch_warp_aux': metrics['branch_warp_aux'],
                'epoch/branch_synth_aux': metrics['branch_synth_aux'],
                'epoch': epoch,
            }
            
            # Validation
            eval_psnr = metrics['psnr']  # fallback: use train PSNR
            if val_dataloader and (epoch + 1) % self.val_every == 0:
                val_metrics = self.validate(val_dataloader)
                eval_psnr = val_metrics['psnr']
                print(f"    Val PSNR:    {val_metrics['psnr']:.2f} dB")
                print(f"    Val Loss:    {val_metrics['loss']:.4f}")
                log_metrics['epoch/val_psnr'] = val_metrics['psnr']
                log_metrics['epoch/val_loss'] = val_metrics['loss']
            
            self._log(log_metrics)
            
            # Save checkpoint
            if (epoch + 1) % self.save_every == 0 or epoch == self.total_epochs - 1:
                self.save_checkpoint()
                print(f"    Saved checkpoint")
            
            # Best model (based on val PSNR when available)
            if eval_psnr > self.best_psnr:
                self.best_psnr = eval_psnr
                self.save_checkpoint('best.pt')
                print(f"    ** New best PSNR: {self.best_psnr:.2f} dB")
        
        print(f"\n{'='*60}")
        print(f"Training complete! Best PSNR: {self.best_psnr:.2f} dB")
        print(f"{'='*60}")
        
        if self.wandb_run:
            import wandb
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='Train V5 Multi-Frame Interpolation')
    parser.add_argument('--config', type=str, default='configs/train_v5_5090.yaml',
                        help='Path to training config YAML')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--auto-resume', action='store_true',
                        help='Auto-resume from latest.pt if it exists')
    parser.add_argument('--fresh-optim', action='store_true',
                        help='Load model weights only, skip corrupted optimizer/scaler states')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    trainer = Trainer(config)
    
    resume_path = args.resume
    if args.auto_resume and resume_path is None:
        latest = os.path.join('checkpoints', 'v5', 'latest.pt')
        if os.path.exists(latest):
            resume_path = latest
            print(f"Auto-resuming from {latest}")
    
    trainer.train(resume=resume_path, fresh_optim=args.fresh_optim)


if __name__ == '__main__':
    main()
