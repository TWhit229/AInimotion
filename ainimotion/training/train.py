"""
Training loop for the LayeredInterpolator model.

Supports:
- GAN training with discriminator
- Mixed precision (FP16)
- Checkpointing and resumption
- TensorBoard logging
- Weights & Biases logging (--wandb flag)
"""

import argparse
import os
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

from ainimotion.models.interp import LayeredInterpolator
from ainimotion.training.losses import VFILoss
from ainimotion.training.discriminator import PatchDiscriminator, GANLoss
from ainimotion.data.dataset import create_dataloader


# Processes that waste GPU memory during training
GPU_HOGGERS = [
    'steamwebhelper', 'Discord', 'chrome', 'msedge',
    'lghub_system_tray', 'AMDRSSrcExt', 'iCUE',
]


def free_gpu_memory():
    """Kill unnecessary background processes that consume GPU memory."""
    import subprocess
    killed = []
    for proc in GPU_HOGGERS:
        try:
            result = subprocess.run(
                ['taskkill', '/F', '/IM', f'{proc}.exe'],
                capture_output=True, timeout=5
            )
            if result.returncode == 0:
                killed.append(proc)
        except Exception:
            pass
    if killed:
        print(f"  [CLEAN] Freed GPU memory by stopping: {', '.join(killed)}")
    else:
        print("  [OK] No unnecessary GPU processes found")


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


class Trainer:
    """
    Trainer for LayeredInterpolator.
    
    Args:
        config: Training configuration dict
        resume_from: Optional checkpoint path to resume from
    """
    
    def __init__(
        self,
        config: dict,
        resume_from: str | None = None,
    ):
        self.config = config
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Kill Steam, Discord, Chrome, etc. to free GPU memory
        free_gpu_memory()
        
        # Build model
        self.generator = LayeredInterpolator(
            base_channels=config.get('base_channels', 32),
            kernel_size=config.get('kernel_size', 7),
            grid_size=config.get('grid_size', 8),
            use_refinement=config.get('use_refinement', True),
        ).to(self.device)
        
        # Build discriminator
        self.discriminator = PatchDiscriminator(
            in_channels=3,
            base_channels=config.get('disc_channels', 64),
        ).to(self.device)
        
        # Losses
        self.vfi_loss = VFILoss(
            l1_weight=config.get('l1_weight', 1.0),
            perceptual_weight=config.get('perceptual_weight', 0.1),
            edge_weight=config.get('edge_weight', 0.5),
            edge_multiplier=config.get('edge_multiplier', 10.0),  # Higher = sharper edges
        ).to(self.device)
        
        self.gan_loss = GANLoss(
            loss_type=config.get('gan_type', 'lsgan'),
            label_smoothing=config.get('label_smoothing', 0.0),
        ).to(self.device)
        
        self.gan_weight = config.get('gan_weight', 0.01)
        
        # Two-phase training: no GAN until gan_start_epoch
        self.gan_start_epoch = config.get('gan_start_epoch', 0)  # 0 = GAN from start
        self.lr_g_phase2 = config.get('lr_g_phase2', None)  # Optional lr_g reduction for Phase 2
        self.current_epoch = 0
        self.gan_activated = self.gan_start_epoch == 0
        self.d_warmup_batches = config.get('d_warmup_batches', 500)  # D-only warmup at Phase 2 start
        self.d_warmup_done = self.gan_start_epoch == 0  # Skip warmup if GAN from start
        
        # Phase 2 gradient accumulation (for VRAM management)
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
        self.phase2_batch_size = config.get('phase2_batch_size', None)
        self._accum_counter = 0  # Tracks mini-batches since last optimizer step
        self._data_path = config.get('data_dir', None)  # Store for dataloader rebuild
        
        # Discriminator update frequency (train D every N batches)
        self.d_update_ratio = config.get('d_update_ratio', 1)
        self.d_update_counter = 0
        
        # Adaptive learning rate for D (self-balancing GAN)
        self.adaptive_lr_d = config.get('adaptive_lr_d', True)
        self.lr_d_min = config.get('lr_d', 1e-5) * 0.1  # Floor: 10% of initial
        self.lr_d_max = config.get('lr_d', 1e-5) * 10   # Ceiling: 10x initial
        self.d_acc_target_high = 0.85  # If d_acc > this, slow D down
        self.d_acc_target_low = 0.50   # If d_acc < this, speed D up
        self.d_acc_window = []         # Rolling window of d_acc values
        self.d_acc_window_size = 100   # Check over last 100 batches
        self.lr_d_adjust_factor = 0.95 # Multiply/divide by this when adjusting
        
        # Adaptive training settings (legacy throttling)
        self.d_throttle_threshold = config.get('d_throttle_threshold', 0.95)
        self.d_throttle_patience = config.get('d_throttle_patience', 100)
        self.d_throttle_counter = 0  # Consecutive batches with high d_acc
        self.d_throttled = False  # Whether D updates are currently skipped
        
        # Gradient clipping
        self.grad_clip = config.get('grad_clip', 0.0)  # 0 = disabled
        
        # Early stopping
        self.early_stop_patience = config.get('early_stop_patience', 0)  # 0 = disabled
        self.min_psnr_delta = config.get('min_psnr_delta', 0.1)
        self.best_psnr = 0.0
        self.epochs_without_improvement = 0
        
        # Optimizers
        self.optimizer_g = torch.optim.AdamW(
            self.generator.parameters(),
            lr=config.get('lr_g', 1e-4),
            betas=(0.9, 0.999),
            weight_decay=config.get('weight_decay', 0.01),
        )
        
        self.optimizer_d = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=config.get('lr_d', 1e-4),
            betas=(0.9, 0.999),
            weight_decay=config.get('weight_decay', 0.01),
        )
        
        # Learning rate schedulers
        self.scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_g,
            T_max=config.get('epochs', 100),
        )
        self.scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_d,
            T_max=config.get('epochs', 100),
        )
        
        # Mixed precision
        self.use_amp = config.get('use_amp', True)
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        
        # Logging
        self.log_dir = Path(config.get('log_dir', 'runs')) / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        
        # Weights & Biases (optional)
        self.use_wandb = config.get('use_wandb', False)
        if self.use_wandb:
            import wandb
            self.wandb = wandb
            wandb.init(
                project=config.get('wandb_project', 'AInimotion'),
                name=config.get('wandb_run_name', f"train_{datetime.now().strftime('%m%d_%H%M')}"),
                config=config,
                resume='allow',
            )
            wandb.watch(self.generator, log='gradients', log_freq=500)
            print(f"  [OK] W&B logging enabled: {wandb.run.url}")
        else:
            self.wandb = None
        
        # Checkpointing
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.start_epoch = 0
        self.global_step = 0
        self.start_batch = 0  # For batch-level resume
        
        # Resume if specified
        if resume_from:
            self._load_checkpoint(resume_from)
    
    def _load_checkpoint(self, path: str):
        """Load checkpoint and resume training.
        
        Handles architecture mismatches gracefully: if the checkpoint was
        trained with a different model config (e.g., kernel_size=7 vs 9),
        compatible weights are loaded and incompatible ones are re-initialized.
        Training restarts from epoch 0 in this case.
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        # Try strict loading first; fall back to partial on architecture mismatch
        try:
            self.generator.load_state_dict(checkpoint['generator'])
            self.discriminator.load_state_dict(checkpoint['discriminator'])
        except RuntimeError as e:
            if "size mismatch" in str(e):
                print(f"\n[!]  Architecture mismatch detected in checkpoint!")
                print(f"     (likely kernel_size or base_channels changed)")
                print(f"     Loading compatible weights only; mismatched layers re-initialized.")
                # strict=False only handles missing/extra keys, NOT size mismatches.
                # We must manually filter out keys with incompatible shapes.
                def _filter_compatible(model, ckpt_state):
                    """Return only checkpoint keys whose shapes match the model."""
                    model_state = model.state_dict()
                    compatible = {}
                    skipped = []
                    for k, v in ckpt_state.items():
                        if k in model_state and v.shape == model_state[k].shape:
                            compatible[k] = v
                        else:
                            skipped.append(k)
                    return compatible, skipped
                
                compat_g, skip_g = _filter_compatible(self.generator, checkpoint['generator'])
                compat_d, skip_d = _filter_compatible(self.discriminator, checkpoint['discriminator'])
                
                self.generator.load_state_dict(compat_g, strict=False)
                self.discriminator.load_state_dict(compat_d, strict=False)
                
                print(f"     Generator:     {len(compat_g)} loaded, {len(skip_g)} skipped")
                if skip_g:
                    print(f"       Skipped: {', '.join(skip_g[:5])}{'...' if len(skip_g) > 5 else ''}")
                print(f"     Discriminator: {len(compat_d)} loaded, {len(skip_d)} skipped")
                print(f"     Starting training from epoch 0 (fresh optimizers)\n")
                
                # Don't load optimizer/scheduler states — they won't match
                self.start_epoch = 0
                self.start_batch = 0
                self.global_step = 0
                return
            else:
                raise
        
        # Full match — restore optimizer/scheduler states too
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d'])
        self.scheduler_g.load_state_dict(checkpoint['scheduler_g'])
        self.scheduler_d.load_state_dict(checkpoint['scheduler_d'])
        self.scaler.load_state_dict(checkpoint['scaler'])
        
        # Calculate resume position
        saved_epoch = checkpoint['epoch']
        saved_global_step = checkpoint.get('global_step', 0)
        batches_per_epoch = checkpoint.get('batches_per_epoch', 0)
        
        if batches_per_epoch > 0:
            # Calculate which batch within epoch we were at
            self.start_batch = saved_global_step % batches_per_epoch
            if self.start_batch > 0:
                # Resume mid-epoch
                self.start_epoch = saved_epoch
                print(f"Resumed from epoch {self.start_epoch}, batch {self.start_batch}")
            else:
                # Start fresh at next epoch
                self.start_epoch = saved_epoch + 1
                print(f"Resumed from epoch {self.start_epoch}")
        else:
            # Fallback for old checkpoints without batches_per_epoch
            self.start_epoch = saved_epoch + 1
            self.start_batch = 0
            print(f"Resumed from epoch {self.start_epoch}")
        
        self.global_step = saved_global_step
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False, metrics: dict = None):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'batches_per_epoch': getattr(self, '_batches_per_epoch', 0),
            'metrics': metrics or {},
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'optimizer_g': self.optimizer_g.state_dict(),
            'optimizer_d': self.optimizer_d.state_dict(),
            'scheduler_g': self.scheduler_g.state_dict(),
            'scheduler_d': self.scheduler_d.state_dict(),
            'scaler': self.scaler.state_dict(),
            'config': self.config,
        }
        
        path = self.checkpoint_dir / f'checkpoint_epoch_{epoch:04d}.pt'
        torch.save(checkpoint, path)
        
        # Also save as latest
        latest_path = self.checkpoint_dir / 'checkpoint_latest.pt'
        torch.save(checkpoint, latest_path)
        
        if is_best:
            best_path = self.checkpoint_dir / 'checkpoint_best.pt'
            torch.save(checkpoint, best_path)
    
    def train_step(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        """
        Single training step.
        
        Args:
            batch: Dictionary with 'frame1', 'frame2', 'frame3', 'inputs'
            
        Returns:
            Dictionary of loss values and metrics
        """
        frame1 = batch['frame1'].to(self.device, non_blocking=True)
        frame2 = batch['frame2'].to(self.device, non_blocking=True)  # Target
        frame3 = batch['frame3'].to(self.device, non_blocking=True)
        
        losses = {}
        
        # ============ Two-Phase Training Check ============
        # GAN loss only applies to G after D warmup is complete
        gan_active = self.current_epoch >= self.gan_start_epoch and self.d_warmup_done
        
        if not gan_active:
            # Phase 1: Skip ALL discriminator code
            losses['d_acc'] = 0.0
            losses['d_throttled'] = 0.0
            losses['d_real'] = 0.0
            losses['d_fake'] = 0.0
            losses['d_total'] = 0.0
        
        # ============ Train Discriminator ============
        # D trains whenever we're in Phase 2 epoch range (even during warmup)
        d_should_train = self.current_epoch >= self.gan_start_epoch
        if d_should_train:
            # Check if we should skip discriminator update
            skip_d_update = False
            
            # D update ratio: only update D every N batches
            self.d_update_counter += 1
            if self.d_update_ratio > 1 and self.d_update_counter % self.d_update_ratio != 0:
                skip_d_update = True
            
            # Throttling based on d_acc (if D is too strong)
            if self.d_throttle_threshold < 1.0:  # Throttling enabled
                if self.d_throttled:
                    # Already throttling - skip D update
                    skip_d_update = True
            
            if not skip_d_update:
                self.optimizer_d.zero_grad()
                
                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    # Generate fake frame
                    with torch.no_grad():
                        output = self.generator(frame1, frame3)
                        fake_frame = output['output']
                    
                    # Real loss
                    pred_real = self.discriminator(frame2)
                    loss_d_real = self.gan_loss(pred_real, is_real=True)
                    
                    # Fake loss
                    pred_fake = self.discriminator(fake_frame.detach())
                    loss_d_fake = self.gan_loss(pred_fake, is_real=False)
                    
                    loss_d = (loss_d_real + loss_d_fake) * 0.5
                
                self.scaler.scale(loss_d).backward()
            
                # Gradient clipping for discriminator
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer_d)
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.grad_clip)
                
                self.scaler.step(self.optimizer_d)
                
                # Discriminator accuracy (how well it distinguishes real/fake)
                with torch.no_grad():
                    d_acc_real = (pred_real.mean() > 0.5).float().item()
                    d_acc_fake = (pred_fake.mean() < 0.5).float().item()
                    d_acc = (d_acc_real + d_acc_fake) / 2
                
                losses['d_real'] = loss_d_real.item()
                losses['d_fake'] = loss_d_fake.item()
                losses['d_total'] = loss_d.item()
                losses['d_acc'] = d_acc
                
                # Update throttle state
                if d_acc >= self.d_throttle_threshold:
                    self.d_throttle_counter += 1
                    if self.d_throttle_counter >= self.d_throttle_patience:
                        self.d_throttled = True
                        self.d_throttle_counter = 0
                else:
                    self.d_throttle_counter = 0
                    self.d_throttled = False
                
                # Adaptive lr_d adjustment (self-balancing GAN)
                if self.adaptive_lr_d:
                    self.d_acc_window.append(d_acc)
                    if len(self.d_acc_window) > self.d_acc_window_size:
                        self.d_acc_window.pop(0)
                    
                    # Only adjust every window_size batches
                    if len(self.d_acc_window) == self.d_acc_window_size:
                        avg_d_acc = sum(self.d_acc_window) / len(self.d_acc_window)
                        
                        current_lr = self.optimizer_d.param_groups[0]['lr']
                        new_lr = current_lr
                        
                        if avg_d_acc > self.d_acc_target_high:
                            # D is too strong, slow it down
                            new_lr = current_lr * self.lr_d_adjust_factor
                            new_lr = max(new_lr, self.lr_d_min)
                        elif avg_d_acc < self.d_acc_target_low:
                            # D is too weak, speed it up
                            new_lr = current_lr / self.lr_d_adjust_factor
                            new_lr = min(new_lr, self.lr_d_max)
                        
                        if new_lr != current_lr:
                            for param_group in self.optimizer_d.param_groups:
                                param_group['lr'] = new_lr
                            losses['lr_d_adjusted'] = new_lr
                        
                        # Clear window for next check
                        self.d_acc_window = []
            else:
                # Skipping D update - still need to generate for G training
                with torch.no_grad():
                    output = self.generator(frame1, frame3)
                    fake_frame = output['output']
                    pred_real = self.discriminator(frame2)
                    pred_fake = self.discriminator(fake_frame)
                    d_acc_real = (pred_real.mean() > 0.5).float().item()
                    d_acc_fake = (pred_fake.mean() < 0.5).float().item()
                    d_acc = (d_acc_real + d_acc_fake) / 2
                
                losses['d_real'] = 0.0
                losses['d_fake'] = 0.0
                losses['d_total'] = 0.0
                losses['d_acc'] = d_acc
                losses['d_throttled'] = 1.0
                
                # Check if we should resume D training
                if d_acc < self.d_throttle_threshold - 0.1:  # Hysteresis
                    self.d_throttled = False
        
        # ============ Train Generator (with gradient accumulation) ============
        # Determine active accumulation steps (only accumulate during Phase 2)
        accum_steps = self.gradient_accumulation_steps if gan_active else 1
        is_accum_step = accum_steps > 1  # Whether we're in accumulation mode
        
        # Only zero gradients at the start of an accumulation cycle
        if self._accum_counter == 0:
            self.optimizer_g.zero_grad()
        
        with torch.amp.autocast('cuda', enabled=self.use_amp):
            # Forward pass
            output = self.generator(frame1, frame3)
            fake_frame = output['output']
            
            # Reconstruction loss
            loss_vfi, vfi_components = self.vfi_loss(
                fake_frame, frame2, return_components=True
            )
            
            # GAN loss (fool discriminator) - only in Phase 2
            if gan_active:
                pred_fake = self.discriminator(fake_frame)
                loss_g_gan = self.gan_loss(
                    pred_fake, is_real=True, for_discriminator=False
                )
                loss_g = loss_vfi + self.gan_weight * loss_g_gan
            else:
                loss_g_gan = torch.tensor(0.0)
                loss_g = loss_vfi
            
            # Scale loss by accumulation steps so gradients average correctly
            if accum_steps > 1:
                loss_g_scaled = loss_g / accum_steps
            else:
                loss_g_scaled = loss_g
        
        self.scaler.scale(loss_g_scaled).backward()
        
        # Track accumulation progress
        self._accum_counter += 1
        should_step = self._accum_counter >= accum_steps
        
        if should_step:
            # All mini-batches accumulated — now step the optimizer
            self._accum_counter = 0
            
            # Unscale gradients for clipping and norm computation
            self.scaler.unscale_(self.optimizer_g)
            
            # Compute gradient norm before clipping (cap to avoid inf in averages)
            grad_norm_g = 0.0
            for p in self.generator.parameters():
                if p.grad is not None:
                    pn = p.grad.data.norm(2).item()
                    if pn == float('inf') or pn != pn:  # inf or nan
                        grad_norm_g = float('inf')
                        break
                    grad_norm_g += pn ** 2
            grad_norm_g = grad_norm_g ** 0.5 if grad_norm_g != float('inf') else float('inf')
            # Cap inf for cleaner epoch averaging
            if grad_norm_g == float('inf') or grad_norm_g != grad_norm_g:
                grad_norm_g = 999.0
            
            # Gradient clipping for generator
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.grad_clip)
            
            self.scaler.step(self.optimizer_g)
            self.scaler.update()
        else:
            # Still accumulating — report deferred metrics
            grad_norm_g = 0.0  # Norm not meaningful mid-accumulation
        
        # Compute PSNR
        with torch.no_grad():
            mse = F.mse_loss(fake_frame, frame2)
            psnr = 10 * torch.log10(1.0 / (mse + 1e-8)).item()
            
            # Scene cut detection rate
            scene_cuts = output.get('is_scene_cut', torch.zeros(1))
            scene_cut_rate = scene_cuts.float().mean().item()
        
        losses['g_l1'] = vfi_components['l1'].item()
        losses['g_perceptual'] = vfi_components['perceptual'].item()
        losses['g_edge'] = vfi_components['edge'].item()
        losses['g_gan'] = loss_g_gan.item()
        losses['g_total'] = loss_g.item()  # Report unscaled loss for metrics
        losses['grad_norm'] = grad_norm_g
        losses['psnr'] = psnr
        losses['scene_cut_rate'] = scene_cut_rate
        losses['lr'] = self.scheduler_g.get_last_lr()[0]
        losses['accum_step'] = int(should_step)  # 1 if optimizer stepped, 0 if accumulating
        
        return losses
    
    def train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        epoch: int,
    ) -> dict[str, float]:
        """Train for one epoch."""
        self.generator.train()
        self.discriminator.train()
        
        epoch_losses = {}
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}', leave=False)
        for batch in pbar:
            losses = self.train_step(batch)
            self.global_step += 1
            
            # Accumulate losses
            for k, v in losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0) + v
            
            # Update progress bar with key metrics
            pbar.set_postfix({
                'g': f"{losses['g_total']:.3f}",
                'd': f"{losses['d_total']:.3f}",
                'psnr': f"{losses['psnr']:.1f}",
                'd_acc': f"{losses['d_acc']:.0%}",
            })
            
            # Log to tensorboard
            if self.global_step % 100 == 0:
                for k, v in losses.items():
                    self.writer.add_scalar(f'train/{k}', v, self.global_step)
        
        # Average losses
        num_batches = len(dataloader)
        for k in epoch_losses:
            epoch_losses[k] /= num_batches
        
        return epoch_losses
    
    def train(self, dataloader: torch.utils.data.DataLoader):
        """Main training loop."""
        epochs = self.config.get('epochs', 100)
        
        # Outer progress bar for epochs
        epoch_pbar = tqdm(
            range(self.start_epoch, epochs),
            desc="Training",
            unit="epoch",
            position=0,
        )
        
        for epoch in epoch_pbar:
            losses = self.train_epoch(dataloader, epoch)
            
            # Step schedulers
            self.scheduler_g.step()
            self.scheduler_d.step()
            
            # Update outer progress bar
            epoch_pbar.set_postfix({
                "g_loss": f"{losses['g_total']:.4f}",
                "d_loss": f"{losses['d_total']:.4f}",
                "lr": f"{self.scheduler_g.get_last_lr()[0]:.2e}",
            })
            
            # Log epoch summary
            print(f"\nEpoch {epoch}/{epochs-1} complete:")
            for k, v in losses.items():
                print(f"  {k}: {v:.4f}")
                self.writer.add_scalar(f'epoch/{k}', v, epoch)
            
            # Save checkpoint
            if (epoch + 1) % self.config.get('save_every', 10) == 0:
                self._save_checkpoint(epoch, metrics=losses)
                print(f"  Saved checkpoint: epoch {epoch}")
        
        # Final checkpoint
        self._save_checkpoint(epochs - 1, metrics=losses)
        self.writer.close()
        print(f"\n{'='*50}")
        print(f"Training complete! {epochs} epochs")
        print(f"Checkpoints saved to: {self.checkpoint_dir}")
        print(f"{'='*50}")
    
    def train_with_recovery(self, dataloader: torch.utils.data.DataLoader):
        """
        Main training loop with error recovery and graceful shutdown.
        
        Features:
        - Saves checkpoint on Ctrl+C
        - Saves checkpoint every N batches
        - Catches CUDA OOM and saves checkpoint
        """
        import signal
        
        epochs = self.config.get('epochs', 100)
        save_every_batches = self.config.get('save_every_batches', 500)
        
        # Flag for graceful shutdown
        self._interrupted = False
        
        def signal_handler(signum, frame):
            print("\n\n[!]  Ctrl+C detected! Saving checkpoint...")
            self._interrupted = True
        
        # Register signal handler
        original_handler = signal.signal(signal.SIGINT, signal_handler)
        
        print(f"\n{'='*50}")
        print(f"Training started!")
        print(f"  Epochs: {self.start_epoch} -> {epochs-1}")
        print(f"  Batches/epoch: {len(dataloader)}")
        print(f"  Checkpoint saves: every {save_every_batches} batches + every epoch")
        print(f"  Press Ctrl+C to stop and save")
        print(f"{'='*50}\n")
        
        try:
            # Outer progress bar for epochs
            epoch_pbar = tqdm(
                range(self.start_epoch, epochs),
                desc="Training",
                unit="epoch",
                position=0,
            )
            
            for epoch in epoch_pbar:
                if self._interrupted:
                    break
                
                # Track current epoch for two-phase training
                self.current_epoch = epoch
                
                # Phase 2 activation: switch from reconstruction-only to GAN
                if epoch == self.gan_start_epoch and self.gan_start_epoch > 0 and not self.gan_activated:
                    self.gan_activated = True
                    print(f"\n[>>] PHASE 2 ACTIVATED (epoch {epoch}): GAN training enabled!")
                    if self.lr_g_phase2 is not None:
                        for param_group in self.optimizer_g.param_groups:
                            param_group['lr'] = self.lr_g_phase2
                        print(f"   lr_g reduced to {self.lr_g_phase2} for fine-tuning")
                    
                    # Rebuild dataloader with smaller batch to fit discriminator in VRAM
                    if self.phase2_batch_size is not None and self._data_path is not None:
                        old_bs = self.config.get('batch_size', 6)
                        print(f"   Reducing batch_size {old_bs} → {self.phase2_batch_size} for Phase 2 VRAM")
                        print(f"   Gradient accumulation: {self.gradient_accumulation_steps}x → effective batch = {self.phase2_batch_size * self.gradient_accumulation_steps}")
                        self.config['batch_size'] = self.phase2_batch_size
                        dataloader = create_dataloader(
                            root_dir=self._data_path,
                            batch_size=self.phase2_batch_size,
                            num_workers=self.config.get('num_workers', 4),
                            crop_size=tuple(self.config.get('crop_size', [256, 256])),
                            prefetch_factor=self.config.get('prefetch_factor', 2),
                            persistent_workers=self.config.get('persistent_workers', False),
                            max_samples=self.config.get('max_samples', None),
                        )
                        print(f"   New dataloader: {len(dataloader)} batches/epoch")
                    
                    # === Discriminator Warmup ===
                    # Train D alone before GAN loss flows into G
                    if self.d_warmup_batches > 0:
                        print(f"   [>>] D warmup: training discriminator alone for {self.d_warmup_batches} batches...")
                        warmup_pbar = tqdm(
                            range(self.d_warmup_batches),
                            desc='D Warmup',
                            leave=False,
                            miniters=50,
                            mininterval=5,
                        )
                        data_iter = iter(dataloader)
                        warmup_d_losses = []
                        warmup_d_accs = []
                        for wi in warmup_pbar:
                            try:
                                batch = next(data_iter)
                            except StopIteration:
                                data_iter = iter(dataloader)
                                batch = next(data_iter)
                            
                            frame1 = batch['frame1'].to(self.device)
                            frame2 = batch['frame2'].to(self.device)
                            frame3 = batch['frame3'].to(self.device)
                            
                            self.optimizer_d.zero_grad()
                            with torch.amp.autocast('cuda', enabled=self.use_amp):
                                with torch.no_grad():
                                    output = self.generator(frame1, frame3)
                                    fake_frame = output['output']
                                pred_real = self.discriminator(frame2)
                                loss_d_real = self.gan_loss(pred_real, is_real=True)
                                pred_fake = self.discriminator(fake_frame.detach())
                                loss_d_fake = self.gan_loss(pred_fake, is_real=False)
                                loss_d = (loss_d_real + loss_d_fake) * 0.5
                            
                            self.scaler.scale(loss_d).backward()
                            if self.grad_clip > 0:
                                self.scaler.unscale_(self.optimizer_d)
                                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.grad_clip)
                            self.scaler.step(self.optimizer_d)
                            self.scaler.update()
                            
                            with torch.no_grad():
                                d_acc = ((pred_real.mean() > 0.5).float().item() + (pred_fake.mean() < 0.5).float().item()) / 2
                            warmup_d_losses.append(loss_d.item())
                            warmup_d_accs.append(d_acc)
                            
                            if (wi + 1) % 100 == 0:
                                warmup_pbar.set_postfix_str(
                                    f"d_loss:{sum(warmup_d_losses[-100:])/min(100,len(warmup_d_losses)):.3f}  "
                                    f"d_acc:{sum(warmup_d_accs[-100:])/min(100,len(warmup_d_accs)):.0%}"
                                )
                        
                        avg_d_loss = sum(warmup_d_losses) / len(warmup_d_losses)
                        avg_d_acc = sum(warmup_d_accs) / len(warmup_d_accs)
                        print(f"   [OK] D warmup complete: avg_loss={avg_d_loss:.3f}, avg_acc={avg_d_acc:.0%}")
                    
                    self.d_warmup_done = True
                    print(f"   [OK] GAN loss now active for Generator")
                    
                # Train one epoch with batch checkpointing
                losses = self._train_epoch_with_recovery(
                    dataloader, epoch, save_every_batches
                )
                
                if self._interrupted:
                    break
                
                # Step schedulers
                self.scheduler_g.step()
                self.scheduler_d.step()
                
                # Update outer progress bar
                epoch_pbar.set_postfix({
                    "g_loss": f"{losses['g_total']:.4f}",
                    "d_loss": f"{losses['d_total']:.4f}",
                    "lr": f"{self.scheduler_g.get_last_lr()[0]:.2e}",
                })
                
                # Log epoch summary
                print(f"\nEpoch {epoch}/{epochs-1} complete:")
                epoch_wandb_metrics = {}
                for k, v in losses.items():
                    print(f"  {k}: {v:.4f}")
                    self.writer.add_scalar(f'epoch/{k}', v, epoch)
                    epoch_wandb_metrics[f'epoch/{k}'] = v
                if self.wandb:
                    epoch_wandb_metrics['epoch'] = epoch
                    self.wandb.log(epoch_wandb_metrics, step=self.global_step)
                
                # Save epoch checkpoint
                self._save_checkpoint(epoch, metrics=losses)
                print(f"   Checkpoint saved: epoch {epoch}")
                
                # Early stopping check based on PSNR
                if self.early_stop_patience > 0 and 'psnr' in losses:
                    current_psnr = losses['psnr']
                    if current_psnr > self.best_psnr + self.min_psnr_delta:
                        self.best_psnr = current_psnr
                        self.epochs_without_improvement = 0
                        print(f"  [UP] New best PSNR: {self.best_psnr:.2f}")
                    else:
                        self.epochs_without_improvement += 1
                        print(f"  [...] No PSNR improvement for {self.epochs_without_improvement}/{self.early_stop_patience} epochs")
                        
                        if self.epochs_without_improvement >= self.early_stop_patience:
                            print(f"\n[STOP]  Early stopping: PSNR hasn't improved for {self.early_stop_patience} epochs")
                            print(f"   Best PSNR: {self.best_psnr:.2f}")
                            break
        
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "CUDA" in str(e):
                self._oom_retry_count = getattr(self, '_oom_retry_count', 0) + 1
                max_retries = self.config.get('oom_max_retries', 3)
                
                if self._oom_retry_count <= max_retries:
                    print(f"\n[!]  CUDA Error detected! Attempt {self._oom_retry_count}/{max_retries}")
                    print(f"   Saving checkpoint and clearing GPU memory...")
                    
                    # Try to save checkpoint
                    try:
                        self._save_checkpoint(epoch if 'epoch' in dir() else self.start_epoch, is_best=False)
                        print(f"    Checkpoint saved")
                    except Exception as save_error:
                        print(f"   [!] Could not save checkpoint: {save_error}")
                    
                    # Clear GPU memory
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
                    # Wait for GPU to stabilize
                    import time
                    wait_time = 10 * self._oom_retry_count  # 10s, 20s, 30s
                    print(f"   Waiting {wait_time}s for GPU to stabilize...")
                    time.sleep(wait_time)
                    
                    # Log memory status
                    if torch.cuda.is_available():
                        mem_allocated = torch.cuda.memory_allocated() / 1024**3
                        mem_reserved = torch.cuda.memory_reserved() / 1024**3
                        print(f"   GPU Memory: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")
                    
                    print(f"    Auto-resuming training...")
                    
                    # Reload from checkpoint and retry
                    checkpoint_path = self.checkpoint_dir / "checkpoint_latest.pt"
                    if checkpoint_path.exists():
                        self._load_checkpoint(str(checkpoint_path))
                        # Recursive call with fresh state
                        return self.train_with_recovery(dataloader)
                    else:
                        print(f"    No checkpoint found to resume from")
                        raise
                else:
                    print(f"\n CUDA Error after {max_retries} retries. Saving and exiting...")
                    try:
                        self._save_checkpoint(epoch if 'epoch' in dir() else self.start_epoch, is_best=False)
                    except:
                        pass
                    print(f" Try reducing batch_size in config or restart your computer")
                    raise
            else:
                print(f"\n Error: {e}")
                try:
                    self._save_checkpoint(epoch if 'epoch' in dir() else self.start_epoch, is_best=False)
                except:
                    pass
                raise
        
        finally:
            # Restore original signal handler
            signal.signal(signal.SIGINT, original_handler)
            
            # Save final checkpoint
            if self._interrupted:
                self._save_checkpoint(epoch if 'epoch' in dir() else self.start_epoch)
                print(f"\n Checkpoint saved on interrupt!")
                print(f"  Resume with: --resume {self.checkpoint_dir}/checkpoint_latest.pt")
            
            self.writer.close()
        
        if not self._interrupted:
            print(f"\n{'='*50}")
            print(f" Training complete! {epochs} epochs")
            print(f"  Checkpoints: {self.checkpoint_dir}")
            print(f"{'='*50}")
    
    def _train_epoch_with_recovery(
        self,
        dataloader: torch.utils.data.DataLoader,
        epoch: int,
        save_every_batches: int,
    ) -> dict[str, float]:
        """Train one epoch with batch checkpointing."""
        import time
        
        self.generator.train()
        self.discriminator.train()
        
        # Store batches_per_epoch for checkpoint resume
        self._batches_per_epoch = len(dataloader)
        total_batches = len(dataloader)
        
        epoch_losses = {}
        batches_processed = 0
        
        # Running averages for smoother display
        running_psnr = []
        running_g_loss = []
        running_d_loss = []
        avg_window = 100  # Average over last 100 batches
        
        # Timing
        epoch_start_time = time.time()
        batch_times = []
        
        # Calculate how many batches to skip on resume (first epoch only)
        skip_batches = 0
        if epoch == self.start_epoch and self.start_batch > 0:
            skip_batches = self.start_batch
            print(f"   Skipping first {skip_batches} batches (already processed)")
        
        # Calculate total epochs for overall progress
        total_epochs = self.config.get('epochs', 100)
        epochs_done = epoch - self.start_epoch
        epochs_remaining = total_epochs - epoch - 1
        
        # Progress reporting interval (prints one clean line every N batches)
        report_every = 200
        
        print(f"  Epoch {epoch}/{total_epochs-1}: {total_batches} batches", flush=True)
        
        for batch_idx, batch in enumerate(dataloader):
            if self._interrupted:
                break
            
            # Skip already-processed batches on resume
            if batch_idx < skip_batches:
                if batch_idx % 1000 == 0:
                    print(f"    Skipping {batch_idx}/{skip_batches}...")
                continue
            
            batch_start = time.time()
            losses = self.train_step(batch)
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            
            self.global_step += 1
            batches_processed += 1
            
            # Accumulate losses
            for k, v in losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0) + v
            
            # Update running averages
            running_psnr.append(losses['psnr'])
            running_g_loss.append(losses['g_total'])
            running_d_loss.append(losses['d_total'])
            if len(running_psnr) > avg_window:
                running_psnr.pop(0)
                running_g_loss.pop(0)
                running_d_loss.pop(0)
            
            avg_psnr = sum(running_psnr) / len(running_psnr)
            avg_g = sum(running_g_loss) / len(running_g_loss)
            avg_d = sum(running_d_loss) / len(running_d_loss)
            
            # Periodic progress report (clean single line, no tqdm)
            if (batch_idx + 1) % report_every == 0 or batch_idx == total_batches - 1:
                # Calculate ETA
                avg_batch_time = sum(batch_times[-100:]) / len(batch_times[-100:])
                remaining_batches = total_batches - batch_idx - 1
                eta_epoch_sec = remaining_batches * avg_batch_time
                eta_total_sec = eta_epoch_sec + (epochs_remaining * total_batches * avg_batch_time)
                
                # Format ETA
                def format_time(seconds):
                    if seconds < 60:
                        return f"{seconds:.0f}s"
                    elif seconds < 3600:
                        return f"{seconds/60:.0f}m"
                    else:
                        hours = seconds // 3600
                        mins = (seconds % 3600) // 60
                        return f"{hours:.0f}h{mins:.0f}m"
                
                # GPU memory
                if torch.cuda.is_available():
                    mem_gb = torch.cuda.memory_reserved() / 1024**3
                    mem_str = f"{mem_gb:.1f}GB"
                else:
                    mem_str = "N/A"
                
                samples_per_sec = self.config.get('batch_size', 1) / avg_batch_time
                pct = 100 * (batch_idx + 1) / total_batches
                
                print(
                    f"    [{batch_idx+1}/{total_batches} {pct:.0f}%] "
                    f"PSNR:{avg_psnr:.1f}  G:{avg_g:.3f} D:{avg_d:.3f}  "
                    f"d_acc:{losses['d_acc']:.0%}  grad:{losses.get('grad_norm', 0):.1f}  "
                    f"GPU:{mem_str}  {samples_per_sec:.1f}samp/s  "
                    f"ETA:{format_time(eta_epoch_sec)}(ep)/{format_time(eta_total_sec)}(tot)",
                    flush=True
                )
            
            # Log to tensorboard + wandb
            if self.global_step % 100 == 0:
                for k, v in losses.items():
                    self.writer.add_scalar(f'train/{k}', v, self.global_step)
                if self.wandb:
                    self.wandb.log({
                        f'train/{k}': v for k, v in losses.items()
                    }, step=self.global_step)
            
            # Batch checkpoint
            if save_every_batches > 0 and (batch_idx + 1) % save_every_batches == 0:
                self._save_checkpoint(epoch)
                print(f"    Batch checkpoint saved ({batch_idx + 1}/{total_batches})")
        
        # Average losses (use actual batches processed, not total)
        if batches_processed > 0:
            for k in epoch_losses:
                epoch_losses[k] /= batches_processed
        
        # Print epoch summary with timing
        epoch_time = time.time() - epoch_start_time
        print(f"\n    Epoch time: {epoch_time/60:.1f} min ({batches_processed} batches)")
        
        # Clear start_batch after first resume epoch
        if skip_batches > 0:
            self.start_batch = 0
        
        return epoch_losses


def main():
    parser = argparse.ArgumentParser(description='Train LayeredInterpolator')
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/interp_training.yaml',
        help='Path to config file',
    )
    parser.add_argument(
        '--resume', '-r',
        type=str,
        default=None,
        help='Path to checkpoint to resume from',
    )
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        help='Automatically resume from latest checkpoint if exists',
    )
    parser.add_argument(
        '--data', '-d',
        type=str,
        required=True,
        help='Path to triplet dataset directory',
    )
    parser.add_argument(
        '--wandb',
        action='store_true',
        help='Enable Weights & Biases logging',
    )
    
    args = parser.parse_args()
    
    # Load config
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        print(f"Config not found: {args.config}, using defaults")
        config = {}
    
    # Auto-resume: find latest checkpoint
    resume_path = args.resume
    if args.auto_resume and resume_path is None:
        checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        latest_checkpoint = checkpoint_dir / 'checkpoint_latest.pt'
        if latest_checkpoint.exists():
            resume_path = str(latest_checkpoint)
            print(f"Auto-resuming from: {resume_path}")
    
    # Enable cuDNN benchmark for faster training (finds optimal algorithms)
    if config.get('cudnn_benchmark', True):
        torch.backends.cudnn.benchmark = True
        print("cuDNN benchmark enabled for optimal performance")
    
    # Create dataloader with optimizations
    dataloader = create_dataloader(
        root_dir=args.data,
        batch_size=config.get('batch_size', 8),
        num_workers=config.get('num_workers', 4),
        crop_size=tuple(config.get('crop_size', [256, 256])),
        prefetch_factor=config.get('prefetch_factor', 2),
        persistent_workers=config.get('persistent_workers', False),
        max_samples=config.get('max_samples', None),
    )
    
    print(f"Dataset size: {len(dataloader.dataset)} triplets")
    print(f"Batches per epoch: {len(dataloader)}")
    
    # Enable W&B if flag is set
    if args.wandb:
        config['use_wandb'] = True
    
    # Store data path in config for Phase 2 dataloader rebuild
    config['data_dir'] = args.data
    
    # Create trainer and train with recovery
    trainer = Trainer(config, resume_from=resume_path)
    trainer.train_with_recovery(dataloader)


if __name__ == '__main__':
    main()

