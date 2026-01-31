"""
Training loop for the LayeredInterpolator model.

Supports:
- GAN training with discriminator
- Mixed precision (FP16)
- Checkpointing and resumption
- TensorBoard logging
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
        ).to(self.device)
        
        self.gan_loss = GANLoss(
            loss_type=config.get('gan_type', 'lsgan'),
            label_smoothing=config.get('label_smoothing', 0.0),
        ).to(self.device)
        
        self.gan_weight = config.get('gan_weight', 0.01)
        
        # Adaptive training settings
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
        """Load checkpoint and resume training."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
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
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'batches_per_epoch': getattr(self, '_batches_per_epoch', 0),
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
        
        # ============ Train Discriminator ============
        # Check if we should throttle (skip) discriminator updates
        skip_d_update = False
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
        
        # ============ Train Generator ============
        self.optimizer_g.zero_grad()
        
        with torch.amp.autocast('cuda', enabled=self.use_amp):
            # Forward pass
            output = self.generator(frame1, frame3)
            fake_frame = output['output']
            
            # Reconstruction loss
            loss_vfi, vfi_components = self.vfi_loss(
                fake_frame, frame2, return_components=True
            )
            
            # GAN loss (fool discriminator)
            pred_fake = self.discriminator(fake_frame)
            loss_g_gan = self.gan_loss(
                pred_fake, is_real=True, for_discriminator=False
            )
            
            # Total generator loss
            loss_g = loss_vfi + self.gan_weight * loss_g_gan
        
        self.scaler.scale(loss_g).backward()
        
        # Unscale gradients for clipping and norm computation
        self.scaler.unscale_(self.optimizer_g)
        
        # Compute gradient norm before clipping
        grad_norm_g = 0.0
        for p in self.generator.parameters():
            if p.grad is not None:
                grad_norm_g += p.grad.data.norm(2).item() ** 2
        grad_norm_g = grad_norm_g ** 0.5
        
        # Gradient clipping for generator
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.grad_clip)
        
        self.scaler.step(self.optimizer_g)
        self.scaler.update()
        
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
        losses['g_total'] = loss_g.item()
        losses['grad_norm'] = grad_norm_g
        losses['psnr'] = psnr
        losses['scene_cut_rate'] = scene_cut_rate
        losses['lr'] = self.scheduler_g.get_last_lr()[0]
        
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
                self._save_checkpoint(epoch)
                print(f"  Saved checkpoint: epoch {epoch}")
        
        # Final checkpoint
        self._save_checkpoint(epochs - 1)
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
            print("\n\n⚠️  Ctrl+C detected! Saving checkpoint...")
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
                for k, v in losses.items():
                    print(f"  {k}: {v:.4f}")
                    self.writer.add_scalar(f'epoch/{k}', v, epoch)
                
                # Save epoch checkpoint
                self._save_checkpoint(epoch)
                print(f"  ✓ Checkpoint saved: epoch {epoch}")
                
                # Early stopping check based on PSNR
                if self.early_stop_patience > 0 and 'psnr' in losses:
                    current_psnr = losses['psnr']
                    if current_psnr > self.best_psnr + self.min_psnr_delta:
                        self.best_psnr = current_psnr
                        self.epochs_without_improvement = 0
                        print(f"  📈 New best PSNR: {self.best_psnr:.2f}")
                    else:
                        self.epochs_without_improvement += 1
                        print(f"  ⏳ No PSNR improvement for {self.epochs_without_improvement}/{self.early_stop_patience} epochs")
                        
                        if self.epochs_without_improvement >= self.early_stop_patience:
                            print(f"\n⏹️  Early stopping: PSNR hasn't improved for {self.early_stop_patience} epochs")
                            print(f"   Best PSNR: {self.best_psnr:.2f}")
                            break
        
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "CUDA" in str(e):
                self._oom_retry_count = getattr(self, '_oom_retry_count', 0) + 1
                max_retries = self.config.get('oom_max_retries', 3)
                
                if self._oom_retry_count <= max_retries:
                    print(f"\n⚠️  CUDA Error detected! Attempt {self._oom_retry_count}/{max_retries}")
                    print(f"   Saving checkpoint and clearing GPU memory...")
                    
                    # Try to save checkpoint
                    try:
                        self._save_checkpoint(epoch if 'epoch' in dir() else self.start_epoch, is_best=False)
                        print(f"   ✓ Checkpoint saved")
                    except Exception as save_error:
                        print(f"   ⚠️ Could not save checkpoint: {save_error}")
                    
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
                    
                    print(f"   🔄 Auto-resuming training...")
                    
                    # Reload from checkpoint and retry
                    checkpoint_path = self.checkpoint_dir / "checkpoint_latest.pt"
                    if checkpoint_path.exists():
                        self._load_checkpoint(str(checkpoint_path))
                        # Recursive call with fresh state
                        return self.train_with_recovery(dataloader)
                    else:
                        print(f"   ❌ No checkpoint found to resume from")
                        raise
                else:
                    print(f"\n❌ CUDA Error after {max_retries} retries. Saving and exiting...")
                    try:
                        self._save_checkpoint(epoch if 'epoch' in dir() else self.start_epoch, is_best=False)
                    except:
                        pass
                    print(f"💡 Try reducing batch_size in config or restart your computer")
                    raise
            else:
                print(f"\n❌ Error: {e}")
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
                print(f"\n✓ Checkpoint saved on interrupt!")
                print(f"  Resume with: --resume {self.checkpoint_dir}/checkpoint_latest.pt")
            
            self.writer.close()
        
        if not self._interrupted:
            print(f"\n{'='*50}")
            print(f"✓ Training complete! {epochs} epochs")
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
            print(f"  ⏩ Skipping first {skip_batches} batches (already processed)")
        
        # Calculate total epochs for overall progress
        total_epochs = self.config.get('epochs', 100)
        epochs_done = epoch - self.start_epoch
        epochs_remaining = total_epochs - epoch - 1
        
        pbar = tqdm(
            dataloader, 
            desc=f'Epoch {epoch}/{total_epochs-1}',
            leave=False,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
        )
        
        for batch_idx, batch in enumerate(pbar):
            if self._interrupted:
                break
            
            # Skip already-processed batches on resume
            if batch_idx < skip_batches:
                if batch_idx % 1000 == 0:
                    pbar.set_postfix_str(f'⏩ Skipping {batch_idx}/{skip_batches}')
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
            
            # Get GPU memory
            if torch.cuda.is_available():
                mem_gb = torch.cuda.memory_allocated() / 1024**3
                mem_str = f"{mem_gb:.1f}GB"
            else:
                mem_str = "N/A"
            
            # Samples per second
            samples_per_sec = self.config.get('batch_size', 1) / avg_batch_time
            
            # Build detailed progress string
            pbar.set_postfix_str(
                f"PSNR:{avg_psnr:.1f} │ G:{avg_g:.3f} D:{avg_d:.3f} │ "
                f"d_acc:{losses['d_acc']:.0%} │ grad:{losses.get('grad_norm', 0):.1f} │ "
                f"GPU:{mem_str} │ {samples_per_sec:.1f}samp/s │ "
                f"ETA:{format_time(eta_epoch_sec)}(ep)/{format_time(eta_total_sec)}(tot)"
            )
            
            # Log to tensorboard
            if self.global_step % 100 == 0:
                for k, v in losses.items():
                    self.writer.add_scalar(f'train/{k}', v, self.global_step)
            
            # Batch checkpoint
            if save_every_batches > 0 and (batch_idx + 1) % save_every_batches == 0:
                self._save_checkpoint(epoch)
                pbar.write(f"  ✓ Batch checkpoint saved ({batch_idx + 1}/{total_batches})")
        
        # Average losses (use actual batches processed, not total)
        if batches_processed > 0:
            for k in epoch_losses:
                epoch_losses[k] /= batches_processed
        
        # Print epoch summary with timing
        epoch_time = time.time() - epoch_start_time
        print(f"\n  ⏱️  Epoch time: {epoch_time/60:.1f} min ({batches_processed} batches)")
        
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
    
    # Create trainer and train with recovery
    trainer = Trainer(config, resume_from=resume_path)
    trainer.train_with_recovery(dataloader)


if __name__ == '__main__':
    main()

