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
            loss_type=config.get('gan_type', 'lsgan')
        ).to(self.device)
        
        self.gan_weight = config.get('gan_weight', 0.01)
        
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
        self.start_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        
        print(f"Resumed from epoch {self.start_epoch}")
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
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
        
        # Compute gradient norm before stepping
        grad_norm_g = 0.0
        for p in self.generator.parameters():
            if p.grad is not None:
                grad_norm_g += p.grad.data.norm(2).item() ** 2
        grad_norm_g = grad_norm_g ** 0.5
        
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
        
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n❌ CUDA Out of Memory! Saving checkpoint...")
                self._save_checkpoint(epoch, is_best=False)
                print(f"💡 Try reducing batch_size in config")
                raise
            else:
                print(f"\n❌ Error: {e}")
                self._save_checkpoint(epoch, is_best=False)
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
        self.generator.train()
        self.discriminator.train()
        
        epoch_losses = {}
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}', leave=False)
        for batch_idx, batch in enumerate(pbar):
            if self._interrupted:
                break
            
            losses = self.train_step(batch)
            self.global_step += 1
            
            # Accumulate losses
            for k, v in losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0) + v
            
            # Update progress bar
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
            
            # Batch checkpoint
            if save_every_batches > 0 and (batch_idx + 1) % save_every_batches == 0:
                self._save_checkpoint(epoch)
                pbar.write(f"  ✓ Batch checkpoint saved ({batch_idx + 1}/{len(dataloader)})")
        
        # Average losses
        num_batches = batch_idx + 1 if batch_idx else len(dataloader)
        for k in epoch_losses:
            epoch_losses[k] /= num_batches
        
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
    
    # Create dataloader with optimizations
    dataloader = create_dataloader(
        root_dir=args.data,
        batch_size=config.get('batch_size', 8),
        num_workers=config.get('num_workers', 4),
        crop_size=tuple(config.get('crop_size', [256, 256])),
    )
    
    # Enable pin_memory for faster GPU transfer
    dataloader.pin_memory = True
    
    print(f"Dataset size: {len(dataloader.dataset)} triplets")
    print(f"Batches per epoch: {len(dataloader)}")
    
    # Create trainer and train with recovery
    trainer = Trainer(config, resume_from=resume_path)
    trainer.train_with_recovery(dataloader)


if __name__ == '__main__':
    main()

