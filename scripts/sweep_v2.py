"""
W&B Hyperparameter Sweep for AInimotion v2.

Runs short training sessions (10 epochs) across different hyperparameter
combinations to find the optimal configuration before full training.

Usage:
    # First, run VRAM benchmark to find max batch size:
    python scripts/benchmark_vram_v2.py

    # Then create and run the sweep:
    python scripts/sweep_v2.py --data training_data/train_10k

    # Or run a single configuration manually:
    python scripts/sweep_v2.py --data training_data/train_10k --no-sweep --batch-size 10
"""

import argparse
import gc
import os
import sys
from copy import deepcopy
from pathlib import Path

import torch
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import wandb
from ainimotion.training.train import Trainer
from ainimotion.data.dataset import create_dataloader


# ============ Sweep Configuration ============

SWEEP_CONFIG = {
    "method": "bayes",  # Bayesian optimization
    "metric": {
        "name": "epoch/psnr",
        "goal": "maximize",
    },
    "parameters": {
        # Batch size — must fit worst-case kernel_size in 31.5GB VRAM.
        # K=5 at bs=14 was ~27GB, K=7 at bs=14 was 34.5GB (overflowed!).
        # Safe: bs=8 fits any kernel size comfortably.
        "batch_size": {
            "values": [6, 8, 10],
        },
        # Learning rates
        "lr_g": {
            "distribution": "log_uniform_values",
            "min": 3e-5,
            "max": 3e-4,
        },
        # Loss weights
        "freq_weight": {
            "values": [0.0, 0.25, 0.5, 1.0],
        },
        "edge_weight": {
            "values": [0.25, 0.5, 1.0],
        },
        "census_weight": {
            "values": [0.5, 1.0, 2.0],
        },
        # Architecture — K=3 vs K=5 (K=7 too expensive, marginal gain)
        "kernel_size": {
            "values": [3, 5],
        },
        # Weight decay
        "weight_decay": {
            "values": [0.001, 0.01, 0.05],
        },
    },
}

# Base config — loaded from YAML, sweep overrides specific fields
BASE_CONFIG_PATH = "configs/interp_training_5090.yaml"
SWEEP_EPOCHS = 5   # Short runs — enough to see convergence direction
SWEEP_SAMPLES = 2000  # Subset of data for fast trials (~3 min/epoch vs 50 min)


def load_base_config() -> dict:
    """Load the base training config."""
    with open(BASE_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def run_sweep_trial(data_dir: str):
    """Run a single sweep trial. Called by W&B agent."""
    # Initialize W&B run
    run = wandb.init()
    sweep_params = dict(wandb.config)
    
    print(f"\n{'='*60}")
    print(f"  Sweep Trial: {run.name}")
    print(f"  Parameters: {sweep_params}")
    print(f"{'='*60}\n")
    
    # Build config from base + sweep overrides
    config = load_base_config()
    config.update(sweep_params)
    config['epochs'] = SWEEP_EPOCHS
    config['data_dir'] = data_dir
    # Don't call wandb.init in Trainer — sweep agent already did it
    # Instead, pass the existing wandb module so Trainer can log
    config['use_wandb'] = True
    config['wandb_project'] = 'ainimotion-v2'
    config['max_samples'] = SWEEP_SAMPLES  # Fast trials on data subset
    config['save_every'] = SWEEP_EPOCHS  # Only save at end
    config['save_every_batches'] = 999999  # Disable mid-epoch saves
    config['gan_start_epoch'] = 999  # No GAN in sweep (Phase 1 only)
    
    # Separate checkpoint dir per sweep run
    config['checkpoint_dir'] = f"sweep_checkpoints/{run.id}"
    config['log_dir'] = f"sweep_runs/{run.id}"
    
    try:
        # Create dataloader
        dataloader = create_dataloader(
            root_dir=data_dir,
            batch_size=config['batch_size'],
            num_workers=config.get('num_workers', 8),
            crop_size=tuple(config.get('crop_size', [256, 256])),
            prefetch_factor=config.get('prefetch_factor', 2),
            persistent_workers=config.get('persistent_workers', False),
            max_samples=config.get('max_samples', None),
        )
        
        print(f"Dataset: {len(dataloader.dataset)} triplets, {len(dataloader)} batches/epoch")
        
        # Train
        trainer = Trainer(config)
        trainer.train_with_recovery(dataloader)
        
    except torch.cuda.OutOfMemoryError:
        print(f"OOM at batch_size={config['batch_size']}! Marking as failed.")
        torch.cuda.empty_cache()
        wandb.log({"epoch/psnr": 0.0, "oom": True})
    except Exception as e:
        print(f"Trial failed: {e}")
        wandb.log({"epoch/psnr": 0.0, "error": str(e)})
    finally:
        # Aggressive GPU cleanup between trials
        if 'trainer' in locals():
            del trainer
        if 'dataloader' in locals():
            del dataloader
            
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()  # Double-tap after gc
        
        # Clean up sweep checkpoints to save disk
        sweep_ckpt = Path(f"sweep_checkpoints/{run.id}")
        if sweep_ckpt.exists():
            import shutil
            shutil.rmtree(sweep_ckpt, ignore_errors=True)
        sweep_run = Path(f"sweep_runs/{run.id}")
        if sweep_run.exists():
            import shutil
            shutil.rmtree(sweep_run, ignore_errors=True)
        
        wandb.finish()


def run_single(data_dir: str, batch_size: int | None = None):
    """Run a single training with the base config (no sweep)."""
    config = load_base_config()
    config['data_dir'] = data_dir
    config['use_wandb'] = True
    
    if batch_size:
        config['batch_size'] = batch_size
    
    dataloader = create_dataloader(
        root_dir=data_dir,
        batch_size=config['batch_size'],
        num_workers=config.get('num_workers', 8),
        crop_size=tuple(config.get('crop_size', [256, 256])),
        prefetch_factor=config.get('prefetch_factor', 2),
        persistent_workers=config.get('persistent_workers', False),
    )
    
    print(f"Dataset: {len(dataloader.dataset)} triplets")
    print(f"Batch size: {config['batch_size']}")
    print(f"Batches/epoch: {len(dataloader)}")
    
    trainer = Trainer(config)
    trainer.train_with_recovery(dataloader)


def main():
    parser = argparse.ArgumentParser(description="AInimotion v2 W&B Sweep")
    parser.add_argument(
        "--data", "-d",
        default="training_data/train_10k",
        help="Path to training data",
    )
    parser.add_argument(
        "--no-sweep",
        action="store_true",
        help="Run single training instead of sweep",
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=None,
        help="Override batch size (for single run)",
    )
    parser.add_argument(
        "--count", "-c",
        type=int,
        default=20,
        help="Number of sweep trials (default: 20)",
    )
    parser.add_argument(
        "--sweep-id",
        type=str,
        default=None,
        help="Resume an existing W&B sweep by ID",
    )
    args = parser.parse_args()
    
    if args.no_sweep:
        # Single run with base config
        run_single(args.data, args.batch_size)
    else:
        # W&B sweep
        if args.sweep_id:
            sweep_id = args.sweep_id
            print(f"Resuming sweep: {sweep_id}")
        else:
            sweep_id = wandb.sweep(
                SWEEP_CONFIG,
                project="ainimotion-v2",
            )
            print(f"Created sweep: {sweep_id}")
        
        wandb.agent(
            sweep_id,
            function=lambda: run_sweep_trial(args.data),
            count=args.count,
            project="ainimotion-v2",
        )


if __name__ == "__main__":
    main()
