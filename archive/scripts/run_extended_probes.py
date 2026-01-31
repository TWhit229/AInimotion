"""
Extended Hyperparameter Probe Runner

Longer tests (7 epochs) with more setting combinations.
Each probe takes ~1.5-2 hours with 10K samples.

Usage:
    python scripts/run_extended_probes.py
"""

import subprocess
import sys
import yaml
from pathlib import Path
from datetime import datetime


# Extended probe configurations - more combinations, longer training
PROBES = {
    "ext_32ch_high_gan": {
        # Best PSNR from round 1 + higher GAN weight
        "base_channels": 32,
        "label_smoothing": 0.15,
        "d_throttle_threshold": 0.85,
        "lr_g": 0.0001,
        "lr_d": 0.00002,
        "gan_weight": 0.1,           # Higher GAN weight
    },
    "ext_32ch_very_high_gan": {
        # Push GAN weight even higher  
        "base_channels": 32,
        "label_smoothing": 0.15,
        "d_throttle_threshold": 0.85,
        "lr_g": 0.0001,
        "lr_d": 0.00002,
        "gan_weight": 0.15,          # Even higher
    },
    "ext_32ch_equal_lr": {
        # What if G and D have same LR?
        "base_channels": 32,
        "label_smoothing": 0.15,
        "d_throttle_threshold": 0.85,
        "lr_g": 0.0001,
        "lr_d": 0.0001,              # Same as G
        "gan_weight": 0.1,
    },
    "ext_48ch_high_gan": {
        # Larger model + high GAN weight
        "base_channels": 48,
        "label_smoothing": 0.15,
        "d_throttle_threshold": 0.85,
        "lr_g": 0.0001,
        "lr_d": 0.00002,
        "gan_weight": 0.1,
    },
    "ext_32ch_aggressive_throttle": {
        # Very aggressive throttling
        "base_channels": 32,
        "label_smoothing": 0.2,
        "d_throttle_threshold": 0.75,  # Very early throttle
        "d_throttle_patience": 20,
        "lr_g": 0.0001,
        "lr_d": 0.00002,
        "gan_weight": 0.1,
    },
}

# Extended probe settings
EXTENDED_CONFIG = {
    "epochs": 7,              # Longer training
    "max_samples": 10000,     # More samples
    "save_every_batches": 500,
}


def load_base_config():
    """Load the probe base config."""
    with open("configs/probe_training.yaml") as f:
        return yaml.safe_load(f)


def run_probe(probe_name: str, overrides: dict):
    """Run a single probe configuration."""
    print(f"\n{'='*60}")
    print(f"EXTENDED PROBE: {probe_name}")
    print(f"{'='*60}")
    
    # Load base config and apply extended settings + overrides
    config = load_base_config()
    config.update(EXTENDED_CONFIG)
    config.update(overrides)
    
    # Disable early stopping for probes
    config["early_stop_patience"] = 0
    
    # Set unique output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config["log_dir"] = f"runs/ext_{probe_name}_{timestamp}"
    config["checkpoint_dir"] = f"checkpoints/ext_{probe_name}_{timestamp}"
    
    # Write temporary config
    temp_config = Path(f"configs/_temp_{probe_name}.yaml")
    with open(temp_config, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Settings: {overrides}")
    print(f"Epochs: {config['epochs']}, Samples: {config['max_samples']}")
    print(f"Log dir: {config['log_dir']}")
    print()
    
    # Run training
    cmd = [
        sys.executable, "-m", "ainimotion.training.train",
        "--config", str(temp_config),
        "--data", "D:\\Triplets",
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        success = True
    except subprocess.CalledProcessError as e:
        print(f"Probe failed with error: {e}")
        success = False
    except KeyboardInterrupt:
        print("\nProbe interrupted by user")
        success = False
    finally:
        temp_config.unlink(missing_ok=True)
    
    return success


def main():
    print("="*60)
    print("EXTENDED HYPERPARAMETER PROBE RUNNER")
    print("="*60)
    print(f"\nProbes to run: {list(PROBES.keys())}")
    print(f"Each probe: ~1.5-2 hours (10K samples, 7 epochs)")
    print(f"Total estimated time: ~{len(PROBES) * 1.5}-{len(PROBES) * 2} hours")
    print()
    
    input("Press Enter to start, or Ctrl+C to cancel...")
    
    results = {}
    for probe_name, overrides in PROBES.items():
        try:
            success = run_probe(probe_name, overrides)
            results[probe_name] = "SUCCESS" if success else "FAILED"
        except KeyboardInterrupt:
            print("\n\nStopping probe runner...")
            break
    
    print("\n" + "="*60)
    print("EXTENDED PROBE RESULTS SUMMARY")
    print("="*60)
    for name, status in results.items():
        print(f"  {name}: {status}")
    
    print("\nCheck TensorBoard for detailed metrics:")
    print("  tensorboard --logdir runs/")
    print("\nLook for:")
    print("  - d_acc stabilizing at 60-75%")
    print("  - PSNR trending upward")
    print("  - g_gan decreasing (G fooling D better)")


if __name__ == "__main__":
    main()
