"""
Hyperparameter Probe Runner

Quickly test different training configurations to find optimal settings.
Each probe takes ~30-45 minutes with 5K samples and 3 epochs.

Usage:
    python scripts/run_probes.py

Results are logged to runs/probe_* directories.
"""

import subprocess
import sys
import yaml
import shutil
from pathlib import Path
from datetime import datetime


# Define probe configurations to test
PROBES = {
    "probe_baseline": {
        # Baseline with V2 settings
        "base_channels": 48,
        "label_smoothing": 0.15,
        "d_throttle_threshold": 0.85,
        "lr_g": 0.0001,
        "lr_d": 0.00002,
        "gan_weight": 0.05,
    },
    "probe_aggressive_d_weaken": {
        # Even more D weakening
        "base_channels": 48,
        "label_smoothing": 0.2,          # Higher smoothing
        "d_throttle_threshold": 0.80,     # Earlier throttle
        "lr_g": 0.0001,
        "lr_d": 0.00001,                  # Even lower D LR
        "gan_weight": 0.05,
    },
    "probe_higher_gan_weight": {
        # Stronger adversarial signal
        "base_channels": 48,
        "label_smoothing": 0.15,
        "d_throttle_threshold": 0.85,
        "lr_g": 0.0001,
        "lr_d": 0.00002,
        "gan_weight": 0.1,                # Double GAN weight
    },
    "probe_smaller_model": {
        # Original model size with new D settings
        "base_channels": 32,
        "label_smoothing": 0.15,
        "d_throttle_threshold": 0.85,
        "lr_g": 0.0001,
        "lr_d": 0.00002,
        "gan_weight": 0.05,
    },
}


def load_base_config():
    """Load the probe base config."""
    with open("configs/probe_training.yaml") as f:
        return yaml.safe_load(f)


def run_probe(probe_name: str, overrides: dict):
    """Run a single probe configuration."""
    print(f"\n{'='*60}")
    print(f"PROBE: {probe_name}")
    print(f"{'='*60}")
    
    # Load base config and apply overrides
    config = load_base_config()
    config.update(overrides)
    
    # Set unique output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config["log_dir"] = f"runs/probe_{probe_name}_{timestamp}"
    config["checkpoint_dir"] = f"checkpoints/probe_{probe_name}_{timestamp}"
    
    # Write temporary config
    temp_config = Path(f"configs/_temp_{probe_name}.yaml")
    with open(temp_config, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Settings: {overrides}")
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
        # Cleanup temp config
        temp_config.unlink(missing_ok=True)
    
    return success


def main():
    print("="*60)
    print("HYPERPARAMETER PROBE RUNNER")
    print("="*60)
    print(f"\nProbes to run: {list(PROBES.keys())}")
    print("Each probe: ~30-45 minutes (5K samples, 3 epochs)")
    print(f"Total estimated time: ~{len(PROBES) * 0.5}-{len(PROBES) * 0.75} hours")
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
    print("PROBE RESULTS SUMMARY")
    print("="*60)
    for name, status in results.items():
        print(f"  {name}: {status}")
    
    print("\nCheck TensorBoard for detailed metrics:")
    print("  tensorboard --logdir runs/")


if __name__ == "__main__":
    main()
