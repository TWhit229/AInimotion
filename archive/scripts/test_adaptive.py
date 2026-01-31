"""Test adaptive training features."""
import sys

def main():
    print("Testing adaptive training features...")
    
    # Test GANLoss with label smoothing
    from ainimotion.training.discriminator import GANLoss
    loss = GANLoss(loss_type='lsgan', label_smoothing=0.1)
    print(f"GANLoss with label_smoothing={loss.label_smoothing}")
    
    # Test config loading
    import yaml
    with open('configs/finetune_training.yaml') as f:
        config = yaml.safe_load(f)
    
    print(f"Fine-tune config loaded:")
    print(f"  label_smoothing: {config.get('label_smoothing')}")
    print(f"  d_throttle_threshold: {config.get('d_throttle_threshold')}")
    print(f"  early_stop_patience: {config.get('early_stop_patience')}")
    print(f"  lr_g: {config.get('lr_g')}")
    print(f"  lr_d: {config.get('lr_d')}")
    
    print("All imports and config OK!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
