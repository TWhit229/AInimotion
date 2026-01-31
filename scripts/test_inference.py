"""
Quick inference test - generate interpolated frames from the trained model.
"""
import torch
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as TF
import sys

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ainimotion.models.interp import LayeredInterpolator


def load_model(checkpoint_path: str, device: torch.device):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    
    model = LayeredInterpolator(
        base_channels=config.get('base_channels', 32),
        kernel_size=config.get('kernel_size', 7),
        grid_size=config.get('grid_size', 8),
        use_refinement=config.get('use_refinement', True),
    ).to(device)
    
    model.load_state_dict(checkpoint['generator'])
    model.eval()
    
    epoch = checkpoint.get('epoch', 'unknown')
    print(f"Loaded model from epoch {epoch}")
    print(f"  Config: {config.get('base_channels', 32)} channels")
    
    return model


def interpolate_triplet(model, triplet_dir: Path, device: torch.device, output_dir: Path):
    """Interpolate a single triplet and save results."""
    # Find image extension
    ext = "png" if (triplet_dir / "f1.png").exists() else "jpg"
    
    # Load frames
    f1 = Image.open(triplet_dir / f"f1.{ext}").convert("RGB")
    f2_gt = Image.open(triplet_dir / f"f2.{ext}").convert("RGB")  # Ground truth
    f3 = Image.open(triplet_dir / f"f3.{ext}").convert("RGB")
    
    # Convert to tensors
    f1_t = TF.to_tensor(f1).unsqueeze(0).to(device)
    f3_t = TF.to_tensor(f3).unsqueeze(0).to(device)
    
    # Inference - model takes frame1 and frame2 (which is frame3 for us)
    with torch.no_grad():
        output = model(f1_t, f3_t)
        f2_pred = output['output']  # (1, C, H, W)
    
    # Convert back to PIL
    f2_pred_pil = TF.to_pil_image(f2_pred.squeeze(0).cpu().clamp(0, 1))
    
    # Calculate PSNR
    mse = torch.nn.functional.mse_loss(
        TF.to_tensor(f2_pred_pil),
        TF.to_tensor(f2_gt)
    )
    psnr = 10 * torch.log10(1.0 / (mse + 1e-8)).item()
    
    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    triplet_name = triplet_dir.name
    
    f1.save(output_dir / f"{triplet_name}_1_input_first.png")
    f2_gt.save(output_dir / f"{triplet_name}_2_ground_truth.png")
    f2_pred_pil.save(output_dir / f"{triplet_name}_3_predicted.png")
    f3.save(output_dir / f"{triplet_name}_4_input_last.png")
    
    print(f"  {triplet_name}: PSNR = {psnr:.2f} dB")
    
    return psnr


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    checkpoint_path = "checkpoints/checkpoint_latest.pt"
    model = load_model(checkpoint_path, device)
    
    # Find test triplets (grab a few random ones)
    data_dir = Path("D:/Triplets")
    triplet_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])[:10]
    
    # Test on sample triplets
    output_dir = Path("test_outputs")
    print(f"\nTesting on {len(triplet_dirs)} triplets...")
    
    psnrs = []
    for triplet_dir in triplet_dirs:
        try:
            psnr = interpolate_triplet(model, triplet_dir, device, output_dir)
            psnrs.append(psnr)
        except Exception as e:
            print(f"  {triplet_dir.name}: Error - {e}")
    
    # Summary
    if psnrs:
        avg_psnr = sum(psnrs) / len(psnrs)
        print(f"\n{'='*50}")
        print(f"Average PSNR: {avg_psnr:.2f} dB")
        print(f"Min PSNR: {min(psnrs):.2f} dB")
        print(f"Max PSNR: {max(psnrs):.2f} dB")
        print(f"{'='*50}")
        print(f"\nOutputs saved to: {output_dir.absolute()}")
        print("Compare the '_2_ground_truth.png' vs '_3_predicted.png' files!")


if __name__ == "__main__":
    main()
