"""
Diagnostic: dump all intermediate model layers to see where blurriness comes from.

For each test triplet, saves:
  - background.png (warped+stitched background)
  - foreground.png (AdaCoF synthesis)
  - alpha.png (foreground/background mask)
  - composite_raw.png (before refinement)
  - output.png (final after refinement)
  - ground_truth.png
  - comparison.png (all panels in one image)
"""

import argparse
import random
import torch
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as TF

from ainimotion.models.interp import LayeredInterpolator


def load_model(checkpoint_path: str, device: torch.device):
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
    epoch = checkpoint.get('epoch', '?')
    print(f"Loaded epoch {epoch}, base_channels={config.get('base_channels', 32)}")
    return model


def add_label(img: Image.Image, text: str) -> Image.Image:
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.rectangle([0, 0, tw + 8, th + 8], fill=(0, 0, 0))
    draw.text((4, 4), text, fill=(255, 255, 255), font=font)
    return img


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """(1,C,H,W) or (1,1,H,W) tensor to PIL."""
    t = t.squeeze(0).cpu().clamp(0, 1)
    if t.shape[0] == 1:
        t = t.repeat(3, 1, 1)  # grayscale to RGB
    return TF.to_pil_image(t)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", "-c", default="checkpoints/checkpoint_latest.pt")
    parser.add_argument("--data", "-d", default="training_data/train_10k")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--output", "-o", default="test_outputs_diagnostic")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    
    # Find triplets
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Data directory '{data_path}' does not exist.")
        return
        
    triplet_dirs = sorted([
        d for d in data_path.iterdir()
        if d.is_dir() and all((d / f"frame{i}.jpg").exists() for i in (1, 2, 3))
    ])
    
    if not triplet_dirs:
        print(f"No valid triplets found in {data_path}")
        return
        
    random.seed(args.seed)
    selected = random.sample(triplet_dirs, min(args.n, len(triplet_dirs)))
    
    out_path = Path(args.output)
    out_path.mkdir(parents=True, exist_ok=True)
    
    for i, triplet_dir in enumerate(selected):
        name = triplet_dir.name
        print(f"\n[{i+1}/{len(selected)}] {name}")
        
        # Load
        f1 = Image.open(triplet_dir / "frame1.jpg").convert("RGB")
        f2_gt = Image.open(triplet_dir / "frame2.jpg").convert("RGB")
        f3 = Image.open(triplet_dir / "frame3.jpg").convert("RGB")
        
        f1_t = TF.to_tensor(f1).unsqueeze(0).to(device)
        f3_t = TF.to_tensor(f3).unsqueeze(0).to(device)
        f2_gt_t = TF.to_tensor(f2_gt).unsqueeze(0).to(device)
        
        # Run model and capture intermediates
        with torch.no_grad(), torch.amp.autocast(device.type, dtype=torch.bfloat16):
            output = model(f1_t, f3_t)
        
        # Extract all intermediates
        final = tensor_to_pil(output['output'].float())
        bg = tensor_to_pil(output['background'].float())
        fg = tensor_to_pil(output['foreground'].float())
        alpha = tensor_to_pil(output['alpha'].float())
        
        # Compute PSNR for each stage
        def psnr(pred, gt):
            mse = torch.nn.functional.mse_loss(pred.float(), gt.float())
            return (10 * torch.log10(1.0 / (mse + 1e-8))).item()
        
        psnr_bg = psnr(output['background'], f2_gt_t)
        psnr_fg = psnr(output['foreground'], f2_gt_t)
        psnr_final = psnr(output['output'], f2_gt_t)
        
        print(f"  Background PSNR: {psnr_bg:.2f} dB")
        print(f"  Foreground PSNR: {psnr_fg:.2f} dB")
        print(f"  Final output PSNR: {psnr_final:.2f} dB")
        print(f"  Alpha range: [{output['alpha'].min():.3f}, {output['alpha'].max():.3f}]")
        
        if 'is_scene_cut' in output and 'scene_confidence' in output:
            cut_val = output['is_scene_cut'].item() if hasattr(output['is_scene_cut'], 'item') else output['is_scene_cut']
            conf_val = output['scene_confidence'].item() if hasattr(output['scene_confidence'], 'item') else output['scene_confidence']
            print(f"  Scene cut: {cut_val}, confidence: {conf_val:.4f}")
        
        # Save individual images
        triplet_out = out_path / name
        triplet_out.mkdir(parents=True, exist_ok=True)
        
        f1.save(triplet_out / "1_frame1_input.png")
        f2_gt.save(triplet_out / "2_ground_truth.png")
        f3.save(triplet_out / "3_frame3_input.png")
        bg.save(triplet_out / "4_background.png")
        fg.save(triplet_out / "5_foreground.png")
        alpha.save(triplet_out / "6_alpha_mask.png")
        final.save(triplet_out / "7_final_output.png")
        
        # Build comparison strip
        w, h = f1.size
        gap = 2
        panels = [
            ("Frame 1", f1),
            (f"Background ({psnr_bg:.1f}dB)", bg),
            (f"Foreground ({psnr_fg:.1f}dB)", fg),
            ("Alpha Mask", alpha),
            (f"Final ({psnr_final:.1f}dB)", final),
            ("Ground Truth", f2_gt),
            ("Frame 3", f3),
        ]
        
        grid = Image.new("RGB", (w * len(panels) + gap * (len(panels) - 1), h), (40, 40, 40))
        for j, (label, img) in enumerate(panels):
            labeled = img.copy()
            add_label(labeled, label)
            grid.paste(labeled, (j * (w + gap), 0))
        
        grid.save(triplet_out / "comparison_all_stages.png")
        print(f"  Saved to {triplet_out}")


if __name__ == "__main__":
    main()
