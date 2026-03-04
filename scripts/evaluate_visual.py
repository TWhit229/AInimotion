"""
Visual evaluation: compare model predictions against ground truth.

Picks random triplets from the dataset, runs inference, and generates
side-by-side comparison images (input1 | predicted | ground truth | input3).

Usage:
    python scripts/evaluate_visual.py [--checkpoint PATH] [--data PATH] [--n NUM]

Examples:
    python scripts/evaluate_visual.py
    python scripts/evaluate_visual.py --checkpoint checkpoints/checkpoint_epoch_0299.pt --n 20
    python scripts/evaluate_visual.py --data training_data/train_10k --n 10
"""

import argparse
import random
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as TF

from ainimotion.models.interp import LayeredInterpolator


def load_model(checkpoint_path: str, device: torch.device):
    """Load trained model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
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
    print(f"  Model: base_channels={config.get('base_channels', 32)}, epoch={epoch}")
    return model, epoch


def load_frame(path: Path) -> Image.Image:
    """Load an image as RGB PIL."""
    return Image.open(path).convert("RGB")


def compute_psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Compute PSNR between two tensors."""
    mse = F.mse_loss(pred, gt)
    return (10 * torch.log10(1.0 / (mse + 1e-8))).item()


def compute_ssim_simple(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Compute a simple SSIM approximation between two tensors."""
    # Use mean/var based SSIM
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_pred = pred.mean()
    mu_gt = gt.mean()
    sigma_pred = pred.var()
    sigma_gt = gt.var()
    sigma_cross = ((pred - mu_pred) * (gt - mu_gt)).mean()

    ssim = ((2 * mu_pred * mu_gt + c1) * (2 * sigma_cross + c2)) / \
           ((mu_pred ** 2 + mu_gt ** 2 + c1) * (sigma_pred + sigma_gt + c2))
    return ssim.item()


def add_label(img: Image.Image, text: str, font_size: int = 20) -> Image.Image:
    """Add a text label at the top of an image."""
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    # Draw background rectangle
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = 4
    draw.rectangle([0, 0, tw + pad * 2, th + pad * 2], fill=(0, 0, 0, 180))
    draw.text((pad, pad), text, fill=(255, 255, 255), font=font)
    return img


def create_comparison(
    f1: Image.Image,
    f2_gt: Image.Image,
    f2_pred: Image.Image,
    f3: Image.Image,
    triplet_name: str,
    psnr: float,
    ssim: float,
) -> Image.Image:
    """Create a 4-panel side-by-side comparison image."""
    w, h = f1.size

    # Add labels
    f1_labeled = f1.copy()
    f2_gt_labeled = f2_gt.copy()
    f2_pred_labeled = f2_pred.copy()
    f3_labeled = f3.copy()

    add_label(f1_labeled, "Frame 1 (input)")
    add_label(f2_pred_labeled, f"Predicted (PSNR: {psnr:.1f} dB)")
    add_label(f2_gt_labeled, "Ground Truth (frame 2)")
    add_label(f3_labeled, "Frame 3 (input)")

    # Create 4-panel grid: F1 | Predicted | Ground Truth | F3
    gap = 4
    grid = Image.new("RGB", (w * 4 + gap * 3, h), color=(40, 40, 40))
    grid.paste(f1_labeled, (0, 0))
    grid.paste(f2_pred_labeled, (w + gap, 0))
    grid.paste(f2_gt_labeled, (w * 2 + gap * 2, 0))
    grid.paste(f3_labeled, (w * 3 + gap * 3, 0))

    return grid


def evaluate(
    checkpoint_path: str,
    data_dir: str,
    n_samples: int = 10,
    output_dir: str = "eval_outputs",
    seed: int = 42,
):
    """Run visual evaluation on random triplets."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model, epoch = load_model(checkpoint_path, device)

    # Find all triplet directories
    data_path = Path(data_dir)
    triplet_dirs = sorted([
        d for d in data_path.iterdir()
        if d.is_dir() and (d / "frame1.jpg").exists()
    ])
    print(f"Found {len(triplet_dirs)} triplets in {data_dir}")

    # Pick random samples
    random.seed(seed)
    if n_samples < len(triplet_dirs):
        selected = random.sample(triplet_dirs, n_samples)
    else:
        selected = triplet_dirs
    print(f"Evaluating {len(selected)} triplets...\n")

    # Output directory
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    psnrs = []
    ssims = []

    for i, triplet_dir in enumerate(selected):
        name = triplet_dir.name

        # Load frames at original resolution
        f1_orig = load_frame(triplet_dir / "frame1.jpg")
        f2_gt_orig = load_frame(triplet_dir / "frame2.jpg")
        f3_orig = load_frame(triplet_dir / "frame3.jpg")

        # Resize to training resolution (model trained on 256x256 crops)
        eval_size = (256, 256)
        f1 = f1_orig.resize(eval_size, Image.LANCZOS)
        f2_gt = f2_gt_orig.resize(eval_size, Image.LANCZOS)
        f3 = f3_orig.resize(eval_size, Image.LANCZOS)

        # To tensors
        f1_t = TF.to_tensor(f1).unsqueeze(0).to(device)
        f3_t = TF.to_tensor(f3).unsqueeze(0).to(device)
        f2_gt_t = TF.to_tensor(f2_gt).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = model(f1_t, f3_t)
            f2_pred_t = output["output"].float().clamp(0, 1)

        # Metrics
        psnr = compute_psnr(f2_pred_t, f2_gt_t)
        ssim = compute_ssim_simple(f2_pred_t, f2_gt_t)
        psnrs.append(psnr)
        ssims.append(ssim)

        # Convert prediction to PIL
        f2_pred = TF.to_pil_image(f2_pred_t.squeeze(0).cpu())

        # Save individual images
        f2_pred.save(out_path / f"{name}_predicted.png")
        f2_gt.save(out_path / f"{name}_ground_truth.png")

        # Save comparison grid
        grid = create_comparison(f1, f2_gt, f2_pred, f3, name, psnr, ssim)
        grid.save(out_path / f"{name}_comparison.png")

        print(f"  [{i+1}/{len(selected)}] {name}: PSNR={psnr:.2f} dB, SSIM={ssim:.4f}")

    # Summary
    avg_psnr = sum(psnrs) / len(psnrs)
    avg_ssim = sum(ssims) / len(ssims)
    print(f"\n{'=' * 55}")
    print(f"  Results (epoch {epoch}, {len(selected)} triplets)")
    print(f"  Avg PSNR:  {avg_psnr:.2f} dB")
    print(f"  Avg SSIM:  {avg_ssim:.4f}")
    print(f"  Min PSNR:  {min(psnrs):.2f} dB")
    print(f"  Max PSNR:  {max(psnrs):.2f} dB")
    print(f"{'=' * 55}")
    print(f"\nOutputs saved to: {out_path.absolute()}")
    print(f"  *_comparison.png  = side-by-side grids (F1 | Predicted | GT | F3)")
    print(f"  *_predicted.png   = model output only")
    print(f"  *_ground_truth.png = original frame 2")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visual evaluation of trained model")
    parser.add_argument(
        "--checkpoint", "-c",
        default="checkpoints/checkpoint_latest.pt",
        help="Path to model checkpoint (default: checkpoint_latest.pt)",
    )
    parser.add_argument(
        "--data", "-d",
        default="training_data/train_10k",
        help="Path to triplet dataset directory",
    )
    parser.add_argument(
        "--n", "-n",
        type=int,
        default=10,
        help="Number of random triplets to evaluate (default: 10)",
    )
    parser.add_argument(
        "--output", "-o",
        default="eval_outputs",
        help="Output directory for results (default: eval_outputs)",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for triplet selection (default: 42)",
    )
    args = parser.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        data_dir=args.data,
        n_samples=args.n,
        output_dir=args.output,
        seed=args.seed,
    )
