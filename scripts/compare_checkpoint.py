"""Generate visual comparisons from a checkpoint across diverse anime triplets."""

import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ainimotion.models.interp.layered_interp import LayeredInterpolator
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from pathlib import Path

device = torch.device('cpu')  # CPU to avoid competing with training
transform = transforms.ToTensor()

# Get all triplet dirs and sample widely across the dataset
all_dirs = sorted([d for d in Path('D:/Triplets').iterdir() if d.is_dir()])
total = len(all_dirs)

# Sample 15 triplets evenly spread across the dataset (different anime)
indices = [int(i * total / 15) for i in range(15)]
sample_dirs = [all_dirs[i] for i in indices]
print(f"Sampling {len(sample_dirs)} triplets from {total} total")
for i, idx in enumerate(indices):
    print(f"  {i+1:2d}. {all_dirs[idx].name} (index {idx})")
print()

# Load latest checkpoint
ckpt = 'checkpoints/checkpoint_epoch_0030.pt'
cp = torch.load(ckpt, map_location='cpu', weights_only=False)
config = cp['config']
epoch = cp['epoch']
channels = config.get('base_channels', 64)
print(f"Loaded epoch {epoch}, {channels} channels")
print()

model = LayeredInterpolator(
    base_channels=channels,
    kernel_size=config.get('kernel_size', 7),
    grid_size=config.get('grid_size', 8),
    use_refinement=config.get('use_refinement', True),
)
model.load_state_dict(cp['generator'])
model.eval()

# Generate comparisons
out_dir = Path('test_outputs')
out_dir.mkdir(exist_ok=True)

psnrs = []
for i, td in enumerate(sample_dirs):
    f1 = transform(Image.open(td / 'f1.jpg').convert('RGB')).unsqueeze(0)
    f2 = transform(Image.open(td / 'f2.jpg').convert('RGB')).unsqueeze(0)
    f3 = transform(Image.open(td / 'f3.jpg').convert('RGB')).unsqueeze(0)

    with torch.no_grad():
        pred = model(f1, f3)['output']
        mse = F.mse_loss(pred, f2)
        psnr = 10 * torch.log10(1.0 / (mse + 1e-8)).item()
        psnrs.append(psnr)

    name = td.name
    save_image(f1[0], out_dir / f'{name}_1_input_first.png')
    save_image(f2[0], out_dir / f'{name}_2_ground_truth.png')
    save_image(pred[0].clamp(0, 1), out_dir / f'{name}_3_predicted.png')
    save_image(f3[0], out_dir / f'{name}_4_input_last.png')
    print(f"  {i+1:2d}. {name}: PSNR = {psnr:.2f} dB")

print()
avg = sum(psnrs) / len(psnrs)
print(f"Average PSNR: {avg:.2f} dB")
print(f"Min: {min(psnrs):.2f} dB | Max: {max(psnrs):.2f} dB")
print(f"Outputs saved to: {out_dir.absolute()}")
