import sys
import os
import torch
from torch.amp import autocast

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ainimotion.models.interp_v5.losses import CharbonnierLoss
from ainimotion.models.interp_v5.discriminator import MultiScalePatchGAN, r1_gradient_penalty
from ainimotion.models.interp_v5 import build_model

def test_charbonnier():
    print("Testing CharbonnierLoss with FP16 underflow prevention...")
    loss_fn = CharbonnierLoss()
    pred = torch.zeros(2, 3, 64, 64, device='cuda')
    gt = torch.zeros(2, 3, 64, 64, device='cuda')
    
    with autocast('cuda'):
        loss = loss_fn(pred, gt)
        print(f"  Loss for identical FP16 inputs: {loss.item()} (Should NOT be NaN)")
        
        # Test backward to ensure gradients are not NaN
        pred.requires_grad_(True)
        loss = loss_fn(pred, gt)
        loss.backward()
        if torch.isnan(pred.grad).any():
            print("  FAIL: CharbonnierLoss grad is NaN")
        else:
            print(f"  PASS: CharbonnierLoss grad sum = {pred.grad.sum().item()}")

def test_r1_penalty():
    print("\nTesting R1 Penalty with AMP...")
    disc = MultiScalePatchGAN().cuda()
    real = torch.randn(2, 3, 64, 64, device='cuda')
    
    r1 = r1_gradient_penalty(disc, real)
    print(f"  R1 Penalty value (should be > 0): {r1.item():.6f}")

def test_compile():
    print("\nTesting torch.compile...")
    # Use smaller channels to fit in memory for quick test
    model = build_model({'base_channels': 16, 'n_synth_points': 4}).cuda()
    model = torch.compile(model)
    frames = [torch.randn(1, 3, 64, 64, device='cuda') for _ in range(7)]
    
    with torch.no_grad():
        with autocast('cuda'):
            out = model(frames)
    print(f"  Compiled model output shape: {out['output'].shape}")
    print("  PASS: Compilation successful")

if __name__ == '__main__':
    test_charbonnier()
    test_r1_penalty()
    test_compile()
