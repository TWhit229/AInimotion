"""
VRAM benchmark for v2 architecture.

Tests different batch sizes to find the maximum that fits in 32GB VRAM.
Simulates a full training step (forward + backward + GAN) at each batch size.
"""

import gc
import torch
import torch.nn.functional as F
from ainimotion.models.interp import LayeredInterpolator
from ainimotion.training.losses import VFILoss
from ainimotion.training.discriminator import (
    PatchDiscriminator, GANLoss,
    compute_r1_penalty, compute_r2_penalty,
)


def benchmark_batch_size(batch_size: int, base_channels: int = 96) -> float | None:
    """Run one full training step and return peak VRAM in GB."""
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    
    try:
        device = torch.device("cuda")
        
        # Build v2 model
        model = LayeredInterpolator(
            base_channels=base_channels, kernel_size=5
        ).to(device).train()
        
        disc = PatchDiscriminator(base_channels=64).to(device).train()
        vfi_loss = VFILoss(freq_weight=0.5).to(device)
        gan_loss = GANLoss(loss_type='relativistic').to(device)
        
        opt_g = torch.optim.AdamW(model.parameters(), lr=1e-4)
        opt_d = torch.optim.AdamW(disc.parameters(), lr=2e-5)
        scaler = torch.amp.GradScaler("cuda")
        
        # Simulate training step
        f1 = torch.randn(batch_size, 3, 256, 256, device=device)
        f2 = torch.randn(batch_size, 3, 256, 256, device=device)
        f3 = torch.randn(batch_size, 3, 256, 256, device=device)
        
        # Discriminator step
        opt_d.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                output = model(f1, f3)
                fake = output['output'].float()
            
            pred_real = disc(f2)
            pred_fake = disc(fake.detach())
            loss_d = gan_loss(pred_real, pred_other=pred_fake, for_discriminator=True)
        
        scaler.scale(loss_d).backward()
        
        # R1 penalty
        f2.requires_grad_(True)
        r1 = compute_r1_penalty(disc, f2)
        scaler.scale(r1 * 10.0).backward()
        f2.requires_grad_(False)
        
        scaler.step(opt_d)
        scaler.update()
        
        # Generator step
        opt_g.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = model(f1, f3)
            fake = output['output'].float()
            loss_vfi, _ = vfi_loss(fake, f2, return_components=True)
            pred_fake_g = disc(fake)
            with torch.no_grad():
                pred_real_g = disc(f2)
            loss_g_gan = gan_loss(pred_real_g, pred_other=pred_fake_g, for_discriminator=False)
            loss_g = loss_vfi + 0.005 * loss_g_gan
        
        scaler.scale(loss_g).backward()
        scaler.step(opt_g)
        scaler.update()
        
        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        
        # Cleanup
        del model, disc, vfi_loss, gan_loss, opt_g, opt_d, scaler
        del f1, f2, f3, output, fake
        torch.cuda.empty_cache()
        gc.collect()
        
        return peak_gb
        
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        gc.collect()
        return None


def main():
    print("=" * 60)
    print("  AInimotion v2 VRAM Benchmark (RTX 5090, 32GB)")
    print("  Model: base_channels=96, kernel_size=5, bicubic")
    print("=" * 60)
    
    # Test batch sizes
    batch_sizes = [4, 6, 8, 10, 12, 14, 16]
    results = {}
    
    for bs in batch_sizes:
        print(f"\n  Batch size {bs}...", end=" ", flush=True)
        peak = benchmark_batch_size(bs)
        if peak is not None:
            results[bs] = peak
            status = "OK" if peak < 30 else "TIGHT"
            print(f"{peak:.1f} GB [{status}]")
        else:
            print("OOM!")
            break
    
    print(f"\n{'=' * 60}")
    print("  Results Summary:")
    print(f"  {'BS':>4} | {'VRAM (GB)':>10} | {'Status':>8}")
    print(f"  {'-'*4}-+-{'-'*10}-+-{'-'*8}")
    
    best_bs = 4
    for bs, vram in results.items():
        status = "OPTIMAL" if 26 <= vram <= 30 else ("TIGHT" if vram > 30 else "OK")
        print(f"  {bs:>4} | {vram:>10.1f} | {status:>8}")
        if 26 <= vram <= 30:
            best_bs = bs
        elif vram < 26:
            best_bs = bs
    
    print(f"\n  Recommended batch_size: {best_bs}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
