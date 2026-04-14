"""
VRAM profiling test for V5 model.

Tests forward + backward pass at full model size (base_channels=64)
with 256x256 crops at increasing batch sizes to find max BS.
"""

import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ainimotion.models.interp_v5 import LayeredInterpolatorV5


def profile_vram(base_channels=64, resolution=256, max_bs=8):
    if not torch.cuda.is_available():
        print("CUDA not available, skipping VRAM profile")
        return
    
    device = torch.device('cuda')
    torch.cuda.reset_peak_memory_stats()
    
    print(f"=== V5 VRAM Profiling ===")
    print(f"Base channels: {base_channels}")
    print(f"Resolution: {resolution}x{resolution}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()
    
    # Build model
    model = LayeredInterpolatorV5(base_channels=base_channels).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {params:,}")
    
    model_mem = torch.cuda.memory_allocated() / 1e9
    print(f"Model memory: {model_mem:.2f} GB")
    print()
    
    # Test increasing batch sizes
    for bs in range(1, max_bs + 1):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        try:
            frames = [torch.randn(bs, 3, resolution, resolution, device=device) for _ in range(7)]
            
            # Forward pass
            out = model(frames)
            fwd_mem = torch.cuda.max_memory_allocated() / 1e9
            
            # Backward pass
            loss = out['output'].mean()
            loss.backward()
            
            peak_mem = torch.cuda.max_memory_allocated() / 1e9
            
            print(f"BS={bs}: Forward={fwd_mem:.2f}GB, Peak(fwd+bwd)={peak_mem:.2f}GB")
            
            # Cleanup
            del frames, out, loss
            model.zero_grad()
            torch.cuda.empty_cache()
            
        except torch.cuda.OutOfMemoryError:
            print(f"BS={bs}: OOM!")
            torch.cuda.empty_cache()
            break
    
    print()
    print(f"Recommended max batch size: BS={bs-1}")


if __name__ == '__main__':
    profile_vram(base_channels=64, resolution=256, max_bs=8)
