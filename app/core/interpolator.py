"""
V5 model inference for frame interpolation.

Uses fp16 autocast + torch.compile for maximum throughput.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import torch

from ainimotion.models.interp_v5 import build_model


class Interpolator:
    """
    Loads the V5 model and runs batched interpolation with fp16 + torch.compile.

    Args:
        model_path: Path to checkpoint (.pt file).
        device: CUDA device (default: 'cuda').
        use_ema: Use EMA weights for slightly better quality (default: True).
        use_compile: Use torch.compile for faster inference (default: True).
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str = 'cuda',
        use_ema: bool = True,
        use_compile: bool = True,
    ):
        self.device = torch.device(device)
        self.model = self._load_model(model_path, use_ema)
        self._compiled = False
        self._use_compile = use_compile
        self._batch_size: int | None = None
        self._batch_size_res: tuple[int, int] | None = None
        self._infer_lock = threading.Lock()

    def _load_model(self, path: str | Path, use_ema: bool) -> torch.nn.Module:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {path}")

        print(f"Loading model from {path}...")
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)

        config = checkpoint.get('config', {}).get('model', {})
        model = build_model(config)

        if use_ema and 'ema_state_dict' in checkpoint:
            state = checkpoint['ema_state_dict']
            state = {
                k.replace('module.', ''): v
                for k, v in state.items()
                if k != 'n_averaged'
            }
            print(f"  Using EMA weights ({len(state)} params)")
        elif 'generator_state_dict' in checkpoint:
            state = checkpoint['generator_state_dict']
            print(f"  Using generator weights ({len(state)} params)")
        else:
            raise ValueError("Checkpoint has no generator or EMA weights")

        state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}

        # Remap affine_head keys if checkpoint predates ONNX-compatible BackgroundPath
        # Old: AdaptiveAvgPool2d(0), Flatten(1), Linear(2)... -> New: Flatten(0), Linear(1)...
        remapped = {}
        for k, v in state.items():
            if 'background_path.affine_head.' in k:
                parts = k.split('.')
                idx_pos = parts.index('affine_head') + 1
                old_idx = int(parts[idx_pos])
                if old_idx >= 2:  # skip pool layer (idx 0), shift rest down by 1
                    parts[idx_pos] = str(old_idx - 1)
                    remapped['.'.join(parts)] = v
                # idx 0 and 1 are AdaptiveAvgPool2d and Flatten (no params) — skip
            else:
                remapped[k] = v
        state = remapped

        model.load_state_dict(state, strict=True)
        model = model.to(self.device).eval()

        # Enable TF32 for faster matmuls on Ampere+ GPUs
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        params = sum(p.numel() for p in model.parameters())
        print(f"  Model loaded: {params:,} parameters on {self.device}")
        return model

    def _compile_model(self):
        """Compile model after batch size is determined (one-time cost)."""
        if self._compiled or not self._use_compile:
            return

        # torch.compile requires Triton (Linux only as of 2026)
        import sys
        if sys.platform == 'win32':
            print("  torch.compile skipped (Windows, no Triton)")
            return

        try:
            print("  Compiling model (one-time, ~30s)...")
            self.model = torch.compile(self.model, mode='reduce-overhead')
            self._compiled = True
            print("  Compiled successfully")
        except Exception as e:
            print(f"  Compile skipped ({e}), using eager mode")

    def find_batch_size(self, height: int, width: int) -> int:
        """Auto-detect largest batch size that fits in VRAM with fp16."""
        if self._batch_size is not None and self._batch_size_res == (height, width):
            return self._batch_size
        self._batch_size = None

        with self._infer_lock:
            for bs in [4, 2, 1]:
                torch.cuda.empty_cache()
                dummy = None
                try:
                    dummy = torch.randn(bs, 7, 3, height, width, device=self.device)
                    with torch.inference_mode():
                        with torch.amp.autocast('cuda', dtype=torch.float16):
                            self.model(dummy)
                    del dummy
                    torch.cuda.empty_cache()
                    self._batch_size = bs
                    self._batch_size_res = (height, width)
                    print(f"  Batch size: {bs} (at {width}x{height}, fp16)")
                    self._compile_model()
                    return bs
                except torch.cuda.OutOfMemoryError:
                    del dummy
                    torch.cuda.empty_cache()
                    continue

        raise RuntimeError(
            f"Cannot fit even batch_size=1 at {width}x{height} in VRAM. "
            f"Reduce resolution in Settings or free GPU memory."
        )

    def interpolate_batch(
        self,
        windows: list[list[np.ndarray]],
        timestep: float = 0.5,
    ) -> list[np.ndarray]:
        """
        Interpolate a batch of 7-frame windows at a given timestep.
        Uses fp16 autocast for ~2x speed vs fp32.
        """
        B = len(windows)
        assert B > 0, "Empty batch"
        assert all(len(w) == 7 for w in windows), "Each window must have 7 frames"

        batch = torch.stack([
            torch.stack([
                torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                for frame in window
            ])
            for window in windows
        ]).to(self.device)

        with self._infer_lock:
            with torch.inference_mode():
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    result = self.model(batch, timestep=timestep)

                output = result['output'].float()  # back to fp32 for clamp/convert
                output = output.clamp(0, 1).mul(255).byte()
                output = output.permute(0, 2, 3, 1).cpu().numpy()

        return [output[i] for i in range(B)]

    def interpolate_single(self, window: list[np.ndarray]) -> np.ndarray:
        return self.interpolate_batch([window])[0]

    def interpolate_multi_timestep(
        self,
        windows: list[list[np.ndarray]],
        timesteps: list[float],
    ) -> list[list[np.ndarray]]:
        """
        Interpolate a batch of windows at multiple timesteps efficiently.
        Encodes once (~65% of compute), then decodes per timestep (~35%).

        Returns: list of B lists, each containing len(timesteps) frames.
        """
        B = len(windows)
        assert B > 0
        assert all(len(w) == 7 for w in windows)

        batch = torch.stack([
            torch.stack([
                torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                for frame in window
            ])
            for window in windows
        ]).to(self.device)

        with self._infer_lock:
            with torch.inference_mode():
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    # Encode once (expensive)
                    shared = self.model.encode(batch)

                    # Decode per timestep (cheap)
                    all_results = []
                    for t in timesteps:
                        result = self.model.decode_timestep(shared, timestep=t)
                        out = result['output'].float().clamp(0, 1).mul(255).byte()
                        out = out.permute(0, 2, 3, 1).cpu().numpy()
                        all_results.append([out[i] for i in range(B)])

        # Reshape: all_results[timestep][batch] -> per_window[batch][timestep]
        return [[all_results[t][i] for t in range(len(timesteps))] for i in range(B)]

    @property
    def batch_size(self) -> int | None:
        return self._batch_size

    def unload(self):
        self.model = self.model.cpu()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    import sys
    import time

    model_path = sys.argv[1] if len(sys.argv) > 1 else 'model/ainimotion.pt'
    interp = Interpolator(model_path)

    H, W = 720, 1280
    bs = interp.find_batch_size(H, W)

    print(f"\nBenchmarking BS={bs} at {W}x{H} (fp16 + compile)...")
    dummy_window = [np.random.randint(0, 255, (H, W, 3), dtype=np.uint8) for _ in range(7)]
    batch = [dummy_window] * bs

    # Warmup (includes compile)
    _ = interp.interpolate_batch(batch)
    _ = interp.interpolate_batch(batch)

    n_iters = 10
    t0 = time.time()
    for _ in range(n_iters):
        results = interp.interpolate_batch(batch)
    elapsed = time.time() - t0

    total_frames = n_iters * bs
    fps = total_frames / elapsed
    print(f"  {total_frames} frames in {elapsed:.2f}s = {fps:.2f} frames/sec")
    print(f"  Output: {results[0].shape}, range [{results[0].min()}, {results[0].max()}]")

    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"  VRAM peak: {peak:.2f} GB")
