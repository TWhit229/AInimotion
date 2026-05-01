"""
V5 model inference for frame interpolation.

Tries TensorRT (6x faster) first, falls back to PyTorch fp16.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import numpy as np
import torch

from ainimotion.models.interp_v5 import build_model


class OnnxTrtInference:
    """ONNX Runtime with TensorRT EP — ~6x faster than PyTorch."""

    def __init__(self, onnx_path: str, cache_dir: str):
        import onnxruntime as ort

        # Add TensorRT + CUDA libs to PATH
        # Check both: site-packages (dev) and exe directory (packaged)
        exe_dir = Path(sys.executable).parent
        sp = exe_dir / 'Lib' / 'site-packages'
        for lib_dir in [
            sp / 'tensorrt_libs', sp / 'torch' / 'lib',  # dev
            exe_dir,  # packaged (DLLs next to exe)
        ]:
            if lib_dir.exists():
                os.environ['PATH'] = str(lib_dir) + os.pathsep + os.environ.get('PATH', '')

        os.makedirs(cache_dir, exist_ok=True)

        trt_opts = {
            'trt_fp16_enable': True,
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': cache_dir,
        }

        sess_opts = ort.SessionOptions()
        sess_opts.log_severity_level = 3  # suppress warnings

        self.session = ort.InferenceSession(
            onnx_path, sess_opts,
            providers=[
                ('TensorrtExecutionProvider', trt_opts),
                'CUDAExecutionProvider',
            ]
        )
        self.active_provider = self.session.get_providers()[0]

    def run(self, frames: np.ndarray) -> np.ndarray:
        """frames: (B, 7, 3, H, W) float32 → output: (B, 3, H, W) float32"""
        return self.session.run(['output'], {'frames': frames})[0]


class Interpolator:
    """
    Loads the V5 model and runs inference.

    Automatically uses TensorRT if available (6x faster), falls back to PyTorch fp16.
    First TRT run builds the engine (~60-90s), subsequent runs use cached engine.

    Args:
        model_path: Path to checkpoint (.pt file).
        device: CUDA device (default: 'cuda').
        use_ema: Use EMA weights (default: True).
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str = 'cuda',
        use_ema: bool = True,
        use_compile: bool = True,
    ):
        self.device = torch.device(device)
        self._batch_size: int | None = None
        self._batch_size_res: tuple[int, int] | None = None
        self._batch_size_backend: str | None = None  # 'trt' or 'pytorch'
        self._infer_lock = threading.Lock()

        # Try TensorRT first
        self._trt: OnnxTrtInference | None = None
        self._use_trt = False
        model_dir = Path(model_path).parent

        onnx_path = model_dir / 'ainimotion_720p.onnx'
        if onnx_path.exists():
            try:
                print(f"Loading TensorRT backend from {onnx_path}...")
                self._trt = OnnxTrtInference(
                    str(onnx_path),
                    str(model_dir / 'trt_cache'),
                )
                if 'TensorrtExecutionProvider' in self._trt.active_provider:
                    self._use_trt = True
                    print(f"  TensorRT active (engine cached for instant startup)")
                else:
                    print(f"  TensorRT not available, using {self._trt.active_provider}")
                    self._trt = None
            except Exception as e:
                print(f"  TensorRT failed: {e}")
                self._trt = None

        # Always load PyTorch model (needed for multi-timestep encode/decode + fallback)
        self.model = self._load_model(model_path, use_ema)

        if self._use_trt:
            print("  Mode: TensorRT fp16 (~6x faster)")
        else:
            print("  Mode: PyTorch fp16")

    def _load_model(self, path: str | Path, use_ema: bool) -> torch.nn.Module:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {path}")

        print(f"Loading PyTorch model from {path}...")
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

        # Remap affine_head keys (ONNX-compatible BackgroundPath has different Sequential indices)
        remapped = {}
        for k, v in state.items():
            if 'background_path.affine_head.' in k:
                parts = k.split('.')
                idx_pos = parts.index('affine_head') + 1
                old_idx = int(parts[idx_pos])
                if old_idx >= 2:
                    parts[idx_pos] = str(old_idx - 1)
                    remapped['.'.join(parts)] = v
            else:
                remapped[k] = v
        state = remapped

        model.load_state_dict(state, strict=True)
        model = model.to(self.device).eval()

        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        params = sum(p.numel() for p in model.parameters())
        print(f"  Model loaded: {params:,} parameters on {self.device}")
        return model

    def find_batch_size(self, height: int, width: int) -> int:
        """Auto-detect largest batch size that fits in VRAM."""
        if self._batch_size is not None and self._batch_size_res == (height, width):
            return self._batch_size
        self._batch_size = None

        if self._use_trt:
            # TRT: test with actual session (only works at exported resolution)
            for bs in [4, 2, 1]:
                try:
                    dummy = np.random.randn(bs, 7, 3, height, width).astype(np.float32)
                    self._trt.run(dummy)
                    self._batch_size = bs
                    self._batch_size_res = (height, width)
                    self._batch_size_backend = 'trt'
                    print(f"  Batch size: {bs} (at {width}x{height}, TensorRT)")
                    return bs
                except Exception:
                    continue
            # TRT failed at this resolution — fall through to PyTorch
            print(f"  TRT failed at {width}x{height}, falling back to PyTorch")
            # PyTorch: test with model
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
                        self._batch_size_backend = 'pytorch'
                        print(f"  Batch size: {bs} (at {width}x{height}, PyTorch fp16)")
                        return bs
                    except torch.cuda.OutOfMemoryError:
                        del dummy
                        torch.cuda.empty_cache()
                        continue

        # Debug info for the error
        try:
            free = torch.cuda.mem_get_info()[0] / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            gpu_info = f"GPU: {free:.1f}/{total:.0f} GB free"
        except Exception:
            gpu_info = "GPU: unable to query"

        raise RuntimeError(
            f"Cannot fit batch_size=1 at {width}x{height}. {gpu_info}. "
            f"Close other GPU apps or reduce resolution in Settings."
        )

    def interpolate_batch(
        self,
        windows: list[list[np.ndarray]],
        timestep: float = 0.5,
    ) -> list[np.ndarray]:
        """Interpolate a batch of 7-frame windows at a given timestep."""
        B = len(windows)
        assert B > 0
        assert all(len(w) == 7 for w in windows)

        if self._use_trt and timestep == 0.5 and self._batch_size_backend == 'trt':
            # TensorRT path (fixed timestep=0.5, only when TRT batch size detection succeeded)
            H, W = windows[0][0].shape[:2]
            if (H, W) == self._batch_size_res:
                batch = np.stack([
                    np.stack([frame.astype(np.float32) / 255.0 for frame in window])
                    for window in windows
                ])
                batch = batch.transpose(0, 1, 4, 2, 3)  # (B, 7, 3, H, W)

                with self._infer_lock:
                    output = self._trt.run(batch)

                output = np.clip(output, 0, 1)
                output = (output * 255).astype(np.uint8)
                output = output.transpose(0, 2, 3, 1)
                return [output[i] for i in range(B)]

        # PyTorch path (any timestep, or TRT not available)
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

                output = result['output'].float()
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
        Interpolate at multiple timesteps efficiently.
        Uses TRT for t=0.5, PyTorch encode/decode for others.
        """
        B = len(windows)

        if len(timesteps) == 1 and timesteps[0] == 0.5 and self._use_trt:
            results = self.interpolate_batch(windows, 0.5)
            return [[r] for r in results]

        # PyTorch path: encode once, decode per timestep
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
                    shared = self.model.encode(batch)

                    all_results = []
                    for t in timesteps:
                        result = self.model.decode_timestep(shared, timestep=t)
                        out = result['output'].float().clamp(0, 1).mul(255).byte()
                        out = out.permute(0, 2, 3, 1).cpu().numpy()
                        all_results.append([out[i] for i in range(B)])

        return [[all_results[t][i] for t in range(len(timesteps))] for i in range(B)]

    @property
    def batch_size(self) -> int | None:
        return self._batch_size

    @property
    def backend(self) -> str:
        return "TensorRT" if self._use_trt else "PyTorch"

    def unload(self):
        self.model = self.model.cpu()
        torch.cuda.empty_cache()
