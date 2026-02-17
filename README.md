# AInimotion

**AInimotion** is a GPU-accelerated, offline anime video enhancer that combines:

- **Frame interpolation**: *24 → 48 FPS* (motion-adaptive, anime-safe)
- **Upscale/restoration**: *1080p → 1440p (2K)* (line-art preserving)
- **Anime-first safeguards**: **scene-cut detection** + **hold/still detection** (so you don't get wobbly outlines or "fake 60fps" artifacts)

> Goal: "buttery smooth" motion and cleaner frames **without** bending line art, shimmering edges, or interpolating across hard cuts.

---

## Current Status: Full Training In Progress 🚀

**65,906 triplets extracted.** Hyperparameter sweep complete (13 experiments). Training with optimized config on RTX 5090.

### Quick Start Training

```powershell
cd "C:\Projects\AInimotion"
python -m ainimotion.training.train --config configs/interp_training_5090.yaml --data "D:\Triplets" --wandb --auto-resume
```

### Sweep-Optimized Config

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Architecture** | K=9, 96 base channels | AdaCoF kernel 9×9 (sweep winner) |
| **Crop size** | 384×384 | +0.74 dB PSNR over 256 crops |
| **Learning rate** | 3e-4 (Phase 1), 5e-5 (Phase 2) | Two-phase training |
| **Loss weights** | L1=1.5, Perceptual=0.1, Edge=1.0 | Sweep-optimized balance |
| **Phase 2 GAN** | Batch 3 + 2× gradient accumulation | Effective batch 6, fits in 32 GB |

### Training Features

| Feature | Description |
|---------|-------------|
| **LayeredInterpolator** | FPN + Scene Gate + AdaCoF + Background Flow + Compositor |
| **Two-Phase Training** | Phase 1: reconstruction only → Phase 2: GAN fine-tuning |
| **GAN Stabilization** | PatchDiscriminator, LSGAN, label smoothing, adaptive lr_d |
| **Gradient Accumulation** | Physical batch 3, accumulate 2× for VRAM-safe Phase 2 |
| **Mixed Precision** | FP16 forward, FP32 gradients via GradScaler |
| **OOM Recovery** | Auto-checkpoint + retry with backoff on CUDA OOM |
| **Checkpointing** | Per-epoch + per-N-batches saves, auto-resume support |
| **W&B + TensorBoard** | Real-time loss/PSNR/gradient monitoring |

### Requirements

- **GPU**: NVIDIA RTX 5090 (32 GB VRAM) — or similar with 24+ GB
- **PyTorch**: 2.9+ with CUDA 12.8+
- **Dataset**: 65,906 triplets in D:\Triplets

---

## Triplet Extraction Tool

The `extract_triplets.py` script extracts high-quality training triplets (F1, F2, F3) from anime videos:

```bash
python scripts/extract_triplets.py \
  --input "/path/to/anime/videos" \
  --output "/path/to/triplets" \
  --temp-dir "/path/to/temp" \
  --workers 8
```

#### Extraction Features

| Feature | Description |
|---------|-------------|
| **JPEG Output** | Default format, ~1.5 MB/triplet (10x smaller than PNG) |
| **Parallel Processing** | 8-worker parallel validation for 8x speedup |
| **720p Output** | Frames extracted at 720p to reduce storage/processing |
| **Chunked Processing** | 5-minute chunks to manage temp storage |
| **Skip Intro** | Skips first 120s to avoid logos/credits |
| **Motion Filtering** | SSIM-based filtering removes static/duplicate frames |
| **Hardsub Support** | Top/bottom cropping to remove burned-in subtitles |

#### Helper Scripts

| Script | Purpose |
|--------|---------|
| `run_all_overnight.ps1` | Process all videos (clean + hardsubs) |
| `run_clean.ps1` | Process clean (no subtitle) sources |
| `run_hardsubs.ps1` | Process videos with burned-in subs |
| `run_test.ps1` | Test on a single video |

---

## Why 48 FPS (not 60)?

Anime is often authored at ~24 FPS (and frequently animated "on 2s" / held frames). Doubling to **48 FPS** is a clean, consistent **2×** step:

- Simpler timing
- Fewer artifacts than forcing 24 → 60
- Preserves the "anime feel" better than aggressive interpolation

---

## Key Features

### Motion-adaptive interpolation (Model A)

- Generates **one in-between** frame between adjacent frames when motion is real
- **Skips interpolation** when:
  - the interval is a **scene cut**
  - the interval is a **hold** / near-still (duplicates instead of hallucinating)

### Anime-tuned upscale/restoration (Model B)

- Upscales frames to **1440p**
- Optional restoration steps (preset-dependent):
  - deblock / mild denoise
  - artifact cleanup for web encodes
  - edge-friendly sharpening (conservative)

### Offline pipeline (not real-time)

- Takes an input file (mp4/mkv) and produces a new encoded file
- Preserves **audio** and (optionally) **subtitles** while re-encoding video

---

## Installation

```bash
# Clone the repository
git clone https://github.com/TWhit229/AInimotion.git
cd AInimotion

# Install dependencies
pip install -e .

# Additional requirements for triplet extraction
pip install numpy pillow scikit-image tqdm
```

### Requirements

- Python 3.10+
- FFmpeg (must be in PATH)
- For training: NVIDIA GPU with CUDA support

---

## How It Works

**Decode → Gate → Interpolate → Upscale → Encode**

1. **Decode** video frames (FFmpeg)
2. **Gate** each frame interval:
   - **Scene-cut detection**: don't interpolate across cuts
   - **Hold/still detection**: if frames are essentially identical, just duplicate
3. **Interpolate** only where motion is real (Model A)
4. **Upscale** every output frame (Model B)
5. **Encode** video + remux audio/subs (FFmpeg / NVENC optional)

---

## Presets

| Preset | Description |
|--------|-------------|
| `anime-clean` | Most conservative. Best for line stability. **(default)** |
| `anime-strong` | More aggressive restoration (denoise/deblock). |
| `fast` | Prioritizes speed. Uses lighter models. |

---

## Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Decode    │ -> │   Analyze   │ -> │ Interpolate │ -> │   Upscale   │ -> │   Encode    │
│    (CPU)    │    │    (Gate)   │    │  (GPU, A)   │    │  (GPU, B)   │    │ (CPU/NVENC) │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

---

## Project Structure

```text
AInimotion/
  README.md
  ROADMAP.md
  LICENSE
  pyproject.toml
  ainimotion/
    __init__.py
    data/
      dataset.py           # Training dataset loader
    models/
      interp/              # Interpolation model
        layered_interp.py  #   Main model orchestrator
        feature_extractor.py # FPN + correlation volumes
        background_flow.py #   Affine grid background
        foreground_flow.py #   AdaCoF deformable foreground
        compositor.py      #   Alpha blend + refinement U-Net
    training/
      train.py             # Training loop (2-phase, accum, OOM recovery)
      losses.py            # Loss functions (L1, perceptual, edge)
      discriminator.py     # PatchGAN discriminator
  scripts/
    extract_triplets.py    # Training data extraction ✅
    run_all_overnight.ps1  # Batch processing script ✅
    run_clean.ps1          # Clean sources script ✅
    run_hardsubs.ps1       # Hardsub sources script ✅
    run_test.ps1           # Test script ✅
  configs/
    interp_training_5090.yaml  # RTX 5090 sweep-optimized config
    interp_training.yaml       # Base config
```

---

## Roadmap

- [x] Phase 0 — Foundations (platform target, inference runtime, presets)
- [x] **Phase 0.5 — Training Data Pipeline** ✅
  - [x] Triplet extraction script
  - [x] 65,906 triplets from diverse anime sources
- [x] Phase 1 — MVP CLI (end-to-end pipeline)
- [x] Phase 2 — Gating (scene-cut + hold detection)
- [ ] **Phase 5 — Model Training** ⚡ CURRENT
  - [x] Training infrastructure (2-phase, GAN, mixed precision, OOM recovery)
  - [x] Hyperparameter sweep (13 experiments, sweep winner: 21.8 dB PSNR)
  - [x] Gradient accumulation for Phase 2 VRAM management
  - [ ] Full training run (50 epochs)
  - [ ] Validation on held-out clips
- [ ] Phase 6 — Quality hardening + performance
- [ ] Phase 7 — GUI + packaging

---

## Legal Note

AInimotion is intended for processing video **you have the rights/permission to use**. It does not include DRM bypassing, ripping tools, or downloading functionality.

---

## Contributing

PRs welcome! Useful contributions:

- Better cut/hold heuristics
- More presets / model configs
- Benchmark suite clips (rights-cleared)
- Packaging + UI improvements

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
