# AInimotion

**AInimotion** is a GPU-accelerated, offline anime video enhancer that combines:

- **Frame interpolation**: *24 → 48 FPS* (motion-adaptive, anime-safe)
- **Upscale/restoration**: *1080p → 1440p (2K)* (line-art preserving)
- **Anime-first safeguards**: **scene-cut detection** + **hold/still detection** (so you don't get wobbly outlines or "fake 60fps" artifacts)

> Goal: "buttery smooth" motion and cleaner frames **without** bending line art, shimmering edges, or interpolating across hard cuts.

---

## Current Status: Ready for Training 🚀

**65,906 triplets extracted** from diverse anime sources. Training infrastructure ready!

### Quick Start Training

```powershell
cd "C:\Projects\AInimotion"
python -m ainimotion.training.train --config configs/interp_training.yaml --data "E:\Triplets"
```

### Training Features

| Feature | Description |
|---------|-------------|
| **LayeredInterpolator** | FPN + Scene Gate + AdaCoF + Background Flow |
| **GAN Training** | PatchDiscriminator with LSGAN loss |
| **Mixed Precision** | FP16 for faster training, lower VRAM |
| **Progress Bars** | tqdm progress for epochs and batches |
| **Checkpointing** | Auto-save every 10 epochs, resume support |
| **TensorBoard** | Real-time loss visualization |

### Requirements

- **GPU**: NVIDIA RTX 3070 or better (16GB+ VRAM recommended)
- **PyTorch**: 2.9+ with CUDA 12.8+
- **Dataset**: 65,906 triplets in E:\Triplets

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
      dataset.py          # Training dataset loader
    models/
      interp/             # Interpolation model
    training/
      train.py            # Training loop
      losses.py           # Loss functions
      discriminator.py    # GAN discriminator
  scripts/
    extract_triplets.py   # Training data extraction ✅
    run_all_overnight.ps1 # Batch processing script ✅
    run_clean.ps1         # Clean sources script ✅
    run_hardsubs.ps1      # Hardsub sources script ✅
    run_test.ps1          # Test script ✅
  configs/
    interp_training.yaml  # Training configuration
```

---

## Roadmap

- [x] Phase 0 — Foundations (platform target, inference runtime, presets)
- [x] **Phase 0.5 — Training Data Pipeline** ✅
  - [x] Triplet extraction script
  - [x] Parallel processing (8 workers)
  - [x] 720p output for efficiency
  - [x] Chunked processing for memory
  - [x] Skip intro (logos/credits)
  - [x] Hardsub support (crop top/bottom)
- [ ] Phase 1 — MVP CLI (end-to-end pipeline)
- [ ] Phase 2 — Quality hardening (anime-specific tuning)
- [ ] Phase 3 — Performance + VRAM stability
- [ ] Phase 4 — Model training on extracted triplets
- [ ] Phase 5 — GUI + packaging

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
