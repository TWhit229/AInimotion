# AInimotion V5 — Multi-Frame Anime Video Interpolation

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)](https://pytorch.org/)

**AInimotion** is a deep learning system for anime video frame interpolation (VFI). V5 is the current architecture: a **multi-frame, layered interpolation model** that uses 7 context frames and temporal attention to produce sharp, artifact-free intermediate frames — specifically designed for anime's unique challenges (line art, flat colors, large inter-frame motion).

> **Goal**: Produce buttery-smooth 48 FPS anime from 24 FPS sources without bending lines, creating ghosting, or hallucinating across scene cuts.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [V5 Pipeline (9 Stages)](#v5-pipeline-9-stages)
- [Loss Functions & Training Schedule](#loss-functions--training-schedule)
- [Dataset Format](#dataset-format)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [How to Train](#how-to-train)
- [Quick Start](#quick-start)
- [Version History](#version-history)
- [License](#license)

---

## Architecture Overview

```
Input: 7 context frames (3 past + 2 anchor + 2 future)
                              |
+---------------------------------------------------------------------+
|                    LayeredInterpolatorV5                              |
|                                                                      |
|  +------------------+                                                |
|  | 1. Spatial Encoder|  Shared FPN, 3 scales: 1x, 1/2x, 1/4x       |
|  |    (per frame)    |  (B,3,H,W) -> [C, 2C, 4C] features           |
|  +--------+---------+                                                |
|           |                                                          |
|  +------------------+                                                |
|  | 2. Temporal Attn  |  Windowed cross-frame attention at 1/4 res    |
|  |    Fusion         |  7 frames x 8x8 windows, 2 layers, 4 heads   |
|  +--------+---------+                                                |
|           |                                                          |
|  +------------------+                                                |
|  | 3. Scene Gate     |  Hard cut detection from correlation stats    |
|  +--------+---------+                                                |
|           |                                                          |
|     +-----+-----+                                                    |
|     |           |                                                    |
|  +--------+  +--------------------------------------------+          |
|  | 4. BG  |  |           Foreground Path                  |          |
|  | Path   |  |  +-----------+     +----------------+      |          |
|  |(affine)|  |  | 5. Warp   |     | 6. Synthesis   |      |          |
|  |        |  |  |  (RIFE)   |     | (Deformable    |      |          |
|  |        |  |  |           |     |  Cross-Temp)   |      |          |
|  |        |  |  +-----+-----+     +-------+--------+      |          |
|  |        |  |        |                   |               |          |
|  |        |  |  +-----------------------------------+     |          |
|  |        |  |  | 7. Motion Router                  |     |          |
|  |        |  |  |    (per-pixel warp vs synthesis)   |     |          |
|  |        |  |  +------------+----------------------+     |          |
|  +---+----+  +---------------+----------------------------+          |
|      |                       |                                       |
|  +----------------------------------+                                |
|  | 8. Layer Compositor              |                                |
|  |    (learned alpha BG/FG blend)   |                                |
|  +------------+---------------------+                                |
|               |                                                      |
|  +----------------------------------+                                |
|  | 9. Edge-Guided Refinement        |                                |
|  |    (line art sharpening residual)|                                |
|  +------------+---------------------+                                |
|               |                                                      |
|         Output: (B, 3, H, W)                                        |
+----------------------------------------------------------------------+
```

### Key V5 Innovations Over V1-V4

| Feature | V1-V4 | V5 |
|---------|-------|----|
| Context | 2 frames (anchor pair only) | **7 frames** (+-3 temporal context) |
| Temporal modeling | None | **Windowed cross-frame attention** (RVRT-inspired) |
| Warp method | AdaCoF (KxK kernel) | **RIFE-style coarse-to-fine** (3-stage) |
| Synthesis | Single-frame decoder | **Deformable cross-temporal attention** (borrow pixels across time) |
| Anti-blur | Laplacian loss | **FFT amplitude loss** (frequency-domain) |
| Line art | No special handling | **Edge-guided refinement** (Sobel-injected residual net) |
| GAN | PatchGAN | **Multi-scale PatchGAN** (full + half res, spectral norm, R1 penalty) |
| Training | 2-phase | **3-phase progressive** (recon -> anti-blur -> GAN) |

---

## V5 Pipeline (9 Stages)

### Stage 1 — Spatial Encoder (`spatial_encoder.py`)

Shared-weight FPN that encodes each of the 7 frames independently.

- **Input**: `(B, 3, H, W)` RGB frame
- **Output**: 3-scale features `[(B, C, H, W), (B, 2C, H/2, W/2), (B, 4C, H/4, W/4)]`
- **Default**: `C = 64`, so scales are 64 / 128 / 256 channels
- **Architecture**: Stem (Conv + GroupNorm + GELU + ResBlock) -> DownBlock x2

**Design rationale**: GroupNorm (groups=8) instead of BatchNorm for stability with small batch sizes on high-VRAM GPUs. GELU everywhere for smooth gradients.

---

### Stage 2 — Temporal Attention Fusion (`temporal_attention.py`)

After spatial encoding, this module fuses information across all 7 frames at 1/4 resolution using windowed cross-frame attention.

- **Input**: List of 7 tensors `(B, 4C, H/4, W/4)`
- **Output**: List of 7 tensors `(B, 4C, H/4, W/4)` — each frame now "knows" about the others
- **Mechanism**: Each spatial window (8x8) attends across all 7 frames simultaneously
- **Complexity**: `O(7 x window^2 x n_windows)` — NOT global attention
- **Layers**: 2 attention layers, each with 4 heads + FFN post-block

**Design rationale**: Windowed (not global) attention keeps VRAM manageable. RVRT showed 2 layers is sufficient for video restoration. Only at 1/4 resolution to save memory.

---

### Stage 3 — Scene Gate (`scene_gate.py`)

Detects hard cuts between the anchor pair (frames 3 and 4) to avoid interpolating across scene changes.

- **Input**: `(B, 1, H/4, W/4)` correlation map between anchor features
- **Output**: `(B,)` boolean tensor + confidence score
- **Mechanism**: Global stats (mean, std, min of correlation) -> 3-layer MLP -> sigmoid

**Design rationale**: Simple and cheap. Scene cuts produce uniformly low correlation, so global stats are sufficient — no need for spatial analysis.

---

### Stage 4 — Background Path (`background_path.py`)

Estimates camera motion via affine transforms and warps a clean background.

- **Input**: Anchor frames + 1/4-res features + timestep
- **Output**: `(B, 3, H, W)` background image + blend weights
- **Mechanism**:
  1. Predict 6 affine parameters from feature difference
  2. Interpolate affine by timestep (identity at t=0, full at t=1)
  3. Warp both anchors with forward/backward affine
  4. Occlusion-aware blending (3-layer CNN on warped pair + difference)

**Design rationale**: Affine initialization is identity (zero init on last linear layer) so the model starts with "no camera motion" and learns from there. Anime backgrounds often have simple pan/zoom that affine handles perfectly.

---

### Stage 5 — Warp Branch (`warp_branch.py`)

RIFE-style coarse-to-fine flow estimation for trackable foreground motion.

- **Input**: Anchor frames `(B, 3, H, W)` + timestep
- **Output**: Warped intermediate frame + forward/backward flows + blend mask
- **Architecture**: 3 `IFBlock` stages:
  1. 1/4 resolution (hidden=64)
  2. 1/2 resolution (hidden=64)
  3. Full resolution (hidden=32)
- Each stage: concat inputs -> 4 conv layers -> predict flow delta + mask

**Design rationale**: RIFE's IFNet is proven fast and effective. Self-contained (doesn't need features from the encoder), so it's a lightweight "fast path" for simple motion. The motion router decides when to use this vs synthesis.

---

### Stage 6 — Synthesis Branch (`synthesis_branch.py`)

Deformable cross-temporal attention for occlusions and complex motion. This is the **key V5 module** that leverages multi-frame context.

- **Input**: Anchor features at 1/4 res + all 7 temporally-fused features + edge maps
- **Output**: `(B, 3, H, W)` synthesized frame
- **Mechanism**:
  1. Edge injection: concat Sobel edge maps (downsampled to 1/4) -> 1x1 conv fuse
  2. 2 layers of `DeformableCrossTemporalAttention`:
     - Query: target frame features
     - Keys/Values: ALL 7 context frames
     - K=9 sampling points per head, 4 heads
     - Offsets clamped with `tanh x max_offset` (prevents overflow)
  3. Decode to RGB: 4 conv blocks -> PixelShuffle 4x upsample -> sigmoid

**Design rationale**: When objects are occluded in the anchor pair but visible in neighboring frames, this module can "borrow" those pixels through learned deformable sampling. Offset clamping (max_offset=32 at 1/4 res = 128px at full res) is critical for training stability.

---

### Stage 7 — Motion Router (`motion_router.py`)

Per-pixel routing between warp (fast/simple) and synthesis (powerful/expensive).

- **Input**: Correlation + anchor features at 1/4 res
- **Output**: `(B, 1, H, W)` routing map in [0, 1]
  - R ~ 0 -> use warp branch (trackable motion)
  - R ~ 1 -> use synthesis branch (occlusions, complex motion)
- **Architecture**: 3 conv layers (in_ch = 1 + 2xfeat_ch -> 64 -> 32 -> 1)

**Design rationale**: Bias initialized to 0.0 (sigmoid -> 0.5), giving balanced routing between warp and synthesis from the start. Auxiliary balance and entropy losses prevent either branch from collapsing during training.

---

### Stage 8 — Layer Compositor (`compositor.py`)

Blends foreground and background with a learned alpha mask.

- **Input**: Foreground, background, anchor frames, edge map (13 channels total)
- **Output**: Composite frame + alpha mask
- **Architecture**: 3 conv layers -> sigmoid alpha

**Design rationale**: The model learns where background (affine-warped) should show through vs foreground (flow/synthesis). Anime has clear layer separation (painted backgrounds + animated characters), so layer-based compositing is natural.

---

### Stage 9 — Edge-Guided Refinement (`compositor.py`)

Lightweight residual correction focused on sharpening anime line art.

- **Input**: Composite + Sobel edge maps from both anchors (5 channels)
- **Output**: Refined frame clamped to [0, 1]
- **Architecture**: Stem -> 3 residual blocks (32 channels) -> head (initialized near zero)

**Design rationale**: The head weights are initialized at 0.01x scale so the refinement starts as near-identity. This prevents early training instability while still giving the model capacity to sharpen lines later. The Sobel edges guide where refinement is most needed.

---

## Loss Functions & Training Schedule

### Loss Components (`losses.py`)

| Loss | Formula | Purpose |
|------|---------|---------|
| **Charbonnier L1** | `sqrt((pred-gt)^2 + e^2)` | Main reconstruction, smooth near zero |
| **Edge-weighted L1** | `\|pred-gt\| x sobel_weight(gt)` | Focus error on line art regions |
| **FFT Amplitude** | `\|FFT_amp(pred) - FFT_amp(gt)\|` weighted by frequency | Anti-blur: penalizes missing high frequencies |
| **LPIPS** | VGG-based learned perceptual metric | Perceptual quality (run in fp32) |
| **Census** | Patch-based structural consistency | Robust to illumination changes |
| **GAN** | Relativistic average BCE (multi-scale PatchGAN) | Perceptual sharpness at both resolutions |

### 3-Phase Progressive Schedule

```
Epochs    0-100:  Charbonnier (1.0) + Edge L1 (0.5)
                  -> Learn basic reconstruction

Epochs  100-300:  + FFT Amplitude (ramps 0.0 -> 1.0)
                  -> Anti-blur hardening

Epochs  300-500:  FFT at full weight (1.0)
                  -> Consolidate frequency-aware synthesis

Epochs  500-550:  + GAN warmup (weight = 0.01)
                  -> Discriminator initialization

Epochs  550-650:  GAN ramps (0.01 -> 0.1)
                  -> Progressive adversarial fine-tuning

Epochs  650-800:  GAN at full weight (0.1)
                  -> Final adversarial refinement
```

### Routing Auxiliary Losses

To prevent either branch from collapsing during training:

- **Balance loss**: Penalizes deviation of mean routing from 0.5
- **Entropy loss**: Encourages exploration (maximized when R=0.5)
- **Per-branch supervision**: Independent Charbonnier on warp and synthesis outputs

### Discriminator (`discriminator.py`)

**Multi-Scale PatchGAN** with stability measures:

- **Two scales**: Full resolution + 2x downscaled (catches blur at both fine and coarse scales)
- **Spectral normalization** on all conv layers (Lipschitz constraint)
- **Minibatch standard deviation** layer (from ProGAN — detects mode collapse)
- **R1 gradient penalty** (weight=10.0) on real images (prevents discriminator sharpness explosion)
- **Relativistic average** loss formulation (real should be "more real" than fake on average)

---

## Dataset Format

### V5 Sequence Dataset (`ainimotion/data/dataset_v5.py`)

V5 uses **9 consecutive frames** per training sample:

```
Extracted:   frame_000  frame_001  frame_002  frame_003  frame_004  frame_005  frame_006  frame_007  frame_008
                                                            ^ GT
Context:     [ 0          1          2          3     ]  skip  [    5          6          7          8     ]
             |------- 4 frames before --------|                |-------- 4 frames after --------------|

Model sees:  7 context frames (indices 0,1,2,3, 5,6,7,8 -> remapped to 0-6)
             Anchor pair: context[3] and context[4] (original frames 3 and 5)
             Ground truth: frame 4 (the one between anchors)
```

### Directory Structure

```
training_data/v5/
  series_ep01_seg000_seq000001_m0.0234/
    frame_000.png
    frame_001.png
    ...
    frame_008.png
```

The `_m0.0234` suffix encodes **motion score** (mean frame difference), used for weighted sampling — higher motion scenes are sampled more frequently as training progresses.

### Data Augmentation

- Random 256x256 crops
- Random horizontal flip
- Random vertical flip
- Random temporal reversal (sequence played backwards)
- Random rotation (0, 90, 180, 270 degrees)
- Random color jitter (brightness, contrast, saturation, hue)

### Extraction Pipeline

Convert anime videos into 9-frame training sequences with GPU-accelerated deduplication:

```bash
python scripts/extract_sequences.py --help
```

Features:
- **GPU dedup**: Detects duplicate frames (held drawings) in-memory on GPU
- **Hardware decoding**: Auto-detects codec (H264/H265/AV1/VP9) and uses NVIDIA CUVID when available
- **Scene cut detection**: Identifies hard cuts to avoid cross-scene sequences
- **Motion scoring**: Computes per-sequence motion score for curriculum learning

### Dataset Balancing

Stratified sampling to prevent any single anime series from dominating:

```bash
python scripts/balance_dataset.py --help
```

---

## Project Structure

```
AInimotion/
|-- README.md                              # This document
|-- CHANGELOG.md                           # Version history and changes
|-- LICENSE                                # MIT License
|-- pyproject.toml                         # Package metadata & dependencies
|-- requirements.txt                       # Pip requirements
|-- .gitignore
|
|-- ainimotion/                            # Main Python package
|   |-- __init__.py
|   |-- data/
|   |   |-- __init__.py
|   |   +-- dataset_v5.py                  # V5 9-frame sequence dataset
|   +-- models/
|       +-- interp_v5/                     # V5 architecture (12 modules)
|           |-- __init__.py                # Public API + build_model()
|           |-- layered_interp.py          # Main model orchestrator
|           |-- spatial_encoder.py         # Stage 1: Shared FPN encoder
|           |-- temporal_attention.py      # Stage 2: Windowed cross-frame attention
|           |-- scene_gate.py              # Stage 3: Hard cut detection
|           |-- background_path.py         # Stage 4: Affine BG estimation
|           |-- warp_branch.py             # Stage 5: RIFE-style flow
|           |-- synthesis_branch.py        # Stage 6: Deformable cross-temporal
|           |-- motion_router.py           # Stage 7: Per-pixel warp vs synth
|           |-- compositor.py              # Stage 8-9: Layer blend + edge refine
|           |-- discriminator.py           # Multi-scale PatchGAN + R1 penalty
|           +-- losses.py                  # All loss functions + schedule
|
|-- scripts/
|   |-- extract_sequences.py               # Video -> 9-frame sequences (GPU dedup)
|   |-- balance_dataset.py                 # Stratified dataset balancing
|   +-- train_v5.py                        # 3-phase training loop
|
|-- configs/
|   +-- train_v5_5090.yaml                 # RTX 5090 training config
|
+-- archive/                               # Historical V1-V4 artifacts
    |-- checkpoints/                       # Old training checkpoints
    +-- finished_models/                   # V1-V3 final weights
```

---

## Setup & Installation

### Prerequisites

- **Python**: 3.10 or higher
- **GPU**: NVIDIA GPU with 24+ GB VRAM (developed on RTX 5090, 32 GB)
- **CUDA**: 12.1+ (12.8+ recommended for PyTorch 2.9)
- **Storage**: ~50 GB for dataset + model checkpoints

### Installation

```bash
# Clone
git clone https://github.com/TWhit229/AInimotion.git
cd AInimotion

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# Install PyTorch with CUDA (see https://pytorch.org/get-started/locally/)
# Example for CUDA 12.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install AInimotion and remaining dependencies
pip install -e .
```

### Verify Installation

```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

from ainimotion.models.interp_v5 import build_model
model = build_model()
params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {params:,}")
```

---

## How to Train

### Step 1: Prepare Training Data

Extract 9-frame sequences from your anime video collection:

```bash
python scripts/extract_sequences.py \
    --input /path/to/videos \
    --output /path/to/training_data
```

Balance the dataset across series and motion levels:

```bash
python scripts/balance_dataset.py \
    --input /path/to/training_data \
    --output /path/to/balanced_data
```

### Step 2: Configure

Edit `configs/train_v5_5090.yaml` to match your setup:

```yaml
# Key settings to adjust:
train_dir: "/path/to/training_data"
val_dir: "/path/to/validation_data"
batch_size: 7            # Reduce if <32GB VRAM
num_workers: 4           # Match your CPU cores
wandb_project: "my-run"  # Your W&B project name
```

### Step 3: Train

```bash
python scripts/train_v5.py --config configs/train_v5_5090.yaml
```

Training is managed by the loss schedule automatically based on epoch number:
- **Phase 1** (epochs 0-300): Reconstruction + progressive FFT anti-blur
- **Phase 2** (epochs 300-500): Full anti-blur hardening
- **Phase 3** (epochs 500-800): GAN fine-tuning with progressive weight ramp

Checkpoints are saved every epoch (last 5 kept). Validation runs every 5 epochs.

### Resume from Checkpoint

```bash
python scripts/train_v5.py --config configs/train_v5_5090.yaml --resume checkpoints/latest.pt
```

### VRAM Reference

| GPU | Batch Size | Peak VRAM | Notes |
|-----|-----------|-----------|-------|
| RTX 5090 (32 GB) | 7 | ~30.6 GB | Default config |
| RTX 4090 (24 GB) | 4 | ~18 GB | Reduce `batch_size` in config |
| RTX 3090 (24 GB) | 3-4 | ~16-18 GB | May need `gradient_accumulation_steps: 2` |

All training uses mixed precision (bfloat16) by default.

---

## Quick Start

```python
from ainimotion.models.interp_v5 import build_model
import torch

model = build_model()
frames = [torch.randn(1, 3, 256, 256) for _ in range(7)]

with torch.no_grad():
    result = model(frames)
    print(f"Output: {result['output'].shape}")         # (1, 3, 256, 256)
    print(f"Routing: {result['routing_map'].mean():.3f}")  # near 0.5 (balanced)

params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {params:,}")
```

### Model Outputs

The model returns a dictionary with:

| Key | Shape | Description |
|-----|-------|-------------|
| `output` | `(B, 3, H, W)` | Final interpolated frame |
| `routing_map` | `(B, 1, H, W)` | Per-pixel warp (0) vs synthesis (1) |
| `warp_out` | `(B, 3, H, W)` | Warp branch output |
| `synth_out` | `(B, 3, H, W)` | Synthesis branch output |
| `bg_out` | `(B, 3, H, W)` | Background path output |
| `alpha` | `(B, 1, H, W)` | FG/BG compositing mask |
| `scene_cut` | `(B,)` | Scene cut detection flag |
| `flow_f` | `(B, 2, H, W)` | Forward optical flow |
| `flow_b` | `(B, 2, H, W)` | Backward optical flow |

---

## Version History

| Version | Architecture | PSNR (ATD-12K) | Status |
|---------|-------------|----------------|--------|
| V1 | AdaCoF K=9, FPN, Layer Separation | **30.86 dB** | Finished |
| V2 | + Correlation Volume, Improved Flow | — | Finished |
| V3 | + Motion Complexity Router, Deformable Synthesis | — | Finished |
| **V5** | Multi-frame (7), Temporal Attention, RIFE Warp, Deformable Cross-Temporal, FFT Loss, Edge Refinement, Multi-Scale PatchGAN | **33.78 dB** | **Complete** |

---

## Implementation Notes

Key technical details for anyone reading or modifying the code:

- **bfloat16 AMP** — no GradScaler needed; safer than fp16 for this architecture
- **LPIPS runs in fp32** — VGG BatchNorm produces NaN under AMP otherwise
- **FFT loss runs in fp32** — rfft2 precision degrades in half precision
- **torch.compile disabled for affine_grid** — dynamic shape recompile crash in backward pass
- **GroupNorm(8)** everywhere instead of BatchNorm — stable at batch size 4-7
- **Offset clamping** in synthesis branch (`tanh x max_offset`) — prevents deformable overflow
- **Edge refinement head init at 0.01x** — starts near-identity, prevents early instability
- **Routing bias = 0.0** — balanced warp/synthesis exploration (previous -1.0 caused routing collapse)

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Legal Note

AInimotion is intended for processing video you have the rights/permission to use. It does not include DRM bypassing, ripping tools, or downloading functionality.
