# AInimotion V5 — Multi-Frame Anime Video Interpolation

**AInimotion** is a deep learning system for anime video frame interpolation (VFI). V5 is the current architecture: a **multi-frame, layered interpolation model** that uses 7 context frames and temporal attention to produce sharp, artifact-free intermediate frames — specifically designed for anime's unique challenges (line art, flat colors, large inter-frame motion).

> **Goal**: Produce buttery-smooth 48 FPS anime from 24 FPS sources without bending lines, creating ghosting, or hallucinating across scene cuts.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [V5 Pipeline (9 Stages)](#v5-pipeline-9-stages)
- [Module Reference](#module-reference)
- [Loss Functions & Training Schedule](#loss-functions--training-schedule)
- [Dataset Format](#dataset-format)
- [Project Structure](#project-structure)
- [Setup & Requirements](#setup--requirements)
- [How to Train](#how-to-train)
- [Version History](#version-history)
- [License](#license)

---

## Architecture Overview

```
Input: 7 context frames (3 past + 2 anchor + 2 future)
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    LayeredInterpolatorV5                            │
│                                                                     │
│  ┌──────────────────┐                                               │
│  │ 1. Spatial Encoder│  Shared FPN, 3 scales: 1x, 1/2x, 1/4x      │
│  │    (per frame)    │  (B,3,H,W) → [C, 2C, 4C] features           │
│  └────────┬─────────┘                                               │
│           ↓                                                         │
│  ┌──────────────────┐                                               │
│  │ 2. Temporal Attn  │  Windowed cross-frame attention at 1/4 res   │
│  │    Fusion         │  7 frames × 8×8 windows, 2 layers, 4 heads   │
│  └────────┬─────────┘                                               │
│           ↓                                                         │
│  ┌──────────────────┐                                               │
│  │ 3. Scene Gate     │  Hard cut detection from correlation stats   │
│  └────────┬─────────┘                                               │
│           ↓                                                         │
│     ┌─────┴─────┐                                                   │
│     ↓           ↓                                                   │
│  ┌────────┐  ┌────────────────────────────────────────────┐         │
│  │ 4. BG  │  │           Foreground Path                  │         │
│  │ Path   │  │  ┌───────────┐     ┌────────────────┐      │         │
│  │(affine)│  │  │ 5. Warp   │     │ 6. Synthesis   │      │         │
│  │        │  │  │  (RIFE)   │     │ (Deformable    │      │         │
│  │        │  │  │           │     │  Cross-Temp)   │      │         │
│  │        │  │  └─────┬─────┘     └───────┬────────┘      │         │
│  │        │  │        ↓                   ↓               │         │
│  │        │  │  ┌───────────────────────────────────┐     │         │
│  │        │  │  │ 7. Motion Router                  │     │         │
│  │        │  │  │    (per-pixel warp vs synthesis)   │     │         │
│  │        │  │  └────────────┬──────────────────────┘     │         │
│  └───┬────┘  └───────────────┼────────────────────────────┘         │
│      ↓                       ↓                                      │
│  ┌──────────────────────────────────┐                               │
│  │ 8. Layer Compositor              │                               │
│  │    (learned alpha BG/FG blend)   │                               │
│  └────────────┬─────────────────────┘                               │
│               ↓                                                     │
│  ┌──────────────────────────────────┐                               │
│  │ 9. Edge-Guided Refinement        │                               │
│  │    (line art sharpening residual)│                               │
│  └────────────┬─────────────────────┘                               │
│               ↓                                                     │
│         Output: (B, 3, H, W)                                        │
└─────────────────────────────────────────────────────────────────────┘
```

### Key V5 Innovations Over V1-V4

| Feature | V1-V4 | V5 |
|---------|-------|----|
| Context | 2 frames (anchor pair only) | **7 frames** (±3 temporal context) |
| Temporal modeling | None | **Windowed cross-frame attention** (RVRT-inspired) |
| Warp method | AdaCoF (K×K kernel) | **RIFE-style coarse-to-fine** (3-stage) |
| Synthesis | Single-frame decoder | **Deformable cross-temporal attention** (borrow pixels across time) |
| Anti-blur | Laplacian loss | **FFT amplitude loss** (frequency-domain) |
| Line art | No special handling | **Edge-guided refinement** (Sobel-injected residual net) |
| GAN | PatchGAN | **Multi-scale PatchGAN** (full + half res, spectral norm, R1 penalty) |
| Training | 2-phase | **3-phase progressive** (recon → anti-blur → GAN) |

---

## V5 Pipeline (9 Stages)

### Stage 1 — Spatial Encoder (`spatial_encoder.py`)

Shared-weight FPN that encodes each of the 7 frames independently.

- **Input**: `(B, 3, H, W)` RGB frame
- **Output**: 3-scale features `[(B, C, H, W), (B, 2C, H/2, W/2), (B, 4C, H/4, W/4)]`
- **Default**: `C = 64`, so scales are 64 / 128 / 256 channels
- **Architecture**: Stem (Conv + GroupNorm + GELU + ResBlock) → DownBlock ×2

**Design rationale**: GroupNorm (groups=8) instead of BatchNorm for stability with small batch sizes on high-VRAM GPUs. GELU everywhere for smooth gradients.

---

### Stage 2 — Temporal Attention Fusion (`temporal_attention.py`)

After spatial encoding, this module fuses information across all 7 frames at 1/4 resolution using windowed cross-frame attention.

- **Input**: List of 7 tensors `(B, 4C, H/4, W/4)`
- **Output**: List of 7 tensors `(B, 4C, H/4, W/4)` — each frame now "knows" about the others
- **Mechanism**: Each spatial window (8×8) attends across all 7 frames simultaneously
- **Complexity**: `O(7 × window² × n_windows)` — NOT global attention
- **Layers**: 2 attention layers, each with 4 heads + FFN post-block

**Design rationale**: Windowed (not global) attention keeps VRAM manageable. RVRT showed 2 layers is sufficient for video restoration. Only at 1/4 resolution to save memory.

---

### Stage 3 — Scene Gate (`scene_gate.py`)

Detects hard cuts between the anchor pair (frames 3 and 4) to avoid interpolating across scene changes.

- **Input**: `(B, 1, H/4, W/4)` correlation map between anchor features
- **Output**: `(B,)` boolean tensor + confidence score
- **Mechanism**: Global stats (mean, std, min of correlation) → 3-layer MLP → sigmoid

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
- Each stage: concat inputs → 4 conv layers → predict flow delta + mask

**Design rationale**: RIFE's IFNet is proven fast and effective. Self-contained (doesn't need features from the encoder), so it's a lightweight "fast path" for simple motion. The motion router decides when to use this vs synthesis.

---

### Stage 6 — Synthesis Branch (`synthesis_branch.py`)

Deformable cross-temporal attention for occlusions and complex motion. This is the **key V5 module** that leverages multi-frame context.

- **Input**: Anchor features at 1/4 res + all 7 temporally-fused features + edge maps
- **Output**: `(B, 3, H, W)` synthesized frame
- **Mechanism**:
  1. Edge injection: concat Sobel edge maps (downsampled to 1/4) → 1×1 conv fuse
  2. 2 layers of `DeformableCrossTemporalAttention`:
     - Query: target frame features
     - Keys/Values: ALL 7 context frames
     - K=9 sampling points per head, 4 heads
     - Offsets clamped with `tanh × max_offset` (prevents overflow)
  3. Decode to RGB: 4 conv blocks → PixelShuffle 4× upsample → sigmoid

**Design rationale**: When objects are occluded in the anchor pair but visible in neighboring frames, this module can "borrow" those pixels through learned deformable sampling. Offset clamping (max_offset=32 at 1/4 res = 128px at full res) is critical for training stability.

---

### Stage 7 — Motion Router (`motion_router.py`)

Per-pixel routing between warp (fast/simple) and synthesis (powerful/expensive).

- **Input**: Correlation + anchor features at 1/4 res
- **Output**: `(B, 1, H, W)` routing map in [0, 1]
  - R ≈ 0 → use warp branch (trackable motion)
  - R ≈ 1 → use synthesis branch (occlusions, complex motion)
- **Architecture**: 3 conv layers (in_ch = 1 + 2×feat_ch → 64 → 32 → 1)

**Design rationale**: Bias initialized to -1.0 (sigmoid → ~0.27), starting the model biased toward the warp path. This prevents the synthesis branch from dominating early in training when it hasn't learned meaningful representations yet.

---

### Stage 8 — Layer Compositor (`compositor.py`)

Blends foreground and background with a learned alpha mask.

- **Input**: Foreground, background, anchor frames, edge map (13 channels total)
- **Output**: Composite frame + alpha mask
- **Architecture**: 3 conv layers → sigmoid alpha

**Design rationale**: The model learns where background (affine-warped) should show through vs foreground (flow/synthesis). Anime has clear layer separation (painted backgrounds + animated characters), so layer-based compositing is natural.

---

### Stage 9 — Edge-Guided Refinement (`compositor.py`)

Lightweight residual correction focused on sharpening anime line art.

- **Input**: Composite + Sobel edge maps from both anchors (5 channels)
- **Output**: Refined frame clamped to [0, 1]
- **Architecture**: Stem → 3 residual blocks (32 channels) → head (initialized near zero)

**Design rationale**: The head weights are initialized at 0.01× scale so the refinement starts as near-identity. This prevents early training instability while still giving the model capacity to sharpen lines later. The Sobel edges guide where refinement is most needed.

---

## Loss Functions & Training Schedule

### Loss Components (`losses.py`)

| Loss | Formula | Purpose |
|------|---------|---------|
| **Charbonnier L1** | `√((pred-gt)² + ε²)` | Main reconstruction, smooth near zero |
| **Edge-weighted L1** | `|pred-gt| × sobel_weight(gt)` | Focus error on line art regions |
| **FFT Amplitude** | `|FFT_amp(pred) - FFT_amp(gt)|` weighted by frequency | Anti-blur: penalizes missing high frequencies |
| **GAN** | Relativistic average BCE (multi-scale PatchGAN) | Perceptual sharpness at both resolutions |

### 3-Phase Progressive Schedule

```
Epochs    0-100:  Charbonnier (1.0) + Edge L1 (0.5)
                  → Learn basic reconstruction

Epochs  100-300:  + FFT Amplitude (ramps 0.0 → 1.0)
                  → Anti-blur hardening

Epochs  300-500:  FFT at full weight (1.0)
                  → Consolidate frequency-aware synthesis

Epochs  500-550:  + GAN warmup (weight = 0.01)
                  → Discriminator initialization

Epochs  550-650:  GAN ramps (0.01 → 0.1)
                  → Progressive adversarial fine-tuning

Epochs  650-800:  GAN at full weight (0.1)
                  → Final adversarial refinement
```

### Discriminator (`discriminator.py`)

**Multi-Scale PatchGAN** with stability measures:

- **Two scales**: Full resolution + 2× downscaled (catches blur at both fine and coarse scales)
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
                                                            ↑ GT
Context:     [ 0          1          2          3     ]  skip  [    5          6          7          8     ]
             └─────── 4 frames before ────────┘                └──────── 4 frames after ────────────────┘

Model sees:  7 context frames (indices 0,1,2,3, 5,6,7,8 → remapped to 0-6)
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

The `_m0.0234` suffix encodes **motion score** (mean frame difference), used for weighted sampling — higher motion scenes are sampled more frequently.

### Data Augmentation

- Random 256×256 crops
- Random horizontal flip
- Random vertical flip
- Random temporal reversal (sequence played backwards)

### Extraction

```powershell
python scripts/extract_sequences.py --help  # See options
```

---

## Project Structure

```
AInimotion/
├── README.md                              # This document
├── LICENSE                                # MIT License
├── pyproject.toml                         # Package config
├── .gitignore
│
├── ainimotion/                            # Main package
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset_v5.py                  # V5 multi-frame dataset
│   └── models/
│       └── interp_v5/                     # V5 architecture (12 files)
│           ├── __init__.py                # Public API + docstring
│           ├── layered_interp.py          # Main model class + build_model()
│           ├── spatial_encoder.py         # Stage 1: Shared FPN encoder
│           ├── temporal_attention.py      # Stage 2: Windowed cross-frame attention
│           ├── scene_gate.py              # Stage 3: Hard cut detection
│           ├── background_path.py         # Stage 4: Affine BG estimation
│           ├── warp_branch.py             # Stage 5: RIFE-style flow
│           ├── synthesis_branch.py        # Stage 6: Deformable cross-temporal
│           ├── motion_router.py           # Stage 7: Per-pixel warp vs synth
│           ├── compositor.py              # Stage 8-9: Layer blend + edge refine
│           ├── discriminator.py           # Multi-scale PatchGAN + R1 penalty
│           └── losses.py                  # All loss functions + schedule
│
├── scripts/
│   ├── extract_sequences.py               # 9-frame sequence extraction
│   └── profile_v5_vram.py                 # VRAM usage profiling
│
├── configs/                               # Training configs (YAML)
│
├── training_data/                         # ATD-12K dataset
│   ├── train_10k/
│   ├── test_2k_540p/
│   ├── test_2k_original/
│   ├── test_2k_annotations/
│   └── test_2k_pre_calc_sgm_flows/
│
├── finished_models/                       # Final trained weights (v1-v3)
│   ├── v1_epoch299_psnr30.86.pt           # V1 (AdaCoF) — 30.86 dB
│   ├── v2_epoch299.pt                     # V2 (Correlation Volume)
│   └── v3_epoch299.pt                     # V3 (Motion Complexity Router)
│
├── Final Proj/                            # Academic deliverables
│   ├── report/                            # CVPR-format LaTeX report
│   ├── presentation/                      # Beamer slides + speaker notes
│   └── study_guide/                       # 44-page teaching guide
│
└── archive/                               # Archived V1-V4 artifacts
    ├── checkpoints/                       # Old training checkpoints
    ├── ainimotion/                        # Old modules (cli, gating, etc.)
    ├── Clips/                             # Demo videos
    ├── docs/                              # Old architecture docs
    ├── runs/                              # TensorBoard logs
    └── wandb/                             # W&B run logs
```

---

## Setup & Requirements

### Hardware

- **GPU**: NVIDIA RTX 5090 (32 GB VRAM) or similar with 24+ GB
- **Storage**: ~50 GB for dataset + model checkpoints

### Software

```powershell
# Clone
git clone https://github.com/TWhit229/AInimotion.git
cd AInimotion

# Install
pip install -e .

# Additional dependencies
pip install torch torchvision  # PyTorch 2.9+ with CUDA 12.8+
pip install pillow tqdm wandb  # Training utilities
```

### VRAM Profiling

Before training, determine your maximum batch size:

```powershell
python scripts/profile_v5_vram.py
```

This runs forward+backward passes at increasing batch sizes with `base_channels=64` at 256×256 resolution.

---

## How to Train

### Step 1: Extract Training Sequences

Convert ATD-12K triplets into 9-frame sequences:

```powershell
python scripts/extract_sequences.py --input training_data/train_10k --output training_data/v5
```

### Step 2: Profile VRAM

```powershell
python scripts/profile_v5_vram.py
```

### Step 3: Train (3 Phases)

Training is managed by the loss schedule automatically based on epoch number:
- **Phase 1 (epochs 0-300)**: Reconstruction + progressive FFT anti-blur
- **Phase 2 (epochs 300-500)**: Full anti-blur hardening
- **Phase 3 (epochs 500-800)**: GAN fine-tuning with progressive weight ramp

### Quick Model Test

```python
from ainimotion.models.interp_v5 import build_model
import torch

model = build_model()
frames = [torch.randn(1, 3, 256, 256) for _ in range(7)]

with torch.no_grad():
    result = model(frames)
    print(f"Output: {result['output'].shape}")        # (1, 3, 256, 256)
    print(f"Routing: {result['routing_map'].mean():.3f}")  # near 0.27 (warp-biased)

params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {params:,}")
```

---

## Version History

| Version | Key Architecture | PSNR (ATD-12K) | Status |
|---------|-----------------|----------------|--------|
| V1 | AdaCoF K=9, FPN, Layer Separation | 30.86 dB | ✅ Finished |
| V2 | + Correlation Volume, Improved Flow | — | ✅ Finished |
| V3 | + Motion Complexity Router, Deformable Synthesis | — | ✅ Finished |
| **V5** | **Multi-frame (7), Temporal Attention, RIFE Warp, Deformable Cross-Temporal, FFT Loss, Edge Refinement, Multi-Scale PatchGAN** | — | 🔧 Architecture complete, training pending |

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Legal Note

AInimotion is intended for processing video you have the rights/permission to use. It does not include DRM bypassing, ripping tools, or downloading functionality.
