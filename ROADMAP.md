# AInimotion Development Roadmap

A practical guide from empty repo to working anime video enhancer.

---

## Phase 0: Project Setup ✅
*Get the skeleton in place*

- [x] Create the package structure
- [x] Set up `pyproject.toml` — defines dependencies, entry points, metadata
- [x] Create a dev environment
- [x] Verify FFmpeg is available — required for decode/encode

---

## Phase 0.5: Training Data Pipeline ✅ COMPLETE
*Build the triplet extraction system for model training*

| Step | Task | Status |
|------|------|--------|
| 0.5.1 | **Triplet extraction script** — extract F1, F2, F3 frame triplets | ✅ |
| 0.5.2 | **SSIM-based filtering** — filter out static/duplicate frames | ✅ |
| 0.5.3 | **Parallel processing** — 8-worker ThreadPoolExecutor | ✅ |
| 0.5.4 | **720p output** — extract frames at 720p for efficiency | ✅ |
| 0.5.5 | **Chunked processing** — 5-minute chunks to manage temp storage | ✅ |
| 0.5.6 | **Skip intro** — skip first 120s to avoid logos/credits | ✅ |
| 0.5.7 | **Hardsub support** — crop top/bottom to remove burned-in subs | ✅ |
| 0.5.8 | **Helper scripts** — PowerShell scripts for batch processing | ✅ |
| 0.5.9 | **Progress bars** — tqdm progress for video/chunk/filtering | ✅ |
| 0.5.10 | **JPEG output** — 10x smaller files with quality 95 default | ✅ |

**Milestone:** ✅ **65,906 triplets extracted** from diverse anime sources!

### Dataset Statistics
- **Total triplets:** 65,906
- **Format:** PNG (existing) / JPEG (new default)
- **Resolution:** 720p
- **Sources:** Movies, TV series, various studios

---

## Phase 1: Build the Pipeline Skeleton ✅ COMPLETE
*Wire up the stages with placeholder logic*

**Goal:** Input video → goes through all stages → output video (even if no processing happens yet)

| Step | Task | Status |
|------|------|--------|
| 1.1 | **FFmpeg decode** — extract frames to temp dir or memory | ✅ |
| 1.2 | **FFmpeg encode** — take frames and encode back to video + remux audio | ✅ |
| 1.3 | **CLI entry point** — `ainimotion enhance input.mkv --out output.mkv` does round-trip | ✅ |

**Milestone:** ✅ CLI runs with `ainimotion enhance` and `ainimotion info` commands.

---

## Phase 2: Add Gating (The "Anime-Safe" Core) ✅ COMPLETE
*Detect scene cuts and holds BEFORE adding any ML*

| Step | Task | Status |
|------|------|--------|
| 2.1 | **Frame comparison** — downscale + compute luma similarity (SSIM or MAD) | ✅ |
| 2.2 | **Classify intervals** — cut / hold / motion based on thresholds | ✅ |
| 2.3 | **Frame duplication** — for 24→48 FPS, duplicate frames for cuts/holds | ✅ |
| 2.4 | **Unit tests** — test gating logic with synthetic frame pairs | ✅ |

**Milestone:** ✅ Gating module detects CUT/HOLD/MOTION. Ready for interpolation integration.

---

## Phase 3: Plug In Interpolation (Model A)
*Add the ML model for generating in-between frames*

| Step | Task | Status |
|------|------|--------|
| 3.1 | **Choose baseline model** — RIFE, IFRNet, or similar (pretrained) | [ ] |
| 3.2 | **Integrate inference** — PyTorch or ONNX Runtime | [ ] |
| 3.3 | **Wire into pipeline** — when gating says "motion", call the model | [ ] |
| 3.4 | **Test on real clips** — verify interpolation quality on anime footage | [ ] |

**Milestone:** Real interpolation for motion, duplication for holds/cuts. First "enhanced" outputs.

---

## Phase 4: Plug In Upscaling (Model B)
*Add the upscaler/restorer*

| Step | Task | Status |
|------|------|--------|
| 4.1 | **Choose baseline model** — Real-ESRGAN, Compact, or anime-tuned variant | [ ] |
| 4.2 | **Add tiling** — handle large frames without OOM | [ ] |
| 4.3 | **Wire after interpolation** — upscale every output frame | [ ] |
| 4.4 | **Test on real clips** — verify upscale quality, check for artifacts | [ ] |

**Milestone:** Full pipeline working: Interpolate → Upscale. End-to-end 1080p@24fps → 1440p@48fps.

---

## Phase 5: Model Training ⚡ CURRENT
*Train custom models on extracted triplets*

| Step | Task | Status |
|------|------|--------|
| 5.1 | **Dataset loader** — load triplets for training | ✅ |
| 5.2 | **Training loop** — implement training with loss functions | ✅ |
| 5.3 | **Mixed precision** — FP16/FP32 via GradScaler | ✅ |
| 5.4 | **Checkpointing** — per-epoch + per-batch, auto-resume | ✅ |
| 5.5 | **W&B + TensorBoard** — real-time loss/PSNR monitoring | ✅ |
| 5.6 | **Two-phase training** — Phase 1 reconstruction, Phase 2 GAN | ✅ |
| 5.7 | **GAN stabilization** — LSGAN, label smoothing, adaptive lr_d, D warmup | ✅ |
| 5.8 | **OOM recovery** — auto-checkpoint + retry with backoff | ✅ |
| 5.9 | **Hyperparameter sweep** — 13 experiments, automated comparison | ✅ |
| 5.10 | **Gradient accumulation** — Phase 2 VRAM management | ✅ |
| 5.11 | **Full training run** — 50 epochs with sweep-winner config | 🔄 In Progress |
| 5.12 | **Validation** — test on held-out anime clips | [ ] |

### Sweep Results (13 Experiments)

| Rank | Config | PSNR (5 epochs) |
|------|--------|------------------|
| 🥇 1 | **heavy_motion_max** (K=9, crop 384, perc=0.1) | **21.80 dB** |
| 🥈 2 | motion_focused (crop 384, perc=0.1) | 21.73 dB |
| 🥉 3 | crop_384 (384 crops only) | 21.69 dB |
| 11 | lr_baseline (current defaults) | 21.06 dB |

**Key finding:** 384×384 crops (+0.74 dB), K=9 kernels, and higher perceptual weight were the biggest wins.

### Training Configuration (Sweep Winner)
- **Model:** LayeredInterpolator (96 base channels, K=9 AdaCoF)
- **Dataset:** 65,906 triplets at 720p
- **GPU:** RTX 5090 (32 GB VRAM)
- **Phase 1:** Batch 6, crop 384×384, lr=3e-4, 35 epochs
- **Phase 2:** Batch 3 + 2× gradient accumulation (effective 6), lr=5e-5, 15 epochs
- **Losses:** L1=1.5, Perceptual=0.1, Edge=1.0 (20× multiplier), GAN=0.005 (Phase 2)

**Milestone:** Custom anime-trained interpolation model ready for inference.

---

## Phase 6: Harden Quality + Performance
*Make it actually good for anime*

| Step | Task | Status |
|------|------|--------|
| 6.1 | **Tune gating thresholds** — balance between false cuts and false motion | [ ] |
| 6.2 | **Bad mid-frame detection** — if interpolated frame looks wrong, fallback to duplicate | [ ] |
| 6.3 | **FP16 inference** — reduce VRAM usage and improve speed | [ ] |
| 6.4 | **Memory-aware batching** — auto tile sizing based on available VRAM | [ ] |
| 6.5 | **NVENC encoding** — optional GPU encoding for speed presets | [ ] |
| 6.6 | **Benchmark suite** — measure FPS, VRAM, quality metrics | [ ] |

**Milestone:** Pipeline is stable, fast, and produces high-quality anime-safe output.

---

## Phase 7: Polish + UX
*Make it usable*

| Step | Task | Status |
|------|------|--------|
| 7.1 | **Better CLI UX** — progress bars, ETA, logging levels | [ ] |
| 7.2 | **Presets implementation** — `anime-clean`, `anime-strong`, `fast` | [ ] |
| 7.3 | **Error handling** — corrupt files, unsupported codecs, VRAM failures | [ ] |
| 7.4 | **Clip mode** — `--clip 00:10:00-00:10:20` for preview renders | [ ] |
| 7.5 | **Documentation** — usage examples, troubleshooting, FAQ | [ ] |

**Milestone:** AInimotion is ready for public release as a CLI tool.

---

## Phase 8: GUI + Packaging
*Desktop application*

| Step | Task | Status |
|------|------|--------|
| 8.1 | Drag/drop desktop UI | [ ] |
| 8.2 | Short preview renders in-app | [ ] |
| 8.3 | Presets panel + advanced settings | [ ] |
| 8.4 | Bundle FFmpeg + runtime + weights | [ ] |
| 8.5 | Installer for Windows/macOS/Linux | [ ] |

---

## Quick Reference: Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| Video I/O | FFmpeg (subprocess) |
| Training | PyTorch 2.9+ with CUDA 12.8+ |
| Inference (prod) | ONNX Runtime + TensorRT |
| Interpolation | LayeredInterpolator (custom: FPN + AdaCoF + Affine Grid) |
| Upscaling | Real-ESRGAN / Compact (or custom) |
| Encoding | FFmpeg (libx264/libx265) or NVENC |
| Monitoring | Weights & Biases + TensorBoard |
| GPU | NVIDIA RTX 5090 (32 GB) |

---

## Getting Started

**Current focus: Phase 5 (Model Training)**

Full training is in progress with sweep-optimized config:

```powershell
cd "C:\Projects\AInimotion"
python -m ainimotion.training.train --config configs/interp_training_5090.yaml --data "D:\Triplets" --wandb --auto-resume
```
