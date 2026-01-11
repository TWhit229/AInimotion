# AInimotion Development Roadmap

A practical guide from empty repo to working anime video enhancer.

---

## Phase 0: Project Setup
*Get the skeleton in place*

- [ ] Create the package structure
  ```
  ainimotion/
    __init__.py
    cli.py
    pipeline/
      __init__.py
      decode.py
      analyze.py
      interpolate.py
      upscale.py
      encode.py
    models/
      interp/
      upscale/
    utils/
      __init__.py
      ffmpeg.py
      metrics.py
      colorspace.py
      vram.py
      logging.py
  ```
- [ ] Set up `pyproject.toml` — defines dependencies, entry points, metadata
- [ ] Create a dev environment — `python -m venv .venv` + install deps
- [ ] Verify FFmpeg is available — required for decode/encode

---

## Phase 1: Build the Pipeline Skeleton
*Wire up the stages with placeholder logic*

**Goal:** Input video → goes through all stages → output video (even if no processing happens yet)

| Step | Task | Status |
|------|------|--------|
| 1.1 | **FFmpeg decode** — extract frames to temp dir or memory | [ ] |
| 1.2 | **FFmpeg encode** — take frames and encode back to video + remux audio | [ ] |
| 1.3 | **CLI entry point** — `ainimotion enhance input.mkv --out output.mkv` does round-trip | [ ] |

**Milestone:** CLI runs and produces a video output (just a copy, no enhancement yet).

---

## Phase 2: Add Gating (The "Anime-Safe" Core)
*Detect scene cuts and holds BEFORE adding any ML*

| Step | Task | Status |
|------|------|--------|
| 2.1 | **Frame comparison** — downscale + compute luma similarity (SSIM or MAD) | [ ] |
| 2.2 | **Classify intervals** — cut / hold / motion based on thresholds | [ ] |
| 2.3 | **Frame duplication** — for 24→48 FPS, duplicate frames for cuts/holds | [ ] |
| 2.4 | **Unit tests** — test gating logic with synthetic frame pairs | [ ] |

**Milestone:** Output is 48 FPS via intelligent duplication. No interpolation yet, but the gating logic is solid.

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

## Phase 5: Harden Quality + Performance
*Make it actually good for anime*

| Step | Task | Status |
|------|------|--------|
| 5.1 | **Tune gating thresholds** — balance between false cuts and false motion | [ ] |
| 5.2 | **Bad mid-frame detection** — if interpolated frame looks wrong, fallback to duplicate | [ ] |
| 5.3 | **FP16 inference** — reduce VRAM usage and improve speed | [ ] |
| 5.4 | **Memory-aware batching** — auto tile sizing based on available VRAM | [ ] |
| 5.5 | **NVENC encoding** — optional GPU encoding for speed presets | [ ] |
| 5.6 | **Benchmark suite** — measure FPS, VRAM, quality metrics | [ ] |

**Milestone:** Pipeline is stable, fast, and produces high-quality anime-safe output.

---

## Phase 6: Polish + UX
*Make it usable*

| Step | Task | Status |
|------|------|--------|
| 6.1 | **Better CLI UX** — progress bars, ETA, logging levels | [ ] |
| 6.2 | **Presets implementation** — `anime-clean`, `anime-strong`, `fast` | [ ] |
| 6.3 | **Error handling** — corrupt files, unsupported codecs, VRAM failures | [ ] |
| 6.4 | **Clip mode** — `--clip 00:10:00-00:10:20` for preview renders | [ ] |
| 6.5 | **Documentation** — usage examples, troubleshooting, FAQ | [ ] |

**Milestone:** AInimotion is ready for public release as a CLI tool.

---

## Future Phases (Post-MVP)

### Phase 7: Custom Model Training
- [ ] Fine-tune upscaler on anime with realistic codec degradations
- [ ] Fine-tune interpolator on anime triplets with line-stability regularization
- [ ] Build training pipeline and dataset curation tools

### Phase 8: GUI + Packaging
- [ ] Drag/drop desktop UI
- [ ] Short preview renders in-app
- [ ] Presets panel + advanced settings
- [ ] Bundle FFmpeg + runtime + weights
- [ ] Installer for Windows/macOS/Linux

---

## Quick Reference: Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| Video I/O | FFmpeg (subprocess) |
| Inference (dev) | PyTorch + CUDA |
| Inference (prod) | ONNX Runtime + TensorRT |
| Interpolation | RIFE / IFRNet (or custom) |
| Upscaling | Real-ESRGAN / Compact (or custom) |
| Encoding | FFmpeg (libx264/libx265) or NVENC |

---

## Getting Started

To begin development, start with **Phase 0 + Phase 1**:

1. Create the package structure
2. Write FFmpeg decode/encode utilities
3. Wire up a basic CLI that does passthrough (decode → encode)

This gives you a working foundation to iterate on.
