# AInimotion

**AInimotion** is a GPU-accelerated, offline anime video enhancer that combines:

- **Frame interpolation**: *24 → 48 FPS* (motion-adaptive, anime-safe)
- **Upscale/restoration**: *1080p → 1440p (2K)* (line-art preserving)
- **Anime-first safeguards**: **scene-cut detection** + **hold/still detection** (so you don't get wobbly outlines or "fake 60fps" artifacts)

> Goal: "buttery smooth" motion and cleaner frames **without** bending line art, shimmering edges, or interpolating across hard cuts.

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

## How It Works

**Decode → Gate → Interpolate → Upscale → Encode**

1. **Decode** video frames (FFmpeg)
2. **Gate** each frame interval:
   - **Scene-cut detection**: don't interpolate across cuts
   - **Hold/still detection**: if frames are essentially identical, just duplicate
3. **Interpolate** only where motion is real (Model A)
4. **Upscale** every output frame (Model B)
5. **Encode** video + remux audio/subs (FFmpeg / NVENC optional)

### Why "Interpolate → Upscale" is the default

Upscalers can introduce tiny frame-to-frame changes (micro-sharpening, edge halos, texture crawl). If you upscale first, interpolation may treat those differences as motion and create shimmer/wobble.

Interpolate first at source resolution to keep motion estimation stable on clean line art, then upscale the finalized frames.

---

## Presets

| Preset | Description |
|--------|-------------|
| `anime-clean` | Most conservative. Best for line stability (faces, eyes, outlines). Strong gating + safe interpolation settings. **(default)** |
| `anime-strong` | More aggressive restoration (denoise/deblock). Slightly higher risk of "processed" look. |
| `fast` | Prioritizes speed. Uses lighter models. Intended for quick iteration / previews. |

> AInimotion is designed so you can swap models and presets without rewriting the pipeline.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ainimotion.git
cd ainimotion

# Install dependencies
pip install -e .
```

### Requirements

- Python 3.10+
- FFmpeg (bundled later, or available in PATH)
- NVIDIA GPU with CUDA support
- Sufficient VRAM for chosen scale/model (tiling reduces requirements)

---

## CLI Usage

```bash
ainimotion enhance "input.mkv" \
  --out "output_1440p_48fps.mkv" \
  --scale 1.333 \
  --fps 48 \
  --preset anime-clean
```

### Examples

**Anime Clean (recommended):**

```bash
ainimotion enhance input.mkv --out output.mkv --fps 48 --scale 1.333 --preset anime-clean
```

**Fast preview render (short segment):**

```bash
ainimotion enhance input.mkv --out preview.mkv --fps 48 --scale 1.333 --preset fast --clip 00:10:00-00:10:20
```

**Override gating thresholds (advanced):**

```bash
ainimotion enhance input.mkv --out output.mkv --fps 48 --scale 1.333 \
  --hold-threshold 0.003 --cut-threshold 0.35
```

---

## Gating: Scene Cuts + Holds

The "anime-safe" core of AInimotion.

### Scene cut detection

Interpolation across a hard cut produces nightmare blended frames. AInimotion detects cuts and disables interpolation for that interval.

### Hold / still detection

Anime commonly uses held frames. If two adjacent frames are essentially the same, AInimotion duplicates frames to reach 48 FPS instead of inventing motion.

**How it works:**

- Compare **downscaled luma** (Y channel) to ignore tiny compression flicker
- Use a similarity metric (SSIM or mean absolute difference)
- Three bands:
  - **Cut**: large change → no interpolation
  - **Hold**: tiny change → duplicate
  - **Motion**: interpolate

---

## Architecture

AInimotion is built as a modular, pipelined system:

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Decode    │ -> │   Analyze   │ -> │ Interpolate │ -> │   Upscale   │ -> │   Encode    │
│    (CPU)    │    │    (Gate)   │    │  (GPU, A)   │    │  (GPU, B)   │    │ (CPU/NVENC) │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

- **Stage 1: Decode** — FFmpeg decodes frames to a queue; extracts audio/subs metadata.
- **Stage 2: Analyze/Gate** — Computes per-interval cut/hold decisions.
- **Stage 3: Interpolate** — Consumes frame pairs and outputs mid-frames selectively.
- **Stage 4: Upscale/Restore** — Upscales each produced frame (tiling + FP16).
- **Stage 5: Encode** — FFmpeg encodes final frames and remuxes audio/subs.

### Queue-based pipeline

The program keeps the system busy by overlapping:

- Decoding while GPU is working
- Gating while decoding
- Encoding while next frames are processing

---

## Project Structure

```text
ainimotion/
  README.md
  LICENSE
  pyproject.toml
  ainimotion/
    __init__.py
    cli.py
    pipeline/
      decode.py
      analyze.py
      interpolate.py
      upscale.py
      encode.py
    models/
      interp/   (Model A weights/config)
      upscale/  (Model B weights/config)
    utils/
      ffmpeg.py
      metrics.py
      colorspace.py
      vram.py
      logging.py
  scripts/
    benchmark.py
    render_clip.py
  tests/
    test_gating.py
    test_timestamps.py
```

---

## Limitations

- Some anime effects are intentionally non-physical (smears, impact frames). Interpolation can't always "guess right."
- Web encodes with heavy artifacts can confuse motion; gating and light cleanup help.
- For maximum quality, expect offline render times.

---

## Roadmap

- [x] Phase 0 — Foundations (platform target, inference runtime, presets)
- [ ] Phase 1 — MVP CLI (end-to-end pipeline)
- [ ] Phase 2 — Quality hardening (anime-specific tuning)
- [ ] Phase 3 — Performance + VRAM stability
- [ ] Phase 4 — Model strategy (fine-tuning on anime)
- [ ] Phase 5 — GUI + packaging

---

## Legal Note

AInimotion is intended for processing video **you have the rights/permission to use**. It does not include DRM bypassing, ripping tools, or downloading functionality.

---

## Contributing

PRs welcome once the MVP CLI is stable! Useful contributions:

- Better cut/hold heuristics
- More presets / model configs
- Benchmark suite clips (rights-cleared)
- Packaging + UI improvements

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

Model weights may have separate licenses depending on their source.
