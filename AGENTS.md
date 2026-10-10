# AGENTS.md — Protocol for Autonomous Agent Operations

Welcome, Agent. This repository is dedicated to researching and developing neural upscaling models and filters for transforming **pixel art into realistic or semi-realistic high-resolution imagery**.

You are expected to operate autonomously, methodically, and with high engineering rigor. Follow this protocol whenever you perform research, implementation, or experimentation.

---

## 1. Core Operating Principles

1. **Hypothesis-Driven Development:** Every model change, loss function experiment, or data pipeline modification must have an explicit rationale documented in [`EXPERIMENTS.md`](EXPERIMENTS.md).
2. **Never Break Baselines:** Classical algorithms (Nearest, Bicubic) and established baseline models must remain functional and tested. Every new model must be benchmarked against these baselines.
3. **Reproducibility:** All training runs and benchmarks must take configuration files (from `configs/`) or deterministic CLI parameters, with random seeds set.
4. **Hardware Awareness (Apple Silicon MPS / CUDA / CPU):**
   - The primary local machine runs **macOS Apple Silicon (MPS)**.
   - Always route tensors through `nn_upscaler.utils.device.get_device()`.
   - Prefer `float32` for MPS (some ops like certain complex numbers or rare FFT/conv variants lack float64 or bfloat16 on older MPS runtimes).
   - Clean up cache (`torch.mps.empty_cache()` or `torch.cuda.empty_cache()`) between large benchmark passes.
5. **Always Verify Before Declaring Done:**
   - Run `uv run pytest` after any code modification.
   - Run `uv run python scripts/smoke_test.py` to ensure GPU/MPS and models are operational.

---

## 2. The Research Mission: Pixel Art -> Realistic Image

### Why is this hard?
Standard Super-Resolution (SR) assumes continuous natural degradation (bicubic blur + downsampling + noise). In pixel art:
- High frequencies are sharp (single square pixels represent boundaries).
- Color count is strictly quantized (e.g. 16, 32, or 256 color palettes).
- High-level semantic features (eyes, armor, leaves, buildings) are represented by just 2-10 pixels.
- Naive bicubic/ESRGAN simply produces blurry blobs or over-sharp squares without "hallucinating" the realistic continuous textures (skin pores, fabric weaves, foliage, depth).

### Two Complementary Approaches to Explore:
1. **Paired Super-Resolution (Synthetic Degradation):**
   - Take high-resolution realistic photos or 3D renders.
   - Apply a realistic pixel-art synthesizer (`src/nn_upscaler/data/degradation.py`): downscaling + edge-aware palette quantization + nearest-neighbor rasterization.
   - Train an upscale network to invert this degradation back to the original photo.
2. **Generative / Diffusion / Image-to-Image Translation:**
   - Unpaired or conditional translation (e.g., ControlNet-tile, latent diffusion, or GAN-based discriminator with perceptual & perceptual style loss).

---

## 3. Experimentation Lifecycle

When tasked with an experiment or advancing the roadmap:

```
[1. Read ROADMAP.md] ──> [2. Formulate Hypothesis in EXPERIMENTS.md]
                                   │
                                   ▼
                       [3. Implement Code / Config]
                                   │
                                   ▼
                       [4. Run Tests & Smoke Test]
                                   │
                                   ▼
                       [5. Train / Benchmark Run]
                                   │
                                   ▼
                       [6. Record Metrics & Samples]
                                   │
                                   ▼
                       [7. Commit & Update Status]
```

### Logging in `EXPERIMENTS.md`
Every experiment must record:
- **ID:** e.g., `EXP-001`
- **Hypothesis:** What are we testing and why?
- **Architecture / Config:** Model, parameters, loss components, dataset.
- **Metrics:** PSNR, SSIM, LPIPS, visual quality notes, inference speed (ms/frame).
- **Conclusion:** Keep, discard, or iterate.

---

## 4. Repository Structure & Key Entrypoints

| Path | Purpose |
|------|---------|
| `src/nn_upscaler/models/` | Model architectures. Subclass `BaseUpscaler` and register in `registry.py`. |
| `src/nn_upscaler/data/degradation.py` | Pixel-art synthesis pipeline (crucial for paired training data). |
| `src/nn_upscaler/losses/` | Loss functions: L1, LPIPS, gradient/edge losses. |
| `src/nn_upscaler/evaluation/` | PSNR, SSIM, LPIPS metrics and benchmark suite. |
| `src/nn_upscaler/inference/` | Inference pipeline with tiling support to prevent OOM on large images. |
| `scripts/smoke_test.py` | Sanity check script for device, forward pass, and memory. |
| `scripts/upscaler.py` | CLI tool for upscaling images. |
| `scripts/benchmark.py` | Automated comparison across registered models. |
| `scripts/train.py` | Training runner. |

---

## 5. Coding Standards

- Use Python 3.12 syntax and typing (`from typing import Optional, Tuple`, etc.).
- Follow PEP 8 via `ruff`. Run `uv run ruff check .` and `uv run ruff format .`.
- Keep dependencies lean and registered in `pyproject.toml`.
- Document all new models and modules with clear docstrings explaining inputs, outputs, and tensor shapes `[B, C, H, W]`.
