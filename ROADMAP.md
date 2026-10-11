# ROADMAP.md — Research & Development Roadmap

This roadmap outlines the systematic path from baseline super-resolution to state-of-the-art semantic depixelization (turning pixel art into near-realistic imagery).

---

## 📌 Status Legend
- 🟢 `DONE` — Implemented, verified, and benchmarked.
- 🟡 `IN PROGRESS` — Current focus.
- ⚪ `PLANNED` — Queued for subsequent iterations.

---

## 🗺️ Phases

### Phase 1: Foundations, Tooling & Baselines 🟢 `DONE`
- [x] 🟢 **Repository Structure & Dev Environment:** `uv`-managed dependencies, MPS support, CI-ready testing setup.
- [x] 🟢 **Baseline Models:**
  - Classical baselines: Nearest Neighbor, Bicubic, Bilinear.
  - Initial CNN baseline: SRCNN (super-resolution convolutional neural network).
  - Residual baseline: EDSR-Light (Enhanced Deep Residual Networks).
- [x] 🟢 **Evaluation Suite:** PSNR, SSIM, LPIPS metrics calculation, speed benchmarking (`ms/frame`).
- [x] 🟢 **Inference Engine:** CLI tool (`scripts/upscale.py`) with automatic tiling for memory-safe execution on arbitrary image sizes.

### Phase 2: Degradation Modeling & Synthetic Data Pipeline 🟢 `DONE`
- [x] 🟢 **Pixel-Art Synthesizer:** GPU-accelerated and tensor-native degradation pipeline (`src/nn_upscaler/data/degradation.py`):
  - Area downsampling factors (2x, 4x, 8x, 16x) with MPS divisible compatibility.
  - Native tensor Euclidean palette quantization via `torch.cdist` (NES, C64, GameBoy, PICO-8, CGA, ZX Spectrum, EGA).
  - Tensor Ordered Bayer Dithering (2x2, 4x4, 8x8) and Pillow C-backend Floyd-Steinberg dithering.
  - Pixel grid alignment & box subsampling.
- [x] 🟢 **Dataset Aggregation & Curation:**
  - Automated downloader & curator CLI (`scripts/download_dataset.py`, `src/nn_upscaler/data/downloader.py`) for DIV2K / high-res raw images.
  - Curation of authentic pixel art benchmark sprites (`data/benchmarks/authentic_pixel_art/`).
  - Automated SHA-256 deduplication, corruption filtering, and procedural offline fallbacks.
- [x] 🟢 **On-the-Fly Synthetic Dataset Generator:**
  - High-throughput PyTorch `SyntheticPixelArtDataset` & `OnTheFlyPixelArtDataset` generating dynamic degradations during training.
  - RGBA transparency handling with randomized contrasting backgrounds.
  - Batch GPU offline synthesizer (`scripts/generate_synthetic.py`) exporting `manifest.json`.

### Phase 3: Supervised Deep Neural Upscaling 🟢 `DONE`
- [x] 🟢 **Architectures to Explore:**
  - **ESRGAN / Real-ESRGAN:** Residual-in-Residual Dense Block (`RRDBNet`) with residual scaling ($\beta=0.2$). Registered as `rrdb` and `esrgan`.
  - **SwinIR / HAT:** Transformer-based super-resolution capturing non-local relationships (`SwinIRLightUpscaler` with 8x8 Shifted Window Attention). Registered as `swinir`.
  - **Edge-Aware / Grid-Aware Networks:** Explicitly modeling pixel grid boundaries with two-stream architecture and Spatial Feature Modulation (`GridAwareUpscaler`). Registered as `grid_aware`.
- [x] 🟢 **Loss Functions:**
  - Pixel losses: L1, `CharbonnierLoss` ($\epsilon=10^{-6}$).
  - Frequency & Edge losses: Sobel `GradientLoss`, 2D RFFT `FFTLoss` penalizing discrete pixel grid harmonics.
  - Perceptual losses: Multi-layer `VGGPerceptualLoss` (ImageNet normalized, frozen weights), `calculate_lpips` (AlexNet).
  - Adversarial: `RaGANLoss` (Relativistic Average GAN) with `UNetDiscriminatorWithSpectralNorm`.
- [x] 🟢 **Two-Stage Training & Pipeline:**
  - Stage 1 (PSNR pretraining) and Stage 2 (Adversarial perceptual fine-tuning) supported in `SuperResolutionTrainer`.
  - Production configs in `configs/`: `rrdb_esrgan.yaml`, `swinir.yaml`, `grid_aware.yaml`, `esrgan_adversarial.yaml`.
  - Extended CLI (`scripts/train.py`, `scripts/benchmark.py`, `scripts/smoke_test.py`).

### Phase 4: Semantic Depixelization (Realistic Hallucination) 🟢 `DONE`
- [x] 🟢 **Addressing the Information Bottleneck:**
  - Transitioned from pixel interpolation to semantic generation using discrete codebook priors and contrastive mutual information.
- [x] 🟢 **Discrete Codebook Prior (Vector-Quantized Prior):**
  - Implemented `VectorQuantizer` with Straight-Through Estimator (STE) and commitment loss ($\mathcal{L}_{\text{vq}} = \|\text{sg}[z_e] - e\|_2^2 + \beta \|z_e - \text{sg}[e]\|_2^2$).
  - Built `CodebookUpscaler` (`src/nn_upscaler/models/codebook_prior.py`) with Spatial Feature Transform (SFT) silhouette and palette modulation. Registered as `codebook_prior`.
- [x] 🟢 **Semantic & Prompt Conditioning:**
  - Built `SemanticPromptEncoder` and AdaIN modulation in `SemanticConditionedUpscaler` (`src/nn_upscaler/models/semantic_conditioned.py`). Registered as `semantic_conditioned`.
  - Supports categories (`face`, `character`, `armor`, `nature`, `stone_architecture`, etc.) and text prompt embeddings.
  - Integrated `--category` and `--prompt` into `scripts/upscale.py`.
- [x] 🟢 **Unpaired Translation (CUT & PatchNCE):**
  - Implemented `UnpairedPixelArtDataset` (`src/nn_upscaler/data/unpaired_dataset.py`) pairing authentic retro sprites (`data/benchmarks/authentic_pixel_art/`) and real HR photos (`data/hr_raw/`).
  - Implemented multi-layer `PatchNCELoss` ($\tau=0.07$) in `src/nn_upscaler/losses/contrastive.py`.
  - Implemented `CUTUpscaler` (`src/nn_upscaler/models/cut.py`) with multi-layer patch projection heads. Registered as `cut_unpaired`.
- [x] 🟢 **Training Pipeline & Configurations:**
  - Production configs: `configs/codebook_prior.yaml`, `configs/semantic_conditioned.yaml`, `configs/unpaired_cut.yaml`.
  - Full support for `codebook`, `semantic`, and `unpaired_cut` training modes in `SuperResolutionTrainer` and `scripts/train.py`.
  - 96 unit tests passing (100% PASS), verified on Apple Silicon MPS via `scripts/smoke_test.py`.

### Phase 5: Generative & Diffusion Approaches (The "Plan Maximum") 🟢 `DONE`
- [x] 🟢 **Diffusion-Based Restoration & Adapter Conditioning:**
  - `PixelArtTileAdapter` & `PixelArtControlAdapter` (`src/nn_upscaler/models/diffusion_adapter.py`): Zero-Convolutions (`ZeroConv2d`), SFT feature modulation, and dynamic `--tile-weight` conditioning balance (0.5 - 1.0).
  - High-frequency grid noise & dithering suppression: vectorized PyTorch tensor bilateral filter (`bilateral_dither_filter_tensor` via `F.unfold`) and OpenCV image filter (`bilateral_dither_filter_image`) eliminating Bayer & Floyd-Steinberg artifacts before generative conditioning.
- [x] 🟢 **Latent Super-Resolution & Fast Iterative Denoising:**
  - `LatentDiffusionUpscaler` (`src/nn_upscaler/models/latent_diffusion.py`, registered as `latent_diffusion` in `registry.py`):
    - Latent autoencoder compressing high-resolution ground truth by $4\times$ into compact latent space $z \in \mathbb{R}^{B \times 4 \times H \times W}$ matching sprite spatial grid.
    - Conditional denoiser with FiLM modulation, sinusoidal time embeddings, and spatial self-attention.
    - Fast deterministic discrete step scheduler (`DDIMScheduler`) supporting 1–8 step real-time sampling on Apple Silicon MPS (**7.54 ms/frame** at 4 steps).
    - Differentiable training loss mode combining diffusion noise prediction MSE and autoencoder reconstruction L1 loss.
- [x] 🟢 **Prompt-Guided Realism Enhancement Pipeline & Production CLI:**
  - `DiffusionUpscalePipeline` (`src/nn_upscaler/inference/diffusion_pipeline.py`):
    - Dual backend: native lightweight MPS model (offline, fast, zero network downloads) and heavy SDXL ControlNet-Tile pipeline (safe lazy loading, VAE tiling).
    - Game category prompt presets (`face`, `armor`, `character`, `creature`, `monster`, `nature`, `stone_architecture`, `weapon`, `vehicle`, `interior`).
  - Production CLI tools:
    - `scripts/upscale_diffusion.py`: Rich terminal output, parameters table, timing stats, error recovery.
    - `scripts/upscale.py`: Extended with `--steps` and `--tile-weight` for `latent_diffusion`.
  - Production configs: `configs/latent_diffusion.yaml`, `configs/diffusion_controlnet.yaml`.
  - 111 unit tests passing (100% PASS), verified on Apple Silicon MPS via `scripts/smoke_test.py`.

### Phase 6: Productionization & Deployment 🟢 `DONE`
- [x] 🟢 **Optimization & Acceleration:**
  - Post-training quantization: FP16 half-precision (`quantize_fp16`, `FP16Upscaler`) cutting VRAM/storage by 50% and accelerating MPS inference, and Dynamic INT8 PTQ (`quantize_dynamic_int8`, `DynamicINT8Upscaler`) with QNNPACK engine for CPU-only edge inference.
  - Apple Silicon CoreML native export (`src/nn_upscaler/export/coreml_exporter.py`): native `.mlpackage` tracing, metadata configuration, flexible spatial resolution ranges (`ct.RangeDim`), and Apple Neural Engine compute units (`ct.ComputeUnit.ALL` / `ComputeUnit.CPU_AND_NE`).
  - Cross-platform ONNX export & runtime (`src/nn_upscaler/export/onnx_exporter.py`): TorchScript/dynamic axes tracing (`[B, 3, H, W]`), numerical equivalence validation ($\Delta < 10^{-4}$), and `ONNXUpscaler` wrapper supporting `CoreMLExecutionProvider`, `CUDAExecutionProvider`, and `CPUExecutionProvider`.
- [x] 🟢 **Production Tooling & Batch Processing CLI:**
  - `scripts/export.py`: Unified model exporter supporting ONNX, CoreML, and TorchScript with quantization, input shape configuration, and Rich summary tables.
  - `scripts/batch_upscale.py`: High-throughput CLI tool for recursive directory processing of sprite libraries and texture packs with alpha-transparency preservation, periodic memory purging (`clear_device_cache`), and detailed `manifest.json` export.
- [x] 🟢 **Interactive Web UI:**
  - `scripts/web_ui.py`: Gradio web application with interactive split comparison slider (`gr.ImageSlider`), support for all registered architectures, semantic category and prompt steering, bilateral dither filter toggle, real-time device telemetry (MPS/CUDA/CPU, latency, FPS), and built-in authentic sprite benchmark gallery.
- [x] 🟢 **Verification & Testing:**
  - 122 unit tests passing (100% PASS), smoke test verified on Apple Silicon MPS with ONNX and quantization pipelines operational.
