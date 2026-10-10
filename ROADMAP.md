# ROADMAP.md — Research & Development Roadmap

This roadmap outlines the systematic path from baseline super-resolution to state-of-the-art semantic depixelization (turning pixel art into near-realistic imagery).

---

## 📌 Status Legend
- 🟢 `DONE` — Implemented, verified, and benchmarked.
- 🟡 `IN PROGRESS` — Current focus.
- ⚪ `PLANNED` — Queued for subsequent iterations.

---

## 🗺️ Phases

### Phase 1: Foundations, Tooling & Baselines
- [x] 🟢 **Repository Structure & Dev Environment:** `uv`-managed dependencies, MPS support, CI-ready testing setup.
- [x] 🟢 **Baseline Models:**
  - Classical baselines: Nearest Neighbor, Bicubic, Bilinear.
  - Initial CNN baseline: SRCNN (super-resolution convolutional neural network).
  - Residual baseline: EDSR-Light (Enhanced Deep Residual Networks).
- [x] 🟢 **Evaluation Suite:** PSNR, SSIM, LPIPS metrics calculation, speed benchmarking (`ms/frame`).
- [x] 🟢 **Inference Engine:** CLI tool (`scripts/upscale.py`) with automatic tiling for memory-safe execution on arbitrary image sizes.

### Phase 2: Degradation Modeling & Synthetic Data Pipeline
- [x] 🟢 **Pixel-Art Synthesizer:** Realistic -> Pixel Art degradation pipeline (`src/nn_upscaler/data/degradation.py`):
  - Downsampling factors (2x, 4x, 8x, 16x).
  - Palette quantization (K-Means, octree, uniform, retro palettes like NES/GameBoy/C64).
  - Dithering algorithms (Floyd-Steinberg, ordered Bayer dithering).
  - Pixel grid alignment & rasterization.
- [ ] ⚪ **Dataset Aggregation:**
  - Automated downloaders for open-source high-res paired imagery (e.g. DIV2K, Flickr2K).
  - Curation of authentic pixel art benchmark sets (retro sprites, pixel art landscapes).
- [ ] ⚪ **On-the-Fly Synthetic Dataset Generator:** PyTorch `Dataset` that generates dynamic pixel-art degradations during training.

### Phase 3: Supervised Deep Neural Upscaling
- [ ] ⚪ **Architectures to Explore:**
  - **ESRGAN / Real-ESRGAN:** Residual-in-Residual Dense Block (RRDB) with adversarial loss.
  - **SwinIR / HAT:** Transformer-based super-resolution capturing non-local relationships.
  - **Edge-Aware / Grid-Aware Networks:** Explicitly modeling pixel grid boundaries to remove blockiness while reconstructing continuous contours.
- [ ] ⚪ **Loss Functions:**
  - Pixel losses: L1, Charbonnier.
  - Frequency & Edge losses: Sobel gradient loss, FFT high-frequency loss.
  - Perceptual losses: VGG-19, LPIPS (AlexNet/VGG).
  - Adversarial: Relativistic GAN discriminator.

### Phase 4: Semantic Depixelization (Realistic Hallucination)
- [ ] ⚪ **Addressing the Information Bottleneck:**
  - A 16x16 pixel face has only 256 RGB values, whereas a realistic face requires millions. The network must perform *semantic synthesis*, not just interpolation.
- [ ] ⚪ **Conditioning & Prior Knowledge:**
  - Class-conditional or text-prompt-conditional upscaling (e.g., "medieval knight in armor, realistic metal texture").
  - Facial/object priors (e.g. CodeFormer / GFPGAN style codebook priors adapted for sprites).
- [ ] ⚪ **CycleGAN / Unpaired Translation:**
  - Training unpaired translation between real photos and real pixel-art sprites without requiring artificial synthetic pairs.

### Phase 5: Generative & Diffusion Approaches (The "Plan Maximum")
- [ ] ⚪ **Diffusion-Based Restoration:**
  - ControlNet / Adapter conditioning on pixel art inputs.
  - Latent Super-Resolution (SD-Upscaler, CCSR, ResShift).
  - Prompt-guided realism enhancement: transforming 8-bit/16-bit sprites into realistic portraits and landscapes.

### Phase 6: Productionization & Deployment
- [ ] ⚪ **Optimization & Acceleration:**
  - Quantization (FP16, INT8).
  - Apple Silicon CoreML export (`.mlpackage`) for blazing fast native macOS/iOS execution via Neural Engine.
  - ONNX runtime export with TensorRT / DirectML support.
- [ ] ⚪ **User Interface:**
  - Web UI (Streamlit or Gradio) with interactive before/after split slider.
  - Batch processing CLI.
