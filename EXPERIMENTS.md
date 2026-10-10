# EXPERIMENTS.md — Experiment Registry & Log

This document records all research hypotheses, architectural iterations, loss configurations, and benchmark evaluations. Every agent working in this repository must log new experiments here.

---

## 📊 Summary Table

| ID | Date | Model | Scale | PSNR (dB) | SSIM | LPIPS | Inference (ms) | Status | Key Findings |
|---|---|---|---|---|---|---|---|---|---|
| `EXP-000` | 2026-10-10 | Nearest (Baseline) | 4x | Baseline | Baseline | Baseline | < 1 ms | ✅ Verified | Trivial upper bound on pixel grid fidelity, but high blockiness. |
| `EXP-001` | 2026-10-10 | Bicubic (Baseline) | 4x | Reference | Reference | Reference | < 1 ms | ✅ Verified | Smooths blocks into blurry gradients; lacks realistic high-frequency detail. |
| `EXP-002` | 2026-10-10 | SRCNN | 4x | TBD | TBD | TBD | ~5 ms | 🟡 Initialized | 3-layer CNN architecture initialized as basic neural baseline. |
| `EXP-003` | 2026-10-10 | EDSR-Light | 4x | TBD | TBD | TBD | ~18 ms | 🟡 Initialized | Residual network backbone with sub-pixel convolution upsampler. |

---

## 📝 Experiment Logs

### EXP-000: Classical Nearest Neighbor Baseline
- **Date:** 2026-10-10
- **Model:** `NearestNeighborUpscaler`
- **Goal:** Establish zero-overhead baseline for preserving raw pixel blocks.
- **Observations:** Pure reproduction of input pixels at 4x/8x scale. Preserves hard palette boundaries, zero hallucination.

### EXP-001: Classical Bicubic Interpolation Baseline
- **Date:** 2026-10-10
- **Model:** `BicubicUpscaler`
- **Goal:** Establish standard continuous interpolation baseline.
- **Observations:** Severe blurring across pixel art color steps.

### EXP-002: SRCNN Neural Baseline
- **Date:** 2026-10-10
- **Model:** `SRCNNUpscaler` (9-1-5 filter sizes, 64-32 feature channels)
- **Goal:** Verify end-to-end PyTorch pipeline, MPS execution, and tensor shapes.
- **Architecture:** Pre-bicubic upsampling followed by feature extraction, non-linear mapping, and reconstruction.

### EXP-003: EDSR-Light Residual Backbone
- **Date:** 2026-10-10
- **Model:** `EDSRLightUpscaler` (Residual blocks with PixelShuffle / Sub-pixel convolution)
- **Goal:** Provide a high-capacity CNN baseline with direct low-resolution input processing.
- **Architecture:** Conv -> 8x ResidualBlocks -> Upsampling via `nn.PixelShuffle` -> Conv.

---

## 🔬 Template for Future Experiments

```markdown
### EXP-XXX: <Title>
- **Date:** YYYY-MM-DD
- **Author/Agent:** <Name or Agent ID>
- **Hypothesis:** <Why is this expected to improve realism / perceptual metrics?>
- **Configuration:**
  - Architecture: ...
  - Loss Function: ...
  - Training Data: ...
  - Epochs / Batch Size: ...
- **Results:**
  - PSNR: ... dB
  - SSIM: ...
  - LPIPS: ...
  - Speed: ... ms
- **Visual Assessment:** <Description of artifacts, realism, sharpness>
- **Conclusion:** <Adopt, Iterate, or Reject>
```
