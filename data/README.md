# Data Directory Structure

This folder contains dataset samples, benchmarks, and guides for downloading full-scale datasets.

## Directory Layout

```
data/
├── samples/              # Tracked in Git: Minimal sample pixel art images for testing
│   ├── sample_sprite.png
│   └── sample_landscape.png
├── raw/                  # Git-ignored: Original downloaded datasets
└── synthetic/            # Git-ignored: Generated (LR pixel-art, HR photo) pairs
```

## Recommended Datasets for Pixel Art -> Realism Research

1. **High-Resolution Realistic Images (for Synthetic Degradation):**
   - **DIV2K:** 800 2K high-resolution training images (Super-Resolution standard).
     Download: `https://data.vision.ee.ethz.ch/cvl/DIV2K/`
   - **Flickr2K:** 2,650 images extending DIV2K diversity.
   - **CelebA-HQ / FFHQ:** High-quality portraits for testing facial depixelization.

2. **Authentic Pixel Art Datasets:**
   - **Pixel Art Sprite Datasets (Kaggle/OpenGameArt):** Retro game sprites, icons, tilesets.
   - **Depixelizing Pixel Art (Kopf & Lischinski):** Classical benchmark images (Space Invader, Yoshi, Mario, etc.).

## Generating Paired Data

To generate pairs on-the-fly during training or to pre-generate pairs into `data/synthetic/`:
```bash
uv run python scripts/generate_synthetic.py --input-dir data/raw/div2k --output-dir data/synthetic/div2k_pixelart --scale 4 --colors 32
```
