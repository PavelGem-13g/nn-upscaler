"""Offline batch script to generate synthetic pixel art pairs from high-res images."""

import argparse
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nn_upscaler.data.degradation import PixelArtDegradation


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic pixel art dataset from realistic images.")
    parser.add_argument("--input-dir", "-i", type=str, required=True, help="Input directory of high-res images")
    parser.add_argument("--output-dir", "-o", type=str, required=True, help="Output directory for pairs")
    parser.add_argument("--scale", "-s", type=int, default=4, help="Downscale factor")
    parser.add_argument("--colors", "-c", type=int, default=32, help="Target palette size (colors)")
    parser.add_argument("--palette", "-p", type=str, default=None, choices=["gameboy", "pico8", "cga"], help="Preset palette")
    parser.add_argument("--dither", action="store_true", help="Apply Floyd-Steinberg dithering")

    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    lr_dir = out_dir / f"LR_{args.scale}x"
    hr_dir = out_dir / "HR"

    lr_dir.mkdir(parents=True, exist_ok=True)
    hr_dir.mkdir(parents=True, exist_ok=True)

    images = list(in_dir.glob("*.png")) + list(in_dir.glob("*.jpg")) + list(in_dir.glob("*.jpeg"))
    if not images:
        print(f"No images found in {in_dir}")
        sys.exit(1)

    degrader = PixelArtDegradation(
        scale_factor=args.scale,
        num_colors=args.colors,
        palette_preset=args.palette,
        dither=args.dither,
    )

    print(f"Synthesizing pixel art pairs for {len(images)} images...")
    for img_path in tqdm(images):
        try:
            with Image.open(img_path) as im:
                hr_img = im.convert("RGB")
            # Save HR copy
            hr_path = hr_dir / f"{img_path.stem}.png"
            hr_img.save(hr_path)

            # Degrade and save LR pixel art
            lr_img = degrader(hr_img)
            lr_path = lr_dir / f"{img_path.stem}.png"
            lr_img.save(lr_path)
        except Exception as e:
            print(f"Failed processing {img_path.name}: {e}")

    print(f"Dataset generated at: {out_dir}")
    print(f"  - LR pixel art: {lr_dir}")
    print(f"  - HR ground truth: {hr_dir}")


if __name__ == "__main__":
    main()
