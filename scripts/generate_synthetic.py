"""Offline batch script to generate synthetic pixel art pairs from high-res images on GPU/MPS."""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nn_upscaler.data.degradation import (
    PALETTE_TENSORS,
    TensorPaletteQuantizer,
    apply_bayer_dithering,
    quantize_palette_pillow,
    tensor_downscale,
)
from nn_upscaler.data.downloader import compute_file_hash, validate_image_file
from nn_upscaler.utils.device import get_device, get_device_name

console = Console()


def save_image_pair(hr_img: Image.Image, lr_img: Image.Image, hr_path: Path, lr_path: Path) -> None:
    hr_path.parent.mkdir(parents=True, exist_ok=True)
    lr_path.parent.mkdir(parents=True, exist_ok=True)
    hr_img.save(hr_path, format="PNG")
    lr_img.save(lr_path, format="PNG")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="High-throughput GPU-accelerated batch synthesizer for pixel art pairs."
    )
    parser.add_argument(
        "--input-dir", "-i", type=str, required=True, help="Input directory of high-res images"
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, required=True, help="Output directory for generated pairs"
    )
    parser.add_argument("--scale", "-s", type=int, default=4, help="Downscale factor")
    parser.add_argument(
        "--colors",
        "-c",
        type=int,
        default=32,
        help="Target palette size (for adaptive quantization)",
    )
    parser.add_argument(
        "--palette",
        "-p",
        type=str,
        default=None,
        choices=list(PALETTE_TENSORS.keys()),
        help="Preset retro palette ('nes', 'c64', 'gameboy', 'pico8', 'cga', 'zx_spectrum', 'ega')",
    )
    parser.add_argument(
        "--dither-mode",
        type=str,
        default="bayer",
        choices=["bayer", "floyd_steinberg", "none"],
        help="Dithering algorithm",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device ('mps', 'cuda', 'cpu')",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=4,
        help="Batch size for GPU tensor processing",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of workers for image IO",
    )

    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    lr_dir = out_dir / f"LR_{args.scale}x"
    hr_dir = out_dir / "HR"

    lr_dir.mkdir(parents=True, exist_ok=True)
    hr_dir.mkdir(parents=True, exist_ok=True)

    extensions = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")
    images: list[Path] = []
    for ext in extensions:
        images.extend(in_dir.glob(ext))
    images = sorted([f for f in images if not f.name.startswith(".")])

    if not images:
        console.print(f"[bold red]No valid images found in {in_dir}[/bold red]")
        sys.exit(1)

    device = get_device(args.device)
    console.print(f"Device: [bold green]{get_device_name(device)}[/bold green] ({device})")
    console.print(
        f"Processing [bold cyan]{len(images)}[/bold cyan] images at [bold yellow]{args.scale}x[/bold yellow] scale..."
    )

    quantizer = TensorPaletteQuantizer(device=device)
    manifest_records: list[dict] = []

    # Process in batches
    batch_size = args.batch_size
    save_tasks = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
    ) as progress:
        task_id = progress.add_task("Synthesizing pairs...", total=len(images))

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for b_idx in range(0, len(images), batch_size):
                batch_paths = images[b_idx : b_idx + batch_size]
                pil_hrs: list[Image.Image] = []
                valid_paths: list[Path] = []

                for p in batch_paths:
                    if not validate_image_file(p, min_size=(args.scale, args.scale)):
                        continue
                    try:
                        with Image.open(p) as img:
                            pil_hrs.append(img.convert("RGB"))
                            valid_paths.append(p)
                    except Exception as e:
                        console.print(f"[yellow]Warning: Skipping image {p.name}: {e}[/yellow]")
                        continue

                if not pil_hrs:
                    progress.advance(task_id, len(batch_paths))
                    continue

                for img, p in zip(pil_hrs, valid_paths, strict=True):
                    w, h = img.size
                    lr_w = max(1, w // args.scale)
                    lr_h = max(1, h // args.scale)

                    # 1. Convert to tensor on target device [1, 3, H, W]
                    hr_arr = np.array(img, dtype=np.float32) / 255.0
                    hr_t = torch.from_numpy(hr_arr).permute(2, 0, 1).unsqueeze(0).to(device=device)

                    # 2. Area Box Downscale with MPS compatibility
                    lr_t = tensor_downscale(hr_t, args.scale)
                    lr_h, lr_w = lr_t.shape[-2], lr_t.shape[-1]

                    # 3. Dithering
                    if args.dither_mode == "bayer":
                        lr_t = apply_bayer_dithering(lr_t, size=4, spread=0.05)

                    # 4. Color Quantization
                    if args.palette:
                        lr_t = quantizer.quantize(lr_t, palette=args.palette)
                    elif args.dither_mode == "floyd_steinberg":
                        # Convert to PIL for Pillow C-backend Floyd-Steinberg
                        lr_np = (
                            (lr_t.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255.0)
                            .round()
                            .astype(np.uint8)
                        )
                        pil_lr = quantize_palette_pillow(
                            Image.fromarray(lr_np, mode="RGB"), num_colors=args.colors, dither=True
                        )
                    else:
                        levels = max(2, round(args.colors ** (1.0 / 3.0)))
                        lr_t = quantizer.quantize_adaptive(lr_t, levels=levels)

                    if args.dither_mode != "floyd_steinberg" or args.palette:
                        lr_np = (
                            (lr_t.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255.0)
                            .round()
                            .astype(np.uint8)
                        )
                        pil_lr = Image.fromarray(lr_np, mode="RGB")

                    hr_out_path = hr_dir / f"{p.stem}.png"
                    lr_out_path = lr_dir / f"{p.stem}.png"

                    # Async saving
                    future = executor.submit(save_image_pair, img, pil_lr, hr_out_path, lr_out_path)
                    save_tasks.append(future)

                    manifest_records.append(
                        {
                            "id": p.stem,
                            "hr_path": str(hr_out_path.relative_to(out_dir)),
                            "lr_path": str(lr_out_path.relative_to(out_dir)),
                            "hr_resolution": [w, h],
                            "lr_resolution": [lr_w, lr_h],
                            "hr_sha256": compute_file_hash(p),
                        }
                    )

                progress.advance(task_id, len(batch_paths))

            # Wait for all saves to finish
            for fut in save_tasks:
                fut.result()

    # Save manifest.json
    manifest = {
        "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "scale_factor": args.scale,
        "device": str(device),
        "palette_preset": args.palette,
        "dither_mode": args.dither_mode,
        "num_colors": args.colors,
        "total_pairs": len(manifest_records),
        "pairs": manifest_records,
    }

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    console.print(
        f"[bold green]✔ Synthesis complete! Saved {len(manifest_records)} pairs.[/bold green]"
    )
    console.print(f"  - Manifest: [bold]{manifest_path}[/bold]")
    console.print(f"  - LR directory: [bold]{lr_dir}[/bold]")
    console.print(f"  - HR directory: [bold]{hr_dir}[/bold]")


if __name__ == "__main__":
    main()
