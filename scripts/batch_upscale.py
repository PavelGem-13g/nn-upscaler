"""Production Batch Upscaling CLI tool for sprites, tilesets, and texture libraries."""

import argparse
import inspect
import json
import sys
import time
from pathlib import Path
from typing import Any

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from PIL import Image
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from nn_upscaler.export.onnx_exporter import ONNXUpscaler
from nn_upscaler.inference.pipeline import InferencePipeline
from nn_upscaler.models.diffusion_adapter import bilateral_dither_filter_image
from nn_upscaler.models.registry import get_model, list_models
from nn_upscaler.utils.device import clear_device_cache, get_device, get_device_name

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(
        description="High-Throughput Batch Upscaler for Sprite & Texture Libraries."
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        type=str,
        required=True,
        help="Path to directory containing input pixel art images",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        required=True,
        help="Path to output directory for upscaled assets",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="edsr_light",
        help=f"Model architecture name (or path to .onnx file). Registered: {', '.join(list_models())}",
    )
    parser.add_argument(
        "--scale",
        "-s",
        type=int,
        default=4,
        help="Upscale multiplier (e.g. 2, 4, 8)",
    )
    parser.add_argument(
        "--weights",
        "-w",
        type=str,
        default=None,
        help="Optional path to model checkpoint weights (.pt / .pth)",
    )
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="Recursively scan subdirectories for images",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=["png", "jpg", "jpeg", "webp", "bmp"],
        help="Allowed image file extensions (default: png jpg jpeg webp bmp)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="same",
        choices=["same", "png", "webp", "jpg"],
        help="Output image format ('same' preserves original extension)",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=None,
        help="Tile dimension for memory-safe chunked inference",
    )
    parser.add_argument(
        "--tile-overlap",
        type=int,
        default=16,
        help="Overlap margin in pixels between adjacent tiles",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default=None,
        help="Compute device ('mps', 'cuda', 'cpu')",
    )
    parser.add_argument(
        "--category",
        "-c",
        type=str,
        default=None,
        help="Semantic category for models that support semantic conditioning",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        default=None,
        help="Text prompt description for generative models",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Diffusion sampling steps (for diffusion models)",
    )
    parser.add_argument(
        "--tile-weight",
        type=float,
        default=None,
        help="Adapter conditioning strength for diffusion models",
    )
    parser.add_argument(
        "--bilateral-filter",
        action="store_true",
        help="Apply bilateral filtering to eliminate Bayer / Floyd-Steinberg dithering",
    )
    return parser.parse_args()


def collect_images(input_dir: Path, recursive: bool, extensions: list[str]) -> list[Path]:
    """Find all image files matching allowed extensions."""
    ext_set = {f".{e.lower().lstrip('.')}" for e in extensions}
    if recursive:
        candidates = input_dir.rglob("*")
    else:
        candidates = input_dir.glob("*")
    return sorted([f for f in candidates if f.is_file() and f.suffix.lower() in ext_set])


def upscale_single_image(
    pipeline: InferencePipeline,
    img: Image.Image,
    apply_bilateral: bool = False,
    run_kwargs: dict[str, Any] | None = None,
) -> Image.Image:
    """Upscale image with transparency preservation and optional bilateral filtering."""
    run_kwargs = run_kwargs or {}
    has_alpha = img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)

    if has_alpha:
        img_rgba = img.convert("RGBA")
        r, g, b, a = img_rgba.split()
        rgb_img = Image.merge("RGB", (r, g, b))
    else:
        rgb_img = img.convert("RGB")

    if apply_bilateral:
        rgb_img = bilateral_dither_filter_image(rgb_img)

    upscaled_rgb = pipeline.run(rgb_img, **run_kwargs)

    if has_alpha:
        # Scale alpha channel with nearest neighbor to preserve crisp pixel-art alpha cutouts
        upscaled_a = a.resize(
            (upscaled_rgb.width, upscaled_rgb.height),
            resample=Image.Resampling.NEAREST,
        )
        r_up, g_up, b_up = upscaled_rgb.split()
        return Image.merge("RGBA", (r_up, g_up, b_up, upscaled_a))

    return upscaled_rgb


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists() or not input_dir.is_dir():
        console.print(f"[bold red]Error:[/bold red] Input directory '{input_dir}' does not exist.")
        sys.exit(1)

    images = collect_images(input_dir, args.recursive, args.extensions)
    if not images:
        console.print(
            f"[bold yellow]Warning:[/bold yellow] No matching images found in '{input_dir}'."
        )
        sys.exit(0)

    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)

    console.rule("[bold cyan]Batch Asset Upscaler[/bold cyan]")
    console.print(f"Input Directory:  [bold white]{input_dir}[/bold white]")
    console.print(f"Output Directory: [bold white]{output_dir}[/bold white]")
    console.print(f"Total Images:     [bold green]{len(images)}[/bold green]")
    console.print(f"Target Scale:     [bold green]{args.scale}x[/bold green]")
    console.print(f"Compute Device:   [bold green]{get_device_name(device)}[/bold green]")

    # 1. Initialize Upscaler Model
    if args.model.endswith(".onnx"):
        console.print(f"Loading ONNX Model from: [green]{args.model}[/green]")
        model = ONNXUpscaler(args.model, scale_factor=args.scale)
    else:
        console.print(f"Loading Registered Model: [green]{args.model}[/green]")
        model = get_model(args.model, scale_factor=args.scale)
        if args.weights:
            ckpt = torch.load(args.weights, map_location="cpu")
            state_dict = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state_dict)

    pipeline = InferencePipeline(
        model=model,
        device=device,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
    )

    # 2. Filter extra model keyword arguments
    raw_kwargs = {}
    if args.category is not None:
        raw_kwargs["category"] = args.category
    if args.prompt is not None:
        raw_kwargs["prompt"] = args.prompt
    if args.steps is not None:
        raw_kwargs["steps"] = args.steps
    if args.tile_weight is not None:
        raw_kwargs["tile_weight"] = args.tile_weight

    sig = inspect.signature(model.forward)
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    run_kwargs = {k: v for k, v in raw_kwargs.items() if has_var_kw or k in sig.parameters}

    manifest_records: list[dict[str, Any]] = []
    total_in_mp = 0.0
    total_out_mp = 0.0
    success_count = 0
    fail_count = 0
    start_time_all = time.perf_counter()

    # 3. Processing with Rich Progress
    with Progress(
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Upscaling assets...", total=len(images))

        for idx, img_path in enumerate(images):
            rel_path = img_path.relative_to(input_dir)
            target_ext = img_path.suffix if args.format == "same" else f".{args.format}"
            out_file = output_dir / rel_path.with_suffix(target_ext)
            out_file.parent.mkdir(parents=True, exist_ok=True)

            t0 = time.perf_counter()
            try:
                with Image.open(img_path) as img:
                    orig_w, orig_h = img.width, img.height
                    upscaled = upscale_single_image(
                        pipeline=pipeline,
                        img=img,
                        apply_bilateral=args.bilateral_filter,
                        run_kwargs=run_kwargs,
                    )
                    up_w, up_h = upscaled.width, upscaled.height

                    # Save format handling
                    if out_file.suffix.lower() in (".jpg", ".jpeg"):
                        upscaled.convert("RGB").save(out_file, quality=95)
                    else:
                        upscaled.save(out_file)

                latency_ms = (time.perf_counter() - t0) * 1000.0
                in_mp = (orig_w * orig_h) / 1e6
                out_mp = (up_w * up_h) / 1e6
                total_in_mp += in_mp
                total_out_mp += out_mp
                success_count += 1

                manifest_records.append(
                    {
                        "input_rel_path": str(rel_path),
                        "output_rel_path": str(out_file.relative_to(output_dir)),
                        "input_size": [orig_w, orig_h],
                        "output_size": [up_w, up_h],
                        "latency_ms": round(latency_ms, 2),
                        "status": "success",
                    }
                )

            except Exception as e:
                latency_ms = (time.perf_counter() - t0) * 1000.0
                fail_count += 1
                manifest_records.append(
                    {
                        "input_rel_path": str(rel_path),
                        "output_rel_path": str(out_file.relative_to(output_dir)),
                        "error": str(e),
                        "latency_ms": round(latency_ms, 2),
                        "status": "error",
                    }
                )

            # Clear MPS/CUDA cache periodically every 5 images to keep Unified Memory footprint flat
            if (idx + 1) % 5 == 0:
                clear_device_cache(device)

            progress.advance(task)

    total_time_s = time.perf_counter() - start_time_all
    avg_latency_ms = (total_time_s / max(len(images), 1)) * 1000.0
    throughput_fps = len(images) / max(total_time_s, 1e-6)
    throughput_mp_s = total_out_mp / max(total_time_s, 1e-6)

    # 4. Generate manifest.json
    manifest_data = {
        "model": args.model,
        "scale": args.scale,
        "device": get_device_name(device),
        "total_images": len(images),
        "successful_images": success_count,
        "failed_images": fail_count,
        "total_time_seconds": round(total_time_s, 3),
        "average_latency_ms": round(avg_latency_ms, 2),
        "throughput_fps": round(throughput_fps, 2),
        "throughput_out_mp_s": round(throughput_mp_s, 2),
        "total_output_megapixels": round(total_out_mp, 3),
        "records": manifest_records,
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)

    # 5. Summary Table
    table = Table(title="Batch Upscale Processing Summary")
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value", style="bold white")

    table.add_row("Processed Assets", f"{success_count} / {len(images)} successful")
    table.add_row("Total Processing Time", f"{total_time_s:.2f} s")
    table.add_row("Average Latency per Asset", f"{avg_latency_ms:.1f} ms")
    table.add_row("Throughput (FPS)", f"{throughput_fps:.2f} images/s")
    table.add_row("Throughput (Output Megapixels)", f"{throughput_mp_s:.2f} MP/s")
    table.add_row("Manifest JSON Export", str(manifest_path))

    console.print()
    console.print(table)
    console.print("[bold green]✔ Batch processing completed successfully![/bold green]")


if __name__ == "__main__":
    main()
