"""Benchmark all registered models on sample images, measuring speed and visual quality."""

import argparse
import sys
import time
from pathlib import Path

from PIL import Image
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nn_upscaler.inference.pipeline import InferencePipeline
from nn_upscaler.models.registry import get_model, list_models
from nn_upscaler.utils.device import clear_device_cache, get_device, get_device_name
from nn_upscaler.utils.visualization import create_comparison_grid


def main():
    parser = argparse.ArgumentParser(description="Benchmark super-resolution models on sample pixel art images.")
    parser.add_argument("--samples", type=str, default="data/samples", help="Directory with sample test images")
    parser.add_argument("--scale", type=int, default=4, help="Scale factor")
    parser.add_argument("--output-dir", type=str, default="outputs/benchmark", help="Directory to save comparison grids")
    parser.add_argument("--device", type=str, default=None, help="Device to run benchmark on")
    parser.add_argument("--runs", type=int, default=5, help="Timing repetitions for stability")

    args = parser.parse_args()
    console = Console()
    device = get_device(args.device)

    samples_dir = Path(args.samples)
    image_files = sorted(list(samples_dir.glob("*.png")) + list(samples_dir.glob("*.jpg")))
    if not image_files:
        console.print(f"[bold red]No images found in {samples_dir}[/bold red]")
        sys.exit(1)

    console.print(f"\n[bold cyan]=== Neural Upscaler Model Benchmark ({get_device_name(device)}) ===[/bold cyan]\n")

    models = list_models()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in image_files:
        console.print(f"[bold yellow]Benchmarking on image: {img_path.name}[/bold yellow]")
        img = Image.open(img_path).convert("RGB")

        table = Table(title=f"Results for {img_path.name} ({img.width}x{img.height} -> {img.width*args.scale}x{img.height*args.scale})")
        table.add_column("Model", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Inference Time (ms)", style="green")
        table.add_column("Output Size", style="white")

        comparison_dict = {"Original (LR)": img}

        for m_name in models:
            model = get_model(m_name, scale_factor=args.scale)
            pipeline = InferencePipeline(model=model, device=device)

            # Warmup
            _ = pipeline.run(img)
            clear_device_cache(device)

            # Timing
            t0 = time.perf_counter()
            for _ in range(args.runs):
                result = pipeline.run(img)
            t1 = time.perf_counter()
            avg_ms = ((t1 - t0) / args.runs) * 1000.0

            comparison_dict[m_name] = result
            m_type = "Classical" if m_name in ("nearest", "bicubic", "bilinear") else "Neural"
            table.add_row(m_name, m_type, f"{avg_ms:.2f} ms", f"{result.width}x{result.height}")

        console.print(table)

        # Save visual comparison grid
        grid_path = out_dir / f"compare_{img_path.stem}_{args.scale}x.png"
        create_comparison_grid(comparison_dict, output_path=grid_path)
        console.print(f"Saved side-by-side comparison to: [bold green]{grid_path}[/bold green]\n")


if __name__ == "__main__":
    main()
