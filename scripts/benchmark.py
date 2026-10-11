"""Benchmark all registered models on sample images, measuring speed and visual quality."""

import argparse
import sys
import time
from pathlib import Path

from PIL import Image
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nn_upscaler.data.degradation import PixelArtDegradation
from nn_upscaler.evaluation.metrics import calculate_lpips, calculate_psnr, calculate_ssim
from nn_upscaler.inference.pipeline import InferencePipeline
from nn_upscaler.models.registry import get_model, list_models
from nn_upscaler.utils.device import clear_device_cache, get_device, get_device_name
from nn_upscaler.utils.visualization import create_comparison_grid, pil_to_tensor


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark super-resolution models on sample pixel art images."
    )
    parser.add_argument(
        "--samples",
        type=str,
        default="data/samples",
        help="Directory with sample test images or authentic pixel art",
    )
    parser.add_argument("--scale", type=int, default=4, help="Scale factor")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/benchmark",
        help="Directory to save comparison grids",
    )
    parser.add_argument("--device", type=str, default=None, help="Device to run benchmark on")
    parser.add_argument("--runs", type=int, default=3, help="Timing repetitions for stability")
    parser.add_argument(
        "--eval-metrics",
        action="store_true",
        help="Synthesize pixel art from HR images to compute ground-truth PSNR/SSIM/LPIPS",
    )

    args = parser.parse_args()
    console = Console()
    device = get_device(args.device)

    samples_dir = Path(args.samples)
    image_files = sorted(list(samples_dir.glob("*.png")) + list(samples_dir.glob("*.jpg")))
    if not image_files:
        console.print(f"[bold red]No images found in {samples_dir}[/bold red]")
        sys.exit(1)

    console.print(
        f"\n[bold cyan]=== Neural Upscaler Model Benchmark ({get_device_name(device)}) ===[/bold cyan]\n"
    )

    models = list_models()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    degrader = PixelArtDegradation(scale_factor=args.scale, num_colors=32)

    for img_path in image_files:
        console.print(f"[bold yellow]Benchmarking on image: {img_path.name}[/bold yellow]")
        raw_img = Image.open(img_path).convert("RGB")

        # Determine if we test as paired synthetic degradation or direct input
        has_gt = args.eval_metrics
        if has_gt:
            # raw_img is ground truth HR; degrade to LR
            hr_img = raw_img
            # Ensure dimensions divisible by scale
            w_crop = hr_img.width - (hr_img.width % args.scale)
            h_crop = hr_img.height - (hr_img.height % args.scale)
            if w_crop != hr_img.width or h_crop != hr_img.height:
                hr_img = hr_img.crop((0, 0, w_crop, h_crop))
            lr_img = degrader(hr_img)
            target_tensor = pil_to_tensor(hr_img)
        else:
            hr_img = None
            lr_img = raw_img
            target_tensor = None

        table = Table(
            title=f"Results for {img_path.name} (LR: {lr_img.width}x{lr_img.height} -> SR: {lr_img.width * args.scale}x{lr_img.height * args.scale})"
        )
        table.add_column("Model", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Speed (ms)", style="green")
        if has_gt:
            table.add_column("PSNR (dB)", style="yellow")
            table.add_column("SSIM", style="blue")
            table.add_column("LPIPS", style="red")
        table.add_column("Output Size", style="white")

        comparison_dict = {"Original (LR)": lr_img}
        if hr_img is not None:
            comparison_dict["Ground Truth (HR)"] = hr_img

        for m_name in models:
            try:
                model = get_model(m_name, scale_factor=args.scale)
                pipeline = InferencePipeline(model=model, device=device)

                # Warmup
                _ = pipeline.run(lr_img)
                clear_device_cache(device)

                # Timing
                t0 = time.perf_counter()
                for _ in range(args.runs):
                    result = pipeline.run(lr_img)
                t1 = time.perf_counter()
                avg_ms = ((t1 - t0) / args.runs) * 1000.0

                comparison_dict[m_name] = result
                m_type = "Classical" if m_name in ("nearest", "bicubic", "bilinear") else "Neural"

                if has_gt and target_tensor is not None:
                    res_tensor = pil_to_tensor(result)
                    # Align shapes if minor rounding occurs
                    min_h = min(res_tensor.shape[1], target_tensor.shape[1])
                    min_w = min(res_tensor.shape[2], target_tensor.shape[2])
                    r_crop = res_tensor[:, :min_h, :min_w]
                    t_crop = target_tensor[:, :min_h, :min_w]

                    psnr_val = calculate_psnr(r_crop, t_crop)
                    ssim_val = calculate_ssim(r_crop, t_crop)
                    lpips_val = calculate_lpips(r_crop, t_crop, device=device)

                    table.add_row(
                        m_name,
                        m_type,
                        f"{avg_ms:.2f} ms",
                        f"{psnr_val:.2f}",
                        f"{ssim_val:.4f}",
                        f"{lpips_val:.4f}",
                        f"{result.width}x{result.height}",
                    )
                else:
                    table.add_row(
                        m_name,
                        m_type,
                        f"{avg_ms:.2f} ms",
                        f"{result.width}x{result.height}",
                    )
            except Exception as e:
                console.print(f"[bold red]Error running model {m_name}: {e}[/bold red]")

        console.print(table)

        # Save visual comparison grid
        grid_path = out_dir / f"compare_{img_path.stem}_{args.scale}x.png"
        create_comparison_grid(comparison_dict, output_path=grid_path)
        console.print(f"Saved side-by-side comparison to: [bold green]{grid_path}[/bold green]\n")


if __name__ == "__main__":
    main()
