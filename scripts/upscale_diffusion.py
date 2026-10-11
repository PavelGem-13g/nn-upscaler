"""Production CLI tool for Pixel Art to Realism conversion using Diffusion Models.

Supports:
- Dual Backend: Native LatentDiffusionUpscaler (fast, offline, MPS-native) & SDXL ControlNet-Tile.
- Category prompt templates (face, armor, creature, nature, architecture, etc.).
- Tile conditioning weight control (0.5 - 1.0) and bilateral dithering suppression.
- Rich terminal UI and full error handling.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from PIL import Image
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add src to pythonpath if run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nn_upscaler.inference.diffusion_pipeline import (
    CATEGORY_PROMPTS,
    DEFAULT_NEGATIVE_PROMPT,
    DiffusionUpscalePipeline,
)
from nn_upscaler.utils.device import get_device, get_device_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Neural Pixel Art to Realism Upscaler (Phase 5 Generative Diffusion)"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to input pixel art sprite/image",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="outputs/result_realistic.png",
        help="Path to save photorealistic upscaled result",
    )
    parser.add_argument(
        "--backend",
        "-b",
        type=str,
        choices=["native", "sdxl"],
        default="native",
        help="Diffusion backend: 'native' (fast repo model) or 'sdxl' (heavy ControlNet-Tile)",
    )
    parser.add_argument(
        "--category",
        "-c",
        type=str,
        default=None,
        choices=list(CATEGORY_PROMPTS.keys()),
        help="Gaming domain category preset for prompt guidance",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        default=None,
        help="Custom text prompt description",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=DEFAULT_NEGATIVE_PROMPT,
        help="Negative prompt for filtering unwanted visual artifacts",
    )
    parser.add_argument(
        "--tile-weight",
        type=float,
        default=0.75,
        help="Conditioning weight multiplier for structure preservation (0.5 - 1.0)",
    )
    parser.add_argument(
        "--steps",
        "-s",
        type=int,
        default=None,
        help="Number of sampling steps (default: 4 for native, 28 for SDXL)",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.0,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=4,
        help="Upscale factor (e.g. 2, 4, 8 for native model)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic generation",
    )
    parser.add_argument(
        "--weights",
        "-w",
        type=str,
        default=None,
        help="Optional path to model weights checkpoint for native model",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default=None,
        help="Compute device ('mps', 'cuda', 'cpu')",
    )
    parser.add_argument(
        "--no-bilateral",
        action="store_true",
        help="Disable bilateral dithering suppression filter",
    )
    return parser.parse_args()


def main():
    console = Console()
    args = parse_args()

    console.print(
        Panel.fit(
            "[bold cyan]Pixel Art -> Realism Depixelization Engine[/bold cyan]\n"
            "[yellow]Phase 5: Generative & Diffusion Approaches[/yellow]",
            border_style="cyan",
        )
    )

    input_path = Path(args.input)
    if not input_path.exists():
        console.print(f"[bold red]Error: Input file '{input_path}' not found![/bold red]")
        sys.exit(1)

    # Determine sampling steps
    default_steps = 4 if args.backend == "native" else 28
    steps = args.steps if args.steps is not None else default_steps

    device = get_device(args.device)
    dev_name = get_device_name(device)

    # Display configuration table
    table = Table(title="Inference Configuration", border_style="dim")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Input Image", str(input_path))
    table.add_row("Output Path", str(args.output))
    table.add_row("Backend", f"[bold]{args.backend.upper()}[/bold]")
    table.add_row("Compute Device", f"{dev_name} ({device})")
    table.add_row("Scale Factor", f"{args.scale}x")
    table.add_row("Category Preset", str(args.category or "None (General)"))
    table.add_row("Tile Weight", f"{args.tile_weight:.2f}")
    table.add_row("Sampling Steps", str(steps))
    table.add_row("Guidance Scale", f"{args.guidance_scale:.1f}")
    table.add_row("Random Seed", str(args.seed))
    table.add_row("Bilateral Filter", "Disabled" if args.no_bilateral else "Enabled")

    console.print(table)

    try:
        raw_sprite = Image.open(input_path).convert("RGB")
        console.print(
            f"[dim]Input sprite resolution: [bold]{raw_sprite.width}x{raw_sprite.height}[/bold][/dim]"
        )

        with console.status("[bold green]Initializing Diffusion Pipeline...[/bold green]"):
            pipeline = DiffusionUpscalePipeline(
                backend=args.backend,
                device=device,
                scale_factor=args.scale,
                weights_path=args.weights,
            )

        with console.status(
            f"[bold green]Generating photorealism ({steps} steps, backend={args.backend})...[/bold green]"
        ):
            start_time = time.perf_counter()
            result = pipeline.run(
                image=raw_sprite,
                category=args.category,
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                tile_weight=args.tile_weight,
                steps=steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed,
                apply_bilateral_dither_filter=not args.no_bilateral,
            )
            elapsed = time.perf_counter() - start_time

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(out_path)

        console.print(
            Panel.fit(
                f"[bold green]✔ Successfully generated realistic image![/bold green]\n"
                f"Saved to: [bold underline]{out_path}[/bold underline]\n"
                f"Resolution: [cyan]{result.width}x{result.height}[/cyan]\n"
                f"Generation time: [yellow]{elapsed:.2f} seconds[/yellow]",
                border_style="green",
            )
        )

    except ImportError as e:
        console.print(
            Panel.fit(
                f"[bold red]Dependency Missing:[/bold red] {e}\n\n"
                f"[yellow]Hint: For the SDXL backend, ensure diffusers is installed:\n"
                f"`uv pip install 'nn-upscaler[diffusion]'`\n"
                f"Or switch to native local mode using: `--backend native`[/yellow]",
                border_style="red",
            )
        )
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Generation Failed: {e}[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
