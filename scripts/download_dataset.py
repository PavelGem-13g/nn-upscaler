"""CLI script to download, curate, and validate datasets for pixel-art super-resolution."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nn_upscaler.data.downloader import (
    DatasetCurator,
    download_authentic_sprites,
    download_div2k_samples,
    validate_image_file,
)

console = Console()


def print_summary_table(title: str, directory: Path, files: list[Path]) -> None:
    table = Table(title=f"Dataset Summary: {title}")
    table.add_column("Filename", style="cyan")
    table.add_column("Resolution (WxH)", style="green")
    table.add_column("File Size", style="magenta")
    table.add_column("Validation", style="bold green")

    from PIL import Image

    for p in files:
        if not p.is_file():
            continue
        try:
            with Image.open(p) as im:
                w, h = im.size
            size_kb = f"{p.stat().st_size / 1024:.1f} KB"
            valid = "PASS" if validate_image_file(p) else "FAIL"
            table.add_row(p.name, f"{w}x{h}", size_kb, valid)
        except Exception:
            table.add_row(p.name, "Corrupt", "0 KB", "FAIL")

    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and curate datasets for pixel art to realism research."
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="all",
        choices=["div2k_sample", "authentic_sprites", "all"],
        help="Dataset subset to download",
    )
    parser.add_argument(
        "--target-dir",
        "-t",
        type=str,
        default=None,
        help="Optional custom target directory override",
    )
    parser.add_argument(
        "--count",
        "-n",
        type=int,
        default=5,
        help="Number of images to download per dataset",
    )

    args = parser.parse_args()

    console.rule("[bold cyan]Dataset Downloader & Curator[/bold cyan]")

    # 1. DIV2K / HR Sample Subset
    if args.dataset in ("div2k_sample", "all"):
        target = (
            Path(args.target_dir)
            if args.target_dir and args.dataset == "div2k_sample"
            else Path("data/hr_raw")
        )
        console.print(
            f"[yellow]Fetching high-resolution realistic samples into:[/yellow] [bold]{target}[/bold]..."
        )
        files = download_div2k_samples(target_dir=target, count=args.count)
        curation = DatasetCurator.curate(target, min_size=(128, 128))
        console.print(
            f"✔ Curated {len(files)} files (Removed {curation['corrupted_removed']} corrupt, "
            f"{curation['duplicates_removed']} duplicates)."
        )
        print_summary_table("High-Resolution Raw (data/hr_raw)", target, files)

    # 2. Authentic Pixel Art Sprites
    if args.dataset in ("authentic_sprites", "all"):
        target = (
            Path(args.target_dir)
            if args.target_dir and args.dataset == "authentic_sprites"
            else Path("data/benchmarks/authentic_pixel_art")
        )
        console.print(
            f"[yellow]Fetching authentic pixel-art benchmark sprites into:[/yellow] [bold]{target}[/bold]..."
        )
        files = download_authentic_sprites(target_dir=target, count=args.count)
        curation = DatasetCurator.curate(target, min_size=(16, 16))
        console.print(
            f"✔ Curated {len(files)} sprites (Removed {curation['corrupted_removed']} corrupt, "
            f"{curation['duplicates_removed']} duplicates)."
        )
        print_summary_table(
            "Authentic Pixel Art (data/benchmarks/authentic_pixel_art)", target, files
        )

    console.print("[bold green]✔ Dataset setup & curation complete![/bold green]")


if __name__ == "__main__":
    main()
