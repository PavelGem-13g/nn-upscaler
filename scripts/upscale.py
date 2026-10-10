"""CLI utility to upscale an image using any registered model."""

import argparse
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from nn_upscaler.inference.pipeline import InferencePipeline
from nn_upscaler.models.registry import get_model, list_models
from nn_upscaler.utils.device import get_device, get_device_name


def main():
    parser = argparse.ArgumentParser(description="Upscale pixel art image using neural or classical models.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path to save upscaled output")
    parser.add_argument(
        "--model", "-m", type=str, default="edsr_light", choices=list_models(),
        help="Model architecture name",
    )
    parser.add_argument("--scale", "-s", type=int, default=4, help="Upscale factor (e.g. 2, 4, 8)")
    parser.add_argument("--weights", "-w", type=str, default=None, help="Optional path to model weights checkpoint")
    parser.add_argument("--device", "-d", type=str, default=None, help="Compute device ('mps', 'cuda', 'cpu')")
    parser.add_argument("--tile-size", type=int, default=None, help="Tile size for memory-safe chunked inference")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input image '{input_path}' not found.", file=sys.stderr)
        sys.exit(1)

    device = get_device(args.device)
    print(f"Using device: {get_device_name(device)} ({device})")
    print(f"Loading model: {args.model} (scale={args.scale}x)...")

    model = get_model(args.model, scale_factor=args.scale)
    if args.weights:
        ckpt_path = Path(args.weights)
        if not ckpt_path.exists():
            print(f"Error: checkpoint '{ckpt_path}' not found.", file=sys.stderr)
            sys.exit(1)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        print(f"Loaded weights from: {ckpt_path}")

    pipeline = InferencePipeline(model=model, device=device, tile_size=args.tile_size)

    img = Image.open(input_path).convert("RGB")
    print(f"Input image resolution: {img.width}x{img.height}")

    result = pipeline.run(img)
    print(f"Output resolution: {result.width}x{result.height}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(out_path)
    print(f"Saved upscaled image to: {out_path}")


if __name__ == "__main__":
    main()
