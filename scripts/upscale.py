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
    parser = argparse.ArgumentParser(
        description="Upscale pixel art image using neural or classical models."
    )
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--output", "-o", type=str, required=True, help="Path to save upscaled output"
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="edsr_light",
        choices=list_models(),
        help="Model architecture name",
    )
    parser.add_argument("--scale", "-s", type=int, default=4, help="Upscale factor (e.g. 2, 4, 8)")
    parser.add_argument(
        "--weights", "-w", type=str, default=None, help="Optional path to model weights checkpoint"
    )
    parser.add_argument(
        "--device", "-d", type=str, default=None, help="Compute device ('mps', 'cuda', 'cpu')"
    )
    parser.add_argument(
        "--tile-size", type=int, default=None, help="Tile size for memory-safe chunked inference"
    )
    parser.add_argument(
        "--category",
        "-c",
        type=str,
        default=None,
        help="Semantic category for conditioning (e.g. 'face', 'character', 'armor', 'nature', 'stone_architecture')",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        default=None,
        help="Text prompt description for semantic texture hallucination",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of sampling steps (for diffusion models, e.g. 1-8)",
    )
    parser.add_argument(
        "--tile-weight",
        type=float,
        default=None,
        help="Conditioning weight multiplier for structure preservation (0.0 - 1.0+)",
    )

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

    run_kwargs = {}
    if args.category is not None:
        run_kwargs["category"] = args.category
    if args.prompt is not None:
        run_kwargs["prompt"] = args.prompt
    if args.steps is not None:
        run_kwargs["steps"] = args.steps
    if args.tile_weight is not None:
        run_kwargs["tile_weight"] = args.tile_weight

    # Filter kwargs supported by model.forward
    import inspect

    sig = inspect.signature(model.forward)
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    filtered_kwargs = {k: v for k, v in run_kwargs.items() if has_var_kw or k in sig.parameters}
    if run_kwargs and not filtered_kwargs:
        print(f"Note: Model '{args.model}' does not accept conditioning arguments, ignoring them.")

    result = pipeline.run(img, **filtered_kwargs)
    print(f"Output resolution: {result.width}x{result.height}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(out_path)
    print(f"Saved upscaled image to: {out_path}")


if __name__ == "__main__":
    main()
