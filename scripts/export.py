"""CLI utility to export and quantize neural upscaler models (ONNX, CoreML, TorchScript)."""

import argparse
import sys
import time
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from rich.console import Console
from rich.table import Table

from nn_upscaler.export.coreml_exporter import (
    COREML_AVAILABLE,
    export_to_coreml,
    verify_coreml_equivalence,
)
from nn_upscaler.export.onnx_exporter import (
    ONNXUpscaler,
    export_to_onnx,
    verify_onnx_equivalence,
)
from nn_upscaler.export.quantization import (
    measure_model_size_mb,
    quantize_dynamic_int8,
    quantize_fp16,
)
from nn_upscaler.models.registry import get_model, list_models

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Production Export & Optimization CLI for Neural Upscalers."
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="edsr_light",
        choices=list_models(),
        help="Model architecture name",
    )
    parser.add_argument(
        "--scale",
        "-s",
        type=int,
        default=4,
        help="Upscale factor (e.g. 2, 4, 8)",
    )
    parser.add_argument(
        "--weights",
        "-w",
        type=str,
        default=None,
        help="Optional path to model checkpoint weights (.pt / .pth)",
    )
    parser.add_argument(
        "--format",
        "-f",
        type=str,
        default="onnx",
        choices=["onnx", "coreml", "torchscript"],
        help="Target export format",
    )
    parser.add_argument(
        "--quantize",
        "-q",
        type=str,
        default="none",
        choices=["none", "fp16", "int8"],
        help="Post-training quantization precision",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Destination path for the exported model artifact",
    )
    parser.add_argument(
        "--input-size",
        type=str,
        default="1,3,64,64",
        help="Input tensor shape as B,C,H,W (e.g. '1,3,64,64')",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )
    parser.add_argument(
        "--compute-units",
        type=str,
        default="ALL",
        choices=["ALL", "CPU_AND_NE", "CPU_AND_GPU", "CPU_ONLY"],
        help="Target compute units for CoreML export",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify numerical output equivalence with PyTorch baseline",
    )
    parser.add_argument(
        "--sample-image",
        type=str,
        default=None,
        help="Optional path to a test image to verify upscaling pass",
    )
    return parser.parse_args()


def parse_shape(shape_str: str) -> tuple[int, int, int, int]:
    parts = [int(p.strip()) for p in shape_str.split(",")]
    if len(parts) == 2:
        return 1, 3, parts[0], parts[1]
    if len(parts) == 4:
        return parts[0], parts[1], parts[2], parts[3]
    raise ValueError(f"Invalid shape string '{shape_str}'. Expected 'H,W' or 'B,C,H,W'.")


def main():
    args = parse_args()
    input_shape = parse_shape(args.input_size)

    console.rule("[bold cyan]Neural Upscaler Model Export & Optimization[/bold cyan]")
    console.print(f"Loading model [bold green]{args.model}[/bold green] (scale={args.scale}x)...")

    # 1. Instantiate PyTorch baseline
    model = get_model(args.model, scale_factor=args.scale)
    if args.weights:
        ckpt_path = Path(args.weights)
        if not ckpt_path.exists():
            console.print(f"[bold red]Error:[/bold red] Weights not found: {ckpt_path}")
            sys.exit(1)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        console.print(f"Loaded weights from: {ckpt_path}")

    model.eval().to("cpu")
    original_size_mb = measure_model_size_mb(model)

    # 2. Apply quantization if requested
    export_model = model
    quant_info = "None (FP32)"
    if args.quantize == "fp16":
        console.print("Applying [bold yellow]FP16[/bold yellow] quantization...")
        export_model = quantize_fp16(model, wrap_io=False)
        quant_info = "FP16 (Half Precision)"
    elif args.quantize == "int8":
        console.print("Applying [bold yellow]Dynamic INT8[/bold yellow] quantization...")
        export_model = quantize_dynamic_int8(model, wrap_io=False)
        quant_info = "INT8 (Dynamic PTQ)"

    # 3. Determine default output path if not specified
    if args.output is None:
        models_dir = Path("exported_models")
        suffix = (
            ".onnx"
            if args.format == "onnx"
            else (".mlpackage" if args.format == "coreml" else ".pt")
        )
        q_tag = f"_{args.quantize}" if args.quantize != "none" else ""
        out_path = models_dir / f"{args.model}_x{args.scale}{q_tag}{suffix}"
    else:
        out_path = Path(args.output)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    verification_status = "Skipped"
    max_diff_str = "N/A"

    # 4. Perform export
    if args.format == "onnx":
        console.print(f"Exporting to ONNX ([green]{out_path}[/green])...")
        export_to_onnx(
            export_model,
            save_path=out_path,
            scale=args.scale,
            input_size=input_shape,
            dynamic_axes=True,
            opset_version=args.opset,
            check_model=True,
        )

        if args.verify:
            console.print("Verifying numerical equivalence PyTorch vs ONNX Runtime...")
            try:
                # If exported model was wrapped or modified, compare against PyTorch baseline
                onnx_up = ONNXUpscaler(out_path, scale_factor=args.scale)
                equiv = verify_onnx_equivalence(
                    pytorch_model=model,
                    onnx_target=onnx_up,
                    scale_factor=args.scale,
                    input_shape=input_shape,
                )
                verification_status = "PASS" if equiv["matches"] else "FAIL (tolerance)"
                max_diff_str = f"{equiv['max_abs_diff']:.2e}"
            except Exception as e:
                verification_status = f"Error: {e}"

    elif args.format == "coreml":
        if not COREML_AVAILABLE:
            console.print("[bold red]Error:[/bold red] coremltools not available.")
            sys.exit(1)
        console.print(f"Exporting to CoreML .mlpackage ([green]{out_path}[/green])...")
        export_to_coreml(
            export_model,
            save_path=out_path,
            scale=args.scale,
            input_size=input_shape,
            compute_units=args.compute_units,
            flexible_shape=True,
        )

        if args.verify:
            console.print("Verifying numerical equivalence PyTorch vs CoreML...")
            try:
                equiv = verify_coreml_equivalence(
                    pytorch_model=model,
                    coreml_target=out_path,
                    scale_factor=args.scale,
                    input_shape=input_shape,
                )
                verification_status = "PASS" if equiv["matches"] else "FAIL (tolerance)"
                max_diff_str = f"{equiv['max_abs_diff']:.2e}"
            except Exception as e:
                verification_status = f"Error: {e}"

    elif args.format == "torchscript":
        console.print(f"Exporting to TorchScript ([green]{out_path}[/green])...")
        dummy = torch.randn(*input_shape, dtype=torch.float32)
        traced = torch.jit.trace(export_model, dummy)
        torch.jit.save(traced, str(out_path))

        if args.verify:
            console.print("Verifying TorchScript numerical equivalence...")
            loaded_ts = torch.jit.load(str(out_path))
            with torch.no_grad():
                y_pt = model(dummy)
                y_ts = loaded_ts(dummy)
                diff = (y_pt - y_ts).abs().max().item()
            verification_status = "PASS" if diff < 1e-4 else "FAIL"
            max_diff_str = f"{diff:.2e}"

    export_duration_ms = (time.perf_counter() - start_time) * 1000.0

    # 5. Measure output file size
    if out_path.is_file():
        file_size_mb = out_path.stat().st_size / (1024 * 1024)
    elif out_path.is_dir():
        file_size_mb = sum(f.stat().st_size for f in out_path.rglob("*") if f.is_file()) / (
            1024 * 1024
        )
    else:
        file_size_mb = 0.0

    # 6. Sample image verification if provided
    if args.sample_image:
        sample_path = Path(args.sample_image)
        if sample_path.exists():
            from PIL import Image

            from nn_upscaler.inference.pipeline import InferencePipeline

            img = Image.open(sample_path).convert("RGB")
            console.print(
                f"Testing sample image '{sample_path.name}' ({img.width}x{img.height})..."
            )
            if args.format == "onnx":
                up_engine = ONNXUpscaler(out_path, scale_factor=args.scale)
            else:
                up_engine = model
            pipe = InferencePipeline(up_engine)
            res = pipe.run(img)
            console.print(f"Sample upscaled resolution: [green]{res.width}x{res.height}[/green]")

    # 7. Print summary table
    table = Table(title="Model Export & Optimization Report")
    table.add_column("Property", style="bold cyan")
    table.add_column("Value", style="bold white")

    table.add_row("Model Architecture", args.model)
    table.add_row("Scale Factor", f"{args.scale}x")
    table.add_row("Export Format", args.format.upper())
    table.add_row("Quantization", quant_info)
    table.add_row("Input Shape", str(list(input_shape)))
    table.add_row("Original Size", f"{original_size_mb:.2f} MB")
    table.add_row("Exported Artifact Size", f"{file_size_mb:.2f} MB")
    table.add_row("Export Latency", f"{export_duration_ms:.1f} ms")
    table.add_row("Verification Status", verification_status)
    table.add_row("Max Numerical Diff", max_diff_str)
    table.add_row("Destination", str(out_path))

    console.print()
    console.print(table)
    console.print(f"[bold green]✔ Export complete:[/bold green] {out_path}")


if __name__ == "__main__":
    main()
