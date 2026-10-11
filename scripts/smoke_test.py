"""Smoke test script: verifies environment, acceleration device, and all models."""

import sys
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table

# Add src to pythonpath if run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nn_upscaler.data.degradation import PixelArtDegradation
from nn_upscaler.evaluation.metrics import calculate_psnr, calculate_ssim
from nn_upscaler.models.registry import get_model, list_models
from nn_upscaler.utils.device import get_device, get_device_name


def run_smoke_test():
    console = Console()
    console.print("\n[bold cyan]=== Neural Upscaler Smoke Test ===[/bold cyan]\n")

    # 1. Device check
    device = get_device()
    dev_name = get_device_name(device)
    console.print(
        f"[bold green]✔[/bold green] Selected Device: [bold yellow]{dev_name}[/bold yellow] ({device})"
    )

    # 2. Test models forward pass
    table = Table(title="Model Registry & Forward Pass Verification")
    table.add_column("Model", style="cyan")
    table.add_column("Scale", style="magenta")
    table.add_column("Input Shape", style="white")
    table.add_column("Output Shape", style="white")
    table.add_column("Status", style="green")

    models = list_models()
    x = torch.rand(1, 3, 16, 16, device=device)

    for m_name in models:
        try:
            model = get_model(m_name, scale_factor=4).to(device)
            model.eval()
            with torch.no_grad():
                out = model(x)
            assert out.shape == (1, 3, 64, 64), f"Unexpected shape {out.shape}"
            table.add_row(
                m_name,
                "4x",
                str(list(x.shape)),
                str(list(out.shape)),
                "[bold green]PASS[/bold green]",
            )
        except Exception as e:
            table.add_row(
                m_name, "4x", str(list(x.shape)), "ERROR", f"[bold red]FAIL: {e}[/bold red]"
            )

    console.print(table)

    # 3. Test degradation pipeline
    from PIL import Image

    test_img = Image.new("RGB", (64, 64), color=(120, 150, 180))
    degrader = PixelArtDegradation(scale_factor=4, num_colors=16)
    lr_art = degrader(test_img)
    assert lr_art.size == (16, 16), f"Degradation size mismatch: {lr_art.size}"
    console.print(
        "[bold green]✔[/bold green] PixelArtDegradation pipeline operational (64x64 -> 16x16 with 16-color palette)"
    )

    # 4. Test metrics
    t1 = torch.rand(3, 32, 32)
    t2 = t1 + torch.randn(3, 32, 32) * 0.05
    psnr = calculate_psnr(t2.clamp(0, 1), t1)
    ssim = calculate_ssim(t2.clamp(0, 1), t1)
    console.print(
        f"[bold green]✔[/bold green] Metrics calculation operational (PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f})"
    )

    # 5. Test ONNX Export and ONNX Runtime inference
    import shutil
    import tempfile

    from nn_upscaler.export.onnx_exporter import (
        ONNXUpscaler,
        export_to_onnx,
        verify_onnx_equivalence,
    )
    from nn_upscaler.export.quantization import quantize_dynamic_int8, quantize_fp16

    tmp_smoke_dir = Path(tempfile.mkdtemp(prefix="smoke_onnx_"))
    try:
        mini_model = get_model("edsr_light", scale_factor=4).eval()
        onnx_file = tmp_smoke_dir / "smoke_edsr.onnx"
        export_to_onnx(mini_model, onnx_file, scale=4, input_size=(1, 3, 16, 16), check_model=True)
        onnx_upscaler = ONNXUpscaler(onnx_file, scale_factor=4)
        x_smoke = torch.rand(1, 3, 16, 16)
        y_smoke = onnx_upscaler(x_smoke)
        assert y_smoke.shape == (1, 3, 64, 64)
        equiv = verify_onnx_equivalence(
            mini_model, onnx_upscaler, scale_factor=4, input_shape=(1, 3, 16, 16)
        )
        assert equiv["matches"], f"ONNX equivalence check failed: diff={equiv['max_abs_diff']}"
        console.print(
            f"[bold green]✔[/bold green] ONNX export & ONNX Runtime execution operational (Max Diff: {equiv['max_abs_diff']:.2e})"
        )

        # 6. Test Quantization (FP16 & Dynamic INT8)
        fp16_model = quantize_fp16(mini_model)
        y_fp16 = fp16_model(x_smoke)
        assert y_fp16.shape == (1, 3, 64, 64)

        int8_model = quantize_dynamic_int8(mini_model)
        y_int8 = int8_model(x_smoke)
        assert y_int8.shape == (1, 3, 64, 64)
        console.print(
            "[bold green]✔[/bold green] Quantization pipelines operational (FP16 & Dynamic INT8)"
        )
    finally:
        shutil.rmtree(tmp_smoke_dir, ignore_errors=True)

    console.print(
        "\n[bold green]All systems operational! The repository is ready for experimentation.[/bold green]\n"
    )


if __name__ == "__main__":
    run_smoke_test()
