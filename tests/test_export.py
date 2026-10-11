"""Unit and integration tests for model export (ONNX, CoreML, Quantization)."""

import platform
import shutil
import tempfile
from pathlib import Path

import pytest
import torch
from PIL import Image

from nn_upscaler.export.coreml_exporter import (
    COREML_AVAILABLE,
    CoreMLUpscaler,
    export_to_coreml,
    verify_coreml_equivalence,
)
from nn_upscaler.export.onnx_exporter import (
    ONNXUpscaler,
    export_to_onnx,
    verify_onnx_equivalence,
)
from nn_upscaler.export.quantization import (
    compare_quantization_size,
    quantize_dynamic_int8,
    quantize_fp16,
)
from nn_upscaler.models.edsr_light import EDSRLightUpscaler
from nn_upscaler.models.registry import get_model


@pytest.fixture
def temp_export_dir():
    """Create a temporary directory for export artifacts."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="test_export_"))
    yield tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_onnx_export_and_checker(temp_export_dir):
    """Test standard ONNX export and onnx.checker validation."""
    model = EDSRLightUpscaler(scale_factor=4, num_features=16, num_blocks=2)
    onnx_path = temp_export_dir / "edsr_mini.onnx"

    exported = export_to_onnx(
        model=model,
        save_path=onnx_path,
        scale=4,
        input_size=(1, 3, 32, 32),
        dynamic_axes=True,
        opset_version=17,
        check_model=True,
    )

    assert exported.exists()
    assert exported.stat().st_size > 0


def test_onnx_dynamic_axes_and_inference(temp_export_dir):
    """Test dynamic batch and spatial dimensions via ONNXUpscaler."""
    model = EDSRLightUpscaler(scale_factor=4, num_features=16, num_blocks=2)
    onnx_path = temp_export_dir / "edsr_dyn.onnx"
    export_to_onnx(model, onnx_path, scale=4, input_size=(1, 3, 16, 16), dynamic_axes=True)

    upscaler = ONNXUpscaler(onnx_path, scale_factor=4)

    # Test shape 1: batch 1, 32x32 -> 128x128
    x1 = torch.rand(1, 3, 32, 32)
    y1 = upscaler(x1)
    assert y1.shape == (1, 3, 128, 128)

    # Test shape 2: batch 2, 48x64 -> 192x256
    x2 = torch.rand(2, 3, 48, 64)
    y2 = upscaler(x2)
    assert y2.shape == (2, 3, 192, 256)


def test_onnx_numerical_equivalence(temp_export_dir):
    """Verify tight numerical equivalence between PyTorch model and ONNX Runtime."""
    model = EDSRLightUpscaler(scale_factor=4, num_features=16, num_blocks=2).eval()
    onnx_path = temp_export_dir / "edsr_equiv.onnx"
    export_to_onnx(model, onnx_path, scale=4, input_size=(1, 3, 24, 24))

    report = verify_onnx_equivalence(
        pytorch_model=model,
        onnx_target=onnx_path,
        scale_factor=4,
        input_shape=(1, 3, 24, 24),
        atol=1e-4,
    )

    assert report["matches"] is True
    assert report["max_abs_diff"] < 1e-4


def test_onnx_upscaler_pil_inference(temp_export_dir):
    """Test high-level upscale method accepting PIL image on ONNXUpscaler."""
    model = EDSRLightUpscaler(scale_factor=4, num_features=16, num_blocks=2)
    onnx_path = temp_export_dir / "edsr_pil.onnx"
    export_to_onnx(model, onnx_path, scale=4, input_size=(1, 3, 16, 16))

    upscaler = ONNXUpscaler(onnx_path, scale_factor=4)
    pil_in = Image.new("RGB", (16, 16), color=(100, 150, 200))
    pil_out = upscaler.upscale(pil_in)

    assert isinstance(pil_out, Image.Image)
    assert pil_out.size == (64, 64)


def test_quantize_fp16():
    """Test FP16 quantization wrapper and size reduction."""
    model = get_model("edsr_light", scale_factor=4)
    fp16_model = quantize_fp16(model, wrap_io=True)

    x = torch.rand(1, 3, 16, 16, dtype=torch.float32)
    y = fp16_model(x)

    assert y.shape == (1, 3, 64, 64)
    assert y.dtype == torch.float32

    size_stats = compare_quantization_size(model, fp16_model)
    assert size_stats["compression_ratio"] >= 1.8
    assert size_stats["savings_percent"] >= 45.0


def test_quantize_dynamic_int8():
    """Test Dynamic INT8 quantization for CPU execution."""
    model = get_model("edsr_light", scale_factor=4)
    int8_model = quantize_dynamic_int8(model, wrap_io=True)

    x = torch.rand(1, 3, 16, 16, dtype=torch.float32)
    y = int8_model(x)

    assert y.shape == (1, 3, 64, 64)
    assert y.dtype == torch.float32


@pytest.mark.skipif(
    platform.system() != "Darwin" or not COREML_AVAILABLE,
    reason="CoreML export requires macOS and coremltools",
)
def test_coreml_export_and_inference(temp_export_dir):
    """Test Apple CoreML export and inference on Apple Silicon/macOS."""
    model = EDSRLightUpscaler(scale_factor=4, num_features=16, num_blocks=2).eval()
    ml_path = temp_export_dir / "edsr_mini.mlpackage"

    exported = export_to_coreml(
        model=model,
        save_path=ml_path,
        scale=4,
        input_size=(1, 3, 24, 24),
        compute_units="ALL",
        flexible_shape=True,
    )

    assert exported.exists()
    assert (exported / "Data").exists() or (exported / "Manifest.json").exists()

    cml_upscaler = CoreMLUpscaler(exported, scale_factor=4)
    x = torch.rand(1, 3, 24, 24)
    y = cml_upscaler(x)
    assert y.shape == (1, 3, 96, 96)

    equiv = verify_coreml_equivalence(
        model, cml_upscaler, scale_factor=4, input_shape=(1, 3, 24, 24)
    )
    assert equiv["matches"] is True
