"""Model quantization utilities for post-training acceleration (FP16 and dynamic INT8)."""

import copy
import io
import warnings
from typing import Any

import torch
from torch import nn

from nn_upscaler.models.base import BaseUpscaler


def setup_quantization_engine() -> str:
    """Configure supported backend engine for PyTorch CPU quantization.

    Returns:
        The name of the configured quantization engine ('qnnpack', 'fbgemm', or 'none').
    """
    supported = torch.backends.quantized.supported_engines
    if "qnnpack" in supported:
        torch.backends.quantized.engine = "qnnpack"
        return "qnnpack"
    if "fbgemm" in supported:
        torch.backends.quantized.engine = "fbgemm"
        return "fbgemm"
    return "none"


class FP16Upscaler(BaseUpscaler):
    """Wrapper that runs model weights and computation in float16 half precision.

    Accepts float32 or float16 input tensors, casts to float16 for internal layers,
    and returns tensor in the original input dtype.
    """

    def __init__(self, model: nn.Module):
        scale = getattr(model, "scale_factor", 4)
        super().__init__(scale_factor=scale)
        self.inner_model = model.half()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        orig_dtype = x.dtype
        x_half = x.to(dtype=torch.float16)
        out = self.inner_model(x_half, **kwargs)
        return out.to(dtype=orig_dtype)


class DynamicINT8Upscaler(BaseUpscaler):
    """Wrapper around dynamically quantized INT8 CPU model.

    Routes input tensors through CPU in float32, executes quantized operators,
    and moves result back to original device.
    """

    def __init__(self, model: nn.Module):
        scale = getattr(model, "scale_factor", 4)
        super().__init__(scale_factor=scale)
        self.inner_model = model

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        orig_device = x.device
        orig_dtype = x.dtype
        x_cpu = x.detach().to(device="cpu", dtype=torch.float32)
        out = self.inner_model(x_cpu, **kwargs)
        return out.to(device=orig_device, dtype=orig_dtype)


def quantize_fp16(model: nn.Module, wrap_io: bool = True) -> nn.Module:
    """Convert model weights to half-precision FP16.

    Args:
        model: PyTorch neural upscaler module.
        wrap_io: If True, wrap with FP16Upscaler to accept standard float32 inputs.

    Returns:
        Quantized FP16 module.
    """
    model_copy = copy.deepcopy(model).eval()
    if wrap_io:
        return FP16Upscaler(model_copy)
    return model_copy.half()


def quantize_dynamic_int8(
    model: nn.Module,
    qconfig_spec: set[type[nn.Module]] | None = None,
    wrap_io: bool = True,
) -> nn.Module:
    """Apply dynamic INT8 post-training quantization (PTQ) to linear layers for CPU acceleration.

    Args:
        model: PyTorch neural upscaler module.
        qconfig_spec: Optional set of module types to quantize. Defaults to {nn.Linear}.
        wrap_io: If True, wrap with DynamicINT8Upscaler for automatic CPU routing.

    Returns:
        Dynamically quantized module.
    """
    setup_quantization_engine()
    model_cpu = copy.deepcopy(model).cpu().eval()

    if qconfig_spec is None:
        qconfig_spec = {nn.Linear}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            quantized = torch.ao.quantization.quantize_dynamic(
                model_cpu,
                qconfig_spec=qconfig_spec,
                dtype=torch.qint8,
            )
        except Exception:
            # Fallback to Linear only if extended spec fails
            quantized = torch.ao.quantization.quantize_dynamic(
                model_cpu,
                qconfig_spec={nn.Linear},
                dtype=torch.qint8,
            )

    if wrap_io:
        return DynamicINT8Upscaler(quantized)
    return quantized


def measure_model_size_mb(model: nn.Module) -> float:
    """Measure serialized state_dict size in Megabytes (MB).

    Args:
        model: PyTorch module.

    Returns:
        Size in Megabytes.
    """
    buffer = io.BytesIO()
    # If wrapped, access inner model for state dict measurement
    target = getattr(model, "inner_model", model)
    torch.save(target.state_dict(), buffer)
    return buffer.tell() / (1024 * 1024)


def compare_quantization_size(
    original_model: nn.Module,
    quantized_model: nn.Module,
) -> dict[str, Any]:
    """Compare storage size between original and quantized models.

    Args:
        original_model: Baseline full-precision model.
        quantized_model: Quantized model.

    Returns:
        Dictionary with original size, quantized size, compression ratio, and savings.
    """
    orig_mb = measure_model_size_mb(original_model)
    quant_mb = measure_model_size_mb(quantized_model)
    ratio = orig_mb / max(quant_mb, 1e-6)
    savings = (1.0 - (quant_mb / max(orig_mb, 1e-6))) * 100.0

    return {
        "original_size_mb": round(orig_mb, 3),
        "quantized_size_mb": round(quant_mb, 3),
        "compression_ratio": round(ratio, 2),
        "savings_percent": round(savings, 2),
    }
