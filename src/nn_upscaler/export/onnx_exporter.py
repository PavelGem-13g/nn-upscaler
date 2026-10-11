"""Cross-platform ONNX export and high-performance ONNX Runtime inference wrapper."""

from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from nn_upscaler.models.base import BaseUpscaler


def export_to_onnx(
    model: nn.Module,
    save_path: str | Path,
    scale: int = 4,
    input_size: tuple[int, int, int, int] = (1, 3, 64, 64),
    dynamic_axes: bool = True,
    opset_version: int = 17,
    check_model: bool = True,
) -> Path:
    """Export a PyTorch upscaler model to ONNX format.

    Args:
        model: PyTorch upscaler model (e.g. EDSRLightUpscaler, RRDBNet, etc.).
        save_path: Destination path for the .onnx model file.
        scale: Upscale factor of the model.
        input_size: Tuple (B, C, H, W) for dummy input tracing.
        dynamic_axes: Whether to declare dynamic batch size and spatial dimensions.
        opset_version: Target ONNX opset version (default: 17).
        check_model: Whether to run onnx.checker.check_model on the exported graph.

    Returns:
        Path to the saved ONNX file.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Move model to CPU and eval mode for stable tracing
    model_cpu = model.to("cpu").eval()

    # Infer parameter dtype (e.g. float32 or float16)
    first_param = next(model_cpu.parameters(), None)
    input_dtype = (
        first_param.dtype
        if first_param is not None and first_param.is_floating_point()
        else torch.float32
    )
    dummy_input = torch.randn(*input_size, dtype=input_dtype)

    dynamic_axes_dict = None
    if dynamic_axes:
        dynamic_axes_dict = {
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "output": {0: "batch_size", 2: "out_height", 3: "out_width"},
        }

    # Use legacy TorchScript-based exporter (dynamo=False) for maximal stability across all PyTorch versions
    try:
        torch.onnx.export(
            model_cpu,
            dummy_input,
            str(save_path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes_dict,
            opset_version=opset_version,
            do_constant_folding=True,
            dynamo=False,
        )
    except TypeError:
        # Fallback if dynamo argument is not accepted in older PyTorch versions
        torch.onnx.export(
            model_cpu,
            dummy_input,
            str(save_path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes_dict,
            opset_version=opset_version,
            do_constant_folding=True,
        )

    if check_model:
        model_proto = onnx.load(str(save_path))
        onnx.checker.check_model(model_proto)

    return save_path


class ONNXUpscaler(BaseUpscaler):
    """BaseUpscaler implementation executing an exported ONNX model via ONNX Runtime."""

    def __init__(
        self,
        model_path: str | Path,
        scale_factor: int = 4,
        providers: list[str] | None = None,
    ):
        """Initialize the ONNX Runtime inference session.

        Args:
            model_path: Path to the .onnx file.
            scale_factor: Upscaling multiplier.
            providers: List of execution providers (e.g. CoreMLExecutionProvider, CUDAExecutionProvider).
        """
        super().__init__(scale_factor=scale_factor)
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model file not found: {self.model_path}")

        available_providers = ort.get_available_providers()
        if providers is None:
            # Prefer hardware-accelerated providers when available
            candidate_providers = [
                "CoreMLExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
            providers = [p for p in candidate_providers if p in available_providers]
            if not providers:
                providers = available_providers

        self.providers = providers
        self.session = ort.InferenceSession(str(self.model_path), providers=self.providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.is_fp16 = "float16" in self.session.get_inputs()[0].type

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Run forward inference pass on batched input tensor [B, C, H, W].

        Args:
            x: Input tensor [B, C, H, W] in [0, 1].

        Returns:
            Upscaled output tensor [B, C, H * scale, W * scale].
        """
        orig_device = x.device
        orig_dtype = x.dtype

        cast_dtype = torch.float16 if self.is_fp16 else torch.float32
        x_np = x.detach().cpu().to(dtype=cast_dtype).numpy()
        ort_inputs = {self.input_name: x_np}
        ort_outputs = self.session.run([self.output_name], ort_inputs)
        out_tensor = torch.from_numpy(ort_outputs[0]).to(device=orig_device, dtype=orig_dtype)
        return out_tensor


def verify_onnx_equivalence(
    pytorch_model: nn.Module,
    onnx_target: str | Path | ONNXUpscaler,
    scale_factor: int = 4,
    input_shape: tuple[int, int, int, int] = (1, 3, 32, 32),
    atol: float = 1e-4,
    rtol: float = 1e-3,
) -> dict[str, Any]:
    """Verify numerical equivalence between PyTorch model and ONNX Runtime session.

    Args:
        pytorch_model: PyTorch model module.
        onnx_target: File path or instantiated ONNXUpscaler instance.
        scale_factor: Expected scale factor.
        input_shape: Shape of test input tensor.
        atol: Absolute tolerance.
        rtol: Relative tolerance.

    Returns:
        Dictionary with comparison metrics (matches, max_abs_diff, mean_abs_diff).
    """
    if isinstance(onnx_target, (str, Path)):
        onnx_model = ONNXUpscaler(onnx_target, scale_factor=scale_factor)
    else:
        onnx_model = onnx_target

    torch.manual_seed(42)
    x = torch.rand(*input_shape, dtype=torch.float32)

    first_param = next(pytorch_model.parameters(), None)
    if first_param is not None and first_param.dtype == torch.float16:
        x_pt = x.half()
        # Relax tolerance slightly for FP16
        atol = max(atol, 2e-3)
        rtol = max(rtol, 2e-2)
    else:
        x_pt = x

    with torch.no_grad():
        y_pt = pytorch_model.to("cpu").eval()(x_pt)
        y_onnx = onnx_model(x)

    diff = (y_pt.float() - y_onnx.float()).abs()
    max_diff = float(diff.max().item())
    mean_diff = float(diff.mean().item())
    matches = bool(
        np.allclose(
            y_pt.detach().cpu().numpy(),
            y_onnx.detach().cpu().numpy(),
            rtol=rtol,
            atol=atol,
        )
    )

    return {
        "matches": matches,
        "max_abs_diff": round(max_diff, 8),
        "mean_abs_diff": round(mean_diff, 8),
        "input_shape": list(input_shape),
        "output_shape": list(y_onnx.shape),
    }
