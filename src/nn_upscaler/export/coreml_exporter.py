"""Apple Silicon CoreML export (.mlpackage) and Apple Neural Engine (ANE) inference."""

import platform
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from nn_upscaler.models.base import BaseUpscaler

# Graceful optional import for coremltools
try:
    import coremltools as ct

    COREML_AVAILABLE = True
except ImportError:
    ct = None
    COREML_AVAILABLE = False


def check_coreml_support() -> tuple[bool, str]:
    """Check whether CoreML export and execution are supported on current environment.

    Returns:
        Tuple of (is_supported, reason_message).
    """
    if platform.system() != "Darwin":
        return False, f"CoreML requires macOS, but current OS is {platform.system()}"
    if not COREML_AVAILABLE:
        return (
            False,
            "coremltools is not installed. Install via `pip install coremltools` or `uv add --optional coreml coremltools`",
        )
    return True, "CoreML is fully supported"


def export_to_coreml(
    model: nn.Module,
    save_path: str | Path,
    scale: int = 4,
    input_size: tuple[int, int, int, int] = (1, 3, 64, 64),
    compute_units: str = "ALL",
    flexible_shape: bool = True,
    author: str = "Neural Upscaler Team",
    license_str: str = "MIT",
    description: str | None = None,
) -> Path:
    """Export a PyTorch upscaler model to Apple CoreML .mlpackage format.

    Args:
        model: PyTorch upscaler module.
        save_path: Target directory path ending in .mlpackage.
        scale: Model upscaling factor.
        input_size: Base input dimensions (B, C, H, W).
        compute_units: Hardware compute unit target ('ALL', 'CPU_AND_NE', 'CPU_AND_GPU', 'CPU_ONLY').
        flexible_shape: Whether to enable flexible input spatial resolutions for sprites.
        author: Model author metadata.
        license_str: Model license metadata.
        description: Model description metadata.

    Returns:
        Path to the saved .mlpackage.
    """
    supported, reason = check_coreml_support()
    if not supported:
        raise RuntimeError(f"CoreML export cannot proceed: {reason}")

    save_path = Path(save_path)
    if save_path.suffix != ".mlpackage":
        save_path = save_path.with_suffix(".mlpackage")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Unit mapping
    unit_map = {
        "ALL": ct.ComputeUnit.ALL,
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
    }
    selected_units = unit_map.get(compute_units.upper(), ct.ComputeUnit.ALL)

    model_cpu = model.to("cpu").eval()
    dummy_input = torch.randn(*input_size, dtype=torch.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        traced_model = torch.jit.trace(model_cpu, dummy_input)

    # Define input tensor specification
    _b, _c, h, w = input_size
    if flexible_shape:
        # Flexible range dimensions for pixel art resolutions (16px to 512px)
        h_dim = ct.RangeDim(16, 512, default=h)
        w_dim = ct.RangeDim(16, 512, default=w)
        shape_spec = (1, 3, h_dim, w_dim)
    else:
        shape_spec = dummy_input.shape

    inputs = [ct.TensorType(name="input", shape=shape_spec, dtype=np.float32)]

    mlmodel = ct.convert(
        traced_model,
        inputs=inputs,
        compute_units=selected_units,
        minimum_deployment_target=ct.target.macOS13,
    )

    # Set model metadata
    mlmodel.author = author
    mlmodel.license = license_str
    mlmodel.short_description = (
        description or f"Pixel art super-resolution neural network (Scale: {scale}x)"
    )
    mlmodel.user_defined_metadata["scale_factor"] = str(scale)
    mlmodel.user_defined_metadata["color_range"] = "[0, 1] RGB"

    mlmodel.save(str(save_path))
    return save_path


class CoreMLUpscaler(BaseUpscaler):
    """BaseUpscaler wrapper around a native Apple CoreML .mlpackage model."""

    def __init__(
        self,
        model_path: str | Path,
        scale_factor: int = 4,
        compute_units: str = "ALL",
    ):
        """Initialize CoreML model session.

        Args:
            model_path: Path to the .mlpackage directory.
            scale_factor: Model upscale factor.
            compute_units: Target execution units ('ALL', 'CPU_AND_NE', 'CPU_AND_GPU', 'CPU_ONLY').
        """
        super().__init__(scale_factor=scale_factor)
        supported, reason = check_coreml_support()
        if not supported:
            raise RuntimeError(f"Cannot initialize CoreMLUpscaler: {reason}")

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"CoreML model not found at {self.model_path}")

        unit_map = {
            "ALL": ct.ComputeUnit.ALL,
            "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
            "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
            "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        }
        selected_units = unit_map.get(compute_units.upper(), ct.ComputeUnit.ALL)

        self.mlmodel = ct.models.MLModel(str(self.model_path), compute_units=selected_units)
        # Inspect model input and output names
        spec = self.mlmodel.get_spec()
        self.input_name = spec.description.input[0].name
        self.output_name = spec.description.output[0].name

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Execute forward prediction via CoreML.

        Args:
            x: Input tensor [B, C, H, W] in [0, 1].

        Returns:
            Upscaled output tensor [B, C, H * scale, W * scale].
        """
        orig_device = x.device
        orig_dtype = x.dtype

        x_np = x.detach().cpu().to(dtype=torch.float32).numpy()
        prediction = self.mlmodel.predict({self.input_name: x_np})
        out_np = prediction[self.output_name]
        out_tensor = torch.from_numpy(out_np).to(device=orig_device, dtype=orig_dtype)
        return out_tensor


def verify_coreml_equivalence(
    pytorch_model: nn.Module,
    coreml_target: str | Path | CoreMLUpscaler,
    scale_factor: int = 4,
    input_shape: tuple[int, int, int, int] = (1, 3, 32, 32),
    atol: float = 1e-3,
    rtol: float = 1e-2,
) -> dict[str, Any]:
    """Verify numerical output equivalence between PyTorch model and CoreML model.

    Args:
        pytorch_model: PyTorch module.
        coreml_target: Path to .mlpackage or CoreMLUpscaler instance.
        scale_factor: Model scale factor.
        input_shape: Shape of test input tensor.
        atol: Absolute tolerance.
        rtol: Relative tolerance.

    Returns:
        Dictionary with numerical difference metrics.
    """
    if isinstance(coreml_target, (str, Path)):
        cml_model = CoreMLUpscaler(coreml_target, scale_factor=scale_factor)
    else:
        cml_model = coreml_target

    torch.manual_seed(42)
    x = torch.rand(*input_shape, dtype=torch.float32)

    with torch.no_grad():
        y_pt = pytorch_model.to("cpu").eval()(x)
        y_cml = cml_model(x)

    diff = (y_pt - y_cml).abs()
    max_diff = float(diff.max().item())
    mean_diff = float(diff.mean().item())
    matches = bool(
        np.allclose(
            y_pt.detach().cpu().numpy(),
            y_cml.detach().cpu().numpy(),
            rtol=rtol,
            atol=atol,
        )
    )

    return {
        "matches": matches,
        "max_abs_diff": round(max_diff, 8),
        "mean_abs_diff": round(mean_diff, 8),
        "input_shape": list(input_shape),
        "output_shape": list(y_cml.shape),
    }
