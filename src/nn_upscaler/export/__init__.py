"""Export and optimization module for ONNX, CoreML, and Quantization."""

from nn_upscaler.export.coreml_exporter import (
    COREML_AVAILABLE,
    CoreMLUpscaler,
    check_coreml_support,
    export_to_coreml,
    verify_coreml_equivalence,
)
from nn_upscaler.export.onnx_exporter import (
    ONNXUpscaler,
    export_to_onnx,
    verify_onnx_equivalence,
)
from nn_upscaler.export.quantization import (
    DynamicINT8Upscaler,
    FP16Upscaler,
    compare_quantization_size,
    measure_model_size_mb,
    quantize_dynamic_int8,
    quantize_fp16,
    setup_quantization_engine,
)

__all__ = [
    "COREML_AVAILABLE",
    "CoreMLUpscaler",
    "DynamicINT8Upscaler",
    "FP16Upscaler",
    "ONNXUpscaler",
    "check_coreml_support",
    "compare_quantization_size",
    "export_to_coreml",
    "export_to_onnx",
    "measure_model_size_mb",
    "quantize_dynamic_int8",
    "quantize_fp16",
    "setup_quantization_engine",
    "verify_coreml_equivalence",
    "verify_onnx_equivalence",
]
