"""Registry and factory for all upscaler models."""

from collections.abc import Callable

from nn_upscaler.models.base import BaseUpscaler
from nn_upscaler.models.classical import (
    BicubicUpscaler,
    BilinearUpscaler,
    NearestNeighborUpscaler,
)
from nn_upscaler.models.codebook_prior import CodebookUpscaler
from nn_upscaler.models.cut import CUTUpscaler
from nn_upscaler.models.edsr_light import EDSRLightUpscaler
from nn_upscaler.models.grid_aware import GridAwareUpscaler
from nn_upscaler.models.latent_diffusion import LatentDiffusionUpscaler
from nn_upscaler.models.rrdb import RRDBNet
from nn_upscaler.models.semantic_conditioned import SemanticConditionedUpscaler
from nn_upscaler.models.srcnn import SRCNNUpscaler
from nn_upscaler.models.swinir import SwinIRLightUpscaler

MODEL_REGISTRY: dict[str, Callable[..., BaseUpscaler]] = {}


def register_model(name: str):
    """Decorator to register an upscaler class."""

    def decorator(cls: Callable[..., BaseUpscaler]):
        MODEL_REGISTRY[name.lower()] = cls
        return cls

    return decorator


def register_builtin_models():
    """Register all standard built-in models."""
    register_model("nearest")(NearestNeighborUpscaler)
    register_model("bicubic")(BicubicUpscaler)
    register_model("bilinear")(BilinearUpscaler)
    register_model("srcnn")(SRCNNUpscaler)
    register_model("edsr_light")(EDSRLightUpscaler)
    register_model("rrdb")(RRDBNet)
    register_model("esrgan")(RRDBNet)
    register_model("swinir")(SwinIRLightUpscaler)
    register_model("grid_aware")(GridAwareUpscaler)
    register_model("codebook_prior")(CodebookUpscaler)
    register_model("semantic_conditioned")(SemanticConditionedUpscaler)
    register_model("cut_unpaired")(CUTUpscaler)
    register_model("latent_diffusion")(LatentDiffusionUpscaler)


# Auto-register on import
register_builtin_models()


def get_model(name: str, scale_factor: int = 4, **kwargs) -> BaseUpscaler:
    """Instantiate a registered model by name.

    Args:
        name: Name of the model (case-insensitive).
        scale_factor: Upscaling scale (default: 4).
        **kwargs: Additional parameters passed to the model constructor.

    Returns:
        BaseUpscaler instance.
    """
    key = name.lower()
    if key not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise KeyError(f"Model '{name}' not found in registry. Available models: {available}")
    return MODEL_REGISTRY[key](scale_factor=scale_factor, **kwargs)


def list_models() -> list[str]:
    """Return a list of all registered model names."""
    return sorted(MODEL_REGISTRY.keys())
