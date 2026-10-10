"""Model architectures and model registry."""

from nn_upscaler.models.base import BaseUpscaler
from nn_upscaler.models.classical import (
    BicubicUpscaler,
    BilinearUpscaler,
    NearestNeighborUpscaler,
)
from nn_upscaler.models.edsr_light import EDSRLightUpscaler
from nn_upscaler.models.registry import get_model, list_models, register_model
from nn_upscaler.models.srcnn import SRCNNUpscaler

__all__ = [
    "BaseUpscaler",
    "BicubicUpscaler",
    "BilinearUpscaler",
    "EDSRLightUpscaler",
    "NearestNeighborUpscaler",
    "SRCNNUpscaler",
    "get_model",
    "list_models",
    "register_model",
]
