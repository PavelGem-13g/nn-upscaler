"""Model architectures and model registry."""

from nn_upscaler.models.base import BaseUpscaler
from nn_upscaler.models.classical import (
    BicubicUpscaler,
    BilinearUpscaler,
    NearestNeighborUpscaler,
)
from nn_upscaler.models.codebook_prior import CodebookUpscaler, VectorQuantizer
from nn_upscaler.models.cut import CUTUpscaler
from nn_upscaler.models.diffusion_adapter import PixelArtControlAdapter, PixelArtTileAdapter
from nn_upscaler.models.discriminator import UNetDiscriminatorWithSpectralNorm
from nn_upscaler.models.edsr_light import EDSRLightUpscaler
from nn_upscaler.models.grid_aware import GridAwareUpscaler
from nn_upscaler.models.latent_diffusion import LatentDiffusionUpscaler
from nn_upscaler.models.registry import get_model, list_models, register_model
from nn_upscaler.models.rrdb import RRDBNet
from nn_upscaler.models.semantic_conditioned import SemanticConditionedUpscaler
from nn_upscaler.models.srcnn import SRCNNUpscaler
from nn_upscaler.models.swinir import SwinIRLightUpscaler

__all__ = [
    "BaseUpscaler",
    "BicubicUpscaler",
    "BilinearUpscaler",
    "CUTUpscaler",
    "CodebookUpscaler",
    "EDSRLightUpscaler",
    "GridAwareUpscaler",
    "LatentDiffusionUpscaler",
    "NearestNeighborUpscaler",
    "PixelArtControlAdapter",
    "PixelArtTileAdapter",
    "RRDBNet",
    "SRCNNUpscaler",
    "SemanticConditionedUpscaler",
    "SwinIRLightUpscaler",
    "UNetDiscriminatorWithSpectralNorm",
    "VectorQuantizer",
    "get_model",
    "list_models",
    "register_model",
]
