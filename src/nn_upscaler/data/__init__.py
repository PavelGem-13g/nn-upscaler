"""Data loading and degradation pipelines."""

from nn_upscaler.data.dataset import SyntheticPixelArtDataset
from nn_upscaler.data.degradation import (
    PRESET_PALETTES,
    PixelArtDegradation,
    quantize_palette_pillow,
    quantize_to_palette,
)

__all__ = [
    "PRESET_PALETTES",
    "PixelArtDegradation",
    "SyntheticPixelArtDataset",
    "quantize_palette_pillow",
    "quantize_to_palette",
]
