"""Data loading, dataset curation, and synthetic degradation pipelines."""

from nn_upscaler.data.dataset import (
    OnTheFlyPixelArtDataset,
    SyntheticPixelArtDataset,
)
from nn_upscaler.data.degradation import (
    PALETTE_TENSORS,
    PRESET_PALETTES,
    PixelArtDegradation,
    RandomPixelArtDegradation,
    TensorPaletteQuantizer,
    apply_bayer_dithering,
    get_bayer_matrix,
    get_palette_tensor,
    quantize_palette_pillow,
    quantize_to_palette,
)
from nn_upscaler.data.downloader import (
    DatasetCurator,
    download_authentic_sprites,
    download_div2k_samples,
    validate_image_file,
)
from nn_upscaler.data.unpaired_dataset import UnpairedPixelArtDataset

__all__ = [
    "PALETTE_TENSORS",
    "PRESET_PALETTES",
    "DatasetCurator",
    "OnTheFlyPixelArtDataset",
    "PixelArtDegradation",
    "RandomPixelArtDegradation",
    "SyntheticPixelArtDataset",
    "TensorPaletteQuantizer",
    "UnpairedPixelArtDataset",
    "apply_bayer_dithering",
    "download_authentic_sprites",
    "download_div2k_samples",
    "get_bayer_matrix",
    "get_palette_tensor",
    "quantize_palette_pillow",
    "quantize_to_palette",
    "validate_image_file",
]
