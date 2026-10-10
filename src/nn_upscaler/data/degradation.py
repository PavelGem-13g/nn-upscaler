"""Synthetic pixel-art degradation pipeline.

Converts high-resolution realistic images into authentic-looking pixel art
to produce paired datasets for supervised training and testing.
"""

import numpy as np
from PIL import Image

# Preset retro color palettes (RGB 0-255)
PRESET_PALETTES = {
    "gameboy": np.array([
        [15, 56, 15],
        [48, 98, 48],
        [139, 172, 15],
        [155, 188, 15],
    ], dtype=np.uint8),
    "pico8": np.array([
        [0, 0, 0], [29, 43, 83], [126, 37, 83], [0, 135, 81],
        [171, 82, 54], [95, 87, 79], [194, 195, 199], [255, 241, 232],
        [255, 0, 77], [255, 163, 0], [255, 236, 39], [0, 228, 54],
        [41, 173, 255], [131, 118, 156], [255, 119, 168], [255, 204, 170],
    ], dtype=np.uint8),
    "cga": np.array([
        [0, 0, 0], [0, 170, 170], [170, 0, 170], [170, 170, 170],
        [85, 85, 85], [85, 255, 255], [255, 85, 255], [255, 255, 255],
    ], dtype=np.uint8),
}


def quantize_palette_pillow(img: Image.Image, num_colors: int = 32, dither: bool = True) -> Image.Image:
    """Quantize image colors using Pillow's adaptive octree palette.

    Args:
        img: Input PIL Image in RGB mode.
        num_colors: Target number of colors (e.g. 16, 32, 64).
        dither: Whether to apply Floyd-Steinberg dithering.

    Returns:
        Quantized RGB PIL Image.
    """
    dither_flag = Image.Dither.FLOYDSTEINBERG if dither else Image.Dither.NONE
    quantized = img.quantize(colors=num_colors, method=Image.Quantize.MEDIANCUT, dither=dither_flag)
    return quantized.convert("RGB")


def quantize_to_palette(img: Image.Image, palette: np.ndarray) -> Image.Image:
    """Map all image pixels to the nearest Euclidean color in a given palette array.

    Args:
        img: Input RGB PIL Image.
        palette: NumPy array of shape [K, 3] with uint8 RGB colors.

    Returns:
        Mapped RGB PIL Image.
    """
    arr = np.array(img, dtype=np.int32)
    h, w, _ = arr.shape
    flat_pixels = arr.reshape(-1, 3)

    # Compute Euclidean distance to each palette color [N, K]
    # (x - p)^2 = x^2 - 2*x*p + p^2
    dists = np.sum((flat_pixels[:, np.newaxis, :] - palette[np.newaxis, :, :]) ** 2, axis=2)
    nearest_idx = np.argmin(dists, axis=1)
    mapped_pixels = palette[nearest_idx].reshape(h, w, 3)

    return Image.fromarray(mapped_pixels.astype(np.uint8), mode="RGB")


class PixelArtDegradation:
    """Full degradation pipeline to transform realistic photos into pixel art."""

    def __init__(
        self,
        scale_factor: int = 4,
        num_colors: int | None = 32,
        palette_preset: str | None = None,
        dither: bool = False,
    ):
        """
        Args:
            scale_factor: Downscaling factor (e.g. 4 for 4x super-resolution training).
            num_colors: Number of colors for adaptive quantization (if palette_preset is None).
            palette_preset: One of 'gameboy', 'pico8', 'cga', or None.
            dither: Whether to apply dithering.
        """
        self.scale_factor = scale_factor
        self.num_colors = num_colors
        self.palette_preset = palette_preset
        self.dither = dither

    def __call__(self, image: Image.Image) -> Image.Image:
        """Apply degradation pipeline to a high-resolution PIL Image.

        Workflow:
            1. Downscale high-res image to low resolution using BOX/BILINEAR sampling.
            2. Quantize colors to simulate limited palette depth.
            3. Return the low-resolution pixel-art image.
        """
        w, h = image.size
        lr_w = max(1, w // self.scale_factor)
        lr_h = max(1, h // self.scale_factor)

        # 1. Downscale
        lr_img = image.resize((lr_w, lr_h), Image.Resampling.BOX)

        # 2. Palette reduction
        if self.palette_preset is not None and self.palette_preset in PRESET_PALETTES:
            lr_img = quantize_to_palette(lr_img, PRESET_PALETTES[self.palette_preset])
        elif self.num_colors is not None and self.num_colors < 256:
            lr_img = quantize_palette_pillow(lr_img, num_colors=self.num_colors, dither=self.dither)

        return lr_img
