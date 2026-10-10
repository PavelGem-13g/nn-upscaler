"""Unit tests for synthetic pixel-art degradation pipeline."""

import numpy as np
from PIL import Image

from nn_upscaler.data.degradation import (
    PRESET_PALETTES,
    PixelArtDegradation,
    quantize_palette_pillow,
    quantize_to_palette,
)


def test_quantize_palette_pillow():
    img = Image.new("RGB", (32, 32), color=(100, 150, 200))
    res = quantize_palette_pillow(img, num_colors=16)
    assert res.size == (32, 32)
    assert res.mode == "RGB"


def test_quantize_to_preset_palette():
    img = Image.new("RGB", (16, 16), color=(255, 255, 255))
    palette = PRESET_PALETTES["gameboy"]
    res = quantize_to_palette(img, palette)
    assert res.size == (16, 16)
    # Check that pixels are strictly in palette
    arr = np.array(res).reshape(-1, 3)
    for p in arr[:5]:
        assert any(np.array_equal(p, c) for c in palette)


def test_pixel_art_degradation_pipeline():
    img = Image.new("RGB", (64, 64), color=(50, 100, 150))
    degrader = PixelArtDegradation(scale_factor=4, num_colors=16)
    lr = degrader(img)
    assert lr.size == (16, 16)
