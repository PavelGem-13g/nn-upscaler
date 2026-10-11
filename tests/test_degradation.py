"""Comprehensive unit tests for GPU/tensor-accelerated pixel-art degradation pipeline."""

import pytest
import torch
from PIL import Image

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
    tensor_downscale,
)

# Detect MPS availability
DEVICES = ["cpu"]
if torch.backends.mps.is_available():
    DEVICES.append("mps")


def test_quantize_palette_pillow():
    img = Image.new("RGB", (32, 32), color=(100, 150, 200))
    res = quantize_palette_pillow(img, num_colors=16)
    assert res.size == (32, 32)
    assert res.mode == "RGB"


def test_quantize_to_preset_palette_numpy():
    img = Image.new("RGB", (16, 16), color=(255, 255, 255))
    palette = PRESET_PALETTES["gameboy"]
    res = quantize_to_palette(img, palette)
    assert res.size == (16, 16)
    import numpy as np

    arr = np.array(res).reshape(-1, 3)
    for p in arr[:5]:
        assert any(np.array_equal(p, c) for c in palette)


@pytest.mark.parametrize("size", [2, 4, 8])
def test_bayer_matrix_generator(size: int):
    m = get_bayer_matrix(size)
    assert m.shape == (size, size)
    assert m.min().item() == 0.0
    assert m.max().item() == float(size * size - 1)
    # Check uniqueness of entries
    vals = set(m.flatten().tolist())
    assert len(vals) == size * size


def test_bayer_matrix_invalid():
    with pytest.raises(ValueError, match="Unsupported Bayer size"):
        get_bayer_matrix(3)


@pytest.mark.parametrize("device", DEVICES)
def test_apply_bayer_dithering(device: str):
    dev = torch.device(device)
    # 3D tensor [C, H, W]
    x3 = torch.rand(3, 32, 32, device=dev)
    d3 = apply_bayer_dithering(x3, size=4, spread=0.05)
    assert d3.shape == x3.shape
    assert d3.device.type == dev.type
    assert d3.min() >= 0.0 and d3.max() <= 1.0

    # 4D tensor [B, C, H, W]
    x4 = torch.rand(2, 3, 24, 24, device=dev)
    d4 = apply_bayer_dithering(x4, size=8, spread=0.08)
    assert d4.shape == x4.shape
    assert d4.device.type == dev.type
    assert d4.min() >= 0.0 and d4.max() <= 1.0


def test_retro_palettes_specifications():
    expected_counts = {
        "nes": 54,
        "c64": 16,
        "gameboy": 4,
        "pico8": 16,
        "cga": 8,
        "zx_spectrum": 15,
        "ega": 16,
    }
    for name, expected_len in expected_counts.items():
        assert name in PALETTE_TENSORS, f"Missing palette {name}"
        t = PALETTE_TENSORS[name]
        assert t.shape == (expected_len, 3), f"Wrong shape for {name}: {t.shape}"
        assert t.dtype == torch.float32
        assert t.min() >= 0.0 and t.max() <= 1.0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("preset", ["nes", "c64", "gameboy", "pico8", "cga", "zx_spectrum", "ega"])
def test_tensor_palette_quantizer(device: str, preset: str):
    dev = torch.device(device)
    quantizer = TensorPaletteQuantizer(device=dev)
    pal_tensor = get_palette_tensor(preset, device=dev)

    # Test 3D tensor
    x3 = torch.rand(3, 16, 16, device=dev)
    q3 = quantizer.quantize(x3, palette=preset)
    assert q3.shape == x3.shape
    assert q3.device.type == dev.type

    # Verify that each pixel is an exact match to a palette entry
    flat_q3 = q3.permute(1, 2, 0).reshape(-1, 3)
    dists = torch.cdist(flat_q3, pal_tensor)
    min_dists = torch.min(dists, dim=1).values
    assert torch.all(min_dists < 1e-3)

    # Test 4D tensor batch
    x4 = torch.rand(2, 3, 16, 16, device=dev)
    q4 = quantizer.quantize(x4, palette=pal_tensor)
    assert q4.shape == x4.shape
    assert q4.device.type == dev.type


@pytest.mark.parametrize("device", DEVICES)
def test_tensor_quantizer_rgba(device: str):
    dev = torch.device(device)
    quantizer = TensorPaletteQuantizer(device=dev)

    # 4 channels: RGB + Alpha
    x_rgba = torch.rand(2, 4, 16, 16, device=dev)
    orig_alpha = x_rgba[:, 3:4, :, :].clone()

    q_rgba = quantizer.quantize(x_rgba, palette="pico8")
    assert q_rgba.shape == (2, 4, 16, 16)
    # Alpha channel must be preserved exactly
    assert torch.allclose(q_rgba[:, 3:4, :, :], orig_alpha)


def test_tensor_quantizer_adaptive():
    quantizer = TensorPaletteQuantizer()
    x = torch.linspace(0.0, 1.0, 100).unsqueeze(0).repeat(3, 1).unsqueeze(0)  # [1, 3, 100]
    q = quantizer.quantize_adaptive(x, levels=3)
    vals = set(torch.round(q * 100).unique().tolist())
    assert vals == {0.0, 50.0, 100.0}


@pytest.mark.parametrize("device", DEVICES)
def test_random_pixel_art_degradation(device: str):
    dev = torch.device(device)
    degrader = RandomPixelArtDegradation(scale_factors=[2, 4, 8]).to(device=dev)

    # 1. Tensor 4D input
    x = torch.rand(2, 3, 64, 64, device=dev)
    lr = degrader(x, forced_scale=4)
    assert lr.shape == (2, 3, 16, 16)
    assert lr.device.type == dev.type
    assert lr.min() >= 0.0 and lr.max() <= 1.0

    # 2. Tensor 3D input
    x3 = torch.rand(3, 32, 32, device=dev)
    lr3 = degrader(x3, forced_scale=2)
    assert lr3.shape == (3, 16, 16)
    assert lr3.device.type == dev.type

    # 3. PIL Image input
    pil_img = Image.new("RGB", (48, 48), color=(120, 80, 200))
    pil_lr = degrader(pil_img, forced_scale=4)
    assert isinstance(pil_lr, Image.Image)
    assert pil_lr.size == (12, 12)


def test_pixel_art_degradation_backward_compat():
    img = Image.new("RGB", (64, 64), color=(50, 100, 150))
    degrader = PixelArtDegradation(scale_factor=4, num_colors=16, dither=True, dither_mode="bayer")
    lr = degrader(img)
    assert isinstance(lr, Image.Image)
    assert lr.size == (16, 16)

    # Tensor input
    t = torch.rand(3, 64, 64)
    lr_t = degrader(t)
    assert isinstance(lr_t, torch.Tensor)
    assert lr_t.shape == (3, 16, 16)


def test_tensor_downscale_mps_compatibility():
    # Non-divisible dimensions (e.g. 857x1280 at 4x downscale)
    x = torch.rand(1, 3, 857, 1280)
    if torch.backends.mps.is_available():
        x = x.to("mps")
    lr = tensor_downscale(x, scale_factor=4)
    assert lr.shape[-2] == 856 // 4
    assert lr.shape[-1] == 1280 // 4
