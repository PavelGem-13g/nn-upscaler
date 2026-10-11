"""Tests for Phase 5 Generative and Diffusion Depixelization modules."""

import numpy as np
import pytest
import torch
from PIL import Image

from nn_upscaler.inference.diffusion_pipeline import (
    CATEGORY_PROMPTS,
    DiffusionUpscalePipeline,
)
from nn_upscaler.models.diffusion_adapter import (
    PixelArtControlAdapter,
    PixelArtTileAdapter,
    ZeroConv2d,
    bilateral_dither_filter_image,
    bilateral_dither_filter_tensor,
)
from nn_upscaler.models.latent_diffusion import (
    DDIMScheduler,
    LatentDecoder,
    LatentDenoiser,
    LatentDiffusionUpscaler,
    LatentEncoder,
)
from nn_upscaler.models.registry import get_model


def test_bilateral_dither_filter_tensor():
    """Verify tensor bilateral filtering suppresses high-frequency dithering while preserving shape."""
    x = torch.rand(2, 3, 16, 16)
    filtered = bilateral_dither_filter_tensor(x, ksize=5, sigma_color=0.15, sigma_space=3.0)

    assert filtered.shape == x.shape
    assert filtered.min() >= 0.0
    assert filtered.max() <= 1.0

    # Test checkerboard dither smoothing
    # Create alternating checkerboard pattern
    checkerboard = torch.zeros(1, 3, 16, 16)
    checkerboard[:, :, ::2, ::2] = 1.0
    checkerboard[:, :, 1::2, 1::2] = 1.0
    smooth_cb = bilateral_dither_filter_tensor(
        checkerboard, ksize=5, sigma_color=0.5, sigma_space=2.0
    )
    # Variance of the flat patch should decrease after filtering
    assert smooth_cb.std() < checkerboard.std()


def test_bilateral_dither_filter_image():
    """Verify PIL bilateral filtering executes and returns proper image."""
    img = Image.new("RGB", (32, 32), color=(100, 150, 200))
    # Add high frequency noise
    arr = np.array(img)
    arr[::2, ::2] = [200, 220, 240]
    noisy_img = Image.fromarray(arr)

    filtered = bilateral_dither_filter_image(noisy_img, d=5, sigma_color=50, sigma_space=50)
    assert isinstance(filtered, Image.Image)
    assert filtered.size == (32, 32)


def test_zero_conv2d():
    """Verify ZeroConv2d outputs zeros at initialization and computes gradients."""
    conv = ZeroConv2d(16, 32)
    inp = torch.randn(2, 16, 8, 8)
    out = conv(inp)

    assert torch.all(out == 0.0)
    assert out.shape == (2, 32, 8, 8)

    # Test gradient flow
    loss = (out + inp.sum()).sum()
    loss.backward()
    assert conv.conv.weight.grad is not None


def test_pixel_art_tile_adapter():
    """Verify PixelArtTileAdapter forward pass and tile_weight modulation."""
    adapter = PixelArtTileAdapter(
        in_channels=3,
        cond_channels=32,
        out_channels=32,
        num_blocks=2,
        use_bilateral_filter=True,
    )
    x = torch.rand(2, 3, 16, 16)

    # tile_weight = 1.0 (default zero-conv initially outputs zero)
    out = adapter(x, tile_weight=1.0)
    assert out.shape == (2, 32, 16, 16)
    assert torch.all(out == 0.0)

    # Test extract_features for SFT
    feats = adapter.extract_features(x)
    assert feats.shape == (2, 32, 16, 16)
    assert feats.abs().sum() > 0.0

    # Test SFT modulation
    base_feat = torch.ones(2, 32, 16, 16)
    modulated = adapter.modulate(base_feat, x)
    assert modulated.shape == base_feat.shape


def test_pixel_art_control_adapter_alias():
    """Verify PixelArtControlAdapter alias behavior."""
    adapter = PixelArtControlAdapter(3, 32, 32, num_blocks=1)
    x = torch.rand(1, 3, 8, 8)
    out = adapter(x)
    assert out.shape == (1, 32, 8, 8)


def test_latent_encoder_and_decoder():
    """Verify LatentEncoder and LatentDecoder shape mappings."""
    # 4x scale factor
    enc = LatentEncoder(in_channels=3, latent_channels=4, num_features=32, scale_factor=4)
    dec = LatentDecoder(latent_channels=4, out_channels=3, num_features=32, scale_factor=4)

    hr = torch.rand(2, 3, 64, 64)
    latent = enc(hr)
    assert latent.shape == (2, 4, 16, 16)

    recon = dec(latent)
    assert recon.shape == (2, 3, 64, 64)

    # 2x scale factor
    enc2 = LatentEncoder(in_channels=3, latent_channels=4, num_features=32, scale_factor=2)
    dec2 = LatentDecoder(latent_channels=4, out_channels=3, num_features=32, scale_factor=2)
    hr2 = torch.rand(1, 3, 32, 32)
    latent2 = enc2(hr2)
    assert latent2.shape == (1, 4, 16, 16)
    recon2 = dec2(latent2)
    assert recon2.shape == (1, 3, 32, 32)


def test_latent_denoiser():
    """Verify LatentDenoiser forward pass with timesteps and conditions."""
    denoiser = LatentDenoiser(
        latent_channels=4,
        spatial_cond_channels=32,
        cond_vec_dim=32,
        num_features=32,
        num_blocks=2,
    )
    z_t = torch.randn(2, 4, 16, 16)
    t = torch.tensor([50, 200], dtype=torch.long)
    spatial_cond = torch.randn(2, 32, 16, 16)
    semantic_vec = torch.randn(2, 32)

    pred_eps = denoiser(z_t, t, spatial_cond, semantic_vec)
    assert pred_eps.shape == (2, 4, 16, 16)


def test_ddim_scheduler():
    """Verify DDIMScheduler timestep generation across step counts."""
    sched = DDIMScheduler(train_timesteps=1000)
    steps_1 = sched.get_timesteps(1)
    steps_4 = sched.get_timesteps(4)
    steps_8 = sched.get_timesteps(8)

    assert len(steps_1) == 1
    assert len(steps_4) == 4
    assert len(steps_8) == 8

    # Must be monotonically descending
    for s in [steps_1, steps_4, steps_8]:
        assert s == sorted(s, reverse=True)


def test_latent_diffusion_upscaler_registry():
    """Verify LatentDiffusionUpscaler is registered as 'latent_diffusion'."""
    model = get_model("latent_diffusion", scale_factor=4)
    assert isinstance(model, LatentDiffusionUpscaler)
    assert model.scale_factor == 4


def test_latent_diffusion_upscaler_forward_steps():
    """Verify LatentDiffusionUpscaler forward inference for different steps and scales."""
    model = LatentDiffusionUpscaler(
        scale_factor=4,
        latent_channels=4,
        num_features=32,
        cond_dim=32,
        num_steps=2,
    )
    x = torch.rand(1, 3, 16, 16)

    # 1 step sampling
    out_1 = model(x, steps=1)
    assert out_1.shape == (1, 3, 64, 64)
    assert out_1.min() >= 0.0 and out_1.max() <= 1.0

    # 2 step sampling
    out_2 = model(x, steps=2)
    assert out_2.shape == (1, 3, 64, 64)

    # Categories and prompt conditioning
    out_armor = model(x, category="armor", prompt="shining steel helmet", steps=2)
    assert out_armor.shape == (1, 3, 64, 64)


def test_latent_diffusion_upscaler_determinism():
    """Verify LatentDiffusionUpscaler produces identical outputs with same seed/generator."""
    model = LatentDiffusionUpscaler(
        scale_factor=4,
        latent_channels=4,
        num_features=32,
        cond_dim=32,
        num_steps=2,
    )
    x = torch.rand(1, 3, 16, 16)

    gen1 = torch.Generator(device="cpu").manual_seed(123)
    out1 = model(x, steps=2, generator=gen1)

    gen2 = torch.Generator(device="cpu").manual_seed(123)
    out2 = model(x, steps=2, generator=gen2)

    assert torch.allclose(out1, out2, atol=1e-5)


def test_latent_diffusion_upscaler_training_loss():
    """Verify LatentDiffusionUpscaler computes valid differentiable loss during training mode."""
    model = LatentDiffusionUpscaler(
        scale_factor=4,
        latent_channels=4,
        num_features=32,
        cond_dim=32,
    )
    x = torch.rand(2, 3, 16, 16)
    target = torch.rand(2, 3, 64, 64)

    recon, loss_dict = model(x, target=target, return_loss=True)

    assert recon.shape == (2, 3, 64, 64)
    assert "loss" in loss_dict
    assert "loss_diff" in loss_dict
    assert "loss_recon" in loss_dict
    assert loss_dict["loss"].item() > 0.0

    # Check backpropagation
    loss_dict["loss"].backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    assert grad_norm > 0.0


def test_diffusion_upscale_pipeline_native():
    """Verify DiffusionUpscalePipeline with native backend upscales PIL image."""
    pipeline = DiffusionUpscalePipeline(backend="native", scale_factor=4)
    img = Image.new("RGB", (16, 16), color=(120, 140, 160))

    result = pipeline.run(
        image=img,
        category="armor",
        prompt="polished steel cuirass",
        steps=2,
        tile_weight=0.75,
        seed=42,
    )

    assert isinstance(result, Image.Image)
    assert result.size == (64, 64)


def test_diffusion_pipeline_prompt_builder():
    """Verify category prompt generation in DiffusionUpscalePipeline."""
    prompt_face = DiffusionUpscalePipeline.get_category_prompt("face", "warrior girl")
    assert "warrior girl" in prompt_face
    assert "skin" in prompt_face

    prompt_armor = DiffusionUpscalePipeline.get_category_prompt("armor")
    assert "armor" in prompt_armor

    prompt_general = DiffusionUpscalePipeline.get_category_prompt("unknown_category")
    assert prompt_general == CATEGORY_PROMPTS["general"]


def test_diffusion_pipeline_invalid_backend():
    """Verify DiffusionUpscalePipeline rejects unknown backends."""
    with pytest.raises(ValueError, match="Unknown backend"):
        DiffusionUpscalePipeline(backend="unsupported_backend")
