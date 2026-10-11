"""Unit tests for models and model registry."""

import pytest
import torch

from nn_upscaler.models.discriminator import UNetDiscriminatorWithSpectralNorm
from nn_upscaler.models.registry import get_model, list_models


@pytest.mark.parametrize(
    "model_name",
    [
        "nearest",
        "bicubic",
        "bilinear",
        "srcnn",
        "edsr_light",
        "rrdb",
        "esrgan",
        "swinir",
        "grid_aware",
        "codebook_prior",
        "semantic_conditioned",
        "cut_unpaired",
    ],
)
def test_model_forward_pass_4x(model_name: str):
    model = get_model(model_name, scale_factor=4)
    model.eval()
    x = torch.rand(2, 3, 16, 16)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 3, 64, 64)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


@pytest.mark.parametrize("model_name", ["rrdb", "swinir", "grid_aware", "edsr_light"])
@pytest.mark.parametrize("scale", [2, 4, 8])
def test_models_different_scales(model_name: str, scale: int):
    # Use compact configurations for fast multi-scale unit testing
    kwargs = {}
    if model_name in ("rrdb", "esrgan"):
        kwargs = {"num_blocks": 2, "num_features": 32, "growth_channels": 16}
    elif model_name == "swinir":
        kwargs = {"embed_dim": 24, "depths": (1, 1), "num_heads": (4, 4)}
    elif model_name == "grid_aware":
        kwargs = {"num_blocks": 2, "num_features": 32, "cond_features": 16}

    model = get_model(model_name, scale_factor=scale, **kwargs)
    model.eval()
    x = torch.rand(1, 3, 16, 16)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 3, 16 * scale, 16 * scale)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_discriminator_forward():
    disc = UNetDiscriminatorWithSpectralNorm(in_channels=3, num_features=16)
    disc.eval()
    x = torch.rand(2, 3, 32, 32)
    with torch.no_grad():
        out = disc(x)
    assert out.shape == (2, 1, 32, 32)


def test_grid_aware_with_mask():
    model = get_model("grid_aware", scale_factor=4, num_blocks=2, num_features=32)
    model.eval()
    x = torch.rand(1, 3, 16, 16)
    with torch.no_grad():
        out, mask = model(x, return_grid_mask=True)
    assert out.shape == (1, 3, 64, 64)
    assert mask.shape == (1, 1, 16, 16)
    assert 0.0 <= mask.min() <= mask.max() <= 1.0


def test_invalid_model_name():
    with pytest.raises(KeyError):
        get_model("non_existent_model")


def test_list_models_contains_all():
    models = list_models()
    expected = [
        "bicubic",
        "bilinear",
        "codebook_prior",
        "cut_unpaired",
        "edsr_light",
        "esrgan",
        "grid_aware",
        "nearest",
        "rrdb",
        "semantic_conditioned",
        "srcnn",
        "swinir",
    ]
    assert all(m in models for m in expected)
