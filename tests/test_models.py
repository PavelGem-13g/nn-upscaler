"""Unit tests for models and model registry."""

import pytest
import torch

from nn_upscaler.models.registry import get_model


@pytest.mark.parametrize("model_name", ["nearest", "bicubic", "bilinear", "srcnn", "edsr_light"])
def test_model_forward_pass_4x(model_name):
    model = get_model(model_name, scale_factor=4)
    model.eval()
    x = torch.rand(2, 3, 16, 16)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 3, 64, 64)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


@pytest.mark.parametrize("scale", [2, 4])
def test_edsr_light_different_scales(scale):
    model = get_model("edsr_light", scale_factor=scale)
    x = torch.rand(1, 3, 16, 16)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 3, 16 * scale, 16 * scale)


def test_invalid_model_name():
    with pytest.raises(KeyError):
        get_model("non_existent_model")
