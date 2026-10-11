"""Unit tests for advanced loss functions."""

import pytest
import torch

from nn_upscaler.losses.frequency import FFTLoss
from nn_upscaler.losses.gan import RaGANLoss
from nn_upscaler.losses.perceptual import CompositeLoss, GradientLoss, VGGPerceptualLoss
from nn_upscaler.losses.pixel import CharbonnierLoss
from nn_upscaler.models.discriminator import UNetDiscriminatorWithSpectralNorm


def test_charbonnier_loss():
    loss_fn = CharbonnierLoss(eps=1e-6)
    x = torch.zeros(2, 3, 16, 16, requires_grad=True)
    y = torch.zeros(2, 3, 16, 16)

    # Identical tensors: sqrt(0 + eps^2) = eps
    loss_val = loss_fn(x, y)
    assert abs(loss_val.item() - 1e-6) < 1e-8

    # Different tensors + gradient flow
    x2 = torch.ones(2, 3, 16, 16, requires_grad=True)
    loss_diff = loss_fn(x2, y)
    loss_diff.backward()
    assert loss_diff.item() > 0.0
    assert x2.grad is not None
    assert not torch.isnan(x2.grad).any()


def test_fft_loss():
    loss_fn = FFTLoss()
    x = torch.randn(2, 3, 32, 32, requires_grad=True)
    y = torch.randn(2, 3, 32, 32)

    loss_val = loss_fn(x, y)
    loss_val.backward()

    assert loss_val.item() >= 0.0
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()

    # Identity check
    identical_loss = loss_fn(y, y)
    assert identical_loss.item() < 1e-6


def test_gradient_loss():
    loss_fn = GradientLoss()
    x = torch.randn(2, 3, 16, 16, requires_grad=True)
    y = torch.randn(2, 3, 16, 16)

    loss_val = loss_fn(x, y)
    loss_val.backward()

    assert loss_val.item() >= 0.0
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_vgg_perceptual_loss():
    loss_fn = VGGPerceptualLoss()
    x = torch.rand(1, 3, 64, 64, requires_grad=True)
    y = torch.rand(1, 3, 64, 64)

    loss_val = loss_fn(x, y)
    loss_val.backward()

    assert loss_val.item() > 0.0
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()

    # Identical check
    identical_loss = loss_fn(y, y)
    assert identical_loss.item() < 1e-5


def test_ragan_loss():
    loss_fn = RaGANLoss()
    d_real = torch.randn(2, 1, 16, 16, requires_grad=True)
    d_fake = torch.randn(2, 1, 16, 16, requires_grad=True)

    # Discriminator loss
    loss_d = loss_fn.discriminator_loss(d_real, d_fake)
    loss_d.backward(retain_graph=True)
    assert loss_d.item() > 0.0

    # Generator loss
    loss_g = loss_fn.generator_loss(d_real, d_fake)
    loss_g.backward()
    assert loss_g.item() > 0.0


def test_composite_loss_backward():
    comp_loss = CompositeLoss(
        pixel_weight=1.0,
        gradient_weight=0.1,
        fft_weight=0.05,
        vgg_weight=0.1,
        lpips_weight=0.0,
    )
    x = torch.rand(1, 3, 32, 32, requires_grad=True)
    y = torch.rand(1, 3, 32, 32)

    losses = comp_loss(x, y)
    assert "total" in losses
    assert "pixel" in losses
    assert "gradient" in losses
    assert "fft" in losses
    assert "vgg" in losses

    losses["total"].backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_losses_on_mps():
    device = torch.device("mps")
    charb = CharbonnierLoss().to(device)
    fft_l = FFTLoss().to(device)
    vgg_l = VGGPerceptualLoss(device=device).to(device)
    ragan = RaGANLoss().to(device)

    x = torch.rand(2, 3, 32, 32, device=device, requires_grad=True)
    y = torch.rand(2, 3, 32, 32, device=device)

    loss_charb = charb(x, y)
    loss_fft = fft_l(x, y)
    loss_vgg = vgg_l(x, y)

    total = loss_charb + loss_fft + loss_vgg
    total.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()

    # Discriminator on MPS
    disc = UNetDiscriminatorWithSpectralNorm(in_channels=3, num_features=16).to(device)
    d_out_real = disc(y)
    d_out_fake = disc(x)
    loss_d = ragan.discriminator_loss(d_out_real, d_out_fake)
    loss_d.backward()
    assert loss_d.item() > 0.0
