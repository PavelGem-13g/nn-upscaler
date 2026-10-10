"""Unit tests for image quality assessment metrics."""

import torch

from nn_upscaler.evaluation.metrics import calculate_psnr, calculate_ssim, evaluate_pair


def test_perfect_reconstruction_psnr():
    img = torch.rand(3, 32, 32)
    psnr = calculate_psnr(img, img)
    assert psnr == float("inf")


def test_ssim_range_and_identity():
    img = torch.rand(3, 32, 32)
    ssim = calculate_ssim(img, img)
    assert abs(ssim - 1.0) < 1e-4


def test_evaluate_pair():
    img1 = torch.rand(3, 32, 32)
    img2 = (img1 + 0.1).clamp(0, 1)
    metrics = evaluate_pair(img2, img1)
    assert "psnr" in metrics
    assert "ssim" in metrics
    assert metrics["psnr"] > 0
    assert 0.0 <= metrics["ssim"] <= 1.0
