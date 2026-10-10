"""Quality assessment metrics: PSNR, SSIM, and error maps."""

import math

import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim_fn


def calculate_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """Calculate Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).

    Args:
        pred: Predicted image tensor [C, H, W] or [B, C, H, W] in [0, max_val].
        target: Ground truth tensor in [0, max_val].
        max_val: Maximum possible pixel value (default: 1.0).

    Returns:
        PSNR value in dB (float).
    """
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(max_val) - 10.0 * math.log10(mse)


def calculate_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Calculate Structural Similarity Index (SSIM).

    Args:
        pred: Predicted tensor [C, H, W] or [B, C, H, W] in range [0, 1].
        target: Ground truth tensor in range [0, 1].

    Returns:
        Mean SSIM score between -1.0 and 1.0.
    """
    if pred.dim() == 4:
        pred = pred.squeeze(0)
    if target.dim() == 4:
        target = target.squeeze(0)

    # Convert to [H, W, C] numpy arrays in [0, 1]
    pred_np = pred.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    target_np = target.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()

    # skimage ssim over multichannel RGB
    val = ssim_fn(
        pred_np,
        target_np,
        data_range=1.0,
        channel_axis=2,
        win_size=7,
    )
    return float(val)


def evaluate_pair(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Calculate all standard metrics for an output and reference pair."""
    return {
        "psnr": calculate_psnr(pred, target),
        "ssim": calculate_ssim(pred, target),
    }
