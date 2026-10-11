"""Quality assessment metrics: PSNR, SSIM, LPIPS, and error maps."""

import math
from typing import Any

import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim_fn

# Cached LPIPS model instances per device
_LPIPS_CACHE: dict[str, Any] = {}  # type: ignore


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


def calculate_lpips(
    pred: torch.Tensor,
    target: torch.Tensor,
    net_type: str = "alex",
    device: torch.device | None = None,
) -> float:
    """Calculate Learned Perceptual Image Patch Similarity (LPIPS).

    Args:
        pred: Predicted tensor [C, H, W] or [B, C, H, W] in range [0, 1].
        target: Ground truth tensor in range [0, 1].
        net_type: Feature extractor backbone ('alex' or 'vgg').
        device: Device to run computation on.

    Returns:
        LPIPS perceptual distance (lower is better, 0.0 is identical).
    """
    import lpips

    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
    if target.dim() == 3:
        target = target.unsqueeze(0)

    target_dev = device or pred.device
    dev_key = f"{net_type}_{target_dev}"
    if dev_key not in _LPIPS_CACHE:
        lpips_model = lpips.LPIPS(net=net_type).to(target_dev)
        lpips_model.eval()
        _LPIPS_CACHE[dev_key] = lpips_model
    else:
        lpips_model = _LPIPS_CACHE[dev_key]

    # LPIPS expects input range [-1, 1]
    p_norm = (pred.to(target_dev) * 2.0 - 1.0).clamp(-1.0, 1.0)
    t_norm = (target.to(target_dev) * 2.0 - 1.0).clamp(-1.0, 1.0)

    # AlexNet expects spatial dimension at least 64x64
    if p_norm.shape[-2] < 64 or p_norm.shape[-1] < 64:
        p_norm = F.interpolate(p_norm, size=(64, 64), mode="bicubic", align_corners=False)
        t_norm = F.interpolate(t_norm, size=(64, 64), mode="bicubic", align_corners=False)

    with torch.no_grad():
        dist = lpips_model(p_norm, t_norm).mean().item()
    return float(dist)


def evaluate_pair(
    pred: torch.Tensor,
    target: torch.Tensor,
    compute_lpips: bool = False,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Calculate standard metrics for an output and reference pair."""
    results = {
        "psnr": calculate_psnr(pred, target),
        "ssim": calculate_ssim(pred, target),
    }
    if compute_lpips:
        try:
            results["lpips"] = calculate_lpips(pred, target, device=device)
        except Exception as e:
            print(f"[Warning] Failed to compute LPIPS: {e}")
    return results
