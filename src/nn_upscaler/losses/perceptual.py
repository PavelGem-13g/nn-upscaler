"""Loss functions for super-resolution and depixelization."""

import torch
import torch.nn.functional as F
from torch import nn


class GradientLoss(nn.Module):
    """Computes image gradient difference using Sobel operators to enforce edge consistency."""

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32
        )
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32
        )
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3).repeat(3, 1, 1, 1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        sx = self.sobel_x.to(device=pred.device, dtype=pred.dtype)
        sy = self.sobel_y.to(device=pred.device, dtype=pred.dtype)
        # Grouped conv to compute per-channel gradients
        grad_x_pred = F.conv2d(pred, sx, padding=1, groups=3)
        grad_y_pred = F.conv2d(pred, sy, padding=1, groups=3)
        grad_x_tgt = F.conv2d(target, sx, padding=1, groups=3)
        grad_y_tgt = F.conv2d(target, sy, padding=1, groups=3)

        loss_x = F.l1_loss(grad_x_pred, grad_x_tgt)
        loss_y = F.l1_loss(grad_y_pred, grad_y_tgt)
        return loss_x + loss_y


class CompositeLoss(nn.Module):
    """Combines Pixel Loss (L1/L2), Edge Gradient Loss, and optional Perceptual Loss (LPIPS)."""

    def __init__(
        self,
        pixel_weight: float = 1.0,
        gradient_weight: float = 0.1,
        lpips_weight: float = 0.0,
        net_type: str = "alex",
        device: torch.device | None = None,
    ):
        super().__init__()
        self.pixel_weight = pixel_weight
        self.gradient_weight = gradient_weight
        self.lpips_weight = lpips_weight
        self.l1 = nn.L1Loss()
        self.grad_loss = GradientLoss() if gradient_weight > 0 else None

        self.lpips_fn = None
        if lpips_weight > 0:
            try:
                import lpips

                self.lpips_fn = lpips.LPIPS(net=net_type)
                if device is not None:
                    self.lpips_fn = self.lpips_fn.to(device)
            except Exception as e:
                print(f"[Warning] Could not initialize LPIPS: {e}. Skipping perceptual loss.")
                self.lpips_fn = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        losses = {}
        total = torch.tensor(0.0, device=pred.device)

        if self.pixel_weight > 0:
            l1_val = self.l1(pred, target)
            losses["pixel_l1"] = l1_val
            total = total + self.pixel_weight * l1_val

        if self.grad_loss is not None and self.gradient_weight > 0:
            g_val = self.grad_loss(pred, target)
            losses["gradient"] = g_val
            total = total + self.gradient_weight * g_val

        if self.lpips_fn is not None and self.lpips_weight > 0:
            # LPIPS expects input range [-1, 1]
            pred_norm = pred * 2.0 - 1.0
            target_norm = target * 2.0 - 1.0
            lpips_val = self.lpips_fn(pred_norm, target_norm).mean()
            losses["lpips"] = lpips_val
            total = total + self.lpips_weight * lpips_val

        losses["total"] = total
        return losses
