"""Pixel-level reconstruction loss functions."""

import torch
from torch import nn


class CharbonnierLoss(nn.Module):
    """Charbonnier loss (smooth L1 / pseudo-Huber loss).

    Formula:
        L_char(x, y) = sqrt((x - y)^2 + eps^2)

    Provides robust gradients near zero without the over-smoothing
    associated with L2 (MSE) or discontinuity at zero of pure L1.
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        super().__init__()
        self.eps2 = float(eps) ** 2
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Charbonnier loss between prediction and target.

        Args:
            pred: Predicted tensor [B, C, H, W] or arbitrary shape.
            target: Ground truth tensor of matching shape.

        Returns:
            Scalar tensor if reduction in ('mean', 'sum'), else elementwise tensor.
        """
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps2)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"Unsupported reduction mode: {self.reduction}")
