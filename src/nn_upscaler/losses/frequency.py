"""Frequency domain loss functions for penalizing pixel grid harmonics."""

import torch
import torch.nn.functional as F
from torch import nn


class FFTLoss(nn.Module):
    """Frequency Domain Loss based on 2D Real Fast Fourier Transform.

    Formula:
        L_fft = || |F(I_pred)| - |F(I_target)| ||_1

    Pixel art exhibits periodic spectral harmonics due to its discrete block grid.
    This loss suppresses unnatural high-frequency grid frequencies and encourages
    natural spectral decay in reconstructed imagery.
    """

    def __init__(self, norm: str = "ortho", loss_type: str = "l1"):
        super().__init__()
        self.norm = norm
        self.loss_type = loss_type

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute frequency amplitude difference.

        Args:
            pred: Predicted tensor [B, C, H, W] in [0, 1].
            target: Ground truth tensor [B, C, H, W] in [0, 1].

        Returns:
            Scalar loss tensor.
        """
        # Ensure float32 for stable MPS FFT operations
        pred_f32 = pred.to(dtype=torch.float32)
        target_f32 = target.to(dtype=torch.float32)

        # 2D real FFT across spatial dimensions [H, W]
        pred_fft = torch.fft.rfft2(pred_f32, norm=self.norm)
        target_fft = torch.fft.rfft2(target_f32, norm=self.norm)

        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)

        if self.loss_type == "l1":
            return F.l1_loss(pred_mag, target_mag)
        elif self.loss_type == "l2":
            return F.mse_loss(pred_mag, target_mag)
        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")
