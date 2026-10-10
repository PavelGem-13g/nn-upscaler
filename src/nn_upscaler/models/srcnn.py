"""Super-Resolution Convolutional Neural Network (SRCNN) baseline."""

import torch
import torch.nn.functional as F
from torch import nn

from nn_upscaler.models.base import BaseUpscaler


class SRCNNUpscaler(BaseUpscaler):
    """Classic 3-layer SRCNN architecture (Dong et al., 2014).

    Workflow:
        1. Bicubic pre-upsampling to target resolution.
        2. Feature extraction (9x9 conv).
        3. Non-linear mapping (5x5 conv).
        4. Reconstruction (5x5 conv) + residual connection.
    """

    def __init__(self, scale_factor: int = 4, in_channels: int = 3, num_channels: int = 64):
        super().__init__(scale_factor=scale_factor)
        self.conv1 = nn.Conv2d(in_channels, num_channels, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_channels, num_channels // 2, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(num_channels // 2, in_channels, kernel_size=5, padding=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-upsample via bicubic to target resolution
        x_up = F.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        out = self.relu1(self.conv1(x_up))
        out = self.relu2(self.conv2(out))
        out = self.conv3(out)
        # Residual addition with input pre-upsampled tensor
        return (x_up + out).clamp(0.0, 1.0)
