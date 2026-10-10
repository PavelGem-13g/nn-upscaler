"""Enhanced Deep Residual Network (EDSR-Light) baseline."""

import math

import torch
from torch import nn

from nn_upscaler.models.base import BaseUpscaler


class ResidualBlock(nn.Module):
    """Residual block without Batch Normalization (standard for Super-Resolution)."""

    def __init__(self, num_features: int, res_scale: float = 0.1):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.conv2(self.relu(self.conv1(x)))
        return x + res * self.res_scale


class UpsampleBlock(nn.Sequential):
    """Upsampling module using PixelShuffle for scale factors 2, 4, 8."""

    def __init__(self, num_features: int, scale: int):
        layers = []
        if (scale & (scale - 1)) == 0:  # Power of 2 (2, 4, 8)
            num_steps = int(math.log2(scale))
            for _ in range(num_steps):
                layers.extend([
                    nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1),
                    nn.PixelShuffle(2),
                    nn.PReLU(),
                ])
        elif scale == 3:
            layers.extend([
                nn.Conv2d(num_features, num_features * 9, kernel_size=3, padding=1),
                nn.PixelShuffle(3),
                nn.PReLU(),
            ])
        else:
            raise ValueError(f"Unsupported scale factor: {scale}")
        super().__init__(*layers)


class EDSRLightUpscaler(BaseUpscaler):
    """Lightweight Enhanced Deep Residual Network (EDSR-Light).

    Features:
        - Operates in low-resolution space for efficiency.
        - 8 residual blocks without batch normalization.
        - Long skip connection across all residual blocks.
        - Sub-pixel convolution (PixelShuffle) upsampling.
    """

    def __init__(
        self,
        scale_factor: int = 4,
        in_channels: int = 3,
        num_features: int = 64,
        num_blocks: int = 8,
        res_scale: float = 0.1,
    ):
        super().__init__(scale_factor=scale_factor)
        self.head = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)

        blocks = [ResidualBlock(num_features, res_scale=res_scale) for _ in range(num_blocks)]
        self.body = nn.Sequential(*blocks)
        self.body_tail = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

        self.upsample = UpsampleBlock(num_features, scale_factor)
        self.tail = nn.Conv2d(num_features, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.head(x)
        res = self.body(feat)
        res = self.body_tail(res)
        feat = feat + res
        out = self.upsample(feat)
        out = self.tail(out)
        return out.clamp(0.0, 1.0)
