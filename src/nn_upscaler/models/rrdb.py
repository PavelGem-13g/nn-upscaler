"""Residual-in-Residual Dense Block Network (RRDBNet / Real-ESRGAN backbone)."""

import math

import torch
from torch import nn

from nn_upscaler.models.base import BaseUpscaler


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block (RDB) with 5 densely connected convolutional layers."""

    def __init__(
        self,
        num_features: int = 64,
        growth_channels: int = 32,
        res_scale: float = 0.2,
    ):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_features, growth_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            num_features + growth_channels, growth_channels, kernel_size=3, padding=1
        )
        self.conv3 = nn.Conv2d(
            num_features + 2 * growth_channels, growth_channels, kernel_size=3, padding=1
        )
        self.conv4 = nn.Conv2d(
            num_features + 3 * growth_channels, growth_channels, kernel_size=3, padding=1
        )
        self.conv5 = nn.Conv2d(
            num_features + 4 * growth_channels, num_features, kernel_size=3, padding=1
        )
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], dim=1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], dim=1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], dim=1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], dim=1))
        return x + x5 * self.res_scale


class RRDB(nn.Module):
    """Residual in Residual Dense Block combining three RDB blocks."""

    def __init__(
        self,
        num_features: int = 64,
        growth_channels: int = 32,
        res_scale: float = 0.2,
    ):
        super().__init__()
        self.res_scale = res_scale
        self.rdb1 = ResidualDenseBlock(num_features, growth_channels, res_scale)
        self.rdb2 = ResidualDenseBlock(num_features, growth_channels, res_scale)
        self.rdb3 = ResidualDenseBlock(num_features, growth_channels, res_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return x + out * self.res_scale


class RRDBNet(BaseUpscaler):
    """Residual-in-Residual Dense Network (ESRGAN / Real-ESRGAN generator backbone).

    Reference:
        Wang et al., 'ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks',
        ECCVW 2018.

    Features:
        - Deep feature extraction with multiple RRDB units.
        - Residual scaling (beta = 0.2) for stable high-capacity training.
        - PixelShuffle upsampling supporting scales 2x, 4x, 8x.
        - Native float32 for Apple Silicon MPS and CUDA.
    """

    def __init__(
        self,
        scale_factor: int = 4,
        in_channels: int = 3,
        out_channels: int = 3,
        num_features: int = 64,
        num_blocks: int = 6,
        growth_channels: int = 32,
        res_scale: float = 0.2,
    ):
        super().__init__(scale_factor=scale_factor)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_features = num_features

        # Shallow feature extraction
        self.conv_first = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)

        # Deep feature extraction
        body_blocks = [
            RRDB(
                num_features=num_features,
                growth_channels=growth_channels,
                res_scale=res_scale,
            )
            for _ in range(num_blocks)
        ]
        self.body = nn.Sequential(*body_blocks)
        self.conv_body = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

        # Upsampling module
        upsample_layers = []
        if (scale_factor & (scale_factor - 1)) == 0 and scale_factor > 1:
            num_steps = int(math.log2(scale_factor))
            for _ in range(num_steps):
                upsample_layers.extend(
                    [
                        nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1),
                        nn.PixelShuffle(2),
                        nn.LeakyReLU(0.2, inplace=True),
                    ]
                )
        else:
            raise ValueError(
                f"Unsupported scale_factor for RRDBNet: {scale_factor}. Use 2, 4, or 8."
            )
        self.upsample = nn.Sequential(*upsample_layers)

        # High-resolution reconstruction
        self.conv_hr = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.conv_last = nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, C, H, W] in range [0, 1].

        Returns:
            Reconstructed tensor [B, C, H * scale, W * scale] in range [0, 1].
        """
        feat_first = self.conv_first(x)
        feat_body = self.conv_body(self.body(feat_first))
        feat = feat_first + feat_body

        out = self.upsample(feat)
        out = self.conv_last(self.lrelu(self.conv_hr(out)))
        return out.clamp(0.0, 1.0)
