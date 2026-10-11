"""Contrastive Unpaired Translation (CUT) Upscaler for Unpaired Depixelization.

Based on Park et al. (ECCV 2020) 'Contrastive Learning for Unpaired Image-to-Image Translation'.
Allows training on real retro game sprites and real photos without requiring paired ground truth.
Uses an encoder-bottleneck-upsampler generator with multi-layer patch projection heads.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from nn_upscaler.models.base import BaseUpscaler


class ResidualBlock(nn.Module):
    """Standard residual block with LeakyReLU activations and residual scaling."""

    def __init__(self, channels: int, res_scale: float = 0.2):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.conv2(self.lrelu(self.conv1(x)))
        return x + res * self.res_scale


class PatchProjectionHead(nn.Module):
    """Two-layer 1x1 conv MLP projection head for PatchNCE feature projection."""

    def __init__(self, in_channels: int, proj_dim: int = 128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, proj_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(proj_dim, proj_dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class CUTUpscaler(BaseUpscaler):
    """Contrastive Unpaired Translation (CUT) Neural Upscaler.

    Generator G: X (Pixel Art) -> Y (Realistic Photo).
    Maintains intermediate feature taps and projection heads to compute PatchNCE loss.
    """

    def __init__(
        self,
        scale_factor: int = 4,
        in_channels: int = 3,
        out_channels: int = 3,
        num_features: int = 64,
        num_blocks: int = 6,
        proj_dim: int = 128,
        res_scale: float = 0.2,
    ):
        super().__init__(scale_factor=scale_factor)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_features = num_features
        self.proj_dim = proj_dim

        # 1. Input shallow feature extractor
        self.conv_in = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        # 2. Residual bottleneck blocks
        self.blocks = nn.ModuleList(
            [ResidualBlock(channels=num_features, res_scale=res_scale) for _ in range(num_blocks)]
        )
        self.conv_after_body = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

        # 3. Multi-scale PixelShuffle upsampler
        upsample_layers: list[nn.Module] = []
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
            raise ValueError(f"Unsupported scale_factor: {scale_factor}. Supported: 2, 4, 8.")
        self.upsample = nn.Sequential(*upsample_layers)

        # 4. Final output layer
        self.conv_last = nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1)

        # 5. Patch projection heads for multi-layer contrastive estimation:
        # Layer 0: Input conv feature
        # Layer 1: Mid-body feature
        # Layer 2: Deep-body feature
        # Layer 3: Upsampled feature
        self.proj_head_in = PatchProjectionHead(num_features, proj_dim=proj_dim)
        self.proj_head_mid = PatchProjectionHead(num_features, proj_dim=proj_dim)
        self.proj_head_deep = PatchProjectionHead(num_features, proj_dim=proj_dim)
        self.proj_head_up = PatchProjectionHead(num_features, proj_dim=proj_dim)

    def extract_features(
        self, x: torch.Tensor, encode_only: bool = False
    ) -> tuple[torch.Tensor | None, list[torch.Tensor]]:
        """Extract multi-layer projected features for PatchNCE computation.

        Args:
            x: Input tensor [B, C, H, W]
            encode_only: If True, do not run upsampler or output conv (saves memory for keys)

        Returns:
            Tuple of (output_image_or_None, list_of_projected_features)
        """
        feats: list[torch.Tensor] = []

        # Layer 0: Initial feature
        f_in = self.lrelu(self.conv_in(x))
        feats.append(self.proj_head_in(f_in))

        # Pass through bottleneck
        f_mid = f_in
        half = len(self.blocks) // 2
        for i, block in enumerate(self.blocks):
            f_mid = block(f_mid)
            if i == half - 1:
                feats.append(self.proj_head_mid(f_mid))

        f_deep = f_in + self.conv_after_body(f_mid)
        feats.append(self.proj_head_deep(f_deep))

        if encode_only:
            return None, feats

        f_up = self.upsample(f_deep)
        feats.append(self.proj_head_up(f_up))
        out = self.conv_last(f_up).clamp(0.0, 1.0)

        return out, feats

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass.

        Args:
            x: Input tensor [B, C, H, W] in [0, 1].
            return_features: If True, return (output, list_of_projected_features).

        Returns:
            Upscaled tensor [B, C, H * scale, W * scale] in [0, 1].
        """
        if return_features:
            out, feats = self.extract_features(x, encode_only=False)
            assert out is not None
            return out, feats

        f = self.lrelu(self.conv_in(x))
        f_body = f
        for block in self.blocks:
            f_body = block(f_body)
        f = f + self.conv_after_body(f_body)
        f_up = self.upsample(f)
        return self.conv_last(f_up).clamp(0.0, 1.0)
