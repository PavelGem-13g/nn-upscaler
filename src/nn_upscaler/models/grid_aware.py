"""Grid-Aware Upscaler: Edge-guided pixel-art depixelization with spatial feature modulation."""

import math

import torch
import torch.nn.functional as F
from torch import nn

from nn_upscaler.models.base import BaseUpscaler


class SpatialFeatureModulation(nn.Module):
    """Spatial Feature Modulation (SFT) conditioned on pixel-grid boundary features."""

    def __init__(self, in_features: int, cond_features: int):
        super().__init__()
        self.conv_gamma = nn.Conv2d(cond_features, in_features, kernel_size=3, padding=1)
        self.conv_beta = nn.Conv2d(cond_features, in_features, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma = self.conv_gamma(cond)
        beta = self.conv_beta(cond)
        return x * (1.0 + torch.tanh(gamma)) + beta


class GridModulatedResidualBlock(nn.Module):
    """Residual block dynamically modulated by grid boundary predictions."""

    def __init__(self, num_features: int, cond_features: int, res_scale: float = 0.2):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.sft = SpatialFeatureModulation(num_features, cond_features)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        res = self.conv1(x)
        res = self.sft(res, cond)
        res = self.conv2(self.lrelu(res))
        return x + res * self.res_scale


class EdgeGridGuidanceBranch(nn.Module):
    """Extracts step-grid boundaries and structural contour guidance."""

    def __init__(self, in_channels: int = 3, cond_features: int = 32):
        super().__init__()
        # Laplacian / differential kernel for high-frequency step edge detection
        laplacian = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32
        )
        self.register_buffer("laplacian", laplacian.view(1, 1, 3, 3))

        self.conv_in = nn.Conv2d(in_channels + 1, cond_features, kernel_size=3, padding=1)
        self.conv_mid = nn.Sequential(
            nn.Conv2d(cond_features, cond_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(cond_features, cond_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Grid transition mask predictor [0, 1]
        self.mask_out = nn.Sequential(
            nn.Conv2d(cond_features, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # High-frequency edge detection
        gray = x.mean(dim=1, keepdim=True)
        edges = F.conv2d(gray, self.laplacian.to(device=x.device, dtype=x.dtype), padding=1)

        feat = torch.cat([x, edges], dim=1)
        cond = self.conv_mid(self.conv_in(feat))
        mask = self.mask_out(cond)
        return cond, mask


class GridAwareUpscaler(BaseUpscaler):
    """Grid-Aware Neural Upscaler for Pixel-Art to Realistic Image Restoration.

    Features:
        - Two-stream architecture: Edge/Grid Guidance Branch + Content Restoration Branch.
        - Explicit prediction of pixel-grid boundary transition mask M_grid.
        - Spatial Feature Modulation (SFT) de-blocking flat staircased artifacts
          while preserving structural semantic edges.
        - Multi-scale PixelShuffle upsampler for 2x, 4x, 8x.
    """

    def __init__(
        self,
        scale_factor: int = 4,
        in_channels: int = 3,
        out_channels: int = 3,
        num_features: int = 64,
        cond_features: int = 32,
        num_blocks: int = 6,
        res_scale: float = 0.2,
    ):
        super().__init__(scale_factor=scale_factor)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_features = num_features

        # 1. Edge/Grid Guidance Stream
        self.guidance_branch = EdgeGridGuidanceBranch(
            in_channels=in_channels, cond_features=cond_features
        )

        # 2. Content Restoration Stream
        self.conv_first = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            [
                GridModulatedResidualBlock(
                    num_features=num_features,
                    cond_features=cond_features,
                    res_scale=res_scale,
                )
                for _ in range(num_blocks)
            ]
        )
        self.conv_after_body = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

        # 3. Upsampling
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
                f"Unsupported scale_factor for GridAwareUpscaler: {scale_factor}. Use 2, 4, or 8."
            )
        self.upsample = nn.Sequential(*upsample_layers)

        # 4. Tail reconstruction
        self.conv_last = nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1)

    def forward(
        self, x: torch.Tensor, return_grid_mask: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor [B, C, H, W] in [0, 1].
            return_grid_mask: If True, also return predicted grid mask [B, 1, H, W].

        Returns:
            Reconstructed tensor [B, C, H * scale, W * scale] in [0, 1].
        """
        cond, grid_mask = self.guidance_branch(x)

        feat_first = self.conv_first(x)
        feat = feat_first
        for block in self.blocks:
            feat = block(feat, cond)
        feat = feat_first + self.conv_after_body(feat)

        out = self.upsample(feat)
        out = self.conv_last(out).clamp(0.0, 1.0)

        if return_grid_mask:
            return out, grid_mask
        return out
