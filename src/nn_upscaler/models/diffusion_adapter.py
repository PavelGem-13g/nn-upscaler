"""Pixel-Art Conditioning Adapters for Diffusion Depixelization.

Provides:
- Bilateral dither filtering for suppressing Bayer/Floyd-Steinberg grid noise.
- Zero-convolutions for safe conditioning injection.
- PixelArtTileAdapter & PixelArtControlAdapter for spatial feature projection into diffusion models.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn


def bilateral_dither_filter_tensor(
    x: torch.Tensor,
    ksize: int = 5,
    sigma_color: float = 0.15,
    sigma_space: float = 3.0,
) -> torch.Tensor:
    """Vectorized bilateral filter operating directly on PyTorch tensors [B, C, H, W].

    Suppresses high-frequency checkerboard dithering (Bayer, Floyd-Steinberg)
    and color quantization noise while preserving structural silhouette edges.

    Args:
        x: Tensor of shape [B, C, H, W] in [0, 1].
        ksize: Filter kernel size (odd integer, e.g. 3, 5, 7).
        sigma_color: Gaussian standard deviation for color distance.
        sigma_space: Gaussian standard deviation for spatial distance.

    Returns:
        Filtered tensor of identical shape [B, C, H, W] in [0, 1].
    """
    b, c, h, w = x.shape
    p = ksize // 2
    pad_x = F.pad(x, (p, p, p, p), mode="reflect")
    unfold = F.unfold(pad_x, kernel_size=ksize)
    patches = unfold.view(b, c, ksize * ksize, h * w)

    # Spatial Gaussian weights [1, 1, ksize^2, 1]
    coords = torch.arange(ksize, device=x.device, dtype=x.dtype) - p
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    spatial_dist_sq = (yy**2 + xx**2).view(1, 1, ksize * ksize, 1)
    spatial_weights = torch.exp(-spatial_dist_sq / (2.0 * sigma_space**2))

    # Color difference from center pixel
    center = x.view(b, c, 1, h * w)
    color_dist_sq = (patches - center) ** 2
    color_dist_sq = color_dist_sq.mean(dim=1, keepdim=True)
    color_weights = torch.exp(-color_dist_sq / (2.0 * sigma_color**2))

    # Combined bilateral weights
    weights = spatial_weights * color_weights
    out = (patches * weights).sum(dim=2) / weights.sum(dim=2).clamp(min=1e-6)
    return out.view(b, c, h, w).clamp(0.0, 1.0)


def bilateral_dither_filter_image(
    image: Image.Image | np.ndarray,
    d: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
) -> Image.Image:
    """Applies bilateral filtering on PIL Image or NumPy array via OpenCV.

    Args:
        image: PIL Image or [H, W, 3] NumPy array in [0, 255].
        d: Diameter of each pixel neighborhood.
        sigma_color: Filter sigma in color space.
        sigma_space: Filter sigma in coordinate space.

    Returns:
        Filtered PIL Image.
    """
    if isinstance(image, Image.Image):
        arr = np.array(image.convert("RGB"))
        is_pil = True
    else:
        arr = image.copy()
        is_pil = False

    smoothed = cv2.bilateralFilter(arr, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    if is_pil:
        return Image.fromarray(smoothed)
    return Image.fromarray(smoothed.astype(np.uint8))


class ZeroConv2d(nn.Module):
    """1x1 Convolution initialized with zeros on weights and biases.

    Guarantees zero initial contribution when attaching adapters to diffusion models,
    preventing catastrophic drift of pretrained generative representations.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SpatialFeatureTransform(nn.Module):
    """Spatial Feature Transform (SFT) layer for condition-guided feature modulation."""

    def __init__(self, in_channels: int, cond_channels: int):
        super().__init__()
        self.conv_gamma = nn.Sequential(
            nn.Conv2d(cond_channels, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
        )
        self.conv_beta = nn.Sequential(
            nn.Conv2d(cond_channels, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma = self.conv_gamma(cond)
        beta = self.conv_beta(cond)
        return x * (1.0 + torch.tanh(gamma)) + beta


class PixelArtTileAdapter(nn.Module):
    """Spatial Tile Adapter for injecting pixel art structural priors into diffusion models.

    Extracts multi-scale or single-scale feature representations from pixel-art sprites,
    applies optional bilateral dithering suppression, and projects features via Zero-Convolutions
    scaled by a tile conditioning weight.
    """

    def __init__(
        self,
        in_channels: int = 3,
        cond_channels: int = 64,
        out_channels: int = 64,
        num_blocks: int = 3,
        use_bilateral_filter: bool = True,
        ksize: int = 5,
        sigma_color: float = 0.15,
        sigma_space: float = 3.0,
    ):
        """
        Args:
            in_channels: Input image channels (typically 3 for RGB).
            cond_channels: Intermediate adapter feature dimensionality.
            out_channels: Output feature dimensionality matching diffusion backbone.
            num_blocks: Number of residual processing blocks in the adapter.
            use_bilateral_filter: Whether to apply tensor bilateral filtering to input.
            ksize: Bilateral filter kernel size.
            sigma_color: Bilateral filter color sigma.
            sigma_space: Bilateral filter spatial sigma.
        """
        super().__init__()
        self.use_bilateral_filter = use_bilateral_filter
        self.ksize = ksize
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space
        self.cond_channels = cond_channels
        self.out_channels = out_channels

        # 1. Feature extractor head
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, cond_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(cond_channels, cond_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 2. Residual processing blocks
        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(cond_channels, cond_channels, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(cond_channels, cond_channels, kernel_size=3, padding=1),
                )
            )
        self.body_blocks = nn.ModuleList(blocks)

        # 3. Zero-convolution output projection
        self.zero_conv = ZeroConv2d(cond_channels, out_channels)

        # 4. Optional SFT modulation
        self.sft = SpatialFeatureTransform(out_channels, cond_channels)

    def preprocess_sprite(self, x: torch.Tensor) -> torch.Tensor:
        """Applies bilateral dithering suppression filter if enabled."""
        if self.use_bilateral_filter and x.shape[-1] >= self.ksize and x.shape[-2] >= self.ksize:
            return bilateral_dither_filter_tensor(
                x,
                ksize=self.ksize,
                sigma_color=self.sigma_color,
                sigma_space=self.sigma_space,
            )
        return x

    def forward(
        self,
        x: torch.Tensor,
        tile_weight: float = 1.0,
    ) -> torch.Tensor:
        """Extracts spatial condition features from pixel art sprite.

        Args:
            x: Input sprite tensor [B, 3, H, W] in [0, 1].
            tile_weight: Conditioning weight multiplier in [0.0, 1.0+].

        Returns:
            Condition feature map [B, out_channels, H, W] modulated by tile_weight.
        """
        feat = self.preprocess_sprite(x)
        feat = self.head(feat)

        for block in self.body_blocks:
            feat = feat + 0.2 * block(feat)

        cond_proj = self.zero_conv(feat)
        return cond_proj * float(tile_weight)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Returns intermediate un-projected condition features for SFT modulation."""
        feat = self.preprocess_sprite(x)
        feat = self.head(feat)
        for block in self.body_blocks:
            feat = feat + 0.2 * block(feat)
        return feat

    def modulate(self, feature_map: torch.Tensor, sprite: torch.Tensor) -> torch.Tensor:
        """Directly modulates feature map with sprite using Spatial Feature Transform."""
        cond = self.extract_features(sprite)
        return self.sft(feature_map, cond)


class PixelArtControlAdapter(PixelArtTileAdapter):
    """Alias for PixelArtTileAdapter representing ControlNet/Adapter conditioning."""
