"""Discriminator architectures with spectral normalization for adversarial training."""

import torch
import torch.nn.functional as F
from torch import nn

try:
    from torch.nn.utils.parametrizations import spectral_norm
except ImportError:
    from torch.nn.utils import spectral_norm  # type: ignore


class UNetDiscriminatorWithSpectralNorm(nn.Module):
    """U-Net discriminator with spectral normalization.

    Reference:
        Wang et al., 'Real-ESRGAN: Training Real-World Blind Super-Resolution
        with Pure Synthetic Data', ICCV Workshops 2021.

    Features:
        - Downsampling path capturing global context and composition.
        - Dense skip-connections preserving local pixel boundaries.
        - Spectral normalization on all convolutional layers for Lipschitz continuity
          and training stability on Apple Silicon MPS / CUDA / CPU.
        - Outputs dense per-pixel / patch logits [B, 1, H, W].
    """

    def __init__(self, in_channels: int = 3, num_features: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.num_features = num_features

        # Downsampling path
        self.conv0 = spectral_norm(nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1))
        self.conv1 = spectral_norm(
            nn.Conv2d(num_features, num_features * 2, kernel_size=4, stride=2, padding=1)
        )
        self.conv2 = spectral_norm(
            nn.Conv2d(num_features * 2, num_features * 4, kernel_size=4, stride=2, padding=1)
        )
        self.conv3 = spectral_norm(
            nn.Conv2d(num_features * 4, num_features * 8, kernel_size=4, stride=2, padding=1)
        )

        # Upsampling path with skip connections
        self.upconv3 = spectral_norm(
            nn.Conv2d(num_features * 8, num_features * 4, kernel_size=3, padding=1)
        )
        self.fuse3 = spectral_norm(
            nn.Conv2d(num_features * 8, num_features * 4, kernel_size=3, padding=1)
        )

        self.upconv2 = spectral_norm(
            nn.Conv2d(num_features * 4, num_features * 2, kernel_size=3, padding=1)
        )
        self.fuse2 = spectral_norm(
            nn.Conv2d(num_features * 4, num_features * 2, kernel_size=3, padding=1)
        )

        self.upconv1 = spectral_norm(
            nn.Conv2d(num_features * 2, num_features, kernel_size=3, padding=1)
        )
        self.fuse1 = spectral_norm(
            nn.Conv2d(num_features * 2, num_features, kernel_size=3, padding=1)
        )

        # Output projection
        self.conv_out = nn.Sequential(
            spectral_norm(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features, 1, kernel_size=3, padding=1),
        )

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input image tensor [B, C, H, W] in range [0, 1].

        Returns:
            Logit map [B, 1, H, W].
        """
        # Downsample
        d0 = self.lrelu(self.conv0(x))
        d1 = self.lrelu(self.conv1(d0))
        d2 = self.lrelu(self.conv2(d1))
        d3 = self.lrelu(self.conv3(d2))

        # Upsample with alignment
        u3 = self.lrelu(
            self.upconv3(F.interpolate(d3, size=d2.shape[2:], mode="bilinear", align_corners=False))
        )
        u3 = self.lrelu(self.fuse3(torch.cat([u3, d2], dim=1)))

        u2 = self.lrelu(
            self.upconv2(F.interpolate(u3, size=d1.shape[2:], mode="bilinear", align_corners=False))
        )
        u2 = self.lrelu(self.fuse2(torch.cat([u2, d1], dim=1)))

        u1 = self.lrelu(
            self.upconv1(F.interpolate(u2, size=d0.shape[2:], mode="bilinear", align_corners=False))
        )
        u1 = self.lrelu(self.fuse1(torch.cat([u1, d0], dim=1)))

        return self.conv_out(u1)
