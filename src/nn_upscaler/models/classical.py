"""Classical interpolation baselines."""

import torch
import torch.nn.functional as F

from nn_upscaler.models.base import BaseUpscaler


class NearestNeighborUpscaler(BaseUpscaler):
    """Nearest Neighbor upscaler baseline (standard pixel art scaling)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")


class BicubicUpscaler(BaseUpscaler):
    """Bicubic interpolation baseline."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).clamp(0.0, 1.0)


class BilinearUpscaler(BaseUpscaler):
    """Bilinear interpolation baseline."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).clamp(0.0, 1.0)
