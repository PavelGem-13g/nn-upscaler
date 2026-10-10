"""Abstract base class for all upscaler models."""

from abc import ABC, abstractmethod

import numpy as np
import torch
from PIL import Image
from torch import nn

from nn_upscaler.utils.device import get_device
from nn_upscaler.utils.visualization import pil_to_tensor, tensor_to_pil


class BaseUpscaler(nn.Module, ABC):
    """Abstract base class for neural and classical upscalers."""

    def __init__(self, scale_factor: int = 4):
        super().__init__()
        self.scale_factor = int(scale_factor)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for batched tensor [B, C, H, W] in range [0, 1].

        Args:
            x: Input tensor [B, C, H, W].

        Returns:
            Upscaled tensor [B, C, H * scale, W * scale].
        """

    def upscale(
        self,
        image: Image.Image | np.ndarray | torch.Tensor,
        device: torch.device | None = None,
    ) -> Image.Image:
        """High-level inference method that accepts PIL, NumPy, or Tensor and returns PIL.

        Args:
            image: Input image.
            device: Optional compute device.

        Returns:
            Upscaled PIL Image.
        """
        dev = device or get_device()
        self.to(dev)
        self.eval()

        if isinstance(image, Image.Image):
            tensor = pil_to_tensor(image).unsqueeze(0).to(dev)
        elif isinstance(image, np.ndarray):
            tensor = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(dev)
        elif isinstance(image, torch.Tensor):
            tensor = image.to(dev)
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        with torch.no_grad():
            output_tensor = self.forward(tensor)

        return tensor_to_pil(output_tensor.squeeze(0))
