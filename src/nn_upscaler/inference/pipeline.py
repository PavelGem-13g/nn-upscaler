"""Inference pipeline with optional tile-based execution for arbitrary image sizes."""

import numpy as np
import torch
from PIL import Image

from nn_upscaler.models.base import BaseUpscaler
from nn_upscaler.utils.device import get_device
from nn_upscaler.utils.visualization import pil_to_tensor, tensor_to_pil


class InferencePipeline:
    """Manages inference execution with device transfer and tiling support."""

    def __init__(
        self,
        model: BaseUpscaler,
        device: torch.device | None = None,
        tile_size: int | None = None,
        tile_overlap: int = 16,
    ):
        """
        Args:
            model: An instance of BaseUpscaler.
            device: Compute device (MPS, CUDA, or CPU).
            tile_size: Max tile dimension for chunked inference (None = process whole image).
            tile_overlap: Overlapping pixel margin between adjacent tiles to prevent seams.
        """
        self.device = device or get_device()
        self.model = model.to(self.device).eval()
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.scale = model.scale_factor

    def run(
        self,
        image: Image.Image | np.ndarray | torch.Tensor,
        **kwargs,
    ) -> Image.Image:
        """Upscale an input image using the configured model.

        Args:
            image: PIL Image, NumPy array, or Tensor.
            **kwargs: Extra model-specific inference arguments (e.g. prompt, category).

        Returns:
            Upscaled PIL Image.
        """
        if isinstance(image, Image.Image):
            tensor = pil_to_tensor(image)
        elif isinstance(image, np.ndarray):
            tensor = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)
        elif isinstance(image, torch.Tensor):
            tensor = image
            if tensor.dim() == 4:
                tensor = tensor.squeeze(0)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        _c, h, w = tensor.shape

        # If no tiling or image is small enough, process directly
        if self.tile_size is None or (h <= self.tile_size and w <= self.tile_size):
            with torch.no_grad():
                inp = tensor.unsqueeze(0).to(self.device)
                out = self.model(inp, **kwargs).squeeze(0)
            return tensor_to_pil(out)

        # Otherwise, process via overlapping tiles
        out_tensor = self._run_tiled(tensor, **kwargs)
        return tensor_to_pil(out_tensor)

    def _run_tiled(self, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        """Splits tensor into overlapping tiles, runs inference, and blends output."""
        c, h, w = tensor.shape
        out_h, out_w = h * self.scale, w * self.scale

        output = torch.zeros((c, out_h, out_w), dtype=torch.float32, device="cpu")
        weights = torch.zeros((1, out_h, out_w), dtype=torch.float32, device="cpu")

        stride = self.tile_size - self.tile_overlap

        for y in range(0, h, stride):
            for x in range(0, w, stride):
                # Calculate tile bounding box with clamping
                y1 = min(y, max(0, h - self.tile_size))
                x1 = min(x, max(0, w - self.tile_size))
                y2 = min(y1 + self.tile_size, h)
                x2 = min(x1 + self.tile_size, w)

                tile = tensor[:, y1:y2, x1:x2].unsqueeze(0).to(self.device)
                with torch.no_grad():
                    out_tile = self.model(tile, **kwargs).squeeze(0).cpu()

                oy1, oy2 = y1 * self.scale, y2 * self.scale
                ox1, ox2 = x1 * self.scale, x2 * self.scale

                output[:, oy1:oy2, ox1:ox2] += out_tile
                weights[:, oy1:oy2, ox1:ox2] += 1.0

                if x2 >= w:
                    break
            if y2 >= h:
                break

        weights = weights.clamp(min=1.0)
        output = output / weights
        return output.clamp(0.0, 1.0)
