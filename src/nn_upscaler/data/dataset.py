"""Datasets and data loaders for pixel-art super-resolution.

Provides high-performance on-the-fly degradation, alpha-transparency handling,
spatial augmentations, and GPU-ready tensor conversion.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from nn_upscaler.data.degradation import (
    PixelArtDegradation,
    RandomPixelArtDegradation,
)


class SyntheticPixelArtDataset(Dataset):
    """Generates (LR pixel-art, HR realistic) pairs on the fly from high-resolution images.

    Features:
    - Dynamic degradation via `RandomPixelArtDegradation` or fixed `PixelArtDegradation`.
    - Transparent RGBA sprite support (compositing on randomized contrasting backgrounds).
    - Data augmentations (random crops, horizontal/vertical flips, 90/180/270 rotations).
    - Tensor output [C, H, W] in [0.0, 1.0] ready for GPU training.
    """

    def __init__(
        self,
        hr_image_paths: list[str | Path],
        patch_size: int = 128,
        scale_factor: int = 4,
        degradation: PixelArtDegradation | RandomPixelArtDegradation | None = None,
        augment: bool = True,
        handle_alpha: Literal["composite", "mask", "keep"] = "composite",
        bg_mode: Literal["random", "black", "white", "gray"] = "random",
        device: torch.device | str | None = None,
    ):
        """
        Args:
            hr_image_paths: List of paths to high-resolution realistic source images.
            patch_size: HR patch crop size (divisible by scale_factor).
            scale_factor: Super-resolution scale factor (e.g. 2, 4, 8).
            degradation: Degradation instance. Defaults to RandomPixelArtDegradation.
            augment: Whether to apply random flips and 90-degree rotations.
            handle_alpha: Strategy for RGBA images ('composite', 'mask', or 'keep').
            bg_mode: Background color when compositing ('random', 'black', 'white', 'gray').
            device: Optional target device for output tensors.
        """
        self.image_paths = [Path(p) for p in hr_image_paths]
        if not self.image_paths:
            raise ValueError("hr_image_paths cannot be empty.")

        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.augment = augment
        self.handle_alpha = handle_alpha
        self.bg_mode = bg_mode
        self.device = torch.device(device) if device is not None else None

        if degradation is None:
            self.degradation = RandomPixelArtDegradation(scale_factors=[scale_factor])
        else:
            self.degradation = degradation

    def __len__(self) -> int:
        return len(self.image_paths)

    def _sample_bg_color(self) -> tuple[int, int, int]:
        """Sample a contrasting solid background color."""
        if self.bg_mode == "black":
            return (0, 0, 0)
        if self.bg_mode == "white":
            return (255, 255, 255)
        if self.bg_mode == "gray":
            return (128, 128, 128)
        # Random choice among black, white, neutral gray, or random bright color
        palette_choice = random.choice(["black", "white", "gray", "random"])
        if palette_choice == "black":
            return (0, 0, 0)
        if palette_choice == "white":
            return (255, 255, 255)
        if palette_choice == "gray":
            return (128, 128, 128)
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    def _handle_rgba(self, img: Image.Image) -> tuple[Image.Image, Image.Image | None]:
        """Process transparent images according to handle_alpha setting."""
        if img.mode != "RGBA":
            return img.convert("RGB"), None

        alpha_mask = img.split()[-1]
        if self.handle_alpha == "composite":
            bg_rgb = self._sample_bg_color()
            bg = Image.new("RGBA", img.size, bg_rgb + (255,))
            composited = Image.alpha_composite(bg, img).convert("RGB")
            return composited, None
        if self.handle_alpha == "keep":
            return img, None
        # "mask"
        return img.convert("RGB"), alpha_mask

    def _random_crop(self, img: Image.Image) -> Image.Image:
        """Crop a patch of size (patch_size, patch_size). Resizes if smaller."""
        w, h = img.size
        if w < self.patch_size or h < self.patch_size:
            factor = max(self.patch_size / float(w), self.patch_size / float(h))
            new_w = int(w * factor) + 1
            new_h = int(h * factor) + 1
            img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
            w, h = img.size

        x = random.randint(0, w - self.patch_size)
        y = random.randint(0, h - self.patch_size)
        return img.crop((x, y, x + self.patch_size, y + self.patch_size))

    def _augment(self, img: Image.Image) -> Image.Image:
        """Apply random dihedral augmentations (flips & rotations)."""
        if random.random() > 0.5:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if random.random() > 0.5:
            img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        rot_choice = random.choice([0, 90, 180, 270])
        if rot_choice != 0:
            img = img.rotate(rot_choice)
        return img

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve a paired (LR pixel-art, HR realistic) patch tensor pair.

        Returns:
            lr_tensor: [C, H / scale, W / scale] in [0, 1].
            hr_tensor: [C, H, W] in [0, 1].
        """
        img_path = self.image_paths[idx]
        with Image.open(img_path) as im:
            # Handle alpha channel
            processed_img, _ = self._handle_rgba(im)

        # 1. Random crop
        hr_patch = self._random_crop(processed_img)

        # 2. Spatial augmentations
        if self.augment:
            hr_patch = self._augment(hr_patch)

        # 3. Convert HR patch to PyTorch Tensor [C, H, W] float32 in [0, 1]
        hr_np = np.array(hr_patch, dtype=np.float32) / 255.0
        hr_tensor = torch.from_numpy(hr_np).permute(2, 0, 1).contiguous()

        # 4. Synthesize LR pixel-art patch
        if isinstance(self.degradation, RandomPixelArtDegradation):
            lr_tensor = self.degradation(hr_tensor, forced_scale=self.scale_factor)
        else:
            lr_tensor = self.degradation(hr_tensor)

        if self.device is not None:
            lr_tensor = lr_tensor.to(device=self.device)
            hr_tensor = hr_tensor.to(device=self.device)

        return lr_tensor, hr_tensor


class OnTheFlyPixelArtDataset(Dataset):
    """Dynamic pixel art dataset scanning an image directory on the fly."""

    def __init__(
        self,
        image_dir: str | Path,
        patch_size: int = 128,
        scale_factors: list[int] | tuple[int, ...] = (2, 4, 8),
        augment: bool = True,
        device: torch.device | str | None = None,
    ):
        p = Path(image_dir)
        extensions = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")
        paths: list[Path] = []
        for ext in extensions:
            paths.extend(p.glob(ext))
        paths = sorted([f for f in paths if not f.name.startswith(".")])

        if not paths:
            raise FileNotFoundError(f"No valid images found in directory: {image_dir}")

        self.dataset = SyntheticPixelArtDataset(
            hr_image_paths=paths,
            patch_size=patch_size,
            scale_factor=scale_factors[0]
            if isinstance(scale_factors, (list, tuple))
            else scale_factors,
            degradation=RandomPixelArtDegradation(scale_factors=scale_factors),
            augment=augment,
            device=device,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.dataset[idx]
