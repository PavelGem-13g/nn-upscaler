"""Datasets and data loaders for pixel-art super-resolution."""

import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from nn_upscaler.data.degradation import PixelArtDegradation
from nn_upscaler.utils.visualization import pil_to_tensor


class SyntheticPixelArtDataset(Dataset):
    """Generates (LR pixel-art, HR realistic) pairs on the fly from HR images.

    This enables training using any folder of high-resolution realistic images
    (e.g., DIV2K, Unsplash, portraits, landscapes) without needing pre-existing
    paired datasets.
    """

    def __init__(
        self,
        hr_image_paths: list[str | Path],
        patch_size: int = 128,
        scale_factor: int = 4,
        degradation: PixelArtDegradation | None = None,
        augment: bool = True,
    ):
        """
        Args:
            hr_image_paths: List of file paths to high-resolution realistic images.
            patch_size: High-resolution crop patch size (must be divisible by scale_factor).
            scale_factor: Super-resolution scale factor.
            degradation: Degradation pipeline instance (defaults to PixelArtDegradation(scale_factor)).
            augment: Whether to apply random flips and rotations.
        """
        self.image_paths = [Path(p) for p in hr_image_paths]
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.degradation = degradation or PixelArtDegradation(scale_factor=scale_factor)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.image_paths)

    def _random_crop(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w < self.patch_size or h < self.patch_size:
            # Upscale if image is smaller than requested patch
            factor = max(self.patch_size / w, self.patch_size / h)
            img = img.resize((int(w * factor) + 1, int(h * factor) + 1), Image.Resampling.BILINEAR)
            w, h = img.size

        x = random.randint(0, w - self.patch_size)
        y = random.randint(0, h - self.patch_size)
        return img.crop((x, y, x + self.patch_size, y + self.patch_size))

    def _augment(self, hr_img: Image.Image) -> Image.Image:
        if random.random() > 0.5:
            hr_img = hr_img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if random.random() > 0.5:
            hr_img = hr_img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        rot_choice = random.choice([0, 90, 180, 270])
        if rot_choice != 0:
            hr_img = hr_img.rotate(rot_choice)
        return hr_img

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        with Image.open(img_path) as img:
            hr_img = img.convert("RGB")

        # 1. Random crop
        hr_patch = self._random_crop(hr_img)

        # 2. Augmentation
        if self.augment:
            hr_patch = self._augment(hr_patch)

        # 3. Degrade HR patch to synthetic LR pixel art
        lr_patch = self.degradation(hr_patch)

        # 4. Convert to tensors [C, H, W] in [0, 1]
        hr_tensor = pil_to_tensor(hr_patch)
        lr_tensor = pil_to_tensor(lr_patch)

        return lr_tensor, hr_tensor
