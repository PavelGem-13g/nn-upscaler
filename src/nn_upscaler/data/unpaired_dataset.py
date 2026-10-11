"""Unpaired Pixel Art Dataset for Contrastive Unpaired Translation (CUT).

Loads authentic retro game sprites (Domain X) and real high-resolution photos (Domain Y)
without requiring 1-to-1 paired correspondences.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class UnpairedPixelArtDataset(Dataset):
    """Dataset for unpaired pixel-art to realistic image translation.

    Domain X: Authentic or synthesized pixel art sprites (e.g. 16x16, 32x32, 64x64).
    Domain Y: High-resolution realistic photographs or renders.

    Returns:
        (lr, hr): Tuple of float32 tensors [C, H_lr, W_lr] and [C, H_hr, W_hr] in [0.0, 1.0].
    """

    def __init__(
        self,
        domain_x_paths: list[str | Path] | str | Path,
        domain_y_paths: list[str | Path] | str | Path,
        lr_patch_size: int = 32,
        scale_factor: int = 4,
        augment: bool = True,
        handle_alpha: Literal["composite", "black", "white"] = "composite",
        device: torch.device | str | None = None,
    ):
        """
        Args:
            domain_x_paths: File paths or directory containing pixel art sprites (Domain X).
            domain_y_paths: File paths or directory containing realistic HR photos (Domain Y).
            lr_patch_size: Size of low-resolution pixel-art patch (e.g. 16, 32, 64).
            scale_factor: Target super-resolution scale (HR size = lr_patch_size * scale_factor).
            augment: Whether to apply horizontal flip augmentations.
            handle_alpha: Strategy for transparent sprites.
            device: Optional compute device for returned tensors.
        """
        self.domain_x = self._resolve_paths(domain_x_paths)
        self.domain_y = self._resolve_paths(domain_y_paths)

        if not self.domain_x:
            raise ValueError(f"Domain X contains no valid images: {domain_x_paths}")
        if not self.domain_y:
            raise ValueError(f"Domain Y contains no valid images: {domain_y_paths}")

        self.lr_patch_size = int(lr_patch_size)
        self.scale_factor = int(scale_factor)
        self.hr_patch_size = self.lr_patch_size * self.scale_factor
        self.augment = augment
        self.handle_alpha = handle_alpha
        self.device = torch.device(device) if device is not None else None

    @staticmethod
    def _resolve_paths(paths: list[str | Path] | str | Path) -> list[Path]:
        """Resolve directory or list of file paths into verified Image paths."""
        if isinstance(paths, (str, Path)):
            p = Path(paths)
            if p.is_dir():
                found = list(p.glob("*.png")) + list(p.glob("*.jpg")) + list(p.glob("*.jpeg"))
                return sorted(found)
            if p.is_file():
                return [p]
            return []
        resolved: list[Path] = []
        for item in paths:
            ip = Path(item)
            if ip.is_file():
                resolved.append(ip)
            elif ip.is_dir():
                resolved.extend(
                    list(ip.glob("*.png")) + list(ip.glob("*.jpg")) + list(ip.glob("*.jpeg"))
                )
        return sorted(set(resolved))

    def __len__(self) -> int:
        """Dataset length is maximum of both domains."""
        return max(len(self.domain_x), len(self.domain_y))

    def _load_image(self, path: Path) -> Image.Image:
        """Load image and composite RGBA if needed."""
        img = Image.open(path)
        if img.mode == "RGBA":
            if self.handle_alpha == "white":
                bg = Image.new("RGB", img.size, (255, 255, 255))
            elif self.handle_alpha == "black":
                bg = Image.new("RGB", img.size, (0, 0, 0))
            else:
                c = random.choice([(0, 0, 0), (255, 255, 255), (128, 128, 128)])
                bg = Image.new("RGB", img.size, c)
            bg.paste(img, mask=img.split()[-1])
            return bg
        return img.convert("RGB")

    def _prepare_domain_x(self, img: Image.Image) -> torch.Tensor:
        """Resize/crop pixel art sprite to [3, lr_patch_size, lr_patch_size]."""
        w, h = img.size
        target = self.lr_patch_size

        if w >= target and h >= target:
            left = random.randint(0, w - target)
            top = random.randint(0, h - target)
            crop = img.crop((left, top, left + target, top + target))
        else:
            # Sprite is smaller than patch size: pad to target with edge or black
            crop = Image.new("RGB", (target, target), (0, 0, 0))
            crop.paste(img, ((target - w) // 2, (target - h) // 2))

        if self.augment and random.random() < 0.5:
            crop = crop.transpose(Image.FLIP_LEFT_RIGHT)

        arr = np.array(crop, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    def _prepare_domain_y(self, img: Image.Image) -> torch.Tensor:
        """Crop HR realistic photo to [3, hr_patch_size, hr_patch_size]."""
        w, h = img.size
        target = self.hr_patch_size

        if w < target or h < target:
            scale = max(target / w, target / h)
            new_w = max(target, int(w * scale + 0.5))
            new_h = max(target, int(h * scale + 0.5))
            img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
            w, h = img.size

        left = random.randint(0, w - target)
        top = random.randint(0, h - target)
        crop = img.crop((left, top, left + target, top + target))

        if self.augment and random.random() < 0.5:
            crop = crop.transpose(Image.FLIP_LEFT_RIGHT)

        arr = np.array(crop, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Sample Domain X (modulo cycling if len(X) < len(Y))
        idx_x = idx % len(self.domain_x)
        path_x = self.domain_x[idx_x]
        img_x = self._load_image(path_x)
        lr_tensor = self._prepare_domain_x(img_x)

        # Sample Domain Y (random or modulo)
        idx_y = random.randint(0, len(self.domain_y) - 1)
        path_y = self.domain_y[idx_y]
        img_y = self._load_image(path_y)
        hr_tensor = self._prepare_domain_y(img_y)

        if self.device is not None:
            lr_tensor = lr_tensor.to(self.device)
            hr_tensor = hr_tensor.to(self.device)

        return lr_tensor, hr_tensor
