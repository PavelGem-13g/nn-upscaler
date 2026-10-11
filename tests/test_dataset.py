"""Unit tests for synthetic pixel-art datasets and PyTorch DataLoader integration."""

from pathlib import Path

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from nn_upscaler.data.dataset import (
    OnTheFlyPixelArtDataset,
    SyntheticPixelArtDataset,
)
from nn_upscaler.data.degradation import (
    PixelArtDegradation,
    RandomPixelArtDegradation,
)


@pytest.fixture
def dummy_dataset_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with RGB and RGBA test images."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    # 1. Standard RGB image (large)
    rgb_img = Image.new("RGB", (256, 256), color=(200, 100, 50))
    rgb_img.save(img_dir / "rgb_01.png")

    # 2. Another RGB image (medium)
    rgb_img2 = Image.new("RGB", (128, 128), color=(50, 150, 200))
    rgb_img2.save(img_dir / "rgb_02.png")

    # 3. Transparent RGBA sprite (semi-transparent border, transparent background)
    rgba_img = Image.new("RGBA", (128, 128), color=(0, 0, 0, 0))
    # Draw a colored square in the center
    for x in range(32, 96):
        for y in range(32, 96):
            rgba_img.putpixel((x, y), (255, 50, 50, 255))
    rgba_img.save(img_dir / "sprite_rgba.png")

    # 4. Small image to test auto-scaling during random crop
    small_img = Image.new("RGB", (32, 32), color=(10, 200, 10))
    small_img.save(img_dir / "small_01.png")

    return img_dir


def test_synthetic_dataset_basic(dummy_dataset_dir: Path):
    image_paths = list(dummy_dataset_dir.glob("*.png"))
    dataset = SyntheticPixelArtDataset(
        hr_image_paths=image_paths,
        patch_size=64,
        scale_factor=4,
        degradation=PixelArtDegradation(scale_factor=4, num_colors=16),
        augment=True,
    )

    assert len(dataset) == 4
    lr, hr = dataset[0]

    assert isinstance(lr, torch.Tensor)
    assert isinstance(hr, torch.Tensor)
    assert hr.shape == (3, 64, 64)
    assert lr.shape == (3, 16, 16)
    assert lr.min() >= 0.0 and lr.max() <= 1.0
    assert hr.min() >= 0.0 and hr.max() <= 1.0


def test_synthetic_dataset_dataloader(dummy_dataset_dir: Path):
    image_paths = list(dummy_dataset_dir.glob("*.png"))
    dataset = SyntheticPixelArtDataset(
        hr_image_paths=image_paths,
        patch_size=64,
        scale_factor=4,
        degradation=RandomPixelArtDegradation(scale_factors=[4]),
        augment=True,
    )

    loader = DataLoader(dataset, batch_size=2, shuffle=True, drop_last=False)
    batch_count = 0
    for lr_batch, hr_batch in loader:
        assert lr_batch.shape == (2, 3, 16, 16)
        assert hr_batch.shape == (2, 3, 64, 64)
        assert not torch.isnan(lr_batch).any()
        assert not torch.isnan(hr_batch).any()
        batch_count += 1

    assert batch_count == 2


def test_synthetic_dataset_rgba_handling(dummy_dataset_dir: Path):
    rgba_path = dummy_dataset_dir / "sprite_rgba.png"

    # 1. Composite mode (default) -> outputs RGB 3-channel tensor
    dataset_comp = SyntheticPixelArtDataset(
        hr_image_paths=[rgba_path],
        patch_size=64,
        scale_factor=4,
        handle_alpha="composite",
        bg_mode="random",
    )
    lr, hr = dataset_comp[0]
    assert lr.shape == (3, 16, 16)
    assert hr.shape == (3, 64, 64)

    # 2. Keep mode -> preserves alpha channel [4, H, W]
    dataset_keep = SyntheticPixelArtDataset(
        hr_image_paths=[rgba_path],
        patch_size=64,
        scale_factor=4,
        handle_alpha="keep",
    )
    lr_k, hr_k = dataset_keep[0]
    assert lr_k.shape == (4, 16, 16)
    assert hr_k.shape == (4, 64, 64)


def test_onthefly_dataset(dummy_dataset_dir: Path):
    dataset = OnTheFlyPixelArtDataset(
        image_dir=dummy_dataset_dir,
        patch_size=64,
        scale_factors=[4],
    )
    assert len(dataset) == 4
    lr, hr = dataset[0]
    assert lr.shape == (3, 16, 16)
    assert hr.shape == (64, 64, 3) or hr.shape == (3, 64, 64)


def test_empty_dataset_error(tmp_path: Path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        OnTheFlyPixelArtDataset(image_dir=empty_dir)

    with pytest.raises(ValueError, match="hr_image_paths cannot be empty"):
        SyntheticPixelArtDataset(hr_image_paths=[])
