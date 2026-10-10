"""Main training script for neural upscalers."""

import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nn_upscaler.data.dataset import SyntheticPixelArtDataset
from nn_upscaler.data.degradation import PixelArtDegradation
from nn_upscaler.losses.perceptual import CompositeLoss
from nn_upscaler.models.registry import get_model
from nn_upscaler.training.trainer import SuperResolutionTrainer
from nn_upscaler.utils.device import get_device, get_device_name


def main():
    parser = argparse.ArgumentParser(
        description="Train neural upscaler on synthetic or paired data."
    )
    parser.add_argument(
        "--config", "-c", type=str, default="configs/default.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--model", "-m", type=str, default="edsr_light", help="Model architecture name"
    )
    parser.add_argument("--scale", "-s", type=int, default=4, help="Upscale factor")
    parser.add_argument(
        "--data-dir", "-d", type=str, default="data/samples", help="Directory of training images"
    )
    parser.add_argument("--epochs", "-e", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch-size", "-b", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument(
        "--device", type=str, default=None, help="Compute device ('mps', 'cuda', 'cpu')"
    )

    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    cfg = {}
    if config_path.exists():
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}

    scale_factor = args.scale or cfg.get("scale_factor", 4)
    epochs = args.epochs or cfg.get("training", {}).get("epochs", 10)
    batch_size = args.batch_size or cfg.get("data", {}).get("batch_size", 4)
    lr = args.lr or cfg.get("training", {}).get("lr", 1e-4)

    device = get_device(args.device or cfg.get("device"))
    print(f"Device: {get_device_name(device)} ({device})")
    print(f"Model: {args.model} ({scale_factor}x)")

    # Data setup
    data_dir = Path(args.data_dir)
    images = list(data_dir.glob("*.png")) + list(data_dir.glob("*.jpg"))
    if not images:
        print(f"No training images found in {data_dir}", file=sys.stderr)
        sys.exit(1)

    degradation = PixelArtDegradation(
        scale_factor=scale_factor,
        num_colors=cfg.get("data", {}).get("num_colors", 32),
        palette_preset=cfg.get("data", {}).get("palette_preset", None),
        dither=cfg.get("data", {}).get("dither", False),
    )

    dataset = SyntheticPixelArtDataset(
        hr_image_paths=images,
        patch_size=cfg.get("data", {}).get("patch_size", 64),
        scale_factor=scale_factor,
        degradation=degradation,
        augment=cfg.get("data", {}).get("augment", True),
    )

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    # Model & Loss
    model = get_model(args.model, scale_factor=scale_factor)
    loss_cfg = cfg.get("loss", {})
    criterion = CompositeLoss(
        pixel_weight=loss_cfg.get("pixel_weight", 1.0),
        gradient_weight=loss_cfg.get("gradient_weight", 0.1),
        lpips_weight=loss_cfg.get("lpips_weight", 0.0),
        device=device,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    trainer = SuperResolutionTrainer(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=cfg.get("training", {}).get("checkpoint_dir", "checkpoints"),
    )

    print(f"Starting training for {epochs} epochs on {len(images)} images...")
    trainer.fit(epochs=epochs, save_freq=cfg.get("training", {}).get("save_freq", 5))
    print("Training complete! Checkpoints saved in:", trainer.checkpoint_dir)


if __name__ == "__main__":
    main()
