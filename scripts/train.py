"""Main training script for supervised and adversarial neural upscalers."""

import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nn_upscaler.data.dataset import SyntheticPixelArtDataset
from nn_upscaler.data.degradation import PixelArtDegradation, RandomPixelArtDegradation
from nn_upscaler.data.unpaired_dataset import UnpairedPixelArtDataset
from nn_upscaler.losses.contrastive import PatchNCELoss
from nn_upscaler.losses.perceptual import CompositeLoss
from nn_upscaler.models.discriminator import UNetDiscriminatorWithSpectralNorm
from nn_upscaler.models.registry import get_model, list_models
from nn_upscaler.training.trainer import SuperResolutionTrainer
from nn_upscaler.utils.device import get_device, get_device_name


def main():
    parser = argparse.ArgumentParser(
        description="Train neural upscaler on synthetic, paired, or unpaired pixel-art data."
    )
    parser.add_argument("--config", "-c", type=str, default=None, help="Path to config file")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="edsr_light",
        choices=list_models(),
        help="Model architecture name",
    )
    parser.add_argument("--scale", "-s", type=int, default=4, help="Upscale factor (2, 4, 8)")
    parser.add_argument(
        "--data-dir", "-d", type=str, default="data/samples", help="Directory of training images"
    )
    parser.add_argument("--epochs", "-e", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch-size", "-b", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override generator learning rate")
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        choices=["psnr", "adversarial", "codebook", "semantic", "unpaired_cut"],
        help="Training stage: 'psnr', 'adversarial', 'codebook', 'semantic', or 'unpaired_cut'",
    )
    parser.add_argument(
        "--gan",
        action="store_true",
        help="Enable adversarial GAN training with UNetDiscriminator",
    )
    parser.add_argument(
        "--load-pretrained",
        type=str,
        default=None,
        help="Path to pretrained model checkpoint to resume or fine-tune from",
    )
    parser.add_argument(
        "--random-degradation",
        action="store_true",
        help="Use dynamic stochastic degradation (RandomPixelArtDegradation) during training",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Compute device ('mps', 'cuda', 'cpu')"
    )

    args = parser.parse_args()

    # Load configuration
    cfg = {}
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f) or {}
    elif args.model in ("rrdb", "esrgan"):
        cfg_file = Path("configs/rrdb_esrgan.yaml")
        if cfg_file.exists():
            with open(cfg_file, "r") as f:
                cfg = yaml.safe_load(f) or {}
    elif args.model == "swinir":
        cfg_file = Path("configs/swinir.yaml")
        if cfg_file.exists():
            with open(cfg_file, "r") as f:
                cfg = yaml.safe_load(f) or {}
    elif args.model == "grid_aware":
        cfg_file = Path("configs/grid_aware.yaml")
        if cfg_file.exists():
            with open(cfg_file, "r") as f:
                cfg = yaml.safe_load(f) or {}
    elif args.model == "codebook_prior":
        cfg_file = Path("configs/codebook_prior.yaml")
        if cfg_file.exists():
            with open(cfg_file, "r") as f:
                cfg = yaml.safe_load(f) or {}
    elif args.model == "semantic_conditioned":
        cfg_file = Path("configs/semantic_conditioned.yaml")
        if cfg_file.exists():
            with open(cfg_file, "r") as f:
                cfg = yaml.safe_load(f) or {}
    elif args.model == "cut_unpaired":
        cfg_file = Path("configs/unpaired_cut.yaml")
        if cfg_file.exists():
            with open(cfg_file, "r") as f:
                cfg = yaml.safe_load(f) or {}

    scale_factor = args.scale or cfg.get("scale_factor", 4)
    training_cfg = cfg.get("training", {})
    epochs = args.epochs or training_cfg.get("epochs", 10)
    batch_size = args.batch_size or cfg.get("data", {}).get("batch_size", 4)
    lr = args.lr or training_cfg.get("lr", training_cfg.get("lr_g", 1e-4))
    stage = args.stage or training_cfg.get("stage", "psnr")
    is_gan = args.gan or training_cfg.get("gan", False) or (stage == "adversarial")

    device = get_device(args.device or cfg.get("device"))
    print(f"Device: {get_device_name(device)} ({device})")
    print(f"Model: {args.model} ({scale_factor}x) | Stage: {stage} | GAN: {is_gan}")

    # Data setup
    is_unpaired = (cfg.get("data", {}).get("mode") == "unpaired") or (args.model == "cut_unpaired")

    if is_unpaired:
        domain_x = cfg.get("data", {}).get("domain_x", "data/benchmarks/authentic_pixel_art")
        domain_y = cfg.get("data", {}).get("domain_y", "data/hr_raw")
        if not Path(domain_y).exists():
            domain_y = "data/samples"
        lr_patch_size = cfg.get("data", {}).get("lr_patch_size", 32)
        dataset = UnpairedPixelArtDataset(
            domain_x_paths=domain_x,
            domain_y_paths=domain_y,
            lr_patch_size=lr_patch_size,
            scale_factor=scale_factor,
            augment=cfg.get("data", {}).get("augment", True),
        )
        total_sample_count = len(dataset)
    else:
        data_dir = Path(args.data_dir)
        images = list(data_dir.glob("*.png")) + list(data_dir.glob("*.jpg"))
        if not images and Path("data/hr_raw").exists():
            data_dir = Path("data/hr_raw")
            images = list(data_dir.glob("*.png")) + list(data_dir.glob("*.jpg"))

        if args.data_dir == "data/samples" and Path("data/hr_raw").exists():
            hr_raw_images = list(Path("data/hr_raw").glob("*.png")) + list(
                Path("data/hr_raw").glob("*.jpg")
            )
            images = sorted(set(images + hr_raw_images))

        if not images:
            print(f"No training images found in {data_dir}", file=sys.stderr)
            sys.exit(1)

        use_random_deg = args.random_degradation or cfg.get("data", {}).get(
            "random_degradation", True
        )
        if use_random_deg:
            degradation = RandomPixelArtDegradation(scale_factors=[scale_factor])
        else:
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
        total_sample_count = len(images)

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    # Instantiate model
    model_kwargs = {}
    if "model" in cfg and isinstance(cfg["model"], dict):
        model_kwargs = {k: v for k, v in cfg["model"].items() if k != "name"}
    model = get_model(args.model, scale_factor=scale_factor, **model_kwargs).to(device)

    # Discriminator setup for Stage 2 GAN / CUT
    discriminator = None
    optimizer_d = None
    if is_gan:
        disc_features = cfg.get("discriminator", {}).get("num_features", 32)
        discriminator = UNetDiscriminatorWithSpectralNorm(
            in_channels=3, num_features=disc_features
        ).to(device)
        lr_d = training_cfg.get("lr_d", 1e-4)
        optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.9, 0.999))

    # Loss configuration
    loss_cfg = cfg.get("loss", {})
    pixel_type = loss_cfg.get("pixel_type", "charbonnier")
    pixel_weight = loss_cfg.get("pixel_weight", 1.0)
    gradient_weight = loss_cfg.get("gradient_weight", 0.1)
    fft_weight = loss_cfg.get("fft_weight", 0.05)
    vgg_weight = loss_cfg.get("vgg_weight", 0.1 if is_gan else 0.0)
    lpips_weight = loss_cfg.get("lpips_weight", 0.1 if is_gan else 0.0)
    gan_weight = loss_cfg.get("gan_weight", 0.005 if is_gan else 0.0)
    patchnce_weight = loss_cfg.get("patchnce_weight", 1.0)
    vq_weight = loss_cfg.get("vq_weight", 0.25)

    criterion = CompositeLoss(
        pixel_type=pixel_type,
        pixel_weight=pixel_weight,
        gradient_weight=gradient_weight,
        fft_weight=fft_weight,
        vgg_weight=vgg_weight,
        lpips_weight=lpips_weight,
        gan_weight=gan_weight,
        device=device,
    ).to(device)

    patch_nce_loss = PatchNCELoss(
        tau=loss_cfg.get("tau", 0.07),
        num_patches=loss_cfg.get("num_patches", 256),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    checkpoint_dir = training_cfg.get("checkpoint_dir", f"checkpoints/{args.model}_{stage}")

    trainer = SuperResolutionTrainer(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        discriminator=discriminator,
        optimizer_d=optimizer_d,
        patch_nce_loss=patch_nce_loss,
        patchnce_weight=patchnce_weight,
        vq_weight=vq_weight,
        gan_weight=gan_weight,
        device=device,
        checkpoint_dir=checkpoint_dir,
        stage=stage,
    )

    # Load pretrained checkpoint if requested
    load_path = args.load_pretrained or training_cfg.get("load_pretrained")
    if load_path:
        trainer.load_checkpoint(load_path)

    print(f"Starting training for {epochs} epochs on {total_sample_count} samples...")
    trainer.fit(epochs=epochs, save_freq=training_cfg.get("save_freq", 5))
    print("Training complete! Checkpoints saved in:", trainer.checkpoint_dir)


if __name__ == "__main__":
    main()
