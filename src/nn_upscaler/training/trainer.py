"""Trainer class for supervised and adversarial super-resolution training."""

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from nn_upscaler.evaluation.metrics import calculate_psnr, calculate_ssim
from nn_upscaler.losses.contrastive import PatchNCELoss
from nn_upscaler.losses.gan import RaGANLoss
from nn_upscaler.losses.perceptual import CompositeLoss
from nn_upscaler.models.base import BaseUpscaler
from nn_upscaler.utils.device import clear_device_cache, get_device


class SuperResolutionTrainer:
    """Orchestrates PSNR pretraining, GAN fine-tuning, Codebook Priors, and CUT."""

    def __init__(
        self,
        model: BaseUpscaler,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        criterion: nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        discriminator: nn.Module | None = None,
        optimizer_d: torch.optim.Optimizer | None = None,
        patch_nce_loss: nn.Module | None = None,
        patchnce_weight: float = 1.0,
        vq_weight: float = 0.25,
        gan_weight: float = 0.005,
        device: torch.device | None = None,
        checkpoint_dir: str = "checkpoints",
        stage: str = "psnr",
    ):
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = (
            criterion.to(self.device)
            if criterion is not None
            else CompositeLoss(device=self.device).to(self.device)
        )
        self.optimizer = optimizer or torch.optim.Adam(self.model.parameters(), lr=1e-4)

        # Stage 2 (Adversarial) and Stage 4 components
        self.stage = stage.lower()
        self.discriminator = discriminator.to(self.device) if discriminator is not None else None
        self.optimizer_d = optimizer_d
        if self.discriminator is not None and self.optimizer_d is None:
            self.optimizer_d = torch.optim.Adam(
                self.discriminator.parameters(), lr=1e-4, betas=(0.9, 0.999)
            )

        self.ragan_loss = RaGANLoss().to(self.device)
        self.patch_nce_loss = (
            patch_nce_loss.to(self.device)
            if patch_nce_loss is not None
            else PatchNCELoss().to(self.device)
        )
        self.patchnce_weight = float(patchnce_weight)
        self.vq_weight = float(vq_weight)
        self.gan_weight = float(gan_weight)

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_psnr = -1.0
        self.history: list[dict[str, Any]] = []

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Run one training epoch supporting PSNR and Adversarial modes."""
        self.model.train()
        if self.discriminator is not None:
            self.discriminator.train()

        loss_accum: dict[str, float] = {
            "loss_G": 0.0,
            "loss_D": 0.0,
            "loss_pixel": 0.0,
            "loss_fft": 0.0,
            "loss_vgg": 0.0,
            "loss_grad": 0.0,
            "loss_nce": 0.0,
            "loss_vq": 0.0,
        }
        total_batches = max(1, len(self.train_loader))

        desc = f"Epoch {epoch} [{self.stage.upper()}]"
        pbar = tqdm(self.train_loader, desc=desc, leave=False)

        is_cut_stage = self.stage in ("unpaired_cut", "cut")

        for lr, hr in pbar:
            lr = lr.to(device=self.device, dtype=torch.float32)
            hr = hr.to(device=self.device, dtype=torch.float32)

            # ---------------------------
            # 1. Discriminator Update (if GAN)
            # ---------------------------
            loss_d_val = 0.0
            d_real_for_g = None
            d_fake_for_g = None

            if self.discriminator is not None and self.optimizer_d is not None:
                self.optimizer_d.zero_grad()
                with torch.no_grad():
                    sr_detached = self.model(lr).detach()

                d_real = self.discriminator(hr)
                d_fake = self.discriminator(sr_detached)

                loss_d = self.ragan_loss.discriminator_loss(d_real, d_fake)
                loss_d.backward()
                self.optimizer_d.step()
                loss_d_val = loss_d.item()
                loss_accum["loss_D"] += loss_d_val

            # ---------------------------
            # 2. Generator Update
            # ---------------------------
            self.optimizer.zero_grad()

            if is_cut_stage:
                # Contrastive Unpaired Translation (PatchNCE + GAN)
                sr, feats_q = self.model(lr, return_features=True)
                _, feats_k = self.model.extract_features(lr, encode_only=True)
                loss_nce = self.patch_nce_loss(feats_q[: len(feats_k)], feats_k)
                loss_accum["loss_nce"] += loss_nce.item()

                loss_g = self.patchnce_weight * loss_nce
                if self.discriminator is not None:
                    d_fake_for_g = self.discriminator(sr)
                    with torch.no_grad():
                        d_real_for_g = self.discriminator(hr)
                    loss_gan = self.ragan_loss.generator_loss(d_real_for_g, d_fake_for_g)
                    loss_g = loss_g + self.gan_weight * loss_gan
            else:
                sr = self.model(lr)

                if self.discriminator is not None:
                    d_fake_for_g = self.discriminator(sr)
                    with torch.no_grad():
                        d_real_for_g = self.discriminator(hr)

                if isinstance(self.criterion, CompositeLoss):
                    loss_dict = self.criterion(
                        pred=sr, target=hr, d_real=d_real_for_g, d_fake=d_fake_for_g
                    )
                    loss_g = loss_dict["total"]
                    if "pixel" in loss_dict:
                        loss_accum["loss_pixel"] += loss_dict["pixel"].item()
                    if "fft" in loss_dict:
                        loss_accum["loss_fft"] += loss_dict["fft"].item()
                    if "vgg" in loss_dict:
                        loss_accum["loss_vgg"] += loss_dict["vgg"].item()
                    if "gradient" in loss_dict:
                        loss_accum["loss_grad"] += loss_dict["gradient"].item()
                else:
                    loss_g = self.criterion(sr, hr)
                    if isinstance(loss_g, dict):
                        loss_g = loss_g["total"]

                # Codebook Prior auxiliary VQ loss
                if hasattr(self.model, "last_vq_loss") and self.model.last_vq_loss is not None:
                    vq_l = self.model.last_vq_loss
                    loss_g = loss_g + self.vq_weight * vq_l
                    loss_accum["loss_vq"] += vq_l.item()

            loss_g.backward()
            self.optimizer.step()

            loss_accum["loss_G"] += loss_g.item()

            postfix_str = f"G:{loss_g.item():.4f}"
            if self.discriminator is not None:
                postfix_str += f" D:{loss_d_val:.4f}"
            if is_cut_stage:
                postfix_str += f" NCE:{loss_accum['loss_nce'] / max(1, pbar.n):.4f}"
            elif loss_accum["loss_vq"] > 0:
                postfix_str += f" VQ:{loss_accum['loss_vq'] / max(1, pbar.n):.4f}"
            pbar.set_postfix_str(postfix_str)

        # Average losses over epoch
        epoch_losses = {k: v / total_batches for k, v in loss_accum.items()}
        return epoch_losses

    def validate(self, epoch: int) -> dict[str, float]:
        """Run validation on held-out dataset."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_psnr = 0.0
        total_ssim = 0.0
        count = 0

        with torch.no_grad():
            for lr, hr in tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]", leave=False):
                lr = lr.to(device=self.device, dtype=torch.float32)
                hr = hr.to(device=self.device, dtype=torch.float32)
                sr = self.model(lr)

                for b in range(lr.shape[0]):
                    psnr = calculate_psnr(sr[b], hr[b])
                    ssim = calculate_ssim(sr[b], hr[b])
                    total_psnr += psnr
                    total_ssim += ssim
                    count += 1

        clear_device_cache(self.device)
        return {
            "psnr": total_psnr / max(1, count),
            "ssim": total_ssim / max(1, count),
        }

    def fit(self, epochs: int = 10, save_freq: int = 5) -> dict[str, Any]:
        """Execute full training schedule across epochs."""
        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)

            record = {"epoch": epoch, **train_metrics, **val_metrics}
            self.history.append(record)

            val_str = ""
            if val_metrics:
                val_str = (
                    f" | Val PSNR: {val_metrics['psnr']:.2f} dB, SSIM: {val_metrics['ssim']:.4f}"
                )
            disc_str = ""
            if self.discriminator is not None:
                disc_str = f", Loss D: {train_metrics['loss_D']:.4f}"

            print(
                f"[Epoch {epoch:03d}/{epochs:03d}] Loss G: {train_metrics['loss_G']:.5f}"
                f"{disc_str}{val_str}"
            )
            clear_device_cache(self.device)

            # Checkpoint best model
            current_score = val_metrics.get("psnr", -train_metrics["loss_G"])
            if current_score > self.best_psnr:
                self.best_psnr = current_score
                self.save_checkpoint("best_model.pt", epoch=epoch, metrics=val_metrics)
                self.save_checkpoint("best.pth", epoch=epoch, metrics=val_metrics)

            # Checkpoint latest model
            self.save_checkpoint("latest.pt", epoch=epoch, metrics=val_metrics)
            self.save_checkpoint("latest.pth", epoch=epoch, metrics=val_metrics)

            if epoch % save_freq == 0 or epoch == epochs:
                self.save_checkpoint(f"epoch_{epoch:03d}.pt", epoch=epoch, metrics=val_metrics)

        return {"history": self.history, "best_psnr": self.best_psnr}

    def save_checkpoint(self, filename: str, epoch: int, metrics: dict[str, float]) -> Path:
        """Save weights and state to disk."""
        path = self.checkpoint_dir / filename
        state: dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "stage": self.stage,
        }
        if self.discriminator is not None:
            state["discriminator_state_dict"] = self.discriminator.state_dict()
        if self.optimizer_d is not None:
            state["optimizer_d_state_dict"] = self.optimizer_d.state_dict()

        torch.save(state, path)
        return path

    def load_checkpoint(self, checkpoint_path: str | Path, load_discriminator: bool = True):
        """Load pretrained weights into model and optional discriminator."""
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {path}")

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if (
            load_discriminator
            and self.discriminator is not None
            and "discriminator_state_dict" in checkpoint
        ):
            self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        print(f"Loaded pretrained checkpoint from: {path} (Epoch: {checkpoint.get('epoch', '?')})")
