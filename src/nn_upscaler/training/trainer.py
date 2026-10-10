"""Trainer class for supervised super-resolution training."""

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from nn_upscaler.evaluation.metrics import calculate_psnr, calculate_ssim
from nn_upscaler.models.base import BaseUpscaler
from nn_upscaler.utils.device import clear_device_cache, get_device


class SuperResolutionTrainer:
    """Orchestrates model training, validation, checkpointing, and metric logging."""

    def __init__(
        self,
        model: BaseUpscaler,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        criterion: nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        device: torch.device | None = None,
        checkpoint_dir: str = "checkpoints",
    ):
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion or nn.L1Loss()
        self.optimizer = optimizer or torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_psnr = -1.0
        self.history = []

    def train_epoch(self, epoch: int) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        for lr, hr in pbar:
            lr, hr = lr.to(self.device), hr.to(self.device)

            self.optimizer.zero_grad()
            sr = self.model(lr)

            loss = self.criterion(sr, hr)
            if isinstance(loss, dict):
                loss_val = loss["total"]
            else:
                loss_val = loss

            loss_val.backward()
            self.optimizer.step()

            total_loss += loss_val.item()
            pbar.set_postfix(loss=f"{loss_val.item():.4f}")

        avg_loss = total_loss / max(1, len(self.train_loader))
        return avg_loss

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
                lr, hr = lr.to(self.device), hr.to(self.device)
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
            train_loss = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)

            record = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
            self.history.append(record)

            val_str = ""
            if val_metrics:
                val_str = f" | Val PSNR: {val_metrics['psnr']:.2f} dB, SSIM: {val_metrics['ssim']:.4f}"
            print(f"[Epoch {epoch:03d}/{epochs:03d}] Train Loss: {train_loss:.5f}{val_str}")

            # Checkpoint best model
            current_psnr = val_metrics.get("psnr", -train_loss)
            if current_psnr > self.best_psnr:
                self.best_psnr = current_psnr
                self.save_checkpoint("best.pth", epoch=epoch, metrics=val_metrics)

            if epoch % save_freq == 0 or epoch == epochs:
                self.save_checkpoint(f"epoch_{epoch:03d}.pth", epoch=epoch, metrics=val_metrics)

        return {"history": self.history, "best_psnr": self.best_psnr}

    def save_checkpoint(self, filename: str, epoch: int, metrics: dict[str, float]) -> Path:
        """Save weights and state to disk."""
        path = self.checkpoint_dir / filename
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": metrics,
            },
            path,
        )
        return path
