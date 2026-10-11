"""Adversarial loss functions (Relativistic Average GAN)."""

import torch
import torch.nn.functional as F
from torch import nn


class RaGANLoss(nn.Module):
    """Relativistic Average GAN (RaGAN) loss for generator and discriminator.

    Reference:
        Jolicoeur-Martineau, 'The relativistic discriminator: a key element towards
        realistic GANs', ICLR 2019.

    Formulas:
        L_D = -E_xr [log(sigmoid(D(xr) - E_xf[D(xf)]))]
              -E_xf [log(1 - sigmoid(D(xf) - E_xr[D(xr)]))]
        L_G = -E_xf [log(sigmoid(D(xf) - E_xr[D(xr)]))]
              -E_xr [log(1 - sigmoid(D(xr) - E_xf[D(xf)]))]
    """

    def __init__(self, loss_weight: float = 1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def discriminator_loss(self, d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
        """Compute RaGAN loss for discriminator update.

        Args:
            d_real: Discriminator output logits for real images [B, ...].
            d_fake: Discriminator output logits for fake images [B, ...].

        Returns:
            Scalar loss for discriminator.
        """
        # Relativistic logits
        real_pred = d_real - torch.mean(d_fake)
        fake_pred = d_fake - torch.mean(d_real)

        loss_real = F.binary_cross_entropy_with_logits(real_pred, torch.ones_like(real_pred))
        loss_fake = F.binary_cross_entropy_with_logits(fake_pred, torch.zeros_like(fake_pred))

        return self.loss_weight * (loss_real + loss_fake) * 0.5

    def generator_loss(self, d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
        """Compute RaGAN loss for generator update.

        Args:
            d_real: Discriminator output logits for real images [B, ...].
            d_fake: Discriminator output logits for fake images [B, ...].

        Returns:
            Scalar loss for generator.
        """
        real_pred = d_real - torch.mean(d_fake)
        fake_pred = d_fake - torch.mean(d_real)

        loss_fake = F.binary_cross_entropy_with_logits(fake_pred, torch.ones_like(fake_pred))
        loss_real = F.binary_cross_entropy_with_logits(real_pred, torch.zeros_like(real_pred))

        return self.loss_weight * (loss_fake + loss_real) * 0.5

    def forward(
        self, d_real: torch.Tensor, d_fake: torch.Tensor, for_discriminator: bool = True
    ) -> torch.Tensor:
        """Unified forward method."""
        if for_discriminator:
            return self.discriminator_loss(d_real, d_fake)
        else:
            return self.generator_loss(d_real, d_fake)
