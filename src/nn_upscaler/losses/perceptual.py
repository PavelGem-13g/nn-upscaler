"""Loss functions for super-resolution and depixelization: Gradient, VGG, and Composite."""

import torch
import torch.nn.functional as F
from torch import nn

from nn_upscaler.losses.frequency import FFTLoss
from nn_upscaler.losses.gan import RaGANLoss
from nn_upscaler.losses.pixel import CharbonnierLoss


class GradientLoss(nn.Module):
    """Computes image gradient difference using Sobel operators to enforce edge consistency."""

    def __init__(self, normalize: bool = True):
        super().__init__()
        self.normalize = normalize
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32
        )
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32
        )
        if normalize:
            sobel_x = sobel_x / 4.0
            sobel_y = sobel_y / 4.0

        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate spatial gradient L1 error between pred and target."""
        c = pred.shape[1]
        sx = self.sobel_x.to(device=pred.device, dtype=pred.dtype).repeat(c, 1, 1, 1)
        sy = self.sobel_y.to(device=pred.device, dtype=pred.dtype).repeat(c, 1, 1, 1)

        # Depthwise convolution across all image channels
        grad_x_pred = F.conv2d(pred, sx, padding=1, groups=c)
        grad_y_pred = F.conv2d(pred, sy, padding=1, groups=c)
        grad_x_tgt = F.conv2d(target, sx, padding=1, groups=c)
        grad_y_tgt = F.conv2d(target, sy, padding=1, groups=c)

        loss_x = F.l1_loss(grad_x_pred, grad_x_tgt)
        loss_y = F.l1_loss(grad_y_pred, grad_y_tgt)
        return loss_x + loss_y


class VGGPerceptualLoss(nn.Module):
    """Multi-layer VGG-19 perceptual feature loss with ImageNet normalization and frozen weights.

    Extracts features from conv1_2, conv2_2, conv3_4, conv4_4, and conv5_4.
    """

    def __init__(
        self,
        layer_weights: dict[str, float] | None = None,
        criterion: str = "l1",
        device: torch.device | None = None,
    ):
        super().__init__()
        from torchvision import models
        from torchvision.models import VGG19_Weights

        default_weights = {
            "conv1_2": 0.1,
            "conv2_2": 0.1,
            "conv3_4": 1.0,
            "conv4_4": 1.0,
            "conv5_4": 1.0,
        }
        self.layer_weights = layer_weights or default_weights
        self.criterion = nn.L1Loss() if criterion == "l1" else nn.MSELoss()

        vgg = models.vgg19(weights=VGG19_Weights.DEFAULT).features
        layer_map = {
            "conv1_2": 3,
            "conv2_2": 8,
            "conv3_4": 17,
            "conv4_4": 26,
            "conv5_4": 35,
        }

        self.stage_names = []
        self.slices = nn.ModuleList()
        prev_idx = 0
        for name, idx in sorted(layer_map.items(), key=lambda item: item[1]):
            if name in self.layer_weights:
                block = nn.Sequential()
                for i in range(prev_idx, idx + 1):
                    block.add_module(str(i), vgg[i])
                self.slices.append(block)
                self.stage_names.append(name)
                prev_idx = idx + 1

        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        )

        if device is not None:
            self.to(device)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute weighted perceptual difference between pred and target.

        Args:
            pred: Generated tensor [B, 3, H, W] in [0, 1].
            target: Reference tensor [B, 3, H, W] in [0, 1].

        Returns:
            Scalar perceptual loss tensor.
        """
        # ImageNet normalization
        pred_norm = (pred - self.mean) / self.std
        target_norm = (target - self.mean) / self.std

        total_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        cur_pred = pred_norm
        cur_target = target_norm

        for block, name in zip(self.slices, self.stage_names, strict=True):
            cur_pred = block(cur_pred)
            with torch.no_grad():
                cur_target = block(cur_target)
            weight = self.layer_weights.get(name, 1.0)
            if weight > 0:
                total_loss = total_loss + weight * self.criterion(cur_pred, cur_target)

        return total_loss


class CompositeLoss(nn.Module):
    """Combines Pixel Loss (Charbonnier/L1), Edge Gradient, Frequency (FFT), VGG, LPIPS, and GAN."""

    def __init__(
        self,
        pixel_type: str = "charbonnier",
        pixel_weight: float = 1.0,
        gradient_weight: float = 0.1,
        fft_weight: float = 0.0,
        vgg_weight: float = 0.0,
        lpips_weight: float = 0.0,
        gan_weight: float = 0.0,
        net_type: str = "alex",
        device: torch.device | None = None,
    ):
        super().__init__()
        self.pixel_type = pixel_type.lower()
        self.pixel_weight = pixel_weight
        self.gradient_weight = gradient_weight
        self.fft_weight = fft_weight
        self.vgg_weight = vgg_weight
        self.lpips_weight = lpips_weight
        self.gan_weight = gan_weight

        # 1. Pixel loss
        if self.pixel_type == "charbonnier":
            self.pixel_loss_fn = CharbonnierLoss()
        else:
            self.pixel_loss_fn = nn.L1Loss()

        # 2. Gradient loss
        self.grad_loss = GradientLoss() if gradient_weight > 0 else None

        # 3. FFT loss
        self.fft_loss = FFTLoss() if fft_weight > 0 else None

        # 4. VGG Perceptual loss
        self.vgg_loss = None
        if vgg_weight > 0:
            try:
                self.vgg_loss = VGGPerceptualLoss(device=device)
            except Exception as e:
                print(f"[Warning] Could not initialize VGGPerceptualLoss: {e}")

        # 5. LPIPS loss
        self.lpips_fn = None
        if lpips_weight > 0:
            try:
                import lpips

                self.lpips_fn = lpips.LPIPS(net=net_type)
                if device is not None:
                    self.lpips_fn = self.lpips_fn.to(device)
            except Exception as e:
                print(f"[Warning] Could not initialize LPIPS: {e}")

        # 6. GAN loss
        self.gan_loss = RaGANLoss() if gan_weight > 0 else None

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        d_real: torch.Tensor | None = None,
        d_fake: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute composite weighted multi-objective loss.

        Args:
            pred: Reconstructed tensor [B, C, H, W].
            target: Ground truth tensor [B, C, H, W].
            d_real: Optional real discriminator logits for adversarial loss.
            d_fake: Optional fake discriminator logits for adversarial loss.

        Returns:
            Dictionary with each component and 'total'.
        """
        losses = {}
        total = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        if self.pixel_weight > 0:
            p_val = self.pixel_loss_fn(pred, target)
            losses["pixel"] = p_val
            total = total + self.pixel_weight * p_val

        if self.grad_loss is not None and self.gradient_weight > 0:
            g_val = self.grad_loss(pred, target)
            losses["gradient"] = g_val
            total = total + self.gradient_weight * g_val

        if self.fft_loss is not None and self.fft_weight > 0:
            f_val = self.fft_loss(pred, target)
            losses["fft"] = f_val
            total = total + self.fft_weight * f_val

        if self.vgg_loss is not None and self.vgg_weight > 0:
            v_val = self.vgg_loss(pred, target)
            losses["vgg"] = v_val
            total = total + self.vgg_weight * v_val

        if self.lpips_fn is not None and self.lpips_weight > 0:
            pred_norm = pred * 2.0 - 1.0
            target_norm = target * 2.0 - 1.0
            l_val = self.lpips_fn(pred_norm, target_norm).mean()
            losses["lpips"] = l_val
            total = total + self.lpips_weight * l_val

        if (
            self.gan_loss is not None
            and self.gan_weight > 0
            and d_real is not None
            and d_fake is not None
        ):
            gan_val = self.gan_loss.generator_loss(d_real, d_fake)
            losses["gan"] = gan_val
            total = total + self.gan_weight * gan_val

        losses["total"] = total
        return losses
