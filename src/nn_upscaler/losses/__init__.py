"""Loss modules for pixel-art-to-realism neural super-resolution."""

from nn_upscaler.losses.contrastive import PatchNCELoss
from nn_upscaler.losses.frequency import FFTLoss
from nn_upscaler.losses.gan import RaGANLoss
from nn_upscaler.losses.perceptual import CompositeLoss, GradientLoss, VGGPerceptualLoss
from nn_upscaler.losses.pixel import CharbonnierLoss

__all__ = [
    "CharbonnierLoss",
    "CompositeLoss",
    "FFTLoss",
    "GradientLoss",
    "PatchNCELoss",
    "RaGANLoss",
    "VGGPerceptualLoss",
]
