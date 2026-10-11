"""Vector-Quantized Codebook Prior Upscaler for Semantic Pixel Art Depixelization.

Inspired by VQ-VAE, CodeFormer, and GFPGAN: utilizes a learned discrete codebook
of high-quality realistic texture priors with straight-through estimator (STE)
and Spatial Feature Transform (SFT) silhouette/color modulation.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from nn_upscaler.models.base import BaseUpscaler


class VectorQuantizer(nn.Module):
    """Vector Quantizer with Straight-Through Estimator (STE) and Commitment Loss.

    Quantizes continuous latent feature maps z_e into nearest codebook entries e_k
    from a learned dictionary Z = {e_k}_{k=1}^K in R^D.

    Loss formula:
        L_vq = ||sg[z_e] - e||_2^2 + beta * ||z_e - sg[e]||_2^2
    """

    def __init__(
        self,
        num_embeddings: int = 512,
        embedding_dim: int = 64,
        beta: float = 0.25,
    ):
        """
        Args:
            num_embeddings: Number of discrete codebook entries (K).
            embedding_dim: Latent channel dimensionality (D).
            beta: Commitment loss weighting factor.
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        # Initialize codebook weights uniformly
        nn.init.uniform_(
            self.embedding.weight,
            -1.0 / math.sqrt(num_embeddings),
            1.0 / math.sqrt(num_embeddings),
        )

    def forward(self, z_e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize continuous latent feature map.

        Args:
            z_e: Continuous latent tensor [B, D, H, W].

        Returns:
            Tuple of:
                - z_q_st: Quantized tensor with straight-through gradient [B, D, H, W].
                - vq_loss: Scalar VQ codebook + commitment loss.
                - indices: Codebook indices [B, H, W].
        """
        b, d, h, w = z_e.shape
        # Permute to [B, H, W, D] and flatten to [B*H*W, D]
        z_perm = z_e.permute(0, 2, 3, 1).contiguous()
        z_flat = z_perm.view(-1, d)

        # Compute Euclidean distance: ||z - e||^2 = ||z||^2 + ||e||^2 - 2 z e^T
        # Fully MPS/Metal compliant tensor ops
        z_sq = torch.sum(z_flat**2, dim=1, keepdim=True)
        e_sq = torch.sum(self.embedding.weight**2, dim=1)
        dists = z_sq + e_sq - 2.0 * torch.matmul(z_flat, self.embedding.weight.t())

        # Find nearest codebook entry
        encoding_indices = torch.argmin(dists, dim=1)
        z_q = self.embedding(encoding_indices).view(b, h, w, d)

        # Codebook loss + Commitment loss
        loss_codebook = F.mse_loss(z_q, z_perm.detach())
        loss_commitment = F.mse_loss(z_perm, z_q.detach())
        vq_loss = loss_codebook + self.beta * loss_commitment

        # Straight-Through Estimator: gradient of z_q_st flows directly to z_e
        z_q_st = z_perm + (z_q - z_perm).detach()
        # Permute back to [B, D, H, W]
        z_q_st = z_q_st.permute(0, 3, 1, 2).contiguous()

        return z_q_st, vq_loss, encoding_indices.view(b, h, w)


class SpatialFeatureTransform(nn.Module):
    """Spatial Feature Transform (SFT) module for texture-to-silhouette alignment."""

    def __init__(self, in_channels: int, cond_channels: int):
        super().__init__()
        self.conv_gamma = nn.Sequential(
            nn.Conv2d(cond_channels, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
        )
        self.conv_beta = nn.Sequential(
            nn.Conv2d(cond_channels, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Modulate feature x with spatial condition cond."""
        gamma = self.conv_gamma(cond)
        beta = self.conv_beta(cond)
        return x * (1.0 + torch.tanh(gamma)) + beta


class SFTResidualBlock(nn.Module):
    """Residual block with Spatial Feature Transform modulation."""

    def __init__(self, channels: int, cond_channels: int, res_scale: float = 0.2):
        super().__init__()
        self.res_scale = res_scale
        self.sft1 = SpatialFeatureTransform(channels, cond_channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.sft2 = SpatialFeatureTransform(channels, cond_channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        res = self.sft1(x, cond)
        res = self.lrelu(self.conv1(res))
        res = self.sft2(res, cond)
        res = self.conv2(res)
        return x + res * self.res_scale


class CodebookUpscaler(BaseUpscaler):
    """Vector-Quantized Codebook Prior Upscaler for Realistic Pixel-Art Depixelization.

    Architecture:
        1. Content Guidance Branch: Extracts structural silhouette & palette features from sprite.
        2. Latent Prior Encoder: Maps sprite into continuous latent space z_e.
        3. Vector Quantizer: Discretizes z_e to nearest realistic texture tokens z_q in codebook Z.
        4. SFT Decoder: Reconstructs realistic textures modulated by the original sprite silhouette.
        5. PixelShuffle Upsampler: Scales up by 2x, 4x, or 8x.
    """

    def __init__(
        self,
        scale_factor: int = 4,
        in_channels: int = 3,
        out_channels: int = 3,
        num_features: int = 64,
        num_embeddings: int = 512,
        embedding_dim: int = 64,
        num_blocks: int = 6,
        beta: float = 0.25,
        res_scale: float = 0.2,
    ):
        super().__init__(scale_factor=scale_factor)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_features = num_features
        self.last_vq_loss: torch.Tensor | None = None

        # 1. Content guidance encoder (preserves original silhouette & palette)
        self.guide_encoder = nn.Sequential(
            nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 2. Latent prior encoder (compresses input into codebook latent space)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features, embedding_dim, kernel_size=3, padding=1),
        )

        # 3. Vector Quantizer
        self.quantizer = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            beta=beta,
        )

        # Feature projection from embedding_dim to num_features
        self.proj_quant = (
            nn.Conv2d(embedding_dim, num_features, kernel_size=3, padding=1)
            if embedding_dim != num_features
            else nn.Identity()
        )

        # 4. SFT Decoder Blocks
        self.decoder_blocks = nn.ModuleList(
            [
                SFTResidualBlock(
                    channels=num_features,
                    cond_channels=num_features,
                    res_scale=res_scale,
                )
                for _ in range(num_blocks)
            ]
        )
        self.conv_after_body = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

        # 5. Multi-scale PixelShuffle upsampler
        upsample_layers: list[nn.Module] = []
        if (scale_factor & (scale_factor - 1)) == 0 and scale_factor > 1:
            num_steps = int(math.log2(scale_factor))
            for _ in range(num_steps):
                upsample_layers.extend(
                    [
                        nn.Conv2d(
                            num_features,
                            num_features * 4,
                            kernel_size=3,
                            padding=1,
                        ),
                        nn.PixelShuffle(2),
                        nn.LeakyReLU(0.2, inplace=True),
                    ]
                )
        else:
            raise ValueError(f"Unsupported scale_factor: {scale_factor}. Supported: 2, 4, 8.")
        self.upsample = nn.Sequential(*upsample_layers)

        # 6. Final output convolution
        self.conv_last = nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        return_vq_loss: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor [B, C, H, W] in [0, 1].
            return_vq_loss: If True, also return the VQ loss tensor.

        Returns:
            Upscaled output [B, C, H * scale, W * scale] in [0, 1],
            or tuple (output, vq_loss) if return_vq_loss=True.
        """
        # Guidance features from sprite
        guide = self.guide_encoder(x)

        # Continuous latent tokens
        z_e = self.encoder(x)

        # Quantize to realistic prior codebook
        z_q, vq_loss, _indices = self.quantizer(z_e)
        self.last_vq_loss = vq_loss

        # Decode & modulate with sprite silhouette
        feat = self.proj_quant(z_q)
        feat_init = feat
        for block in self.decoder_blocks:
            feat = block(feat, guide)
        feat = feat_init + self.conv_after_body(feat)

        out = self.upsample(feat)
        out = self.conv_last(out).clamp(0.0, 1.0)

        if return_vq_loss:
            return out, vq_loss
        return out
