"""Native Latent Diffusion Upscaler (LatentDiffusionUpscaler) for Pixel Art Depixelization.

Implements multi-step latent super-resolution diffusion with:
- Latent autoencoder compressing high-resolution targets into a compact latent space z.
- PixelArtTileAdapter injecting structural silhouette & palette priors.
- SemanticPromptEncoder injecting category & prompt conditioning via FiLM/AdaIN.
- Discrete step DDIM scheduler for fast deterministic sampling (1-8 steps on Apple Silicon MPS).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from nn_upscaler.models.base import BaseUpscaler
from nn_upscaler.models.diffusion_adapter import PixelArtTileAdapter
from nn_upscaler.models.semantic_conditioned import SemanticPromptEncoder


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Embeds integer timesteps [B] into continuous vectors [B, dim]."""
        if timesteps.dim() == 0:
            timesteps = timesteps.unsqueeze(0)
        half_dim = self.dim // 2
        device = timesteps.device
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_dim, dtype=torch.float32, device=device)
            / half_dim
        )
        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class LatentEncoder(nn.Module):
    """Compresses target high-resolution image [B, 3, sH, sW] into latent space [B, C_z, H, W]."""

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        num_features: int = 64,
        scale_factor: int = 4,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        num_down = int(math.log2(scale_factor))

        self.conv_in = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)

        down_layers: list[nn.Module] = []
        ch = num_features
        for _ in range(num_down):
            down_layers.extend(
                [
                    nn.Conv2d(ch, ch * 2, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(ch * 2, ch * 2, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            ch *= 2

        self.down = nn.Sequential(*down_layers)
        self.conv_out = nn.Conv2d(ch, latent_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.leaky_relu(self.conv_in(x), 0.2)
        h = self.down(h)
        return self.conv_out(h)


class LatentDecoder(nn.Module):
    """Reconstructs high-resolution image [B, 3, sH, sW] from latent space [B, C_z, H, W]."""

    def __init__(
        self,
        latent_channels: int = 4,
        out_channels: int = 3,
        num_features: int = 64,
        scale_factor: int = 4,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        num_up = int(math.log2(scale_factor))

        ch = num_features * (2**num_up)
        self.conv_in = nn.Conv2d(latent_channels, ch, kernel_size=3, padding=1)

        up_layers: list[nn.Module] = []
        for _ in range(num_up):
            up_layers.extend(
                [
                    nn.Conv2d(ch, ch * 2, kernel_size=3, padding=1),
                    nn.PixelShuffle(2),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(ch // 2, ch // 2, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            ch //= 2

        self.up = nn.Sequential(*up_layers)
        self.conv_out = nn.Conv2d(ch, out_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = F.leaky_relu(self.conv_in(z), 0.2)
        h = self.up(h)
        return self.conv_out(h)


class FiLMResBlock(nn.Module):
    """Residual block with FiLM (Feature-wise Linear Modulation) conditioning."""

    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        self.film_proj = nn.Linear(cond_dim, channels * 2)

    def forward(self, x: torch.Tensor, cond_vec: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(x))
        h = self.conv1(h)

        film = self.film_proj(cond_vec).unsqueeze(-1).unsqueeze(-1)
        gamma, beta = film.chunk(2, dim=1)
        h = self.norm2(h) * (1.0 + torch.tanh(gamma)) + beta
        h = F.silu(h)
        h = self.conv2(h)

        return x + h


class SpatialSelfAttention(nn.Module):
    """Lightweight spatial self-attention block for non-local texture coherence."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.channels = channels
        self.head_dim = channels // num_heads

        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        norm_x = self.norm(x)
        qkv = self.qkv(norm_x).view(b, 3, self.num_heads, self.head_dim, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # [B, heads, head_dim, HW]

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q.transpose(-2, -1), k) * scale  # [B, heads, HW, HW]
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(v, attn.transpose(-2, -1))  # [B, heads, head_dim, HW]
        out = out.contiguous().view(b, c, h, w)
        return x + self.proj_out(out)


class LatentDenoiser(nn.Module):
    """Conditional Latent Denoiser network predicting noise epsilon_theta."""

    def __init__(
        self,
        latent_channels: int = 4,
        spatial_cond_channels: int = 64,
        cond_vec_dim: int = 64,
        num_features: int = 64,
        num_blocks: int = 4,
    ):
        super().__init__()
        self.latent_channels = latent_channels

        # Combine noisy latent and spatial sprite condition
        self.conv_in = nn.Conv2d(
            latent_channels + spatial_cond_channels,
            num_features,
            kernel_size=3,
            padding=1,
        )

        # Timestep + semantic condition projection
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(num_features),
            nn.Linear(num_features, cond_vec_dim),
            nn.SiLU(),
            nn.Linear(cond_vec_dim, cond_vec_dim),
        )

        self.blocks = nn.ModuleList(
            [FiLMResBlock(num_features, cond_vec_dim) for _ in range(num_blocks)]
        )
        self.attn = SpatialSelfAttention(num_features, num_heads=4)

        self.norm_out = nn.GroupNorm(8, num_features)
        self.conv_out = nn.Conv2d(num_features, latent_channels, kernel_size=3, padding=1)

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        spatial_cond: torch.Tensor,
        semantic_vec: torch.Tensor,
    ) -> torch.Tensor:
        """Predicts noise tensor epsilon [B, C_z, H, W].

        Args:
            z_t: Noisy latent tensor [B, C_z, H, W].
            t: Timesteps [B] (integer tensor).
            spatial_cond: Spatial condition feature map from sprite [B, C_cond, H, W].
            semantic_vec: Category/prompt vector [B, C_vec].

        Returns:
            Predicted noise [B, C_z, H, W].
        """
        # Form joint condition vector
        t_emb = self.time_embed(t)
        cond_vec = t_emb + semantic_vec

        # Concatenate noisy latent and spatial condition
        h = torch.cat([z_t, spatial_cond], dim=1)
        h = self.conv_in(h)

        for i, block in enumerate(self.blocks):
            h = block(h, cond_vec)
            if i == len(self.blocks) // 2:
                h = self.attn(h)

        h = F.silu(self.norm_out(h))
        return self.conv_out(h)


class DDIMScheduler:
    """Discrete DDIM (Denoising Diffusion Implicit Models) step scheduler."""

    def __init__(
        self,
        train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ):
        self.train_timesteps = train_timesteps
        betas = torch.linspace(beta_start, beta_end, train_timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod

    def get_timesteps(self, num_inference_steps: int) -> list[int]:
        """Generate evenly spaced sub-sequence of discrete inference timesteps."""
        if num_inference_steps >= self.train_timesteps:
            return list(range(self.train_timesteps - 1, -1, -1))
        steps = np.linspace(0, self.train_timesteps - 1, num_inference_steps, dtype=int)
        return sorted(set(steps), reverse=True)


class LatentDiffusionUpscaler(BaseUpscaler):
    """Latent Diffusion Upscaler for Realistic Pixel Art Depixelization.

    Combines:
    - Lightweight latent space representation for fast Apple Silicon MPS compute.
    - PixelArtTileAdapter for boundary-preserving silhouette and palette retention.
    - SemanticPromptEncoder for steerable realism across gaming domains.
    - Fast 1-8 step DDIM sampling for real-time inference.
    """

    def __init__(
        self,
        scale_factor: int = 4,
        latent_channels: int = 4,
        num_features: int = 64,
        cond_dim: int = 64,
        num_steps: int = 4,
        train_timesteps: int = 1000,
        tile_weight: float = 0.75,
    ):
        super().__init__(scale_factor=scale_factor)
        self.latent_channels = latent_channels
        self.num_features = num_features
        self.cond_dim = cond_dim
        self.num_steps = num_steps
        self.train_timesteps = train_timesteps
        self.default_tile_weight = tile_weight

        # 1. Latent Autoencoder
        self.encoder = LatentEncoder(
            in_channels=3,
            latent_channels=latent_channels,
            num_features=num_features,
            scale_factor=scale_factor,
        )
        self.decoder = LatentDecoder(
            latent_channels=latent_channels,
            out_channels=3,
            num_features=num_features,
            scale_factor=scale_factor,
        )

        # 2. Conditioning Modules
        self.tile_adapter = PixelArtTileAdapter(
            in_channels=3,
            cond_channels=num_features,
            out_channels=num_features,
            num_blocks=3,
            use_bilateral_filter=True,
        )
        self.prompt_encoder = SemanticPromptEncoder(
            cond_dim=cond_dim,
        )

        # 3. Latent Denoiser
        self.denoiser = LatentDenoiser(
            latent_channels=latent_channels,
            spatial_cond_channels=num_features,
            cond_vec_dim=cond_dim,
            num_features=num_features,
            num_blocks=4,
        )

        # 4. Discrete Step Scheduler
        self.scheduler = DDIMScheduler(train_timesteps=train_timesteps)

    def encode(self, hr_image: torch.Tensor) -> torch.Tensor:
        """Compresses high-resolution ground truth into latent space."""
        return self.encoder(hr_image)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Reconstructs high-resolution RGB image from latent space."""
        return self.decoder(latent).clamp(0.0, 1.0)

    def sample_ddim(
        self,
        spatial_cond: torch.Tensor,
        semantic_vec: torch.Tensor,
        steps: int = 4,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Runs fast iterative DDIM reverse diffusion process.

        Args:
            spatial_cond: Spatial condition feature map [B, C, H, W].
            semantic_vec: Categorical / prompt vector [B, C_vec].
            steps: Number of discrete sampling steps (e.g. 1, 2, 4, 8).
            generator: Deterministic PyTorch random generator.

        Returns:
            Decoded clean latent z_0 [B, C_z, H, W].
        """
        b, _, h, w = spatial_cond.shape
        device = spatial_cond.device

        # Sample initial latent noise z_T
        if generator is not None and generator.device.type != device.type:
            z_t = torch.randn(
                b,
                self.latent_channels,
                h,
                w,
                device="cpu",
                dtype=torch.float32,
                generator=generator,
            ).to(device)
        else:
            z_t = torch.randn(
                b,
                self.latent_channels,
                h,
                w,
                device=device,
                dtype=torch.float32,
                generator=generator,
            )

        timesteps = self.scheduler.get_timesteps(steps)
        alphas_cumprod = self.scheduler.alphas_cumprod.to(device)

        for i, t_val in enumerate(timesteps):
            t_tensor = torch.full((b,), t_val, device=device, dtype=torch.long)
            alpha_t = alphas_cumprod[t_val]

            # Denoiser predicts noise epsilon
            pred_eps = self.denoiser(z_t, t_tensor, spatial_cond, semantic_vec)

            # Predict clean latent z_0
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)
            pred_z0 = (z_t - sqrt_one_minus_alpha_t * pred_eps) / sqrt_alpha_t.clamp(min=1e-5)
            pred_z0 = pred_z0.clamp(-3.0, 3.0)

            # Advance to next timestep
            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
                alpha_next = alphas_cumprod[t_next]
                sqrt_alpha_next = torch.sqrt(alpha_next)
                sqrt_one_minus_alpha_next = torch.sqrt(1.0 - alpha_next)
                z_t = sqrt_alpha_next * pred_z0 + sqrt_one_minus_alpha_next * pred_eps
            else:
                z_t = pred_z0

        return z_t

    def forward(
        self,
        x: torch.Tensor,
        category: str | int | torch.Tensor | None = None,
        prompt: str | torch.Tensor | None = None,
        steps: int | None = None,
        tile_weight: float | None = None,
        generator: torch.Generator | None = None,
        target: torch.Tensor | None = None,
        return_loss: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forward pass for inference or training.

        Args:
            x: Input pixel-art sprite [B, 3, H, W] in [0, 1].
            category: Optional semantic category name or index.
            prompt: Optional text prompt or embedding tensor.
            steps: Number of DDIM inference steps (default: self.num_steps).
            tile_weight: Condition weight multiplier (default: self.default_tile_weight).
            generator: Deterministic PyTorch random generator.
            target: Ground truth high-resolution image [B, 3, sH, sW] for training.
            return_loss: If True, computes diffusion and reconstruction training loss.

        Returns:
            Upscaled image [B, 3, sH, sW] or tuple (upscaled, loss_dict).
        """
        b = x.shape[0]
        device = x.device
        w_tile = self.default_tile_weight if tile_weight is None else float(tile_weight)
        num_s = self.num_steps if steps is None else max(1, int(steps))

        # 1. Condition extraction
        spatial_cond = self.tile_adapter(x, tile_weight=w_tile)
        semantic_vec = self.prompt_encoder(
            batch_size=b,
            device=device,
            category=category,
            prompt=prompt,
        )

        # 2. Training mode: calculate diffusion loss
        if return_loss and target is not None:
            # Encode ground truth to target latent
            z_0 = self.encode(target)

            # Sample random timestep
            if generator is not None and generator.device.type != device.type:
                t_rand = torch.randint(
                    0, self.train_timesteps, (b,), device="cpu", generator=generator
                ).to(device)
                eps = torch.randn(
                    z_0.shape, device="cpu", dtype=torch.float32, generator=generator
                ).to(device)
            else:
                t_rand = torch.randint(
                    0, self.train_timesteps, (b,), device=device, generator=generator
                )
                eps = torch.randn_like(z_0, generator=generator)

            alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
            alpha_t = alphas_cumprod[t_rand].view(b, 1, 1, 1)
            z_t = torch.sqrt(alpha_t) * z_0 + torch.sqrt(1.0 - alpha_t) * eps

            # Predict noise
            pred_eps = self.denoiser(z_t, t_rand, spatial_cond, semantic_vec)
            loss_diff = F.mse_loss(pred_eps, eps)

            # Reconstruction loss of autoencoder
            recon = self.decode(z_0)
            loss_recon = F.l1_loss(recon, target)

            total_loss = loss_diff + 0.1 * loss_recon
            loss_dict = {
                "loss": total_loss,
                "loss_diff": loss_diff,
                "loss_recon": loss_recon,
            }
            return recon, loss_dict

        # 3. Inference mode: fast DDIM sampling
        z_sample = self.sample_ddim(
            spatial_cond=spatial_cond,
            semantic_vec=semantic_vec,
            steps=num_s,
            generator=generator,
        )
        return self.decode(z_sample)
