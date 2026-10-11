"""Semantic and Prompt Conditioned Upscaler for Controlled Pixel Art Depixelization.

Allows directing the hallucinated textures (e.g. face skin, polished armor, stone, foliage)
using category IDs, category names, or text prompt embeddings via Adaptive Feature Modulation.
"""

from __future__ import annotations

import hashlib
import math

import torch
from torch import nn

from nn_upscaler.models.base import BaseUpscaler

# Standard preset semantic categories for retro pixel art
DEFAULT_CATEGORIES: list[str] = [
    "general",
    "face",
    "character",
    "armor",
    "nature",
    "stone_architecture",
    "weapon",
    "creature",
    "vehicle",
    "interior",
]

CATEGORY_TO_IDX: dict[str, int] = {cat: i for i, cat in enumerate(DEFAULT_CATEGORIES)}


class SemanticPromptEncoder(nn.Module):
    """Encodes categorical IDs, text embeddings, or prompt strings into conditioning vectors."""

    def __init__(
        self,
        num_categories: int = len(DEFAULT_CATEGORIES),
        text_dim: int = 512,
        cond_dim: int = 64,
    ):
        super().__init__()
        self.num_categories = num_categories
        self.text_dim = text_dim
        self.cond_dim = cond_dim

        # 1. Discrete category embedding
        self.category_embed = nn.Embedding(num_categories, cond_dim)

        # 2. Continuous prompt projection (for external CLIP or precomputed text embeddings)
        self.prompt_proj = nn.Sequential(
            nn.Linear(text_dim, cond_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(cond_dim, cond_dim),
        )

        # 3. Offline pseudo-hash text projection (fallback for offline arbitrary string prompts)
        self.hash_proj = nn.Sequential(
            nn.Linear(128, cond_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(cond_dim, cond_dim),
        )

    def _string_to_hash_tensor(
        self, text: str, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        """Deterministically encode a string into a pseudo-random fixed vector [B, 128]."""
        # Check if known keyword matches
        text_lower = text.lower()
        for cat, idx in CATEGORY_TO_IDX.items():
            if cat in text_lower:
                return self.category_embed(torch.tensor([idx] * batch_size, device=device))

        # Hash-based deterministic representation
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
        rng = torch.Generator().manual_seed(seed)
        vec = torch.randn(1, 128, generator=rng, device=device, dtype=torch.float32)
        vec = vec.repeat(batch_size, 1)
        return self.hash_proj(vec)

    def forward(
        self,
        batch_size: int,
        device: torch.device,
        category: str | int | torch.Tensor | None = None,
        prompt: str | torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Resolve conditioning inputs into a single condition tensor [B, cond_dim]."""
        if cond is not None:
            if cond.dim() == 1:
                cond = cond.unsqueeze(0).repeat(batch_size, 1)
            return cond.to(device=device, dtype=torch.float32)

        if prompt is not None:
            if isinstance(prompt, torch.Tensor):
                p = prompt.to(device=device, dtype=torch.float32)
                if p.dim() == 1:
                    p = p.unsqueeze(0).repeat(batch_size, 1)
                return self.prompt_proj(p)
            if isinstance(prompt, str):
                return self._string_to_hash_tensor(prompt, batch_size, device)

        if category is not None:
            if isinstance(category, str):
                idx = CATEGORY_TO_IDX.get(category.lower(), 0)
                cat_tensor = torch.tensor([idx] * batch_size, device=device)
            elif isinstance(category, int):
                cat_tensor = torch.tensor([category] * batch_size, device=device)
            elif isinstance(category, torch.Tensor):
                cat_tensor = category.to(device=device)
                if cat_tensor.dim() == 0:
                    cat_tensor = cat_tensor.repeat(batch_size)
            else:
                cat_tensor = torch.zeros(batch_size, dtype=torch.long, device=device)
            return self.category_embed(cat_tensor)

        # Default: Unconditional / General domain (index 0)
        default_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
        return self.category_embed(default_idx)


class AdaINResidualBlock(nn.Module):
    """Residual block with Adaptive Instance Normalization / Feature Modulation."""

    def __init__(self, num_features: int, cond_dim: int, res_scale: float = 0.2):
        super().__init__()
        self.num_features = num_features
        self.res_scale = res_scale

        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm2d(num_features, affine=False)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm2d(num_features, affine=False)

        # Condition projection to scale (gamma) and shift (beta)
        self.cond_to_style1 = nn.Linear(cond_dim, num_features * 2)
        self.cond_to_style2 = nn.Linear(cond_dim, num_features * 2)

    def _modulate(
        self,
        x: torch.Tensor,
        style: torch.Tensor,
        norm_layer: nn.InstanceNorm2d,
    ) -> torch.Tensor:
        b, c, _, _ = x.shape
        gamma, beta = style.chunk(2, dim=1)
        gamma = gamma.view(b, c, 1, 1)
        beta = beta.view(b, c, 1, 1)
        normalized = norm_layer(x)
        return normalized * (1.0 + torch.tanh(gamma)) + beta

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        style1 = self.cond_to_style1(cond)
        style2 = self.cond_to_style2(cond)

        res = self.conv1(x)
        res = self.lrelu(self._modulate(res, style1, self.norm1))
        res = self.conv2(res)
        res = self._modulate(res, style2, self.norm2)

        return x + res * self.res_scale


class SemanticConditionedUpscaler(BaseUpscaler):
    """Semantic and Prompt Conditioned Neural Upscaler for Realistic Depixelization.

    Allows injecting high-level domain priors (e.g. skin pores for faces, metallic
    luster for armor, organic grain for foliage) into the upscaling process.
    """

    def __init__(
        self,
        scale_factor: int = 4,
        in_channels: int = 3,
        out_channels: int = 3,
        num_features: int = 64,
        cond_dim: int = 64,
        num_blocks: int = 6,
        text_dim: int = 512,
        res_scale: float = 0.2,
    ):
        super().__init__(scale_factor=scale_factor)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_features = num_features
        self.cond_dim = cond_dim

        # 1. Semantic prompt & category encoder
        self.prompt_encoder = SemanticPromptEncoder(
            num_categories=len(DEFAULT_CATEGORIES),
            text_dim=text_dim,
            cond_dim=cond_dim,
        )

        # 2. Shallow feature extraction
        self.conv_first = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)

        # 3. Conditional residual blocks with AdaIN modulation
        self.blocks = nn.ModuleList(
            [
                AdaINResidualBlock(
                    num_features=num_features,
                    cond_dim=cond_dim,
                    res_scale=res_scale,
                )
                for _ in range(num_blocks)
            ]
        )
        self.conv_after_body = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

        # 4. Multi-scale PixelShuffle upsampler
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

        # 5. Output reconstruction
        self.conv_last = nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        category: str | int | torch.Tensor | None = None,
        prompt: str | torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, C, H, W] in [0, 1].
            category: Category name (e.g. 'armor', 'face') or integer index.
            prompt: Text prompt string or embedding tensor [B, text_dim].
            cond: Direct conditioning tensor [B, cond_dim].

        Returns:
            Upscaled tensor [B, C, H * scale, W * scale] in [0, 1].
        """
        b = x.shape[0]
        c_vec = self.prompt_encoder(
            batch_size=b,
            device=x.device,
            category=category,
            prompt=prompt,
            cond=cond,
        )

        feat_first = self.conv_first(x)
        feat = feat_first
        for block in self.blocks:
            feat = block(feat, c_vec)
        feat = feat_first + self.conv_after_body(feat)

        out = self.upsample(feat)
        return self.conv_last(out).clamp(0.0, 1.0)
