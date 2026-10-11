"""SwinIR-Light: Image Restoration Using Swin Transformer for Pixel Art Super-Resolution."""

import math

import torch
import torch.nn.functional as F
from torch import nn

from nn_upscaler.models.base import BaseUpscaler


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition input tensor into non-overlapping spatial windows.

    Args:
        x: Input tensor [B, H, W, C].
        window_size: Size of window (M).

    Returns:
        Windows tensor [B * num_windows, M * M, C].
    """
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, c)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, h: int, w: int) -> torch.Tensor:
    """Reconstruct spatial tensor from windowed representation.

    Args:
        windows: Windows tensor [B * num_windows, M * M, C].
        window_size: Window size (M).
        h: Height of output.
        w: Width of output.

    Returns:
        Reconstructed tensor [B, H, W, C].
    """
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class WindowAttention(nn.Module):
    """Window-based Multi-head Self-Attention (W-MSA / SW-MSA) with relative position bias."""

    def __init__(self, dim: int, window_size: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b_, n, c = x.shape
        qkv = (
            self.qkv(x)
            .reshape(b_, n, 3, self.num_heads, c // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size * self.window_size,
            self.window_size * self.window_size,
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            n_w = mask.shape[0]
            attn = attn.view(b_ // n_w, n_w, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)

        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        return self.proj(out)


class Mlp(nn.Module):
    """Multilayer Perceptron for Transformer block."""

    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class SwinTransformerLayer(nn.Module):
    """Single Swin Transformer Layer (STL) with shifted window attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 8,
        shift_size: int = 0,
        mlp_ratio: float = 2.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        # Convert [B, C, H, W] to [B, H, W, C]
        shortcut = x.permute(0, 2, 3, 1).contiguous()
        x_norm = self.norm1(shortcut)

        shift = self.shift_size if (h > self.window_size and w > self.window_size) else 0

        if shift > 0:
            shifted_x = torch.roll(x_norm, shifts=(-shift, -shift), dims=(1, 2))
            # Dynamic attention mask
            img_mask = torch.zeros((1, h, w, 1), device=x.device)
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -shift),
                slice(-shift, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -shift),
                slice(-shift, None),
            )
            cnt = 0
            for hs in h_slices:
                for ws in w_slices:
                    img_mask[:, hs, ws, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size).view(
                -1, self.window_size * self.window_size
            )
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, (-100.0)).masked_fill(
                attn_mask == 0, 0.0
            )
        else:
            shifted_x = x_norm
            attn_mask = None

        # Window partition & attention
        x_windows = window_partition(shifted_x, self.window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # Reverse window partition
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)

        if shift > 0:
            x_out = torch.roll(shifted_x, shifts=(shift, shift), dims=(1, 2))
        else:
            x_out = shifted_x

        # First residual + MLP second residual
        x_res = shortcut + x_out
        x_res = x_res + self.mlp(self.norm2(x_res))

        # Return [B, C, H, W]
        return x_res.permute(0, 3, 1, 2).contiguous()


class ResidualSwinTransformerBlock(nn.Module):
    """Residual Swin Transformer Block (RSTB) grouping multiple STLs with residual conv."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        depth: int = 2,
        window_size: int = 8,
        mlp_ratio: float = 2.0,
    ):
        super().__init__()
        layers = []
        for i in range(depth):
            shift = 0 if (i % 2 == 0) else window_size // 2
            layers.append(
                SwinTransformerLayer(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=shift,
                    mlp_ratio=mlp_ratio,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        for layer in self.layers:
            res = layer(res)
        return x + self.conv(res)


class SwinIRLightUpscaler(BaseUpscaler):
    """SwinIR-Light: Compact Swin Transformer Architecture for Super-Resolution on MPS.

    Reference:
        Liang et al., 'SwinIR: Image Restoration Using Swin Transformer', ICCVW 2021.

    Features:
        - Residual Swin Transformer Blocks (RSTB) with local and shifted window attention.
        - Window size 8x8 with automatic padding/unpadding for arbitrary dimensions.
        - Native float32 execution optimized for Apple Silicon MPS.
        - Multi-scale PixelShuffle upsampler (2x, 4x, 8x).
    """

    def __init__(
        self,
        scale_factor: int = 4,
        in_channels: int = 3,
        out_channels: int = 3,
        embed_dim: int = 48,
        depths: tuple[int, ...] = (2, 2, 2, 2),
        num_heads: tuple[int, ...] = (6, 6, 6, 6),
        window_size: int = 8,
        mlp_ratio: float = 2.0,
    ):
        super().__init__(scale_factor=scale_factor)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dim = embed_dim
        self.window_size = window_size

        # 1. Shallow feature extraction
        self.conv_first = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)

        # 2. Deep feature extraction (RSTBs)
        self.rstb_blocks = nn.ModuleList(
            [
                ResidualSwinTransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads[i],
                    depth=depths[i],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                )
                for i in range(len(depths))
            ]
        )
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)

        # 3. High quality upsampling
        upsample_layers = []
        if (scale_factor & (scale_factor - 1)) == 0 and scale_factor > 1:
            num_steps = int(math.log2(scale_factor))
            for _ in range(num_steps):
                upsample_layers.extend(
                    [
                        nn.Conv2d(embed_dim, embed_dim * 4, kernel_size=3, padding=1),
                        nn.PixelShuffle(2),
                        nn.LeakyReLU(0.2, inplace=True),
                    ]
                )
        else:
            raise ValueError(f"Unsupported scale_factor: {scale_factor}. Use 2, 4, or 8.")
        self.upsample = nn.Sequential(*upsample_layers)

        # 4. Tail reconstruction
        self.conv_last = nn.Conv2d(embed_dim, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with auto-padding to multiple of window_size."""
        _, _, h_old, w_old = x.shape

        # Reflection pad if not divisible by window_size
        h_pad = (self.window_size - h_old % self.window_size) % self.window_size
        w_pad = (self.window_size - w_old % self.window_size) % self.window_size

        if h_pad > 0 or w_pad > 0:
            x_padded = F.pad(x, (0, w_pad, 0, h_pad), mode="replicate")
        else:
            x_padded = x

        feat_first = self.conv_first(x_padded)
        res = feat_first
        for rstb in self.rstb_blocks:
            res = rstb(res)
        res = self.conv_after_body(res)
        feat = feat_first + res

        out = self.upsample(feat)
        out = self.conv_last(out)

        # Crop padding
        out = out[:, :, : h_old * self.scale_factor, : w_old * self.scale_factor]
        return out.clamp(0.0, 1.0)
