"""GPU-accelerated and tensor-native synthetic pixel-art degradation pipeline.

Converts high-resolution realistic images into authentic-looking pixel art
to produce paired datasets for supervised training and testing.
Provides both tensor-native (CUDA / MPS / CPU) and PIL-based operations.
"""

from __future__ import annotations

import random
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

# ---------------------------------------------------------------------------
# Retro Color Palettes (RGB 0-255 uint8)
# ---------------------------------------------------------------------------

PRESET_PALETTES: dict[str, np.ndarray] = {
    # Nintendo Entertainment System (canonical 54 distinct colors)
    "nes": np.array(
        [
            [102, 102, 102],
            [0, 42, 136],
            [20, 18, 167],
            [59, 0, 164],
            [92, 0, 126],
            [110, 0, 64],
            [108, 6, 0],
            [88, 30, 0],
            [50, 60, 0],
            [16, 76, 0],
            [0, 86, 0],
            [0, 84, 44],
            [0, 73, 104],
            [0, 0, 0],
            [173, 173, 173],
            [21, 95, 217],
            [66, 64, 255],
            [117, 39, 254],
            [160, 26, 204],
            [183, 30, 123],
            [181, 49, 32],
            [153, 78, 0],
            [107, 109, 0],
            [56, 135, 0],
            [12, 147, 0],
            [0, 145, 62],
            [0, 133, 147],
            [255, 255, 255],
            [100, 176, 255],
            [144, 148, 255],
            [193, 122, 255],
            [236, 110, 255],
            [254, 114, 201],
            [254, 130, 112],
            [234, 158, 34],
            [188, 190, 0],
            [136, 216, 0],
            [92, 228, 48],
            [69, 224, 130],
            [72, 214, 215],
            [79, 79, 79],
            [192, 223, 255],
            [211, 210, 255],
            [232, 200, 255],
            [251, 194, 255],
            [254, 196, 234],
            [254, 204, 197],
            [247, 216, 165],
            [228, 229, 148],
            [207, 239, 150],
            [189, 244, 171],
            [179, 243, 204],
            [180, 238, 239],
            [184, 184, 184],
        ],
        dtype=np.uint8,
    ),
    # Commodore 64 (16 colors)
    "c64": np.array(
        [
            [0, 0, 0],
            [255, 255, 255],
            [136, 0, 0],
            [170, 255, 238],
            [204, 68, 204],
            [0, 204, 85],
            [0, 0, 170],
            [238, 238, 119],
            [221, 136, 85],
            [102, 68, 0],
            [255, 119, 119],
            [51, 51, 51],
            [119, 119, 119],
            [170, 255, 102],
            [0, 136, 255],
            [187, 187, 187],
        ],
        dtype=np.uint8,
    ),
    # Nintendo Game Boy Original DMG (4 shades of greenish gray)
    "gameboy": np.array(
        [
            [15, 56, 15],
            [48, 98, 48],
            [139, 172, 15],
            [155, 188, 15],
        ],
        dtype=np.uint8,
    ),
    # PICO-8 Fantasy Console (16 colors)
    "pico8": np.array(
        [
            [0, 0, 0],
            [29, 43, 83],
            [126, 37, 83],
            [0, 135, 81],
            [171, 82, 54],
            [95, 87, 79],
            [194, 195, 199],
            [255, 241, 232],
            [255, 0, 77],
            [255, 163, 0],
            [255, 236, 39],
            [0, 228, 54],
            [41, 173, 255],
            [131, 118, 156],
            [255, 119, 168],
            [255, 204, 170],
        ],
        dtype=np.uint8,
    ),
    # Color Graphics Adapter (CGA Mode 1 / 8 standard colors)
    "cga": np.array(
        [
            [0, 0, 0],
            [0, 0, 170],
            [0, 170, 0],
            [0, 170, 170],
            [170, 0, 0],
            [170, 0, 170],
            [170, 85, 0],
            [170, 170, 170],
        ],
        dtype=np.uint8,
    ),
    # Sinclair ZX Spectrum (15 distinct colors: 8 normal + 7 bright, black shared)
    "zx_spectrum": np.array(
        [
            [0, 0, 0],
            [0, 0, 215],
            [215, 0, 0],
            [215, 0, 215],
            [0, 215, 0],
            [0, 215, 215],
            [215, 215, 0],
            [215, 215, 215],
            [0, 0, 255],
            [255, 0, 0],
            [255, 0, 255],
            [0, 255, 0],
            [0, 255, 255],
            [255, 255, 0],
            [255, 255, 255],
        ],
        dtype=np.uint8,
    ),
    # Enhanced Graphics Adapter (EGA 16 colors)
    "ega": np.array(
        [
            [0, 0, 0],
            [0, 0, 170],
            [0, 170, 0],
            [0, 170, 170],
            [170, 0, 0],
            [170, 0, 170],
            [170, 85, 0],
            [170, 170, 170],
            [85, 85, 85],
            [85, 85, 255],
            [85, 255, 85],
            [85, 255, 255],
            [255, 85, 85],
            [255, 85, 255],
            [255, 255, 85],
            [255, 255, 255],
        ],
        dtype=np.uint8,
    ),
}

# Precomputed PyTorch tensor palettes in [0, 1] range float32 [K, 3]
PALETTE_TENSORS: dict[str, torch.Tensor] = {
    name: torch.from_numpy(arr).float() / 255.0 for name, arr in PRESET_PALETTES.items()
}


def get_palette_tensor(
    preset_name: str,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Retrieve a preset retro palette as a float32 PyTorch tensor in [0, 1].

    Args:
        preset_name: Key in PRESET_PALETTES ('nes', 'c64', 'gameboy', 'pico8', 'cga', 'zx_spectrum', 'ega').
        device: PyTorch device ('cpu', 'mps', 'cuda').
        dtype: Desired tensor data type.

    Returns:
        Tensor of shape [K, 3] on the requested device.
    """
    if preset_name not in PALETTE_TENSORS:
        available = ", ".join(PALETTE_TENSORS.keys())
        raise KeyError(f"Unknown palette preset '{preset_name}'. Available: {available}")
    return PALETTE_TENSORS[preset_name].to(device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Tensor-Native Ordered Bayer Dithering
# ---------------------------------------------------------------------------


def get_bayer_matrix(
    size: int = 4,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate recursive Bayer threshold matrix M_N of size 2x2, 4x4, or 8x8.

    M_2 = [[0, 2], [3, 1]]
    M_{2n} = [[4*M_n, 4*M_n + 2], [4*M_n + 3, 4*M_n + 1]]

    Args:
        size: Matrix dimension (2, 4, or 8).
        device: Target compute device.
        dtype: Tensor data type.

    Returns:
        Tensor of shape [size, size] with integer values in [0, size^2 - 1].
    """
    if size == 2:
        return torch.tensor([[0.0, 2.0], [3.0, 1.0]], device=device, dtype=dtype)
    if size == 4:
        m2 = get_bayer_matrix(2, device=device, dtype=dtype)
        top = torch.cat([4.0 * m2, 4.0 * m2 + 2.0], dim=1)
        bottom = torch.cat([4.0 * m2 + 3.0, 4.0 * m2 + 1.0], dim=1)
        return torch.cat([top, bottom], dim=0)
    if size == 8:
        m4 = get_bayer_matrix(4, device=device, dtype=dtype)
        top = torch.cat([4.0 * m4, 4.0 * m4 + 2.0], dim=1)
        bottom = torch.cat([4.0 * m4 + 3.0, 4.0 * m4 + 1.0], dim=1)
        return torch.cat([top, bottom], dim=0)
    raise ValueError(f"Unsupported Bayer size: {size}. Supported sizes: 2, 4, 8.")


def apply_bayer_dithering(
    tensor: torch.Tensor,
    size: int = 4,
    spread: float = 0.05,
) -> torch.Tensor:
    """Apply GPU-accelerated Ordered Bayer Dithering to an image tensor.

    Formula:
        Delta = (M_N + 0.5) / (N^2) - 0.5   (zero-centered threshold offset)
        X_dithered = clamp(X + Delta_bayer * spread, 0.0, 1.0)

    Args:
        tensor: Input tensor of shape [C, H, W] or [B, C, H, W] in [0, 1].
        size: Bayer matrix size (2, 4, or 8).
        spread: Amplitude scaling of the threshold perturbation.

    Returns:
        Dithered tensor with identical shape and device.
    """
    if tensor.ndim not in (3, 4):
        raise ValueError(f"Expected 3D [C, H, W] or 4D [B, C, H, W] tensor, got {tensor.shape}")

    h, w = tensor.shape[-2], tensor.shape[-1]
    m = get_bayer_matrix(size, device=tensor.device, dtype=tensor.dtype)
    delta = (m + 0.5) / float(size * size) - 0.5

    repeats_h = (h + size - 1) // size
    repeats_w = (w + size - 1) // size
    tile = delta.repeat(repeats_h, repeats_w)[:h, :w]

    if tensor.ndim == 3:
        tile = tile.unsqueeze(0)  # [1, H, W]
    else:
        tile = tile.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

    return (tensor + tile * spread).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Vector Palette Quantization on GPU (torch.Tensor)
# ---------------------------------------------------------------------------


class TensorPaletteQuantizer(nn.Module):
    """Vectorized palette color quantizer executed entirely on PyTorch tensors.

    Computes nearest Euclidean distance to target palette without CPU loops:
        argmin_k || x_{b,c,h,w} - p_k ||^2
    via torch.cdist. Runs in milliseconds on GPU / MPS.
    """

    def __init__(
        self,
        palette: torch.Tensor | str | None = None,
        device: torch.device | str = "cpu",
    ):
        """
        Args:
            palette: Optional initial palette name or tensor [K, 3].
            device: Compute device.
        """
        super().__init__()
        self.device = torch.device(device)
        self._palette_cache: dict[str, torch.Tensor] = {}
        if palette is not None:
            self.set_palette(palette)
        else:
            self.current_palette: torch.Tensor | None = None

    def set_palette(self, palette: torch.Tensor | str) -> None:
        """Set the active palette."""
        if isinstance(palette, str):
            self.current_palette = get_palette_tensor(palette, device=self.device)
        else:
            self.current_palette = palette.to(device=self.device, dtype=torch.float32)

    def quantize(
        self,
        tensor: torch.Tensor,
        palette: torch.Tensor | str | None = None,
    ) -> torch.Tensor:
        """Quantize an image tensor to the nearest colors in a palette.

        Args:
            tensor: Image tensor of shape [C, H, W] or [B, C, H, W] in range [0, 1].
                    Supports 3 channels (RGB) or 4 channels (RGBA, preserving alpha).
            palette: Palette name or [K, 3] tensor. If None, uses current_palette.

        Returns:
            Quantized tensor of identical shape and device.
        """
        if palette is not None:
            if isinstance(palette, str):
                pal_tensor = get_palette_tensor(palette, device=tensor.device, dtype=tensor.dtype)
            else:
                pal_tensor = palette.to(device=tensor.device, dtype=tensor.dtype)
        elif self.current_palette is not None:
            pal_tensor = self.current_palette.to(device=tensor.device, dtype=tensor.dtype)
        else:
            raise ValueError("No palette specified and no default palette set.")

        orig_ndim = tensor.ndim
        if orig_ndim == 3:
            t = tensor.unsqueeze(0)  # [1, C, H, W]
        elif orig_ndim == 4:
            t = tensor
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got {tensor.shape}")

        b, c, h, w = t.shape
        # Separate alpha channel if RGBA
        has_alpha = c == 4
        if has_alpha:
            rgb = t[:, :3, :, :]
            alpha = t[:, 3:4, :, :]
        else:
            rgb = t[:, :3, :, :]
            alpha = None

        # Flatten RGB pixels to [N, 3]
        flat_rgb = rgb.permute(0, 2, 3, 1).reshape(-1, 3)

        # Distance matrix [N, K] computed natively via torch.cdist
        # torch.cdist computes Euclidean distances: ||flat_rgb[i] - pal[k]||_2
        dists = torch.cdist(flat_rgb, pal_tensor, p=2.0)
        nearest_indices = torch.argmin(dists, dim=1)  # [N]

        quantized_flat = pal_tensor[nearest_indices]  # [N, 3]
        quantized_rgb = quantized_flat.reshape(b, h, w, 3).permute(0, 3, 1, 2)

        if has_alpha and alpha is not None:
            out = torch.cat([quantized_rgb, alpha], dim=1)
        else:
            out = quantized_rgb

        if orig_ndim == 3:
            return out.squeeze(0)
        return out

    def quantize_adaptive(
        self,
        tensor: torch.Tensor,
        levels: int = 4,
    ) -> torch.Tensor:
        """Quantize image values to a uniform number of levels per channel.

        Produces levels^3 distinct colors across the RGB gamut.
        Args:
            tensor: Image tensor [C, H, W] or [B, C, H, W] in [0, 1].
            levels: Number of quantization levels per channel (e.g. 2, 3, 4, 8).

        Returns:
            Quantized tensor in [0, 1].
        """
        if levels < 2:
            return tensor
        step = levels - 1
        # If RGBA, quantize only RGB
        if tensor.shape[-3] == 4:
            rgb = tensor[..., :3, :, :]
            alpha = tensor[..., 3:4, :, :]
            q_rgb = torch.round(rgb * step) / float(step)
            return torch.cat([q_rgb, alpha], dim=-3)
        return torch.round(tensor * step) / float(step)

    def forward(
        self,
        tensor: torch.Tensor,
        palette: torch.Tensor | str | None = None,
        levels: int | None = None,
    ) -> torch.Tensor:
        """Forward pass for nn.Module compliance."""
        if levels is not None:
            return self.quantize_adaptive(tensor, levels=levels)
        return self.quantize(tensor, palette=palette)


# ---------------------------------------------------------------------------
# Pillow Fallback Quantizers (Backward Compatibility)
# ---------------------------------------------------------------------------


def quantize_palette_pillow(
    img: Image.Image,
    num_colors: int = 32,
    dither: bool = True,
) -> Image.Image:
    """Quantize image colors using Pillow's adaptive median-cut palette with Floyd-Steinberg.

    Args:
        img: Input PIL Image in RGB mode.
        num_colors: Target number of colors (e.g. 16, 32, 64).
        dither: Whether to apply Floyd-Steinberg dithering.

    Returns:
        Quantized RGB PIL Image.
    """
    dither_flag = Image.Dither.FLOYDSTEINBERG if dither else Image.Dither.NONE
    quantized = img.quantize(colors=num_colors, method=Image.Quantize.MEDIANCUT, dither=dither_flag)
    return quantized.convert("RGB")


def quantize_to_palette(img: Image.Image, palette: np.ndarray) -> Image.Image:
    """Map all image pixels to the nearest Euclidean color in a given palette array.

    Args:
        img: Input RGB PIL Image.
        palette: NumPy array of shape [K, 3] with uint8 RGB colors.

    Returns:
        Mapped RGB PIL Image.
    """
    arr = np.array(img, dtype=np.int32)
    h, w, _ = arr.shape
    flat_pixels = arr.reshape(-1, 3)

    dists = np.sum((flat_pixels[:, np.newaxis, :] - palette[np.newaxis, :, :]) ** 2, axis=2)
    nearest_idx = np.argmin(dists, axis=1)
    mapped_pixels = palette[nearest_idx].reshape(h, w, 3)

    return Image.fromarray(mapped_pixels.astype(np.uint8), mode="RGB")


def tensor_downscale(tensor: torch.Tensor, scale_factor: int) -> torch.Tensor:
    """Downscale a 3D or 4D tensor by scale_factor with Apple Silicon MPS compatibility.

    On MPS, adaptive pool requires input sizes to be integer divisible by output sizes.
    If non-divisible, crops to the nearest divisible multiple or falls back gracefully.
    """
    orig_ndim = tensor.ndim
    t = tensor.unsqueeze(0) if orig_ndim == 3 else tensor
    h, w = t.shape[-2], t.shape[-1]
    h_div = max(scale_factor, (h // scale_factor) * scale_factor)
    w_div = max(scale_factor, (w // scale_factor) * scale_factor)
    if h_div != h or w_div != w:
        t = t[..., :h_div, :w_div]
    target_size = (h_div // scale_factor, w_div // scale_factor)
    try:
        out = F.interpolate(t, size=target_size, mode="area")
    except RuntimeError:
        out = F.interpolate(t, size=target_size, mode="bilinear", align_corners=False)
    if orig_ndim == 3:
        return out.squeeze(0)
    return out


# ---------------------------------------------------------------------------
# Dynamic Degradation Pipelines
# ---------------------------------------------------------------------------


class PixelArtDegradation:
    """Full degradation pipeline to transform realistic photos into pixel art."""

    def __init__(
        self,
        scale_factor: int = 4,
        num_colors: int | None = 32,
        palette_preset: str | None = None,
        dither: bool = False,
        dither_mode: Literal["bayer", "floyd_steinberg"] = "floyd_steinberg",
        bayer_size: int = 4,
    ):
        """
        Args:
            scale_factor: Downscaling factor (e.g. 2, 4, 8).
            num_colors: Number of colors for adaptive quantization (if palette_preset is None).
            palette_preset: One of 'nes', 'c64', 'gameboy', 'pico8', 'cga', 'zx_spectrum', 'ega', or None.
            dither: Whether to apply dithering.
            dither_mode: 'bayer' or 'floyd_steinberg'.
            bayer_size: Bayer matrix size (2, 4, 8) when using bayer dithering.
        """
        self.scale_factor = scale_factor
        self.num_colors = num_colors
        self.palette_preset = palette_preset
        self.dither = dither
        self.dither_mode = dither_mode
        self.bayer_size = bayer_size
        self._quantizer = TensorPaletteQuantizer()

    def __call__(self, image: Image.Image | torch.Tensor) -> Image.Image | torch.Tensor:
        """Apply degradation pipeline to a high-resolution PIL Image or PyTorch Tensor.

        Args:
            image: High-res image as PIL.Image or torch.Tensor in [0, 1].

        Returns:
            Low-res pixel-art image in the same container type as input.
        """
        if isinstance(image, torch.Tensor):
            return self._apply_tensor(image)
        return self._apply_pil(image)

    def _apply_pil(self, image: Image.Image) -> Image.Image:
        w, h = image.size
        lr_w = max(1, w // self.scale_factor)
        lr_h = max(1, h // self.scale_factor)

        # 1. Downscale
        lr_img = image.resize((lr_w, lr_h), Image.Resampling.BOX)

        # 2. Dithering (if Bayer selected)
        if self.dither and self.dither_mode == "bayer":
            arr = np.array(lr_img.convert("RGB"), dtype=np.float32) / 255.0
            t = torch.from_numpy(arr).permute(2, 0, 1)  # [3, H, W]
            t = apply_bayer_dithering(t, size=self.bayer_size, spread=0.05)
            arr_dithered = (t.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
            lr_img = Image.fromarray(arr_dithered, mode="RGB")

        # 3. Palette reduction
        if self.palette_preset is not None and self.palette_preset in PRESET_PALETTES:
            lr_img = quantize_to_palette(lr_img, PRESET_PALETTES[self.palette_preset])
        elif self.num_colors is not None and self.num_colors < 256:
            use_fs = self.dither and self.dither_mode == "floyd_steinberg"
            lr_img = quantize_palette_pillow(lr_img, num_colors=self.num_colors, dither=use_fs)

        return lr_img

    def _apply_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        orig_ndim = tensor.ndim
        if orig_ndim == 3:
            t = tensor.unsqueeze(0)
        else:
            t = tensor

        # 1. Downscale via area interpolation (with MPS divisible compatibility)
        lr_t = tensor_downscale(t, self.scale_factor)

        # 2. Dither
        if self.dither and self.dither_mode == "bayer":
            lr_t = apply_bayer_dithering(lr_t, size=self.bayer_size, spread=0.05)

        # 3. Quantize
        if self.palette_preset is not None and self.palette_preset in PALETTE_TENSORS:
            lr_t = self._quantizer.quantize(lr_t, palette=self.palette_preset)
        elif self.num_colors is not None:
            # Map num_colors to uniform channel levels ~ cbrt(num_colors)
            levels = max(2, round(self.num_colors ** (1.0 / 3.0)))
            lr_t = self._quantizer.quantize_adaptive(lr_t, levels=levels)

        if orig_ndim == 3:
            return lr_t.squeeze(0)
        return lr_t


class RandomPixelArtDegradation(nn.Module):
    """Dynamic stochastic degradation generator for on-the-fly training pairs.

    Features:
    - Random downscale factors (e.g. 2x, 4x, 8x).
    - Random palette modes:
        * Retro presets (p = 0.40): 'nes', 'c64', 'gameboy', 'pico8', 'cga', 'zx_spectrum', 'ega'
        * Adaptive levels / color quantization (p = 0.60): 2-8 levels per channel
    - Random dithering modes:
        * Ordered Bayer dithering (40%): random 2x2, 4x4, 8x8 matrix
        * No dithering (40%)
        * Floyd-Steinberg via Pillow C-backend (20%)
    """

    def __init__(
        self,
        scale_factors: list[int] | tuple[int, ...] = (2, 4, 8),
        preset_prob: float = 0.40,
        dither_probs: tuple[float, float, float] = (0.40, 0.40, 0.20),
        bayer_sizes: list[int] | tuple[int, ...] = (2, 4, 8),
    ):
        """
        Args:
            scale_factors: Set of candidate downsampling scales.
            preset_prob: Probability of choosing a retro palette preset vs adaptive levels.
            dither_probs: Probabilities of (Bayer, None, Floyd-Steinberg).
            bayer_sizes: Allowed Bayer matrix dimensions.
        """
        super().__init__()
        self.scale_factors = list(scale_factors)
        self.preset_prob = preset_prob
        self.dither_probs = dither_probs
        self.bayer_sizes = list(bayer_sizes)
        self.preset_names = list(PRESET_PALETTES.keys())
        self.quantizer = TensorPaletteQuantizer()

    def _apply_floyd_steinberg(self, tensor: torch.Tensor, num_colors: int = 16) -> torch.Tensor:
        """Fallback to Pillow C-backend Floyd-Steinberg dithering."""
        device = tensor.device
        dtype = tensor.dtype
        orig_ndim = tensor.ndim
        t = tensor.unsqueeze(0) if orig_ndim == 3 else tensor
        b = t.shape[0]

        outputs = []
        for i in range(b):
            img_np = (
                (t[i, :3].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255.0)
                .round()
                .astype(np.uint8)
            )
            pil_img = Image.fromarray(img_np, mode="RGB")
            quantized = quantize_palette_pillow(pil_img, num_colors=num_colors, dither=True)
            res_np = np.array(quantized, dtype=np.float32) / 255.0
            res_t = torch.from_numpy(res_np).permute(2, 0, 1)  # [3, H, W]
            if t.shape[1] == 4:
                res_t = torch.cat([res_t, t[i, 3:4].cpu()], dim=0)
            outputs.append(res_t)

        out = torch.stack(outputs, dim=0).to(device=device, dtype=dtype)
        if orig_ndim == 3:
            return out.squeeze(0)
        return out

    def forward(
        self,
        image: torch.Tensor | Image.Image,
        forced_scale: int | None = None,
    ) -> torch.Tensor | Image.Image:
        """Apply stochastic pixel-art degradation.

        Args:
            image: High-res image [C, H, W] or [B, C, H, W] tensor in [0, 1], or PIL Image.
            forced_scale: If specified, overrides random scale factor.

        Returns:
            Degraded low-resolution pixel-art image.
        """
        is_pil = isinstance(image, Image.Image)
        if is_pil:
            arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
            x = torch.from_numpy(arr).permute(2, 0, 1)  # [3, H, W]
        else:
            x = image

        # 1. Scale selection
        scale = forced_scale or random.choice(self.scale_factors)
        orig_ndim = x.ndim
        t = x.unsqueeze(0) if orig_ndim == 3 else x

        # 2. Area downsampling with MPS compatibility
        lr = tensor_downscale(t, scale)

        # 3. Dithering mode selection
        p_bayer, p_none, _ = self.dither_probs
        r_dither = random.random()
        dither_choice: str
        if r_dither < p_bayer:
            dither_choice = "bayer"
        elif r_dither < p_bayer + p_none:
            dither_choice = "none"
        else:
            dither_choice = "floyd_steinberg"

        # Apply Bayer dithering prior to quantization
        if dither_choice == "bayer":
            b_size = random.choice(self.bayer_sizes)
            b_spread = random.uniform(0.03, 0.08)
            lr = apply_bayer_dithering(lr, size=b_size, spread=b_spread)

        # 4. Palette selection & quantization
        r_pal = random.random()
        if r_pal < self.preset_prob:
            # Preset retro palette
            chosen_preset = random.choice(self.preset_names)
            lr = self.quantizer.quantize(lr, palette=chosen_preset)
        else:
            # Adaptive quantization
            levels = random.choice([2, 3, 4, 5, 6, 8])
            if dither_choice == "floyd_steinberg":
                # Convert levels to approximate color count L^3
                colors = min(64, levels**3)
                lr = self._apply_floyd_steinberg(lr, num_colors=colors)
            else:
                lr = self.quantizer.quantize_adaptive(lr, levels=levels)

        if orig_ndim == 3:
            lr = lr.squeeze(0)

        if is_pil:
            lr_np = (lr.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
            return Image.fromarray(lr_np, mode="RGB")

        return lr
