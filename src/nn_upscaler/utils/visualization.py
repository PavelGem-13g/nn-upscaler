"""Visualization and comparison utilities."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a PyTorch image tensor [C, H, W] in [0, 1] to a PIL Image."""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    tensor = tensor.detach().cpu().clamp(0.0, 1.0)
    arr = (tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    if arr.shape[2] == 1:
        return Image.fromarray(arr[:, :, 0], mode="L")
    return Image.fromarray(arr, mode="RGB")


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert a PIL Image to a PyTorch float tensor [C, H, W] in [0, 1]."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return tensor


def create_comparison_grid(
    images: dict[str, Image.Image | torch.Tensor],
    output_path: str | Path | None = None,
    target_height: int | None = None,
    padding: int = 8,
) -> Image.Image:
    """Create a side-by-side comparison image labeled with titles.

    Args:
        images: Dict mapping label string to PIL Image or Tensor.
        output_path: Optional path to save the resulting image.
        target_height: Optional uniform height to rescale all images (nearest neighbor for pixel art).
        padding: Padding between images in pixels.

    Returns:
        PIL Image containing the labeled grid.
    """
    pil_images = {}
    for label, img in images.items():
        if isinstance(img, torch.Tensor):
            pil_images[label] = tensor_to_pil(img)
        else:
            pil_images[label] = img

    # Standardize height if requested, or use max height
    if target_height is None:
        target_height = max(im.height for im in pil_images.values())

    resized_images = []
    labels = []
    header_height = 28

    for label, im in pil_images.items():
        if im.height != target_height:
            scale = target_height / im.height
            new_w = max(1, int(im.width * scale))
            # Use NEAREST to preserve crisp pixel art aesthetic in comparison
            im_resized = im.resize((new_w, target_height), Image.Resampling.NEAREST)
        else:
            im_resized = im
        resized_images.append(im_resized)
        labels.append(label)

    total_width = sum(im.width for im in resized_images) + padding * (len(resized_images) + 1)
    total_height = target_height + header_height + padding * 2

    canvas = Image.new("RGB", (total_width, total_height), color=(24, 24, 28))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    x_offset = padding
    for label, im in zip(labels, resized_images):
        draw.text((x_offset + 4, padding + 4), label, fill=(230, 230, 230), font=font)
        canvas.paste(im, (x_offset, padding + header_height))
        x_offset += im.width + padding

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out_p)

    return canvas
