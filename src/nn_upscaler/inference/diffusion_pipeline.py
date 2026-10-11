"""Production Diffusion Inference Pipeline for Pixel Art to Realism Depixelization.

Supports:
1. Native offline LatentDiffusionUpscaler backend (fast Apple Silicon MPS execution, zero external downloads).
2. Heavy HuggingFace diffusers backend (SDXL ControlNet-Tile + DPM-Solver) with safe lazy loading.
3. Adaptive bilateral dithering suppression filtering (Bayer / Floyd-Steinberg grid noise removal).
4. Category prompt presets (face, armor, character, creature, nature, architecture).
5. Tile weight conditioning balance (0.5 - 1.0) and memory-safe MPS execution with VAE tiling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from nn_upscaler.models.diffusion_adapter import bilateral_dither_filter_image
from nn_upscaler.models.latent_diffusion import LatentDiffusionUpscaler
from nn_upscaler.models.registry import get_model
from nn_upscaler.utils.device import clear_device_cache, get_device
from nn_upscaler.utils.visualization import pil_to_tensor, tensor_to_pil

# Category-specific prompt templates for photorealistic depixelization
CATEGORY_PROMPTS: dict[str, str] = {
    "general": (
        "photorealistic highly detailed photograph, 8k resolution, raw photo, physical materials"
    ),
    "face": (
        "photorealistic close-up portrait, highly detailed human face, natural skin pores, "
        "subsurface scattering, realistic sharp eyes, cinematic lighting, 8k"
    ),
    "armor": (
        "photorealistic knight plate armor, polished steel, realistic metallic reflections, "
        "rivets, fine battle scratches, weathered metal, intricate craftsmanship, 8k"
    ),
    "character": (
        "photorealistic character portrait, highly detailed costume, intricate clothing textures, "
        "realistic fabric weave, studio lighting, sharp features, 8k"
    ),
    "creature": (
        "photorealistic mythical creature, highly detailed organic skin, realistic scales and fur, "
        "sharp eyes, anatomical precision, cinematic lighting, 8k"
    ),
    "monster": (
        "photorealistic fearsome monster, highly detailed organic skin, realistic scales and fur, "
        "sharp eyes, anatomical precision, cinematic lighting, 8k"
    ),
    "nature": (
        "photorealistic lush landscape, highly detailed foliage, tree bark, stone textures, "
        "natural sunlight, 8k landscape photography"
    ),
    "landscape": (
        "photorealistic expansive scenery, natural lighting, realistic rocks and water, "
        "detailed atmospheric perspective, 8k photography"
    ),
    "stone_architecture": (
        "photorealistic medieval stone castle, ancient masonry, weathered bricks, "
        "realistic moss and mortar, dramatic architectural lighting, 8k"
    ),
    "retro_architecture": (
        "photorealistic vintage building, realistic brick textures, weathered facade, "
        "authentic architectural details, natural daylight, 8k"
    ),
    "weapon": (
        "photorealistic forged steel weapon, ornate crossguard, detailed grip and leather wrapping, "
        "sharp metallic edge, realistic reflections, 8k"
    ),
    "vehicle": (
        "photorealistic retro vehicle, metallic automotive paint, realistic chrome details, "
        "glass reflections, highly detailed mechanical parts, 8k"
    ),
    "interior": (
        "photorealistic architectural interior, ambient lighting, realistic wood and textile materials, "
        "8k interior photography"
    ),
}

DEFAULT_NEGATIVE_PROMPT: str = (
    "pixelated, low resolution, 8-bit, 16-bit, blocky, cartoon, blurry, bad anatomy, "
    "artifacts, oversaturated, jpeg compression, drawing, illustration"
)


class DiffusionUpscalePipeline:
    """Production diffusion pipeline for Pixel Art to Realism conversion."""

    def __init__(
        self,
        backend: str = "native",
        device: torch.device | None = None,
        scale_factor: int = 4,
        weights_path: str | Path | None = None,
        controlnet_model_id: str = "xinsir/controlnet-tile-sdxl-1.0",
        base_model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        vae_model_id: str = "madebyollin/sdxl-vae-fp16-fix",
        **kwargs: Any,
    ):
        """
        Args:
            backend: Either 'native' (lightweight repo LatentDiffusionUpscaler) or 'sdxl' (ControlNet-Tile).
            device: Compute device (MPS, CUDA, or CPU).
            scale_factor: Upscaling multiplier (default: 4).
            weights_path: Optional path to checkpoint for native model.
            controlnet_model_id: HuggingFace model ID for ControlNet-Tile (SDXL backend).
            base_model_id: HuggingFace model ID for SDXL Base (SDXL backend).
            vae_model_id: HuggingFace model ID for VAE (SDXL backend).
        """
        self.backend = backend.lower()
        self.device = device or get_device()
        self.scale_factor = scale_factor
        self.weights_path = Path(weights_path) if weights_path else None
        self.controlnet_model_id = controlnet_model_id
        self.base_model_id = base_model_id
        self.vae_model_id = vae_model_id

        # Internal model handles
        self.native_model: LatentDiffusionUpscaler | None = None
        self.sdxl_pipe: Any = None

        if self.backend == "native":
            self._init_native_model(**kwargs)
        elif self.backend == "sdxl":
            # Lazy init will be performed on run() or can be preloaded
            pass
        else:
            raise ValueError(f"Unknown backend '{backend}'. Supported backends: 'native', 'sdxl'.")

    def _init_native_model(self, **kwargs: Any) -> None:
        """Initializes repository's native LatentDiffusionUpscaler."""
        model = get_model("latent_diffusion", scale_factor=self.scale_factor, **kwargs)
        if self.weights_path and self.weights_path.exists():
            ckpt = torch.load(self.weights_path, map_location="cpu")
            state_dict = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state_dict)
        self.native_model = model.to(self.device).eval()

    def _init_sdxl_pipeline(self) -> Any:
        """Lazy loader for HuggingFace SDXL + ControlNet-Tile pipeline."""
        if self.sdxl_pipe is not None:
            return self.sdxl_pipe

        try:
            from diffusers import (
                AutoencoderKL,
                ControlNetModel,
                DPMSolverMultistepScheduler,
                StableDiffusionXLControlNetPipeline,
            )
        except ImportError as err:
            raise ImportError(
                "Backend 'sdxl' requires 'diffusers' and 'transformers'. "
                "Install them with `uv pip install 'nn-upscaler[diffusion]'` or use backend='native'."
            ) from err

        dtype = torch.float32 if self.device.type == "mps" else torch.float16

        controlnet = ControlNetModel.from_pretrained(
            self.controlnet_model_id,
            torch_dtype=dtype,
        )
        vae = AutoencoderKL.from_pretrained(
            self.vae_model_id,
            torch_dtype=dtype,
        )
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            self.base_model_id,
            controlnet=controlnet,
            vae=vae,
            torch_dtype=dtype,
        ).to(self.device)

        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            use_karras_sigmas=True,
            algorithm_type="dpmsolver++",
        )

        if self.device.type == "cuda":
            pipe.enable_model_cpu_offload()
            pipe.enable_vae_tiling()
        elif self.device.type == "mps":
            pipe.enable_vae_tiling()

        self.sdxl_pipe = pipe
        return self.sdxl_pipe

    @classmethod
    def get_category_prompt(cls, category: str | None, custom_prompt: str | None = None) -> str:
        """Constructs an enriched prompt given an optional category and user prompt."""
        cat_lower = (category or "").lower().strip()
        base_cat_prompt = CATEGORY_PROMPTS.get(cat_lower, CATEGORY_PROMPTS["general"])

        if custom_prompt and custom_prompt.strip():
            # Combine custom prompt with category prior
            return f"{custom_prompt.strip()}, {base_cat_prompt}"
        return base_cat_prompt

    def run(
        self,
        image: Image.Image | np.ndarray | torch.Tensor,
        category: str | None = None,
        prompt: str | None = None,
        negative_prompt: str | None = None,
        tile_weight: float = 0.75,
        steps: int = 28,
        guidance_scale: float = 7.0,
        seed: int = 42,
        apply_bilateral_dither_filter: bool = True,
        dither_filter_d: int = 9,
        dither_filter_sigma_color: float = 75.0,
        dither_filter_sigma_space: float = 75.0,
        **kwargs: Any,
    ) -> Image.Image:
        """Executes diffusion-guided depixelization.

        Args:
            image: Input pixel-art sprite.
            category: Optional game category (e.g. 'face', 'armor', 'nature', 'stone_architecture').
            prompt: Text prompt description.
            negative_prompt: Negative text prompt.
            tile_weight: Conditioning weight for tile/structure preservation (0.5 - 1.0).
            steps: Diffusion sampling steps.
            guidance_scale: Classifier-free guidance scale.
            seed: Random seed for deterministic generation.
            apply_bilateral_dither_filter: Whether to apply bilateral filtering against dithering.
            dither_filter_d: Diameter for bilateral filter.
            dither_filter_sigma_color: Color sigma for bilateral filter.
            dither_filter_sigma_space: Spatial coordinate sigma for bilateral filter.

        Returns:
            Upscaled, photorealistic PIL Image.
        """
        # 1. Normalize input to PIL Image
        if isinstance(image, Image.Image):
            pil_sprite = image.convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_sprite = Image.fromarray(image.astype(np.uint8)).convert("RGB")
        elif isinstance(image, torch.Tensor):
            pil_sprite = tensor_to_pil(image.squeeze(0) if image.dim() == 4 else image)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        full_prompt = self.get_category_prompt(category, prompt)
        full_negative_prompt = (
            negative_prompt if negative_prompt is not None else DEFAULT_NEGATIVE_PROMPT
        )

        # 2. Branch according to backend
        if self.backend == "native":
            return self._run_native(
                pil_sprite=pil_sprite,
                category=category,
                prompt=full_prompt,
                tile_weight=tile_weight,
                steps=min(steps, 8)
                if steps > 8
                else steps,  # Native runs blazing fast in 1-8 steps
                seed=seed,
            )

        return self._run_sdxl(
            pil_sprite=pil_sprite,
            prompt=full_prompt,
            negative_prompt=full_negative_prompt,
            tile_weight=tile_weight,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            apply_bilateral_dither_filter=apply_bilateral_dither_filter,
            dither_filter_d=dither_filter_d,
            dither_filter_sigma_color=dither_filter_sigma_color,
            dither_filter_sigma_space=dither_filter_sigma_space,
        )

    def _run_native(
        self,
        pil_sprite: Image.Image,
        category: str | None,
        prompt: str,
        tile_weight: float,
        steps: int,
        seed: int,
    ) -> Image.Image:
        """Inference execution using native LatentDiffusionUpscaler."""
        if self.native_model is None:
            self._init_native_model()
            assert self.native_model is not None

        tensor_in = pil_to_tensor(pil_sprite).unsqueeze(0).to(self.device)
        generator = torch.Generator(device="cpu").manual_seed(seed)

        with torch.no_grad():
            out_tensor = self.native_model(
                tensor_in,
                category=category,
                prompt=prompt,
                steps=steps,
                tile_weight=tile_weight,
                generator=generator,
            )

        clear_device_cache(self.device)
        return tensor_to_pil(out_tensor.squeeze(0))

    def _run_sdxl(
        self,
        pil_sprite: Image.Image,
        prompt: str,
        negative_prompt: str,
        tile_weight: float,
        steps: int,
        guidance_scale: float,
        seed: int,
        apply_bilateral_dither_filter: bool,
        dither_filter_d: int,
        dither_filter_sigma_color: float,
        dither_filter_sigma_space: float,
    ) -> Image.Image:
        """Inference execution using heavy SDXL + ControlNet-Tile pipeline."""
        pipe = self._init_sdxl_pipeline()

        # Target resolution for SDXL
        target_size = (1024, 1024)
        upscaled_nearest = pil_sprite.resize(target_size, Image.Resampling.NEAREST)

        # Bilateral dithering suppression filter
        if apply_bilateral_dither_filter:
            condition_image = bilateral_dither_filter_image(
                upscaled_nearest,
                d=dither_filter_d,
                sigma_color=dither_filter_sigma_color,
                sigma_space=dither_filter_sigma_space,
            )
        else:
            condition_image = upscaled_nearest

        generator = torch.Generator(device="cpu").manual_seed(seed)

        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=condition_image,
            controlnet_conditioning_scale=tile_weight,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]

        clear_device_cache(self.device)
        return result
