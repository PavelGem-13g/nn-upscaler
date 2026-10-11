"""Interactive Web UI for Pixel Art Depixelization and Super-Resolution."""

import argparse
import inspect
import sys
import time
from pathlib import Path
from typing import Any

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import gradio as gr
from PIL import Image

from nn_upscaler.inference.pipeline import InferencePipeline
from nn_upscaler.models.diffusion_adapter import bilateral_dither_filter_image
from nn_upscaler.models.registry import get_model, list_models
from nn_upscaler.utils.device import clear_device_cache, get_device, get_device_name

# Global model cache to avoid re-instantiating heavy models on every click
_MODEL_CACHE: dict[tuple[str, int], Any] = {}

CATEGORY_CHOICES = [
    "None",
    "face",
    "character",
    "armor",
    "creature",
    "monster",
    "nature",
    "landscape",
    "stone_architecture",
    "retro_architecture",
    "weapon",
]


def get_cached_model(model_name: str, scale: int, device: Any):
    """Retrieve or instantiate model for given architecture and scale."""
    key = (model_name.lower(), scale)
    if key not in _MODEL_CACHE:
        model = get_model(model_name, scale_factor=scale)
        model = model.to(device).eval()
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


def upscale_image_ui(
    input_image: Image.Image | None,
    model_name: str,
    scale: int,
    denoising_steps: int,
    tile_weight: float,
    category: str,
    prompt: str,
    apply_bilateral: bool,
    tile_size_str: str,
) -> tuple[tuple[Image.Image, Image.Image] | None, str]:
    """Run interactive inference and format before/after slider pair + telemetry."""
    if input_image is None:
        return None, "⚠️ Please upload or select a pixel art image to begin."

    t0 = time.perf_counter()
    device = get_device()
    device_name = get_device_name(device)

    # 1. Tile size parsing
    tile_size = None if tile_size_str == "None" else int(tile_size_str)

    # 2. Retrieve model & build pipeline
    model = get_cached_model(model_name, scale, device)
    pipeline = InferencePipeline(model=model, device=device, tile_size=tile_size)

    # 3. Handle image channels and optional bilateral filtering
    has_alpha = input_image.mode in ("RGBA", "LA") or (
        input_image.mode == "P" and "transparency" in input_image.info
    )
    if has_alpha:
        img_rgba = input_image.convert("RGBA")
        r, g, b, a = img_rgba.split()
        rgb_img = Image.merge("RGB", (r, g, b))
    else:
        rgb_img = input_image.convert("RGB")

    orig_w, orig_h = rgb_img.width, rgb_img.height

    if apply_bilateral:
        rgb_img = bilateral_dither_filter_image(rgb_img)

    # 4. Prepare extra kwargs
    run_kwargs: dict[str, Any] = {}
    if category != "None":
        run_kwargs["category"] = category
    if prompt.strip():
        run_kwargs["prompt"] = prompt.strip()
    if denoising_steps is not None:
        run_kwargs["steps"] = int(denoising_steps)
    if tile_weight is not None:
        run_kwargs["tile_weight"] = float(tile_weight)

    sig = inspect.signature(model.forward)
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    filtered_kwargs = {k: v for k, v in run_kwargs.items() if has_var_kw or k in sig.parameters}

    # 5. Run inference
    upscaled_rgb = pipeline.run(rgb_img, **filtered_kwargs)

    if has_alpha:
        upscaled_a = a.resize(
            (upscaled_rgb.width, upscaled_rgb.height),
            resample=Image.Resampling.NEAREST,
        )
        r_up, g_up, b_up = upscaled_rgb.split()
        output_image = Image.merge("RGBA", (r_up, g_up, b_up, upscaled_a))
        # Nearest reference for slider comparison
        reference_before = input_image.resize(
            (output_image.width, output_image.height),
            resample=Image.Resampling.NEAREST,
        )
    else:
        output_image = upscaled_rgb
        reference_before = input_image.convert("RGB").resize(
            (output_image.width, output_image.height),
            resample=Image.Resampling.NEAREST,
        )

    latency_ms = (time.perf_counter() - t0) * 1000.0
    clear_device_cache(device)

    # 6. Format telemetry report
    telemetry_md = f"""
### ⚡ Inference Telemetry
- **Hardware Device:** `{device_name}`
- **Model Selected:** `{model_name}` (Scale: `{scale}x`)
- **Input Resolution:** `{orig_w} × {orig_h} px`
- **Output Resolution:** `{output_image.width} × {output_image.height} px`
- **Latency:** `{latency_ms:.1f} ms` ({1000.0 / max(latency_ms, 1e-3):.1f} FPS)
- **Bilateral Dither Filter:** `{"Enabled" if apply_bilateral else "Disabled"}`
"""
    if filtered_kwargs:
        params_str = ", ".join(f"`{k}={v}`" for k, v in filtered_kwargs.items())
        telemetry_md += f"- **Active Conditioning:** {params_str}\n"

    return (reference_before, output_image), telemetry_md


def build_ui() -> gr.Blocks:
    """Construct the Gradio interactive web application."""
    available_models = list_models()
    default_model = "edsr_light" if "edsr_light" in available_models else available_models[0]

    benchmark_dir = Path("data/benchmarks/authentic_pixel_art")
    example_images = []
    if benchmark_dir.exists():
        example_images = sorted([str(p) for p in benchmark_dir.glob("*.png")])

    with gr.Blocks(title="Neural Pixel Art Upscaler") as demo:
        gr.Markdown(
            """
# 🎮 Neural Pixel Art to Realism Upscaler
Transform retro pixel art sprites, tilesets, and textures into high-resolution imagery using
deep convolutional backbones, spatial feature modulation, codebook priors, and latent diffusion.
"""
        )

        with gr.Row():
            # Left Column: Inputs & Controls
            with gr.Column(scale=1):
                input_img = gr.Image(
                    label="Input Pixel Art Sprite",
                    type="pil",
                    image_mode="RGBA",
                )

                with gr.Accordion("⚙️ Model & Architecture Settings", open=True):
                    model_dropdown = gr.Dropdown(
                        label="Upscaler Architecture",
                        choices=available_models,
                        value=default_model,
                    )
                    scale_radio = gr.Radio(
                        label="Upscale Factor",
                        choices=[2, 4, 8],
                        value=4,
                    )
                    bilateral_cb = gr.Checkbox(
                        label="Filter Dithering (Bilateral Artifact Suppression)",
                        value=False,
                    )

                with gr.Accordion("🎨 Semantic & Generative Controls", open=False):
                    category_dropdown = gr.Dropdown(
                        label="Semantic Category Preset",
                        choices=CATEGORY_CHOICES,
                        value="None",
                    )
                    prompt_textbox = gr.Textbox(
                        label="Custom Prompt / Description",
                        placeholder="e.g. photorealistic knight armor, ornate steel, 8k",
                        lines=2,
                    )
                    steps_slider = gr.Slider(
                        label="Denoising Steps (Diffusion)",
                        minimum=1,
                        maximum=12,
                        step=1,
                        value=4,
                    )
                    tile_weight_slider = gr.Slider(
                        label="Structure Conditioning Weight (Tile Weight)",
                        minimum=0.1,
                        maximum=2.0,
                        step=0.05,
                        value=0.75,
                    )
                    tile_size_dropdown = gr.Dropdown(
                        label="Tile Size for Large Textures",
                        choices=["None", "64", "128", "256", "512"],
                        value="None",
                    )

                run_btn = gr.Button("✨ Upscale Image", variant="primary", size="lg")

            # Right Column: Output Split Slider & Telemetry
            with gr.Column(scale=2):
                slider_out = gr.ImageSlider(
                    label="Interactive Before (Pixel Art) vs After (Upscaled)",
                    type="pil",
                    interactive=False,
                )
                telemetry_box = gr.Markdown("### ⚡ Telemetry will appear here after inference.")

        # Examples Gallery
        if example_images:
            gr.Markdown("### 🕹️ Authentic Pixel Art Benchmark Gallery")
            gr.Examples(
                examples=[[img_path] for img_path in example_images],
                inputs=[input_img],
                label="Click any sprite to load into workspace",
            )

        # Event Binding
        run_btn.click(
            fn=upscale_image_ui,
            inputs=[
                input_img,
                model_dropdown,
                scale_radio,
                steps_slider,
                tile_weight_slider,
                category_dropdown,
                prompt_textbox,
                bilateral_cb,
                tile_size_dropdown,
            ],
            outputs=[slider_out, telemetry_box],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Run Neural Upscaler Web UI.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to listen on")
    parser.add_argument("--share", action="store_true", help="Create public shareable link")
    args = parser.parse_args()

    demo = build_ui()
    demo.launch(
        server_name=args.host, server_port=args.port, share=args.share, theme=gr.themes.Soft()
    )


if __name__ == "__main__":
    main()
