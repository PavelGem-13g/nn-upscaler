import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr
from PIL import Image

from scripts.web_ui import build_ui, upscale_image_ui


def test_build_ui():
    """Verify that Gradio application initializes with valid component graph."""
    demo = build_ui()
    assert isinstance(demo, gr.Blocks)
    # Check that blocks container holds configured components
    assert len(demo.blocks) > 0


def test_ui_inference_empty_image():
    """Verify graceful handling when user submits without selecting an image."""
    res, telemetry = upscale_image_ui(
        input_image=None,
        model_name="nearest",
        scale=4,
        denoising_steps=4,
        tile_weight=0.75,
        category="None",
        prompt="",
        apply_bilateral=False,
        tile_size_str="None",
    )
    assert res is None
    assert "Please upload" in telemetry


def test_ui_inference_valid_image():
    """Verify before/after comparison pair and telemetry generation."""
    img = Image.new("RGBA", (16, 16), color=(255, 100, 50, 200))
    res, telemetry = upscale_image_ui(
        input_image=img,
        model_name="nearest",
        scale=4,
        denoising_steps=4,
        tile_weight=0.75,
        category="armor",
        prompt="shiny breastplate",
        apply_bilateral=True,
        tile_size_str="None",
    )

    assert res is not None
    before_img, after_img = res
    assert isinstance(before_img, Image.Image)
    assert isinstance(after_img, Image.Image)
    assert before_img.size == (64, 64)
    assert after_img.size == (64, 64)

    # Check telemetry fields
    assert "Inference Telemetry" in telemetry
    assert "16 × 16 px" in telemetry
    assert "64 × 64 px" in telemetry
    assert "nearest" in telemetry
