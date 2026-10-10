"""Unit tests for inference pipeline and tiling."""

import torch
from PIL import Image

from nn_upscaler.inference.pipeline import InferencePipeline
from nn_upscaler.models.classical import NearestNeighborUpscaler
from nn_upscaler.models.edsr_light import EDSRLightUpscaler


def test_inference_pipeline_direct():
    img = Image.new("RGB", (16, 16), color=(255, 0, 0))
    model = NearestNeighborUpscaler(scale_factor=4)
    pipeline = InferencePipeline(model=model, device=torch.device("cpu"))
    out = pipeline.run(img)
    assert out.size == (64, 64)


def test_inference_pipeline_tiled():
    img = Image.new("RGB", (32, 32), color=(0, 255, 0))
    model = EDSRLightUpscaler(scale_factor=2, num_blocks=2, num_features=16)
    pipeline = InferencePipeline(model=model, device=torch.device("cpu"), tile_size=16, tile_overlap=4)
    out = pipeline.run(img)
    assert out.size == (64, 64)
