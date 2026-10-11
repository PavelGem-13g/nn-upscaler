"""Unit tests for Phase 4: Semantic Depixelization & Realistic Hallucination.

Tests:
    - Task 4.1: Vector-Quantized Codebook Prior (VectorQuantizer, CodebookUpscaler)
    - Task 4.2: Semantic & Prompt Conditioning (SemanticPromptEncoder, AdaIN, SemanticConditionedUpscaler)
    - Task 4.3: Unpaired Translation (UnpairedPixelArtDataset, PatchNCELoss, CUTUpscaler)
"""

import pytest
import torch

from nn_upscaler.data.unpaired_dataset import UnpairedPixelArtDataset
from nn_upscaler.losses.contrastive import PatchNCELoss
from nn_upscaler.models.codebook_prior import CodebookUpscaler, VectorQuantizer
from nn_upscaler.models.cut import CUTUpscaler, PatchProjectionHead
from nn_upscaler.models.registry import get_model
from nn_upscaler.models.semantic_conditioned import (
    DEFAULT_CATEGORIES,
    SemanticConditionedUpscaler,
    SemanticPromptEncoder,
)
from nn_upscaler.utils.device import get_device

# ==============================================================================
# Task 4.1 Tests: Vector-Quantized Codebook Prior
# ==============================================================================


def test_vector_quantizer_forward_and_gradient():
    """Verify codebook quantization, straight-through estimator, and loss."""
    num_embeddings = 32
    embedding_dim = 16
    vq = VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=embedding_dim, beta=0.25)

    z_e = torch.randn(2, embedding_dim, 8, 8, requires_grad=True)
    z_q, loss, indices = vq(z_e)

    assert z_q.shape == z_e.shape
    assert indices.shape == (2, 8, 8)
    assert indices.min() >= 0
    assert indices.max() < num_embeddings
    assert loss.dim() == 0
    assert loss.item() >= 0.0

    # Test gradient flow back to z_e via STE
    loss.backward()
    assert z_e.grad is not None
    assert torch.isfinite(z_e.grad).all()


@pytest.mark.parametrize("scale", [2, 4, 8])
def test_codebook_upscaler_forward_shapes(scale: int):
    """Verify CodebookUpscaler multi-scale forward pass and output bounds."""
    model = CodebookUpscaler(
        scale_factor=scale,
        num_features=32,
        num_embeddings=64,
        embedding_dim=32,
        num_blocks=2,
    )
    model.eval()

    x = torch.rand(1, 3, 16, 16)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (1, 3, 16 * scale, 16 * scale)
    assert 0.0 <= out.min() <= out.max() <= 1.0


def test_codebook_upscaler_auxiliary_loss():
    """Verify return_vq_loss parameter and last_vq_loss tracking."""
    model = CodebookUpscaler(
        scale_factor=4,
        num_features=32,
        num_embeddings=64,
        embedding_dim=32,
        num_blocks=2,
    )
    x = torch.rand(2, 3, 16, 16)
    out, vq_loss = model(x, return_vq_loss=True)

    assert out.shape == (2, 3, 64, 64)
    assert vq_loss is not None
    assert vq_loss.item() >= 0.0
    assert model.last_vq_loss is not None


# ==============================================================================
# Task 4.2 Tests: Semantic and Prompt Conditioning
# ==============================================================================


def test_semantic_prompt_encoder():
    """Verify categorical index, text prompt string, and tensor projection."""
    device = torch.device("cpu")
    encoder = SemanticPromptEncoder(
        num_categories=len(DEFAULT_CATEGORIES), text_dim=128, cond_dim=32
    )

    # 1. Unconditional default
    c_default = encoder(batch_size=2, device=device)
    assert c_default.shape == (2, 32)

    # 2. Category name
    c_cat_str = encoder(batch_size=2, device=device, category="armor")
    assert c_cat_str.shape == (2, 32)

    # 3. Category int
    c_cat_int = encoder(batch_size=2, device=device, category=1)
    assert c_cat_int.shape == (2, 32)

    # 4. Text prompt string
    c_prompt_str = encoder(batch_size=2, device=device, prompt="warrior with steel armor")
    assert c_prompt_str.shape == (2, 32)

    # 5. External prompt embedding tensor
    p_tensor = torch.randn(2, 128)
    c_prompt_tensor = encoder(batch_size=2, device=device, prompt=p_tensor)
    assert c_prompt_tensor.shape == (2, 32)


@pytest.mark.parametrize("scale", [2, 4, 8])
def test_semantic_conditioned_upscaler_scales(scale: int):
    """Verify SemanticConditionedUpscaler on multiple scale factors."""
    model = SemanticConditionedUpscaler(
        scale_factor=scale,
        num_features=32,
        cond_dim=32,
        num_blocks=2,
        text_dim=64,
    )
    model.eval()
    x = torch.rand(1, 3, 16, 16)

    # Unconditional
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 3, 16 * scale, 16 * scale)
    assert 0.0 <= out.min() <= out.max() <= 1.0


def test_semantic_conditioned_upscaler_modes():
    """Verify category, text prompt, and explicit condition tensors."""
    model = SemanticConditionedUpscaler(
        scale_factor=4,
        num_features=32,
        cond_dim=32,
        num_blocks=2,
        text_dim=64,
    )
    x = torch.rand(2, 3, 16, 16)

    # Category mode
    out_cat = model(x, category="face")
    assert out_cat.shape == (2, 3, 64, 64)

    # Prompt mode
    out_prompt = model(x, prompt="stone wall texture")
    assert out_prompt.shape == (2, 3, 64, 64)

    # Direct cond vector
    cond_vec = torch.randn(2, 32)
    out_cond = model(x, cond=cond_vec)
    assert out_cond.shape == (2, 3, 64, 64)


# ==============================================================================
# Task 4.3 Tests: Unpaired Translation (CUT & PatchNCE)
# ==============================================================================


def test_patch_nce_loss_computation():
    """Verify PatchNCELoss calculation and gradient flow."""
    criterion = PatchNCELoss(tau=0.07, num_patches=32)

    # Features: [B, C, H, W]
    feat_q = torch.randn(2, 32, 8, 8, requires_grad=True)
    feat_k = torch.randn(2, 32, 8, 8)

    loss = criterion(feat_q, feat_k)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0.0

    loss.backward()
    assert feat_q.grad is not None


def test_patch_projection_head():
    """Verify PatchProjectionHead maps features to normalized proj_dim."""
    head = PatchProjectionHead(in_channels=32, proj_dim=64)
    x = torch.randn(2, 32, 8, 8)
    out = head(x)
    assert out.shape == (2, 64, 8, 8)


@pytest.mark.parametrize("scale", [2, 4, 8])
def test_cut_upscaler_multi_scale(scale: int):
    """Verify CUTUpscaler output shapes and feature extraction."""
    model = CUTUpscaler(
        scale_factor=scale,
        num_features=32,
        num_blocks=2,
        proj_dim=32,
    )
    model.eval()
    x = torch.rand(1, 3, 16, 16)

    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 3, 16 * scale, 16 * scale)
    assert 0.0 <= out.min() <= out.max() <= 1.0


def test_cut_upscaler_features_extraction():
    """Verify forward_features and extract_features for contrastive training."""
    model = CUTUpscaler(
        scale_factor=4,
        num_features=32,
        num_blocks=2,
        proj_dim=32,
    )
    x = torch.rand(2, 3, 16, 16)

    # Both image and features
    out, feats_q = model(x, return_features=True)
    assert out.shape == (2, 3, 64, 64)
    assert len(feats_q) == 4

    # Encode only (for keys)
    none_img, feats_k = model.extract_features(x, encode_only=True)
    assert none_img is None
    assert len(feats_k) == 3


def test_unpaired_pixel_art_dataset():
    """Verify UnpairedPixelArtDataset loads and samples from both domains."""
    ds = UnpairedPixelArtDataset(
        domain_x_paths="data/benchmarks/authentic_pixel_art",
        domain_y_paths="data/hr_raw",
        lr_patch_size=16,
        scale_factor=4,
    )
    assert len(ds) >= 1
    lr, hr = ds[0]
    assert lr.shape == (3, 16, 16)
    assert hr.shape == (3, 64, 64)
    assert 0.0 <= lr.min() <= lr.max() <= 1.0
    assert 0.0 <= hr.min() <= hr.max() <= 1.0


# ==============================================================================
# Model Registry & Device Routing Tests
# ==============================================================================


@pytest.mark.parametrize(
    "model_name",
    [
        "codebook_prior",
        "semantic_conditioned",
        "cut_unpaired",
    ],
)
def test_phase4_registry_and_device(model_name: str):
    """Verify Phase 4 models instantiate through registry and route through device."""
    device = get_device()
    model = get_model(
        model_name,
        scale_factor=4,
        num_features=32,
        num_blocks=2,
    ).to(device)
    model.eval()

    x = torch.rand(1, 3, 16, 16, device=device)
    with torch.no_grad():
        out = model(x)

    assert out.device.type == device.type
    assert out.shape == (1, 3, 64, 64)
    assert 0.0 <= out.min() <= out.max() <= 1.0
