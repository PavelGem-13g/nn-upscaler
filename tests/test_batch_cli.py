"""Tests for batch processing CLI utility (scripts/batch_upscale.py)."""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def temp_batch_dirs():
    """Create input and output directories with dummy test sprites."""
    tmp_in = Path(tempfile.mkdtemp(prefix="test_batch_in_"))
    tmp_out = Path(tempfile.mkdtemp(prefix="test_batch_out_"))

    # Create nested folder structure
    (tmp_in / "sprites").mkdir(parents=True, exist_ok=True)
    (tmp_in / "tilesets").mkdir(parents=True, exist_ok=True)

    # 1. RGB sprite
    im1 = Image.new("RGB", (16, 16), color=(255, 0, 0))
    im1.save(tmp_in / "sprites" / "knight.png")

    # 2. RGBA transparent sprite
    im2 = Image.new("RGBA", (16, 16), color=(0, 255, 0, 128))
    im2.save(tmp_in / "sprites" / "ghost.png")

    # 3. Tileset
    im3 = Image.new("RGB", (32, 32), color=(0, 0, 255))
    im3.save(tmp_in / "tilesets" / "grass.png")

    yield tmp_in, tmp_out

    shutil.rmtree(tmp_in, ignore_errors=True)
    shutil.rmtree(tmp_out, ignore_errors=True)


def test_batch_upscale_e2e(temp_batch_dirs):
    """Test full batch upscale execution with hierarchy and manifest generation."""
    input_dir, output_dir = temp_batch_dirs

    cmd = [
        sys.executable,
        "scripts/batch_upscale.py",
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
        "--model",
        "nearest",  # Fast baseline for test speed
        "--scale",
        "4",
        "--recursive",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"Batch upscale failed: {result.stderr}\n{result.stdout}"

    # Verify hierarchy preservation
    out_knight = output_dir / "sprites" / "knight.png"
    out_ghost = output_dir / "sprites" / "ghost.png"
    out_grass = output_dir / "tilesets" / "grass.png"
    manifest_path = output_dir / "manifest.json"

    assert out_knight.exists()
    assert out_ghost.exists()
    assert out_grass.exists()
    assert manifest_path.exists()

    # Check image geometry
    with Image.open(out_knight) as img:
        assert img.size == (64, 64)
    with Image.open(out_ghost) as img:
        assert img.size == (64, 64)
        assert img.mode == "RGBA"
    with Image.open(out_grass) as img:
        assert img.size == (128, 128)

    # Check manifest.json contents
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["total_images"] == 3
    assert manifest["successful_images"] == 3
    assert manifest["failed_images"] == 0
    assert manifest["scale"] == 4
    assert len(manifest["records"]) == 3
