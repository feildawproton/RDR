"""Unit tests for ``rdr.io.png``."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from rdr.io.png import density_remap, write_png


def test_png_module_does_not_import_pillow_at_import_time() -> None:
    # Re-verify that importing rdr.io.png does not pull PIL into its module globals.
    import rdr.io.png as png_mod

    assert "Image" not in vars(png_mod)
    assert "PIL" not in vars(png_mod)


def test_density_remap_sqrt_and_log_and_reproducibility() -> None:
    rng = np.random.default_rng(42)
    real = rng.normal(size=(32, 48)).astype(np.float32)
    imag = rng.normal(size=(32, 48)).astype(np.float32)
    img = real + 1j * imag
    # Plant a large outlier to exercise clipping.
    img[0, 0] = 1000.0 + 0.0j
    img[1, 1] = 0.0 + 0.0j

    u8_sqrt, params_sqrt = density_remap(img, method="sqrt", clip_percentile=99.0)
    assert u8_sqrt.dtype == np.uint8
    assert u8_sqrt.shape == (32, 48)
    assert u8_sqrt[0, 0] == 255
    assert u8_sqrt[1, 1] == 0
    assert params_sqrt["method"] == "sqrt"
    assert params_sqrt["clip_percentile"] == 99.0
    assert 0.0 < params_sqrt["clip_value"] < 1000.0

    # Reproducing with the returned dict must match bit-for-bit, even if we alter
    # another pixel in a way that would otherwise shift the percentile on a sub-chip.
    u8_repro, params_repro = density_remap(img, **params_sqrt)
    assert np.array_equal(u8_sqrt, u8_repro)
    assert params_repro == params_sqrt

    # Sub-chip reproduced using the full-image params dict:
    chip_u8, _ = density_remap(img[:16, :16], **params_sqrt)
    assert np.array_equal(chip_u8, u8_sqrt[:16, :16])

    # Check log mode and torch Tensor input (CPU and CUDA if available).
    t_cpu = torch.from_numpy(img)
    u8_log_np, params_log = density_remap(img, method="log", clip_percentile=99.5)
    u8_log_torch, _ = density_remap(t_cpu, method="log", clip_percentile=99.5)
    assert np.array_equal(u8_log_np, u8_log_torch)
    assert u8_log_np.dtype == np.uint8
    assert u8_log_np.shape == (32, 48)
    assert params_log["method"] == "log"

    if torch.cuda.is_available():
        u8_log_cuda, _ = density_remap(t_cpu.cuda(), method="log", clip_percentile=99.5)
        assert np.array_equal(u8_log_np, u8_log_cuda)


def test_density_remap_zero_image() -> None:
    zeros = np.zeros((8, 10), dtype=np.complex64)
    u8, params = density_remap(zeros)
    assert u8.dtype == np.uint8
    assert u8.shape == (8, 10)
    assert np.all(u8 == 0)
    assert params["clip_value"] == 0.0


def test_write_png_roundtrip(tmp_path: Path) -> None:
    from PIL import Image

    arr = np.arange(256, dtype=np.uint8).reshape(16, 16)
    out_file = tmp_path / "nested" / "test.png"
    write_png(out_file, arr)
    assert out_file.exists()

    loaded = np.asarray(Image.open(out_file))
    assert loaded.dtype == np.uint8
    assert np.array_equal(loaded, arr)


def test_write_png_missing_pillow_error_names_viz_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "PIL", None)
    monkeypatch.setitem(sys.modules, "PIL.Image", None)
    arr = np.zeros((4, 4), dtype=np.uint8)
    with pytest.raises(ImportError, match="viz"):
        write_png(tmp_path / "x.png", arr)
