"""Amplitude density remapping and 8-bit grayscale PNG output for SAR quicklooks."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

_LOG10_101 = math.log10(101.0)


def density_remap(
    image: np.ndarray | torch.Tensor,
    *,
    method: str = "sqrt",
    clip_percentile: float = 99.5,
    clip_value: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Map a 2D complex (or real) image's amplitude to an 8-bit grayscale array.

    Formula:
    1. Compute amplitude ``amp = |image|`` in ``float64``.
    2. Determine ``actual_clip``: ``clip_value`` if explicitly supplied, otherwise
       ``np.percentile(amp, clip_percentile)``.
    3. Normalize clipped amplitude to ``[0, 1]``:
       ``a = clip(amp, 0, actual_clip) / actual_clip`` (or ``0`` everywhere if
       ``actual_clip == 0``).
    4. Apply tone curve:
       - ``method == "sqrt"``: ``m = sqrt(a)``
       - ``method == "log"``:  ``m = log10(1 + 100 * a) / log10(101)``
    5. Quantize to ``uint8``: ``rint(clip(m, 0, 1) * 255)``.

    The returned parameter dictionary ``{"method", "clip_percentile", "clip_value"}``
    can be splatted back into ``density_remap(other_image, **params)`` to reproduce
    the exact same amplitude mapping.
    """
    if method not in ("sqrt", "log"):
        raise ValueError(
            f"Unsupported density_remap method {method!r}; expected 'sqrt' or 'log'"
        )
    if not (0.0 < clip_percentile <= 100.0):
        raise ValueError(
            f"clip_percentile must be in (0, 100], got {clip_percentile!r}"
        )

    if isinstance(image, torch.Tensor):
        arr = image.detach().cpu().numpy()
    elif isinstance(image, np.ndarray):
        arr = image
    else:
        raise TypeError(
            f"image must be a numpy.ndarray or torch.Tensor, got {type(image).__name__}"
        )

    if arr.ndim != 2:
        raise ValueError(f"image must be 2D, got shape {arr.shape}")

    amp = np.abs(arr).astype(np.float64, copy=False)

    if clip_value is None:
        actual_clip = float(np.percentile(amp, clip_percentile))
    else:
        actual_clip = float(clip_value)
        if actual_clip < 0.0:
            raise ValueError(f"clip_value must be non-negative, got {actual_clip!r}")

    if actual_clip == 0.0:
        out_u8 = np.zeros(amp.shape, dtype=np.uint8)
    else:
        scaled = np.clip(amp, 0.0, actual_clip) / actual_clip
        if method == "sqrt":
            mapped = np.sqrt(scaled)
        else:
            mapped = np.log10(1.0 + 100.0 * scaled) / _LOG10_101
        out_u8 = np.rint(np.clip(mapped, 0.0, 1.0) * 255.0).astype(np.uint8)

    params: dict[str, Any] = {
        "method": method,
        "clip_percentile": float(clip_percentile),
        "clip_value": actual_clip,
    }
    return out_u8, params


def write_png(path: str | os.PathLike[str], image_uint8: np.ndarray) -> None:
    """Write a 2D ``uint8`` grayscale numpy array to ``path`` as a PNG via Pillow."""
    if not isinstance(image_uint8, np.ndarray):
        raise TypeError(
            f"image_uint8 must be a numpy.ndarray, got {type(image_uint8).__name__}"
        )
    if image_uint8.dtype != np.uint8:
        raise ValueError(
            f"image_uint8 must have dtype uint8, got {image_uint8.dtype}"
        )
    if image_uint8.ndim != 2:
        raise ValueError(
            f"image_uint8 must be 2D, got shape {image_uint8.shape}"
        )

    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "Pillow is required for rdr.io.png.write_png; install the 'viz' extra "
            'with `pip install -e ".[viz]"`.'
        ) from exc

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_uint8).save(out_path, format="PNG")
