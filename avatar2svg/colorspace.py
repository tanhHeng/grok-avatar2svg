from __future__ import annotations

import warnings

import numpy as np
from skimage.color import lab2rgb, rgb2lab


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """uint8 sRGB (..., 3) -> float32 CIELAB (D65), L* in 0..100."""
    return rgb2lab(np.asarray(rgb, dtype=np.uint8)).astype(np.float32)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """CIELAB (..., 3) -> uint8 sRGB, out-of-gamut values clipped."""
    lab = np.asarray(lab, dtype=np.float64)
    shape = lab.shape
    with warnings.catch_warnings():
        # lab2rgb warns when a color falls outside the sRGB gamut; clipping is intended.
        warnings.simplefilter("ignore")
        rgb = lab2rgb(lab.reshape(-1, 1, 3)).reshape(shape)
    return np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)


def lab_to_hex(lab: np.ndarray) -> str:
    r, g, b = (int(v) for v in lab_to_rgb(lab))
    return f"#{r:02x}{g:02x}{b:02x}"


def delta_e(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """CIE76 color difference along the last axis."""
    return np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64), axis=-1)
