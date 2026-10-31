"""Rasterize-back regression metrics (design.md section 9). Requires the optional `resvg-py` package."""

from __future__ import annotations

import io
import re

import cv2
import numpy as np
from PIL import Image

from .colorspace import lab_to_hex, rgb_to_lab

EDGE_DE = 5.0


def render(svg: str, background: str | None) -> np.ndarray:
    """Render with resvg at the SVG's own size; returns uint8 RGB composited over `background` (CSS color)."""
    import resvg_py

    png = resvg_py.svg_to_bytes(svg_string=svg, background=background)
    img = Image.open(io.BytesIO(png))
    if background is None:
        return np.asarray(img.convert("RGBA"))
    return np.asarray(img.convert("RGB"))


def _edge_map(lab: np.ndarray) -> np.ndarray:
    e = np.zeros(lab.shape[:2], dtype=bool)
    dx = np.linalg.norm(lab[:, 1:] - lab[:, :-1], axis=2) > EDGE_DE
    dy = np.linalg.norm(lab[1:] - lab[:-1], axis=2) > EDGE_DE
    e[:, 1:] |= dx
    e[1:] |= dy
    return e


def _chamfer(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric mean distance (px) between two binary edge maps."""
    if not a.any() or not b.any():
        return float("nan")
    da = cv2.distanceTransform((~a).astype(np.uint8), cv2.DIST_L2, 5)
    db = cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2, 5)
    return float(0.5 * (db[a].mean() + da[b].mean()))


def metrics(svg: str, rgb: np.ndarray, c_bg: np.ndarray | None) -> dict[str, float]:
    """Compare the SVG against the original image `rgb` (uint8 H x W x 3).

    The SVG is composited over the detected background color so transparent areas
    compare against what the original shows there.
    """
    bg = lab_to_hex(c_bg) if c_bg is not None else "#000000"
    out = render(svg, bg)
    if out.shape[:2] != rgb.shape[:2]:
        raise ValueError(f"rendered size {out.shape[:2]} != image size {rgb.shape[:2]}")
    lab_a = rgb_to_lab(rgb)
    lab_b = rgb_to_lab(out)
    de = np.linalg.norm(lab_a - lab_b, axis=2)
    ea, eb = _edge_map(lab_a), _edge_map(lab_b)
    return {
        "de_mean": float(de.mean()),
        "de_p95": float(np.percentile(de, 95)),
        "de_p99": float(np.percentile(de, 99)),
        "edge_chamfer_px": _chamfer(ea, eb),
        "paths": svg.count("<path"),
        "beziers": len(re.findall(r"c", "".join(re.findall(r'\sd="([^"]*)"', svg)))),
        "bytes": len(svg.encode("utf-8")),
    }
