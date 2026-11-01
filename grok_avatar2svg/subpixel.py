"""Subpixel edge placement (design.md 6.1)."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .colorspace import delta_e
from .geometry import resample
from .params import Params
from .planar import PlanarMap

NORMAL_SIGMA = 2.0
SMOOTH_SIGMA = 1.0
STEP = 0.25
MAX_SHIFT = 1.5
END_GUARD = 2.0


def _smooth(pts: np.ndarray, sigma: float, closed: bool) -> np.ndarray:
    out = ndimage.gaussian_filter1d(pts, sigma, axis=0, mode="wrap" if closed else "nearest")
    if not closed:
        out[0], out[-1] = pts[0], pts[-1]
    return out


def refine_edges(pm: PlanarMap, lab: np.ndarray, medians: np.ndarray, frozen: np.ndarray, p: Params) -> None:
    """Move edge points onto the 50 % blend between the two side colors, then resample at ~1 px.

    Falls back to Gaussian smoothing where the side colors are too close, near nodes, and next to
    `frozen` pixels (stroke pixels whose label was reassigned, so the image there shows the stroke,
    not a blend of the two sides). Edges along the image border stay unchanged.
    """
    h, w = lab.shape[:2]
    near_frozen = ndimage.binary_dilation(frozen, iterations=1) if frozen.any() else frozen
    channels = [np.ascontiguousarray(lab[..., k], dtype=np.float64) for k in range(3)]
    reach = p.px(p.subpixel_range)
    ts = np.arange(-reach, reach + 1e-9, STEP)
    max_shift = p.px(MAX_SHIFT)

    for e in pm.edges:
        if e.left < 0 or e.right < 0:
            continue
        pts = e.points
        closed = e.closed
        if len(pts) < 3:
            continue
        smoothed = _smooth(pts, SMOOTH_SIGMA, closed)
        c_l, c_r = medians[e.left].astype(np.float64), medians[e.right].astype(np.float64)
        if delta_e(c_l, c_r) < p.subpixel_min_de:
            e.points = resample(smoothed, 1.0, closed=closed)
            continue

        guide = _smooth(pts, NORMAL_SIGMA, closed)
        if closed:
            tang = np.roll(guide, -1, axis=0) - np.roll(guide, 1, axis=0)
        else:
            tang = np.gradient(guide, axis=0)
        norm = np.linalg.norm(tang, axis=1, keepdims=True)
        tang = np.divide(tang, norm, out=np.zeros_like(tang), where=norm > 1e-12)
        normal = np.stack([tang[:, 1], -tang[:, 0]], axis=1)  # points to the left label

        q = pts[:, None, :] + ts[None, :, None] * normal[:, None, :]
        coords = [q[..., 1].ravel() - 0.5, q[..., 0].ravel() - 0.5]
        samples = np.stack([ndimage.map_coordinates(ch, coords, order=1, mode="nearest") for ch in channels], axis=-1)
        samples = samples.reshape(len(pts), len(ts), 3)
        diff = c_l - c_r
        g = (samples - c_r) @ diff / float(diff @ diff) - 0.5

        # Rising crossings of alpha = 0.5 (alpha grows toward the left label).
        g0, g1 = g[:, :-1], g[:, 1:]
        rising = (g0 < 0) & (g1 >= 0)
        frac = np.divide(-g0, g1 - g0, out=np.zeros_like(g0), where=rising)
        t_cross = np.where(rising, ts[:-1] + frac * STEP, np.inf)
        k = np.argmin(np.abs(t_cross), axis=1)
        shift = t_cross[np.arange(len(pts)), k]

        ok = np.isfinite(shift) & (np.abs(shift) <= max_shift) & (norm[:, 0] > 1e-12)
        cols = np.clip(np.floor(pts[:, 0]).astype(int), 0, w - 1)
        rows = np.clip(np.floor(pts[:, 1]).astype(int), 0, h - 1)
        ok &= ~near_frozen[rows, cols]
        if not closed:
            s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
            ok &= (s > END_GUARD) & (s < s[-1] - END_GUARD)
        refined = np.where(ok[:, None], pts + np.where(ok, shift, 0.0)[:, None] * normal, smoothed)
        e.points = resample(refined, 1.0, closed=closed)
