"""Region and stroke colors (design.md section 7): robust median fill, optional linear gradient, color unification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

from .colorspace import delta_e
from .params import Params

_MIN_ERODED_SAMPLES = 20
_MIN_GRADIENT_SAMPLES = 20


@dataclass
class Fill:
    lab: np.ndarray  # (3,) flat color (median Lab)
    gradient: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None  # (p0 xy, p1 xy, lab0, lab1)


def region_fills(lab: np.ndarray, labels: np.ndarray, p: Params) -> dict[int, Fill]:
    """Fill for every label >= 1 present in `labels`."""
    h, w = labels.shape
    fills: dict[int, Fill] = {}
    for index, box in enumerate(ndimage.find_objects(labels)):
        if box is None:
            continue
        label = index + 1
        rows, cols = box
        mask = _pad_mask(labels[box] == label, rows.start, rows.stop, cols.start, cols.stop, h, w)
        fills[label] = _fill(lab, mask, rows.start, cols.start, p, p.gradients)
    return fills


def stroke_fill(lab: np.ndarray, pixels: np.ndarray, p: Params) -> Fill:
    """Flat fill of one stroke given its (M, 2) int (row, col) pixels."""
    h, w = lab.shape[:2]
    pixels = np.asarray(pixels, dtype=np.int64)
    r0, c0 = pixels.min(axis=0)
    r1, c1 = pixels.max(axis=0) + 1
    mask = np.zeros((r1 - r0, c1 - c0), dtype=bool)
    mask[pixels[:, 0] - r0, pixels[:, 1] - c0] = True
    return _fill(lab, _pad_mask(mask, r0, r1, c0, c1, h, w), r0, c0, p, False)


def unify_colors(fills: list[Fill], p: Params) -> None:
    """Give flat fills whose colors are within unify_de of a cluster leader the cluster's median color, in place.

    Greedy leader clustering in list order; the cluster color is the unweighted median of its members' colors.
    """
    flat = [f for f in fills if f.gradient is None]
    leaders: list[np.ndarray] = []
    members: list[list[Fill]] = []
    for f in flat:
        if leaders:
            d = delta_e(np.asarray(leaders), f.lab)
            k = int(np.argmin(d))
            if d[k] < p.unify_de:
                members[k].append(f)
                continue
        leaders.append(np.asarray(f.lab, dtype=np.float64))
        members.append([f])
    for group in members:
        if len(group) > 1:
            color = np.median(np.stack([f.lab for f in group]), axis=0)
            for f in group:
                f.lab = color.copy()


def _pad_mask(mask: np.ndarray, r0: int, r1: int, c0: int, c1: int, h: int, w: int) -> np.ndarray:
    """Pad a cropped region mask by 1 px: empty where the crop is inside the image, replicated at the image border.

    Erosion then treats the image border as region interior instead of as a color transition.
    """
    out = np.pad(mask, 1)
    if r0 == 0:
        out[0] = out[1]
    if r1 == h:
        out[-1] = out[-2]
    if c0 == 0:
        out[:, 0] = out[:, 1]
    if c1 == w:
        out[:, -1] = out[:, -2]
    return out


def _fill(lab: np.ndarray, padded: np.ndarray, r0: int, c0: int, p: Params, gradients: bool) -> Fill:
    """Fill of the region given as a 1-px padded crop mask whose unpadded origin is pixel (r0, c0)."""
    mask = padded[1:-1, 1:-1]
    sample = ndimage.distance_transform_edt(padded)[1:-1, 1:-1] > p.px(p.erode_radius)
    if np.count_nonzero(sample) < _MIN_ERODED_SAMPLES:
        sample = skeletonize(mask)
        if not sample.any():
            sample = mask
    rows, cols = np.nonzero(sample)
    colors = lab[rows + r0, cols + c0].astype(np.float64)
    fill = Fill(lab=np.median(colors, axis=0))
    if gradients and len(colors) >= _MIN_GRADIENT_SAMPLES:
        xy = np.stack([cols + c0 + 0.5, rows + r0 + 0.5], axis=1)
        fill.gradient = _linear_gradient(xy, colors, p)
    return fill


def _linear_gradient(
    xy: np.ndarray, colors: np.ndarray, p: Params
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    center = xy.mean(axis=0)
    d = xy - center
    design = np.column_stack([np.ones(len(d)), d])
    coef, *_ = np.linalg.lstsq(design, colors, rcond=None)
    jacobian = coef[1:].T  # (3 channels, 2 spatial)
    _, _, vt = np.linalg.svd(jacobian)
    v = vt[0]
    t = d @ v

    design_1d = np.column_stack([np.ones(len(t)), t])
    coef_1d, *_ = np.linalg.lstsq(design_1d, colors, rcond=None)
    a, b = coef_1d
    t0, t1 = np.percentile(t, [1.0, 99.0])
    span = float(np.linalg.norm(b)) * (t1 - t0)

    var_flat = float(((colors - colors.mean(axis=0)) ** 2).sum(axis=1).mean())
    var_linear = float(((colors - design_1d @ coef_1d) ** 2).sum(axis=1).mean())
    if var_flat <= 0 or span <= p.gradient_min_de or 1.0 - var_linear / var_flat < p.gradient_min_var_drop:
        return None
    return center + t0 * v, center + t1 * v, a + b * t0, a + b * t1
