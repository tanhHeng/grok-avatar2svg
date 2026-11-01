"""Background detection (design.md section 4): border-connected regions close to the border color."""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .colorspace import delta_e
from .params import Params
from .regions import adjacent_pairs, region_medians, relabel_sequential

_BORDER_CLUSTER_DE = 2.5


def find_background(lab: np.ndarray, labels: np.ndarray, p: Params) -> tuple[np.ndarray, np.ndarray | None]:
    """Bool per label marking background regions, and the background color c_bg (None without opaque border).

    Background regions are found only through adjacency starting from regions on the image border, never by
    global color matching, so enclosed regions of background color (e.g. eyes) survive.
    """
    n = int(labels.max()) + 1
    border = np.zeros(labels.shape, dtype=bool)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    border &= labels > 0
    if not border.any():
        return np.zeros(n, dtype=bool), None
    c_bg = _border_color(lab[border])

    medians = region_medians(lab, labels)
    near = delta_e(medians, c_bg) < p.bg_de
    near[0] = False
    on_border = np.zeros(n, dtype=bool)
    on_border[labels[border]] = True

    pairs = adjacent_pairs(labels)
    pairs = pairs[near[pairs[:, 0]] & near[pairs[:, 1]]]
    graph = coo_matrix((np.ones(len(pairs), dtype=np.int8), (pairs[:, 0], pairs[:, 1])), shape=(n, n)).tocsr()
    _, comp = connected_components(graph, directed=False)
    seed_comps = np.unique(comp[near & on_border])
    return near & np.isin(comp, seed_comps), c_bg


def drop_background(labels: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Set background regions (bool per label, as from find_background) to 0 and relabel the rest 1..N."""
    out = labels.copy()
    out[background[labels]] = 0
    return relabel_sequential(out)


def _border_color(colors: np.ndarray) -> np.ndarray:
    """Median of the largest delta_e-2.5 cluster among (M, 3) Lab colors.

    The cluster is centered on the color with the most neighbors within the radius; its members are taken from
    the full set and their median is returned.
    """
    colors = np.asarray(colors, dtype=np.float64)
    step = max(1, len(colors) // 4000)
    sample = colors[::step]
    counts = np.zeros(len(sample), dtype=np.int64)
    for start in range(0, len(sample), 512):
        block = sample[start : start + 512]
        counts[start : start + 512] = (delta_e(block[:, None, :], sample[None, :, :]) < _BORDER_CLUSTER_DE).sum(axis=1)
    center = sample[int(np.argmax(counts))]
    return np.median(colors[delta_e(colors, center) < _BORDER_CLUSTER_DE], axis=0)
