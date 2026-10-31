"""Helpers over label maps: int32 (H, W) arrays where each value is a region id.

Convention used across the package: label 0 is background (not drawn), regions are >= 1.
"""

from __future__ import annotations

import numpy as np


def relabel_sequential(labels: np.ndarray) -> np.ndarray:
    """Renumber labels to 0..N keeping 0 as 0 and preserving the order of the other ids."""
    ids = np.unique(labels)
    ids = ids[ids != 0]
    lut = np.zeros(int(labels.max()) + 1, dtype=np.int32)
    lut[ids] = np.arange(1, len(ids) + 1, dtype=np.int32)
    return lut[labels]


def region_areas(labels: np.ndarray) -> np.ndarray:
    """Pixel count per label id, indexable by id (length max_label + 1)."""
    return np.bincount(labels.ravel(), minlength=int(labels.max()) + 1)


def region_medians(lab: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Per-channel median color per label id, shape (max_label + 1, C); empty ids get 0."""
    flat_labels = labels.ravel()
    values = lab.reshape(flat_labels.size, -1)
    n = int(flat_labels.max()) + 1
    order = np.argsort(flat_labels, kind="stable")
    sorted_labels = flat_labels[order]
    starts = np.searchsorted(sorted_labels, np.arange(n + 1))
    out = np.zeros((n, values.shape[1]), dtype=np.float32)
    for i in range(n):
        a, b = starts[i], starts[i + 1]
        if b > a:
            out[i] = np.median(values[order[a:b]], axis=0)
    return out


def adjacent_pairs(labels: np.ndarray) -> np.ndarray:
    """Unique 4-connected neighbor label pairs (a < b), shape (K, 2)."""
    pairs = []
    for a, b in ((labels[:, :-1], labels[:, 1:]), (labels[:-1, :], labels[1:, :])):
        diff = a != b
        pairs.append(np.stack([a[diff], b[diff]], axis=1))
    if not pairs:
        return np.zeros((0, 2), dtype=labels.dtype)
    p = np.sort(np.concatenate(pairs), axis=1)
    return np.unique(p, axis=0)
