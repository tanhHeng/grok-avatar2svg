"""Color-block segmentation (design.md section 2): flat seeds + thin-line seeds, watershed, RAG merging."""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from skimage.morphology import skeletonize
from skimage.segmentation import watershed

from .colorspace import delta_e
from .params import Params
from .regions import adjacent_pairs, region_areas, region_medians, relabel_sequential


def segment(lab: np.ndarray, opaque: np.ndarray, p: Params) -> np.ndarray:
    """Label map with regions 1..N covering the opaque pixels and 0 elsewhere."""
    opaque = np.asarray(opaque, dtype=bool)
    if not opaque.any():
        return np.zeros(opaque.shape, dtype=np.int32)

    markers = _flat_seeds(lab, opaque, p)
    markers = _add_line_seeds(lab, opaque, markers, p)

    channels = cv2.split(np.ascontiguousarray(lab, dtype=np.float32))
    height = np.zeros(opaque.shape, dtype=np.float32)
    for ch in channels:
        height += cv2.magnitude(cv2.Sobel(ch, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(ch, cv2.CV_32F, 0, 1, ksize=3))
    labels = watershed(height, markers, mask=opaque).astype(np.int32)

    # Opaque components without any marker stay 0 in the watershed; give each its own region.
    orphans = opaque & (labels == 0)
    if orphans.any():
        _, comp = cv2.connectedComponents(orphans.astype(np.uint8), connectivity=4, ltype=cv2.CV_32S)
        labels[orphans] = comp[orphans] + labels.max()

    labels = _merge_similar(lab, labels, p.merge_de)
    labels = _merge_small(lab, labels, p)
    labels[~opaque] = 0
    return relabel_sequential(labels)


def is_elongated(area: float, skeleton_length: float, p: Params) -> bool:
    """Stroke test of design section 3: width w = A / l <= stroke_max_width and l / w >= stroke_min_elongation."""
    if skeleton_length <= 0:
        return False
    width = area / skeleton_length
    return width <= p.px(p.stroke_max_width) and skeleton_length / width >= p.stroke_min_elongation


def flat_mask(lab: np.ndarray, p: Params) -> np.ndarray:
    """Pixels whose L* standard deviation in a flat_window square is below flat_std."""
    lum = lab[..., 0].astype(np.float64)
    size = (int(p.flat_window), int(p.flat_window))
    mean = cv2.blur(lum, size, borderType=cv2.BORDER_REFLECT)
    mean_sq = cv2.blur(lum * lum, size, borderType=cv2.BORDER_REFLECT)
    std = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))
    return std < p.flat_std


def _flat_seeds(lab: np.ndarray, opaque: np.ndarray, p: Params) -> np.ndarray:
    flat = opaque & flat_mask(lab, p)

    h, w = flat.shape
    idx = np.arange(h * w, dtype=np.int64).reshape(h, w)
    right = flat[:, :-1] & flat[:, 1:] & (delta_e(lab[:, :-1], lab[:, 1:]) < p.seed_link_de)
    down = flat[:-1, :] & flat[1:, :] & (delta_e(lab[:-1, :], lab[1:, :]) < p.seed_link_de)
    src = np.concatenate([idx[:, :-1][right], idx[:-1, :][down]])
    dst = np.concatenate([idx[:, 1:][right], idx[1:, :][down]])
    graph = coo_matrix((np.ones(src.size, dtype=np.int8), (src, dst)), shape=(h * w, h * w)).tocsr()
    _, comp = connected_components(graph, directed=False)

    comp_flat = comp.reshape(h, w)[flat]
    counts = np.bincount(comp_flat)
    keep = counts >= p.area(p.min_seed_area)
    lut = np.zeros(counts.size, dtype=np.int32)
    lut[keep] = np.arange(1, int(keep.sum()) + 1, dtype=np.int32)
    markers = np.zeros((h, w), dtype=np.int32)
    markers[flat] = lut[comp_flat]
    return markers


def _add_line_seeds(lab: np.ndarray, opaque: np.ndarray, markers: np.ndarray, p: Params) -> np.ndarray:
    """Skeletons of black-hat (dark) and top-hat (light) line responses outside flat seeds become extra markers."""
    lum = np.ascontiguousarray(lab[..., 0], dtype=np.float32)
    k = max(3, int(round(p.px(p.tophat_kernel))) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    markers = markers.copy()
    taken = markers > 0
    for op in (cv2.MORPH_BLACKHAT, cv2.MORPH_TOPHAT):
        response = cv2.morphologyEx(lum, op, kernel, borderType=cv2.BORDER_REFLECT)
        mask = (opaque & (response > p.tophat_thresh)).astype(np.uint8)
        n, comp, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8, ltype=cv2.CV_32S)
        big = stats[:, cv2.CC_STAT_AREA] >= p.area(p.tophat_min_area)
        big[0] = False
        if not big.any():
            continue
        skeleton = skeletonize(big[comp]) & ~taken
        n_pieces, pieces = cv2.connectedComponents(skeleton.astype(np.uint8), connectivity=8, ltype=cv2.CV_32S)
        if n_pieces <= 1:
            continue
        markers[skeleton] = pieces[skeleton] + markers.max()
        taken |= skeleton
    return markers


def _union_lut(n: int, pairs: np.ndarray) -> np.ndarray:
    """Label lookup table merging the given id pairs; id 0 maps to 0, other ids to >= 1."""
    graph = coo_matrix((np.ones(len(pairs), dtype=np.int8), (pairs[:, 0], pairs[:, 1])), shape=(n, n)).tocsr()
    _, comp = connected_components(graph, directed=False)
    lut = (comp + 1).astype(np.int32)
    lut[0] = 0
    return lut


def _foreground_pairs(labels: np.ndarray) -> np.ndarray:
    pairs = adjacent_pairs(labels)
    return pairs[pairs[:, 0] > 0]


def _merge_similar(lab: np.ndarray, labels: np.ndarray, max_de: float) -> np.ndarray:
    """Union adjacent regions whose median colors differ by < max_de, repeated until stable."""
    while True:
        medians = region_medians(lab, labels)
        pairs = _foreground_pairs(labels)
        close = pairs[delta_e(medians[pairs[:, 0]], medians[pairs[:, 1]]) < max_de]
        if not len(close):
            return labels
        labels = _union_lut(len(medians), close)[labels]


def _merge_small(lab: np.ndarray, labels: np.ndarray, p: Params) -> np.ndarray:
    """Merge regions smaller than small_area that fail the stroke test into their delta_e-nearest neighbor."""
    min_area = p.area(p.small_area)
    while True:
        areas = region_areas(labels)
        small = np.flatnonzero((areas > 0) & (areas < min_area))
        small = small[small > 0]
        if not len(small):
            return labels
        boxes = ndimage.find_objects(labels)
        candidates = set()
        for r in small:
            mask = labels[boxes[r - 1]] == r
            if not is_elongated(areas[r], int(skeletonize(mask).sum()), p):
                candidates.add(int(r))
        if not candidates:
            return labels

        medians = region_medians(lab, labels)
        pairs = _foreground_pairs(labels)
        de = delta_e(medians[pairs[:, 0]], medians[pairs[:, 1]])
        best: dict[int, tuple[float, int]] = {}
        for (a, b), d in zip(pairs.tolist(), de.tolist()):
            for src, dst in ((a, b), (b, a)):
                if src in candidates and (src not in best or d < best[src][0]):
                    best[src] = (d, dst)
        if not best:
            return labels
        merges = np.array([(src, dst) for src, (_, dst) in best.items()], dtype=np.int64)
        labels = _union_lut(len(areas), merges)[labels]
