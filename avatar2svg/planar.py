"""Planar boundary map over a label map (design.md 5.1).

Boundaries live on pixel cracks. Corners are (x, y) with x in 0..W, y in 0..H; the four pixels
around a corner are TL, TR, BL, BR (indices 0..3). Outside the image counts as label -1.
Every edge is shared by exactly two labels and fitted once, so neighbors can never gap or overlap.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from .geometry import arc_lengths
from .params import Params

# Arms leaving a corner: E, S, W, N.
_DX = (1, 0, -1, 0)
_DY = (0, 1, 0, -1)
_OPP = (2, 3, 0, 1)
# Quadrant pixels relative to the heading when arriving at a corner: (behind-left, ahead-left, ahead-right).
_TURN_QUADS = ((0, 1, 3), (1, 3, 2), (3, 2, 0), (2, 0, 1))
# Quadrant on the left / right of the crack taken by leaving along an arm.
_LEFT_Q = (1, 3, 2, 0)
_RIGHT_Q = (3, 2, 0, 1)


@dataclass
class Edge:
    points: np.ndarray  # open: points[0] at nodes[start], points[-1] at nodes[end]; closed: first point not repeated
    left: int  # label on the left walking the points in order (y down: walking +x the left side is -y); -1 = outside
    right: int
    start: int  # node id, -1 if closed
    end: int  # node id, -1 if closed

    @property
    def closed(self) -> bool:
        return self.start < 0


@dataclass
class Node:
    pos: np.ndarray  # (2,) xy
    ends: list[tuple[int, int]] = field(default_factory=list)  # (edge id, 0 = edge starts here, 1 = edge ends here)


@dataclass
class PlanarMap:
    width: int
    height: int
    nodes: list[Node]
    edges: list[Edge]
    # label >= 0 -> boundary cycles; cycle = [(edge id, forward)] with the label on the left of every edge.
    cycles: dict[int, list[list[tuple[int, bool]]]]


@dataclass
class _Raw:
    points: list[tuple[float, float]]
    left: int
    right: int
    start: int = -1  # corner id
    end: int = -1
    first_arm: int = -1
    last_arm: int = -1


def build_planar_map(labels: np.ndarray, p: Params) -> PlanarMap:
    h, w = labels.shape
    pad = np.full((h + 2, w + 2), -1, dtype=np.int64)
    pad[1:-1, 1:-1] = labels
    quads = np.stack([pad[: h + 1, : w + 1], pad[: h + 1, 1:], pad[1:, : w + 1], pad[1:, 1:]], axis=-1)
    tl, tr, bl, br = (quads[..., k] for k in range(4))
    arms_e, arms_s, arms_w, arms_n = tr != br, bl != br, tl != bl, tl != tr
    arms = arms_e.astype(np.uint8) | (arms_s.astype(np.uint8) << 1) | (arms_w.astype(np.uint8) << 2) | (arms_n.astype(np.uint8) << 3)
    s = np.sort(quads, axis=-1)
    distinct = 1 + (s[..., 1] != s[..., 0]) + (s[..., 2] != s[..., 1]) + (s[..., 3] != s[..., 2])
    degree = arms_e.astype(np.int8) + arms_s + arms_w + arms_n
    is_node = (distinct >= 3) | (degree == 4)

    stride = w + 1
    arms_l = arms.ravel().tolist()
    node_l = is_node.ravel().tolist()
    quad_flat = quads.reshape(-1, 4)
    visited = bytearray(2 * (h + 1) * stride)

    def crack_key(c: int, arm: int) -> int:
        if arm == 0:
            return 2 * c
        if arm == 2:
            return 2 * (c - 1)
        if arm == 1:
            return 2 * c + 1
        return 2 * (c - stride) + 1

    def walk(c0: int, arm0: int, stop_at_nodes: bool) -> _Raw:
        q = quad_flat[c0]
        raw = _Raw([], int(q[_LEFT_Q[arm0]]), int(q[_RIGHT_Q[arm0]]), first_arm=arm0)
        c, arm = c0, arm0
        pts = raw.points
        if stop_at_nodes:
            pts.append((float(c0 % stride), float(c0 // stride)))
        start_key = crack_key(c0, arm0)
        while True:
            visited[crack_key(c, arm)] = 1
            x, y = c % stride, c // stride
            pts.append((x + 0.5 * _DX[arm], y + 0.5 * _DY[arm]))
            c = c + _DX[arm] + _DY[arm] * stride
            if stop_at_nodes and node_l[c]:
                raw.start, raw.end, raw.last_arm = c0, c, arm
                pts.append((float(c % stride), float(c // stride)))
                return raw
            rest = arms_l[c] & ~(1 << _OPP[arm])
            arm = (rest & -rest).bit_length() - 1
            if not stop_at_nodes and crack_key(c, arm) == start_key:
                return raw

    raws: list[_Raw] = []
    for c in np.flatnonzero(is_node.ravel()).tolist():
        m = arms_l[c]
        for arm in range(4):
            if m >> arm & 1 and not visited[crack_key(c, arm)]:
                raws.append(walk(c, arm, True))
    n_open = len(raws)
    for c in np.flatnonzero((arms & 3).ravel()).tolist():
        for arm in (0, 1):
            if arms_l[c] >> arm & 1 and not visited[crack_key(c, arm)]:
                raws.append(walk(c, arm, False))

    cycles_raw = _trace_cycles(raws[:n_open], quad_flat)
    return _assemble(raws, n_open, cycles_raw, w, h, p)


def _turn(quad: np.ndarray, heading: int) -> int:
    bl, al, ar = _TURN_QUADS[heading]
    x = quad[bl]
    if quad[al] != x:
        return (heading + 3) % 4
    if quad[ar] != x:
        return heading
    return (heading + 1) % 4


def _trace_cycles(raws: list[_Raw], quad_flat: np.ndarray) -> list[tuple[int, list[tuple[int, bool]]]]:
    """Cycles over directed open edges using the left-first turn rule (4-connected regions)."""
    out_map: dict[tuple[int, int], tuple[int, bool]] = {}
    for i, r in enumerate(raws):
        out_map[(r.start, r.first_arm)] = (i, True)
        out_map[(r.end, _OPP[r.last_arm])] = (i, False)
    seen: set[tuple[int, bool]] = set()
    cycles = []
    for start in ((i, f) for i in range(len(raws)) for f in (True, False)):
        if start in seen:
            continue
        label = raws[start[0]].left if start[1] else raws[start[0]].right
        cyc = []
        cur = start
        while cur not in seen:
            seen.add(cur)
            cyc.append(cur)
            r = raws[cur[0]]
            corner, heading = (r.end, r.last_arm) if cur[1] else (r.start, _OPP[r.first_arm])
            cur = out_map[(corner, _turn(quad_flat[corner], heading))]
        cycles.append((label, cyc))
    return cycles


def _assemble(
    raws: list[_Raw],
    n_open: int,
    cycles_raw: list[tuple[int, list[tuple[int, bool]]]],
    w: int,
    h: int,
    p: Params,
) -> PlanarMap:
    stride = w + 1
    corners = sorted({r.start for r in raws[:n_open]} | {r.end for r in raws[:n_open]})
    corner_xy = np.array([(c % stride, c // stride) for c in corners], dtype=np.float64).reshape(-1, 2)
    radius = p.px(p.node_cluster_radius)

    # Union-find clustering of nearby nodes.
    parent = list(range(len(corners)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    if len(corners) > 1:
        for i, j in cKDTree(corner_xy).query_pairs(radius):
            parent[find(i)] = find(j)
    roots = [find(i) for i in range(len(corners))]
    root_ids = {r: k for k, r in enumerate(sorted(set(roots)))}
    cluster_of_corner = {c: root_ids[roots[i]] for i, c in enumerate(corners)}
    n_nodes = len(root_ids)
    sums = np.zeros((n_nodes, 2))
    counts = np.zeros(n_nodes)
    for i, c in enumerate(corners):
        k = cluster_of_corner[c]
        sums[k] += corner_xy[i]
        counts[k] += 1
    centers = sums / np.maximum(counts, 1)[:, None]

    max_removed_len = 2 * radius + 2
    new_id: list[int] = [-1] * len(raws)
    edges: list[Edge] = []
    for i, r in enumerate(raws):
        pts = np.asarray(r.points, dtype=np.float64)
        if i >= n_open:
            new_id[i] = len(edges)
            edges.append(Edge(pts, r.left, r.right, -1, -1))
            continue
        a, b = cluster_of_corner[r.start], cluster_of_corner[r.end]
        if a == b and arc_lengths(pts)[-1] <= max_removed_len:
            continue
        new_id[i] = len(edges)
        edges.append(Edge(_snap(pts, centers[a], centers[b], radius), r.left, r.right, a, b))

    nodes = [Node(centers[k].copy()) for k in range(n_nodes)]
    for eid, e in enumerate(edges):
        if not e.closed:
            nodes[e.start].ends.append((eid, 0))
            nodes[e.end].ends.append((eid, 1))

    cycles: dict[int, list[list[tuple[int, bool]]]] = {}
    for label, cyc in cycles_raw:
        if label < 0:
            continue
        kept = [(new_id[i], f) for i, f in cyc if new_id[i] >= 0]
        if kept:
            cycles.setdefault(label, []).append(kept)
    for i in range(n_open, len(raws)):
        e = edges[new_id[i]]
        if e.left >= 0:
            cycles.setdefault(e.left, []).append([(new_id[i], True)])
        if e.right >= 0:
            cycles.setdefault(e.right, []).append([(new_id[i], False)])
    return PlanarMap(w, h, nodes, edges, cycles)


def _snap(pts: np.ndarray, a: np.ndarray, b: np.ndarray, radius: float) -> np.ndarray:
    """Move end points onto their cluster centers, dropping interior points the move overtook."""
    inner = pts[1:-1]
    if len(inner):
        keep = (np.linalg.norm(inner - a, axis=1) > radius) & (np.linalg.norm(inner - b, axis=1) > radius)
        lo = int(np.argmax(keep)) if keep.any() else len(inner)
        hi = len(inner) - int(np.argmax(keep[::-1])) if keep.any() else lo
        inner = inner[lo:hi]
    return np.vstack([a, inner, b])
