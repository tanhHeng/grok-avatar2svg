"""Thin-line layer (design.md section 3): centerline + width profile -> filled outline."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

from .fitting import bezier_eval, fit_path, smooth_piece
from .geometry import arc_lengths, direction_from_end, pair_branches, resample, signed_area, unit
from .occlusion import JointCurve, OccludedEnd, joint_curve, pair_occluded_ends, probe_front, slide_onto, snap_toward
from .params import Params
from .segment import flat_mask, is_elongated

Curve = list[np.ndarray]

OUTLINE_STEP = 0.5
TIP_STEP = 0.5
TAPER_RATIO = 0.75
MIN_WIDTH = 0.5
# A free end whose width near the end stays above this fraction of the chain's median is cut, not drawn tapered.
CUT_WIDTH_RATIO = 0.6
# A free end counts as cut only when the occluder's outline crosses the stroke at least this steeply
# (sine of the angle). Shallower contact means the line runs along that outline; a flat cut there
# would stretch the end into a spike about half-width * cot(angle) long.
CUT_MIN_SIN = 0.7
# Pixels of contact a third region needs around a free end to count as the occluder.
TIP_CONTACT_MIN = 3
# Cross-section sampling for ink coverage: step, and how far beyond the half width the side colors are read.
INK_STEP = 0.25
INK_SIDE_GAP = 2.5
# Tip tracing: step along the ridge, how far the ridge may shift sideways per step, and how much of
# each step's direction comes from the ridge (the rest is momentum, which rides over pixel noise).
TRACE_STEP = 0.5
TRACE_SHIFT = 1.0
TRACE_TURN = 0.3


@dataclass
class Stroke:
    label: int  # region id in the input label map
    pixels: np.ndarray  # (M, 2) int (row, col)
    outline: list[Curve]  # closed subpaths for fill-rule nonzero: positive signed_area, ring holes negative
    centerline: list[Curve]  # fitted centerline per chain
    color: np.ndarray  # Lab fill; outline widths put down the image's ink only when drawn in this color


@dataclass
class _Node:
    pos: np.ndarray
    junction: bool


@dataclass
class _Branch:
    pts: np.ndarray  # xy, from node a to node b (closed loops: first point not repeated)
    a: int = -1
    b: int = -1
    alive: bool = True

    @property
    def closed(self) -> bool:
        return self.a < 0

    @property
    def length(self) -> float:
        return float(arc_lengths(self.pts)[-1]) if len(self.pts) > 1 else 0.0


@dataclass
class _Graph:
    nodes: list[_Node] = field(default_factory=list)
    branches: list[_Branch] = field(default_factory=list)

    def ends(self, nid: int) -> list[tuple[int, int]]:
        out = []
        for bid, br in enumerate(self.branches):
            if br.alive and not br.closed:
                if br.a == nid:
                    out.append((bid, 0))
                if br.b == nid:
                    out.append((bid, 1))
        return out

    def away(self, bid: int, which: int) -> np.ndarray:
        pts = self.branches[bid].pts
        return pts if which == 0 else pts[::-1]


def extract_strokes(lab: np.ndarray, labels: np.ndarray, p: Params) -> tuple[np.ndarray, list[Stroke], np.ndarray]:
    """Pull thin elongated regions out into strokes.

    Only ridges count: a region whose centerline mostly runs through flat color (a plateau) is a
    narrow color block, such as a hair strand, and stays a region; its two edges are placed and
    fitted independently, which a centerline + width model cannot match.

    Width and centerline position come from the line's ink in `lab`, not from the region's shape
    (design 3, step 5). Returns the label map with stroke pixels given to the nearest non-stroke
    label (so the regions underneath meet along each stroke's medial axis), the strokes, and the
    stroke pixel mask. Free ends cut by an occluder (design 3, 5.5) end flat along the occluder's
    outline, and cut ends facing each other across the same occluder share one joint curve; other
    free ends follow the line's ridge in the image until it fades out or meets another segment.
    """
    h, w = labels.shape
    flat = flat_mask(lab, p)
    preps: list[_Prepared] = []
    for idx, box in enumerate(ndimage.find_objects(labels)):
        if box is None:
            continue
        label = idx + 1
        crop = labels[box] == label
        skel = skeletonize(crop)
        if not is_elongated(int(crop.sum()), int(skel.sum()), p):
            continue
        if float(flat[box][skel].mean()) >= p.stroke_max_plateau:
            continue
        prep = _prepare(label, lab, labels, box, p)
        if prep is not None:
            preps.append(prep)

    cuts: dict[tuple[int, int, int], _Cut] = {}
    ends: list[OccludedEnd] = []
    refs: list[tuple[int, int, int]] = []
    for pi, prep in enumerate(preps):
        for ci, ch in enumerate(prep.chains):
            for side in (0, 1):
                if ch.closed or not ch.free[side]:
                    continue
                cut = _classify_end(prep, ch, side, labels, p)
                if cut is None:
                    continue
                cuts[(pi, ci, side)] = cut
                pts = ch.poly if side == 0 else ch.poly[::-1]
                ends.append(OccludedEnd(cut.exit, -cut.out, cut.front, -1, -1, np.vstack([cut.exit, pts]),
                                        width=cut.width, color=prep.color))
                refs.append((pi, ci, side))
    pairs = pair_occluded_ends(ends, labels, None, p)
    for i, j in pairs:
        curve = joint_curve(ends[i], ends[j], p)
        shared = 0.5 * (ends[i].width + ends[j].width)
        for k in (i, j):
            cuts[refs[k]].curve = curve
            cuts[refs[k]].width = shared

    mask_all = np.zeros((h, w), dtype=bool)
    strokes: list[Stroke] = []
    for pi, prep in enumerate(preps):
        own = {(ci, side): c for (q, ci, side), c in cuts.items() if q == pi}
        stroke = _finish(prep, own, lab, labels, p)
        if stroke is not None:
            strokes.append(stroke)
            mask_all[stroke.pixels[:, 0], stroke.pixels[:, 1]] = True
    out = labels.copy()
    if strokes and not mask_all.all():
        _, (ri, ci) = ndimage.distance_transform_edt(mask_all, return_indices=True)
        out[mask_all] = labels[ri[mask_all], ci[mask_all]]
    return out, strokes, mask_all


@dataclass
class _Chain:
    poly: np.ndarray  # smoothed centerline, ~1 px spacing
    widths: np.ndarray
    closed: bool
    free: tuple[bool, bool]  # start / end is a free skeleton end (not a junction)


@dataclass
class _Prepared:
    label: int
    r0: int
    c0: int
    mask: np.ndarray  # region crop
    dist: np.ndarray  # distance transform of the crop
    width: float  # mean width A / l
    color: np.ndarray  # median Lab along the skeleton: the line's core color, without its anti-aliased rim
    chains: list[_Chain]

    def width_at(self, xy: np.ndarray) -> np.ndarray:
        coords = [xy[:, 1] - 0.5 - self.r0, xy[:, 0] - 0.5 - self.c0]
        dist = ndimage.map_coordinates(self.dist, coords, order=1, mode="constant")
        # 2 * EDT, no half-pixel correction: watershed puts the region edge mid-way through the
        # anti-aliasing ramp, so the drawn line is that much wider than the region (resvg-measured).
        return np.maximum(2.0 * dist, MIN_WIDTH)

    def inside(self, xy: np.ndarray) -> bool:
        c, r = int(np.floor(xy[0])) - self.c0, int(np.floor(xy[1])) - self.r0
        return 0 <= r < self.mask.shape[0] and 0 <= c < self.mask.shape[1] and bool(self.mask[r, c])

    def pixels(self) -> np.ndarray:
        rows, cols = np.nonzero(self.mask)
        return np.stack([rows + self.r0, cols + self.c0], axis=1)


@dataclass
class _Cut:
    exit: np.ndarray  # where the centerline meets the occluder
    out: np.ndarray  # unit direction out of the stroke, into the occluder
    front: int  # occluder label
    width: float  # width kept up to the cut
    along: np.ndarray  # unit direction of the occluder's outline at the exit: the cut line
    curve: JointCurve | None = None  # set when paired across the occluder


def _prepare(label: int, lab: np.ndarray, labels: np.ndarray, box: tuple[slice, slice], p: Params) -> _Prepared | None:
    h, w = labels.shape
    margin = int(np.ceil(p.px(p.stroke_max_width))) + 2
    r0, r1 = max(box[0].start - margin, 0), min(box[0].stop + margin, h)
    c0, c1 = max(box[1].start - margin, 0), min(box[1].stop + margin, w)
    mask = labels[r0:r1, c0:c1] == label
    dist = ndimage.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
    skel = skeletonize(mask)
    width = int(mask.sum()) / max(int(skel.sum()), 1)
    graph = _skeleton_graph(skel, np.array([c0, r0], dtype=np.float64))
    if not graph.branches:
        return None
    _simplify_graph(graph, width, p)
    color = np.median(lab[r0:r1, c0:c1][skel], axis=0)
    prep = _Prepared(label, r0, c0, mask, dist, width, color, [])
    junction_pos = [n.pos for n in graph.nodes if n.junction]
    for pts, closed, free in _chains(graph, width, p):
        poly = resample(pts, 1.0, closed=closed)
        if len(poly) < 2:
            continue
        poly = _smooth_chain(poly, closed, p.px(p.smooth_stroke))
        normals = _normals(poly, closed)
        widths, offsets = _line_profile(poly, normals, lab, prep, junction_pos, closed, p)
        poly = poly + offsets[:, None] * normals
        prep.chains.append(_Chain(poly, widths, closed, free))
    return prep if prep.chains else None


def _classify_end(prep: _Prepared, ch: _Chain, side: int, labels: np.ndarray, p: Params) -> _Cut | None:
    """A free end is cut by an occluder when its width holds up to the end and the region beyond
    it is neither side of the line; otherwise it is a natural (tapered or round) end."""
    pts = ch.poly if side == 0 else ch.poly[::-1]
    ws = ch.widths if side == 0 else ch.widths[::-1]
    w = prep.width
    s = arc_lengths(pts)
    near = ws[(s >= 0.5 * w) & (s <= 2.0 * w)]
    w_cut = float(np.median(near)) if len(near) else float(np.median(ws))
    if w_cut < CUT_WIDTH_RATIO * float(np.median(ws)):
        return None  # narrows toward the end: a drawn tip
    away = direction_from_end(pts, 0.0, min(2.0 * w, max(float(s[-1]) * 0.5, 1.0)))
    out = -away
    step = 0
    while prep.inside(pts[0] + (step + 1) * TIP_STEP * out) and step < 400:
        step += 1
    exit_pt = pts[0] + (step + 0.5) * TIP_STEP * out
    reach = p.px(p.occlusion_probe)
    iq = min(int(np.searchsorted(s, w)), len(pts) - 1)
    q = pts[iq]
    normal = np.array([away[1], -away[0]])
    side_reach = 0.5 * float(ws[iq]) + p.px(3.0)  # local half width: the mean width of merged regions overshoots
    sides = {probe_front(labels, q, sgn * normal, {prep.label}, side_reach) for sgn in (1.0, -1.0)}
    front = probe_front(labels, exit_pt - TIP_STEP * out, out, {prep.label}, reach)
    if front is None or front in sides:
        # End directions of short chains bend toward their junction; fall back to which region
        # actually touches the stroke around its end, and aim the cut at that contact.
        contact = _tip_contact(labels, prep, pts[0], max(w, p.px(4.0)), sides)
        if contact is None:
            return None
        front, exit_pt = contact
        if np.linalg.norm(exit_pt - pts[0]) > 1e-6:
            out = unit(exit_pt - pts[0])
    along = _occluder_direction(labels, exit_pt, front, max(w, p.px(3.0)), np.array([out[1], -out[0]]))
    if abs(float(np.array([along[1], -along[0]]) @ out)) < CUT_MIN_SIN:
        return None
    return _Cut(exit_pt, out, front, w_cut, along)


def _tip_contact(labels: np.ndarray, prep: _Prepared, end: np.ndarray, radius: float, sides: set) -> tuple[int, np.ndarray] | None:
    """Most common non-side label bordering the stroke region within `radius` of its end, with the
    centroid of its contact pixels."""
    ring = ndimage.binary_dilation(prep.mask) & ~prep.mask
    rows, cols = np.nonzero(ring)
    xy = np.stack([cols + prep.c0 + 0.5, rows + prep.r0 + 0.5], axis=1)
    near = np.linalg.norm(xy - end, axis=1) <= radius
    xy = xy[near]
    found = labels[rows[near] + prep.r0, cols[near] + prep.c0]
    excluded = list({prep.label} | {s for s in sides if s is not None})
    keep = ~np.isin(found, excluded)
    if not keep.any():
        return None
    vals, counts = np.unique(found[keep], return_counts=True)
    k = int(np.argmax(counts))
    if counts[k] < TIP_CONTACT_MIN:
        return None
    return int(vals[k]), xy[found == vals[k]].mean(axis=0)


def _occluder_direction(labels: np.ndarray, at: np.ndarray, front: int, radius: float, fallback: np.ndarray) -> np.ndarray:
    """Local direction of the occluder's outline near `at` (PCA of its boundary pixels)."""
    h, w = labels.shape
    r = int(np.ceil(radius)) + 1
    cx, cy = int(np.floor(at[0])), int(np.floor(at[1]))
    r0, r1, c0, c1 = max(cy - r, 0), min(cy + r + 1, h), max(cx - r, 0), min(cx + r + 1, w)
    occ = np.pad(labels[r0:r1, c0:c1] == front, 1, mode="edge")
    inner = occ[1:-1, 1:-1]
    edge = inner & ~(occ[:-2, 1:-1] & occ[2:, 1:-1] & occ[1:-1, :-2] & occ[1:-1, 2:])
    rows, cols = np.nonzero(edge)
    pts = np.stack([cols + c0 + 0.5, rows + r0 + 0.5], axis=1)
    pts = pts[np.linalg.norm(pts - at, axis=1) <= radius]
    if len(pts) < 3:
        return fallback
    _, _, vt = np.linalg.svd(pts - pts.mean(axis=0), full_matrices=False)
    return unit(vt[0])


def _apply_cut(poly, widths, side, cut: _Cut, prep: _Prepared, p: Params):
    """Run the centerline to the occluder at full width. Returns (poly, widths, cut direction,
    end tangent pointing into the chain or None)."""
    pts = poly if side == 0 else poly[::-1]
    ws = widths if side == 0 else widths[::-1]
    w = prep.width
    exit_pt, tangent = cut.exit, None
    if cut.curve is not None:
        exit_pt = slide_onto(cut.curve, exit_pt, cut.along, p.px(p.occlusion_max_slide))
        t = cut.curve.tangent(exit_pt)
        tangent = t if float(t @ cut.out) <= 0 else -t
    gap = float(np.linalg.norm(pts[0] - exit_pt))
    n = max(int(np.ceil(gap)), 1)
    ext = exit_pt + (np.arange(n) / n)[:, None] * (pts[0] - exit_pt)
    pts = np.vstack([ext, pts])
    ws = np.concatenate([np.full(n, cut.width), ws])
    s = arc_lengths(pts)
    blend = np.clip((s - w) / max(w, 1e-6), 0.0, 1.0)  # full cut width within w, back to measured by 2w
    ws = cut.width * (1.0 - blend) + ws * blend
    if cut.curve is not None:
        snap_toward(pts, 0, 1, False, cut.curve, p.px(p.tangent_r0 + p.tangent_len))
    if side == 1:
        pts, ws = pts[::-1], ws[::-1]
    return pts, ws, cut.along, tangent


def _finish(prep: _Prepared, cuts: dict[tuple[int, int], _Cut], lab: np.ndarray, labels: np.ndarray, p: Params) -> Stroke | None:
    eps = p.px(p.eps_stroke)
    outline: list[Curve] = []
    centerline: list[Curve] = []
    for ci, ch in enumerate(prep.chains):
        poly, widths = ch.poly, ch.widths
        kinds: list = ["round", "round"]
        tangents: list = [None, None]
        if not ch.closed:
            natural = [ch.free[s] and (ci, s) not in cuts for s in (0, 1)]
            untraced = list(natural)
            for s in (0, 1):
                if not natural[s]:
                    continue
                pts, ws = (poly, widths) if s == 0 else (poly[::-1], widths[::-1])
                traced = _trace_tip(lab, labels, pts, ws, prep.color, prep.width, prep.label, p)
                if traced is None:
                    continue
                i0, ext, ext_w, kinds[s] = traced
                pts, ws = np.vstack([ext[::-1], pts[i0 + 1:]]), np.concatenate([ext_w[::-1], ws[i0 + 1:]])
                poly, widths = (pts, ws) if s == 0 else (pts[::-1], ws[::-1])
                untraced[s] = False
            poly, widths, tapered = _complete_tips(poly, widths, untraced, prep.inside, prep.width)
            for s in (0, 1):
                if untraced[s]:
                    kinds[s] = "taper" if tapered[s] else "round"
            for s in (0, 1):
                if (ci, s) in cuts:
                    poly, widths, kinds[s], tangents[s] = _apply_cut(poly, widths, s, cuts[(ci, s)], prep, p)
        center = fit_path(poly, eps, closed=ch.closed, t0=tangents[0], t1=tangents[1])
        centerline.append(center)
        for loop in _outline_loops(center, poly, widths, ch.closed, kinds):
            outline.append(fit_path(loop, eps, closed=True))
    if not outline:
        return None
    return Stroke(prep.label, prep.pixels(), outline, centerline, prep.color)


def _skeleton_graph(skel: np.ndarray, origin: np.ndarray) -> _Graph:
    """Graph over skeleton pixels with m-adjacency (diagonal links only without a shared 4-neighbor)."""
    coords = np.argwhere(skel)
    index = -np.ones(skel.shape, dtype=np.int64)
    index[coords[:, 0], coords[:, 1]] = np.arange(len(coords))
    padded = np.pad(index, 1, constant_values=-1)
    sk = np.pad(skel, 1)
    rr, cc = coords[:, 0] + 1, coords[:, 1] + 1
    nbrs: list[list[int]] = [[] for _ in range(len(coords))]
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            other = padded[rr + dr, cc + dc]
            ok = other >= 0
            if dr and dc:
                ok &= ~sk[rr + dr, cc] & ~sk[rr, cc + dc]
            for i in np.flatnonzero(ok).tolist():
                nbrs[i].append(int(other[i]))
    deg = np.array([len(n) for n in nbrs])
    xy = coords[:, ::-1] + 0.5 + origin

    graph = _Graph()
    node_of = -np.ones(len(coords), dtype=np.int64)
    # Junction pixels grouped by adjacency; end pixels are single-pixel nodes.
    for i in np.flatnonzero(deg >= 3).tolist():
        if node_of[i] >= 0:
            continue
        nid = len(graph.nodes)
        stack, members = [i], []
        node_of[i] = nid
        while stack:
            v = stack.pop()
            members.append(v)
            for u in nbrs[v]:
                if deg[u] >= 3 and node_of[u] < 0:
                    node_of[u] = nid
                    stack.append(u)
        graph.nodes.append(_Node(xy[members].mean(axis=0), True))
    for i in np.flatnonzero(deg == 1).tolist():
        node_of[i] = len(graph.nodes)
        graph.nodes.append(_Node(xy[i].copy(), False))

    visited = np.zeros(len(coords), dtype=bool)
    done: set[tuple[int, int]] = set()
    for v in np.flatnonzero(node_of >= 0).tolist():
        for u in nbrs[v]:
            if node_of[u] == node_of[v] or (v, u) in done:
                continue
            path, prev, cur = [v], v, u
            while node_of[cur] < 0:
                visited[cur] = True
                path.append(cur)
                nxt = [x for x in nbrs[cur] if x != prev]
                if not nxt:
                    break
                prev, cur = cur, nxt[0]
            if node_of[cur] < 0:
                continue
            path.append(cur)
            done.add((v, path[1]))
            done.add((cur, path[-2]))
            a, b = int(node_of[v]), int(node_of[cur])
            pts = np.vstack([graph.nodes[a].pos, xy[path[1:-1]], graph.nodes[b].pos])
            graph.branches.append(_Branch(pts, a, b))
    for s in np.flatnonzero((deg == 2) & ~visited & (node_of < 0)).tolist():
        if visited[s]:
            continue
        loop, prev, cur = [s], -1, s
        visited[s] = True
        while True:
            nxt = [x for x in nbrs[cur] if x != prev and (not visited[x] or (x == s and len(loop) > 2))]
            if not nxt or nxt[0] == s:
                break
            prev, cur = cur, nxt[0]
            visited[cur] = True
            loop.append(cur)
        if len(loop) >= 3:
            graph.branches.append(_Branch(xy[loop]))
    return graph


def _dissolve(graph: _Graph, nid: int) -> None:
    """Join the two branches meeting at a degree-2 node into one."""
    ends = graph.ends(nid)
    if len(ends) != 2:
        return
    (b1, w1), (b2, w2) = ends
    if b1 == b2:
        br = graph.branches[b1]
        br.pts, br.a, br.b = br.pts[:-1], -1, -1
        return
    p1 = graph.away(b1, w1)[::-1]  # ends at the node
    p2 = graph.away(b2, w2)  # starts at the node
    far1 = graph.branches[b1].b if w1 == 0 else graph.branches[b1].a
    far2 = graph.branches[b2].b if w2 == 0 else graph.branches[b2].a
    graph.branches[b2].alive = False
    graph.branches[b1] = _Branch(np.vstack([p1, p2[1:]]), far1, far2)


def _continues(graph: _Graph, bid: int, hub: int, width: float, p: Params) -> bool:
    """Whether branch `bid` pairs with another branch at node `hub` (same pairing as _chains)."""
    ends = graph.ends(hub)
    r0, length = max(p.px(p.tangent_r0), width), p.px(p.tangent_len)
    dirs = [direction_from_end(graph.away(b, w), r0, length) for b, w in ends]
    k = next(i for i, (b, _) in enumerate(ends) if b == bid)
    return any(k in pair for pair in pair_branches(dirs, p.pair_angle_deg))


def _simplify_graph(graph: _Graph, width: float, p: Params) -> None:
    """Prune spurs shorter than spur_factor * width and contract junction links shorter than width.

    A short spur that leaves its junction straight on from another branch is kept: it is the start
    of a tip that segmentation cut short (tip tracing extends it), not a skeleton fork at a corner.
    """
    for nid in range(len(graph.nodes)):
        _dissolve(graph, nid)
    while True:
        deg = [len(graph.ends(n)) for n in range(len(graph.nodes))]
        spurs = [
            (br.length, bid)
            for bid, br in enumerate(graph.branches)
            if br.alive and not br.closed and br.a != br.b
            and sorted((deg[br.a], deg[br.b]))[0] == 1 and max(deg[br.a], deg[br.b]) >= 3
            and br.length < p.spur_factor * width
            and not _continues(graph, bid, br.a if deg[br.a] >= 3 else br.b, width, p)
        ]
        links = [
            (br.length, bid)
            for bid, br in enumerate(graph.branches)
            if br.alive and not br.closed and br.a != br.b
            and deg[br.a] >= 3 and deg[br.b] >= 3 and br.length < width
        ]
        if spurs:
            _, bid = min(spurs)
            br = graph.branches[bid]
            br.alive = False
            hub = br.a if deg[br.a] >= 3 else br.b
            _dissolve(graph, hub)
        elif links:
            _, bid = min(links)
            br = graph.branches[bid]
            br.alive = False
            keep, gone = br.a, br.b
            pos = 0.5 * (graph.nodes[keep].pos + graph.nodes[gone].pos)
            graph.nodes[keep].pos = pos
            for other in graph.branches:
                if not other.alive or other.closed:
                    continue
                if other.a == gone:
                    other.a = keep
                if other.b == gone:
                    other.b = keep
                if other.a == keep:
                    other.pts = np.vstack([pos, other.pts[1:]])
                if other.b == keep:
                    other.pts = np.vstack([other.pts[:-1], pos])
            _dissolve(graph, keep)
        else:
            return


def _chains(graph: _Graph, width: float, p: Params) -> list[tuple[np.ndarray, bool, tuple[bool, bool]]]:
    """Chain branches through paired junction branches (design 3.5).

    Returns (points, closed, (start is a free end, end is a free end)).
    """
    r0, length = max(p.px(p.tangent_r0), width), p.px(p.tangent_len)
    partner: dict[tuple[int, int], tuple[int, int]] = {}
    free: set[int] = set()
    for nid in range(len(graph.nodes)):
        ends = graph.ends(nid)
        if len(ends) == 1:
            free.add(nid)
            continue
        dirs = [direction_from_end(graph.away(b, w), r0, length) for b, w in ends]
        for i, j in pair_branches(dirs, p.pair_angle_deg):
            partner[ends[i]] = ends[j]
            partner[ends[j]] = ends[i]

    used: set[int] = set()
    out = []

    def walk(bid: int, which: int):
        start = (bid, which)
        parts = []
        while True:
            used.add(bid)
            parts.append(graph.away(bid, which))
            nxt = partner.get((bid, 1 - which))
            if nxt is None or nxt[0] in used:
                closed = nxt == start
                break
            bid, which = nxt
        pts = np.vstack([parts[0]] + [q[1:] for q in parts[1:]])
        first = graph.branches[start[0]]
        start_node = first.a if start[1] == 0 else first.b
        last = graph.branches[bid]
        end_node = last.b if which == 0 else last.a
        if closed:
            out.append((pts[:-1], True, (False, False)))
        else:
            out.append((pts, False, (start_node in free, end_node in free)))

    alive = [bid for bid, br in enumerate(graph.branches) if br.alive]
    for bid in alive:
        br = graph.branches[bid]
        if not br.closed and bid not in used:
            if br.a in free:
                walk(bid, 0)
            elif br.b in free:
                walk(bid, 1)
    for bid in alive:
        br = graph.branches[bid]
        if not br.closed and bid not in used:
            if (bid, 0) not in partner:
                walk(bid, 0)
            elif (bid, 1) not in partner:
                walk(bid, 1)
    for bid in alive:
        br = graph.branches[bid]
        if bid in used:
            continue
        if br.closed:
            used.add(bid)
            out.append((br.pts, True, (False, False)))
        else:
            walk(bid, 0)
    return out


def _smooth_chain(poly: np.ndarray, closed: bool, sigma: float) -> np.ndarray:
    """Remove the +-0.5 px skeleton staircase, which exceeds eps_stroke; open chains keep their ends."""
    if closed:
        return ndimage.gaussian_filter1d(poly, sigma, axis=0, mode="wrap")
    return smooth_piece(poly, sigma)


def _normals(poly: np.ndarray, closed: bool) -> np.ndarray:
    tang = np.roll(poly, -1, axis=0) - np.roll(poly, 1, axis=0) if closed else np.gradient(poly, axis=0)
    tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-12)
    return np.stack([tang[:, 1], -tang[:, 0]], axis=1)


def _line_profile(poly, normals, lab, prep: _Prepared, junction_pos, closed, p: Params) -> tuple[np.ndarray, np.ndarray]:
    """Width and centre offset (along `normals`) per centerline sample, from ink coverage.

    The skeleton runs mid-way through the stroke region, which can sit a few tenths of a pixel off
    the line's ink; the offset moves the centerline onto it. Where ink is undefined the width falls
    back to the distance transform and the offset is interpolated. Samples near junctions, where the
    region is wider than any single line through it, are bridged from both sides, then both profiles
    are smoothed over about one stroke width.
    """
    edt = prep.width_at(poly)
    ink, centre, _, _ = _ink(lab, poly, normals, edt, prep.color, p.ink_min_contrast)
    near_junction = np.zeros(len(poly), dtype=bool)
    for jp in junction_pos:
        radius = 0.5 * float(prep.width_at(jp[None, :])[0]) + 1.0
        near_junction |= np.linalg.norm(poly - jp, axis=1) < radius
    widths = _bridge(np.where(np.isnan(ink), edt, np.maximum(ink, MIN_WIDTH)), near_junction, closed)
    unknown = near_junction | np.isnan(centre)
    offsets = np.zeros(len(poly)) if unknown.all() else _bridge(np.nan_to_num(centre), unknown, closed)
    sigma = max(1.0, p.stroke_width_sigma * prep.width)
    mode = "wrap" if closed else "nearest"
    return ndimage.gaussian_filter1d(widths, sigma, mode=mode), ndimage.gaussian_filter1d(offsets, sigma, mode=mode)


def _bridge(values: np.ndarray, bad: np.ndarray, closed: bool) -> np.ndarray:
    """Replace `bad` samples by linear interpolation between the good ones (all bad: unchanged)."""
    if not bad.any() or bad.all():
        return values
    values = values.copy()
    idx = np.arange(len(values))
    good = idx[~bad]
    if closed:
        n = len(values)
        values[bad] = np.interp(idx[bad], np.concatenate([good - n, good, good + n]), np.tile(values[good], 3))
    else:
        values[bad] = np.interp(idx[bad], good, values[good])
    return values


def _complete_tips(poly, widths, free, inside, width):
    """Extend free ends to the region boundary; long extensions become tapered tips (width 0).
    For natural ends that _trace_tip rejects."""
    tapered = [False, False]
    for side in (0, 1):
        if not free[side]:
            continue
        pts = poly if side == 0 else poly[::-1]
        ws = widths if side == 0 else widths[::-1]
        out_dir = -direction_from_end(pts, 0.0, min(2.0 * width, max(float(arc_lengths(pts)[-1]) * 0.5, 1.0)))
        end, w_end = pts[0], float(ws[0])
        step = 0
        while inside(end + (step + 1) * TIP_STEP * out_dir) and step < 200:
            step += 1
        reach = (step + 0.5) * TIP_STEP
        if reach > TAPER_RATIO * w_end:
            n = max(int(np.ceil(reach)), 1)
            frac = np.arange(n, 0, -1) / n  # far ... near
            ext = end + (frac * reach)[:, None] * out_dir
            ext_w = w_end * (1.0 - frac)
            pts = np.vstack([ext, pts])
            ws = np.concatenate([ext_w, ws])
            tapered[side] = True
        poly, widths = (pts, ws) if side == 0 else (pts[::-1], ws[::-1])
    return poly, widths, (tapered[0], tapered[1])


def _sample_lab(lab: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Bilinear Lab at (N, 2) xy points (pixel centers at +0.5)."""
    coords = [xy[:, 1] - 0.5, xy[:, 0] - 0.5]
    return np.stack([ndimage.map_coordinates(lab[..., k], coords, order=1, mode="nearest") for k in range(3)], axis=1)


def _ink(lab, pts, normals, guide, color, min_contrast, search=None):
    """Ink coverage across the line at each point: (width, centre, ridge, peak). Width is in px,
    centre and ridge are offsets along `normals`, peak is the ink fraction at the ridge.

    The ink fraction of a pixel is how far its color has moved from the side color toward the line
    color; integrated across the line it is the width a line of exactly `color` needs to put down the
    same ink, so faint stretches come out thin. Each half of the cross-section, split at the ridge, is
    measured against its own side, so a line on the boundary of two regions works, and the two half
    widths place the line's centre. The ridge is searched within `search` (default: the half width
    + 1 px) of each point. Width, centre and peak are NaN where either side is within `min_contrast`
    of the line color (crossings, line-colored neighbors).
    """
    half = 0.5 * np.maximum(guide, 1.0)
    reach = half + INK_SIDE_GAP
    us = np.arange(-reach.max(), reach.max() + 1e-9, INK_STEP)
    n, m = len(pts), len(us)
    cs = _sample_lab(lab, (pts[:, None, :] + us[None, :, None] * normals[:, None, :]).reshape(-1, 2)).reshape(n, m, 3)
    side_u = (reach[:, None] + np.arange(0.0, 1.0 + 1e-9, INK_STEP)[None, :])
    k = side_u.shape[1]
    fracs, ok = [], np.ones(n, dtype=bool)
    for sgn in (-1.0, 1.0):
        sp = pts[:, None, :] + (sgn * side_u)[..., None] * normals[:, None, :]
        side = _sample_lab(lab, sp.reshape(-1, 2)).reshape(n, k, 3).mean(axis=1)
        v = color[None, :] - side
        vv = (v * v).sum(axis=1)
        ok &= vv >= min_contrast**2
        fracs.append(((cs - side[:, None, :]) * v[:, None, :]).sum(axis=2) / np.maximum(vv, 1e-9)[:, None])
    f_lo, f_hi = fracs
    limit = half + 1.0 if search is None else np.broadcast_to(np.asarray(search, dtype=np.float64), (n,))
    peak_f = np.where(np.abs(us)[None, :] <= limit[:, None], np.minimum(f_lo, f_hi), -np.inf)
    j = np.argmax(peak_f, axis=1)
    idx = np.arange(m)[None, :]
    inside = np.abs(us)[None, :] <= reach[:, None]
    c_lo, c_hi = (np.clip(f, 0.0, 1.0) * inside for f in (f_lo, f_hi))
    at = idx == j[:, None]
    w_lo = ((c_lo * (idx < j[:, None])).sum(axis=1) + 0.5 * (c_lo * at).sum(axis=1)) * INK_STEP
    w_hi = ((c_hi * (idx > j[:, None])).sum(axis=1) + 0.5 * (c_hi * at).sum(axis=1)) * INK_STEP
    width = w_lo + w_hi
    ridge = us[j]
    centre = ridge + 0.5 * (w_hi - w_lo)
    peak = peak_f[np.arange(n), j]
    for a in (width, centre, peak):
        a[~ok] = np.nan
    return width, centre, ridge, peak


def _trace_tip(lab, labels, pts, ws, color, width, label, p):
    """Follow the line's ridge in the image past a natural free end (design 3, step 6).

    Segmentation loses the faint part of a tapering line to the region around it, so the stroke
    region stops short and its end looks cut off. `pts`/`ws` run from the end into the chain. Tracing
    starts `width` back from the end, because skeleton ends hook toward a corner of the region.
    Returns (i0, points, widths, kind): points run outward from pts[i0], where tracing started;
    kind "taper" when the ridge fades out, "round" when it runs into another segment (the ridge
    pixels are neither this stroke nor the regions beside it) or along a line-colored neighbor.
    None when the ridge is still going after tip_max_trace or leaves the image: that is another
    feature, and the end keeps the region-based tip.
    """
    h, w = labels.shape
    s = arc_lengths(pts)
    if s[-1] < 2.0:
        return None
    back = min(width, float(s[-1]) / 3.0)
    i0 = int(np.searchsorted(s, back))
    out = -direction_from_end(pts, back, 3.0 * width)
    pos = pts[i0].copy()
    guide = np.array([max(float(ws[i0]), 1.0)])
    reach = 0.5 * guide[0] + INK_SIDE_GAP + 1.0
    path, widths = [pos.copy()], [float(ws[i0])]
    travelled, kind = 0.0, None
    search = 0.5 * guide[0] + 1.0
    while kind is None:
        if travelled > p.px(p.tip_max_trace):
            return None
        q = pos + TRACE_STEP * out
        if not (reach <= q[0] < w - reach and reach <= q[1] < h - reach):
            return None
        normal = np.array([[out[1], -out[0]]])
        ink_w, _, shift, peak = _ink(lab, q[None, :], normal, guide, color, p.ink_min_contrast, search)
        search = TRACE_SHIFT
        ridge = q + shift[0] * normal[0]
        side_labels = {labels[int(np.clip(np.floor(y), 0, h - 1)), int(np.clip(np.floor(x), 0, w - 1))]
                       for x, y in (ridge + sgn * (0.5 * guide[0] + INK_SIDE_GAP) * normal[0] for sgn in (-1.0, 1.0))}
        here = labels[int(np.floor(ridge[1])), int(np.floor(ridge[0]))]
        if np.isnan(peak[0]) or (here != label and here not in side_labels):
            kind = "round"
        elif peak[0] < p.tip_min_ink:
            kind = "taper"
        else:
            out = unit((1.0 - TRACE_TURN) * out + TRACE_TURN * unit(ridge - pos))
            travelled += float(np.linalg.norm(ridge - pos))
            pos = ridge
            path.append(pos.copy())
            widths.append(float(ink_w[0]))
    if travelled < back + 1.0:
        return None
    path = np.array(path)
    s_path = arc_lengths(path)
    ext = resample(path, 1.0)
    ext_w = np.interp(arc_lengths(ext), s_path, np.array(widths))
    ext = smooth_piece(ext, p.px(p.smooth_stroke))
    ext_w = ndimage.gaussian_filter1d(ext_w, max(1.0, p.stroke_width_sigma * width), mode="nearest")
    ext_w = np.maximum(ext_w, MIN_WIDTH)
    if kind == "taper":
        s_ext = arc_lengths(ext)
        ramp = min(width, float(s_ext[-1]) / 2.0)
        ext_w *= np.clip((s_ext[-1] - s_ext) / max(ramp, 1e-6), 0.0, 1.0)
    return i0, ext, ext_w, kind


def _sample_curve(curve: Curve, closed: bool) -> np.ndarray:
    pieces = []
    for bez in curve:
        n = max(int(np.ceil(np.linalg.norm(np.diff(bez, axis=0), axis=1).sum() / 0.25)), 2)
        pieces.append(bezier_eval(bez, np.linspace(0.0, 1.0, n + 1))[:-1])
    pieces.append(curve[-1][3:4])
    pts = np.vstack(pieces)
    return resample(pts[:-1] if closed else pts, OUTLINE_STEP, closed=closed)


def _outline_loops(center: Curve, poly: np.ndarray, widths: np.ndarray, closed: bool, ends) -> list[np.ndarray]:
    """Offset outline around the fitted centerline. Open chains take per-side `ends`: "round"
    (semicircle cap), "taper" (the width already reaches 0), or a unit vector: a flat cut along
    that direction (the occluder's outline) through the centerline's end point."""
    samples = _sample_curve(center, closed)
    if len(samples) < 2:
        return []
    s_poly = arc_lengths(np.vstack([poly, poly[:1]]) if closed else poly)
    w_poly = np.concatenate([widths, widths[:1]]) if closed else widths
    s_samp = arc_lengths(np.vstack([samples, samples[:1]]) if closed else samples)
    t = s_samp[: len(samples)] / max(s_samp[-1], 1e-9) * s_poly[-1]
    half = 0.5 * np.interp(t, s_poly, w_poly)

    if closed:
        tang = np.roll(samples, -1, axis=0) - np.roll(samples, 1, axis=0)
    else:
        tang = np.gradient(samples, axis=0)
    tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-12)
    normal = np.stack([tang[:, 1], -tang[:, 0]], axis=1)
    left = samples + normal * half[:, None]
    right = samples - normal * half[:, None]

    if closed:
        a, b = left, right[::-1]
        if abs(signed_area(a)) < abs(signed_area(b)):
            a, b = b, a
        if signed_area(a) < 0:
            a = a[::-1]
        if signed_area(b) > 0:
            b = b[::-1]
        return [a, b]

    def cap(center_pt, n_vec, t_vec, r, start_side: bool) -> np.ndarray:
        k = max(int(np.ceil(np.pi * r / OUTLINE_STEP)), 4)
        th = np.linspace(0.0, np.pi, k + 1)[1:-1]
        if start_side:
            return center_pt - (np.cos(th)[:, None] * n_vec + np.sin(th)[:, None] * t_vec) * r
        return center_pt + (np.cos(th)[:, None] * n_vec + np.sin(th)[:, None] * t_vec) * r

    if not isinstance(ends[1], str):
        left = _cut_poly(left, samples[-1], tang[-1], ends[1])
        right = _cut_poly(right, samples[-1], tang[-1], ends[1])
    if not isinstance(ends[0], str):
        left = _cut_poly(left[::-1], samples[0], -tang[0], ends[0])[::-1]
        right = _cut_poly(right[::-1], samples[0], -tang[0], ends[0])[::-1]

    parts = [left]
    if isinstance(ends[1], str) and ends[1] == "round":
        parts.append(cap(samples[-1], normal[-1], tang[-1], half[-1], False))
    parts.append(right[::-1])
    if isinstance(ends[0], str) and ends[0] == "round":
        parts.append(cap(samples[0], normal[0], tang[0], half[0], True))
    loop = np.vstack(parts)
    keep = np.concatenate([[True], np.linalg.norm(np.diff(loop, axis=0), axis=1) > 1e-6])
    loop = loop[keep]
    if signed_area(loop) < 0:
        loop = loop[::-1]
    return [loop]


def _cut_poly(poly: np.ndarray, x: np.ndarray, t_out: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Trim an offset polyline at the cut line through `x` along `f` and end it exactly on the line."""
    m = np.array([f[1], -f[0]])
    if abs(float(m @ t_out)) < CUT_MIN_SIN:
        m = t_out.copy()
    elif float(m @ t_out) < 0:
        m = -m
    end = poly[-1]
    tip = end - float((end - x) @ m) / float(t_out @ m) * t_out
    return np.vstack([poly[(poly - x) @ m < 0], tip])
