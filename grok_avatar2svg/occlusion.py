"""Occlusion continuity (design.md 5.5): contours cut by a front object stay one curve behind it.

An occluded end is a visible contour piece that stops where a front object F begins: at a node,
or at a corner where the contour turns to follow F's outline (or F's rim). Two occluded ends on
opposite sides of the same F that face each other are the visible pieces of one contour, so both
take their end tangent from one curve fitted through both pieces, and their end points slide along
F's outline onto that curve. Nothing hidden is emitted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .colorspace import delta_e
from .geometry import arc_lengths, corner_indices, direction_from_end, unit
from .params import Params
from .planar import PlanarMap

PROBE_STEP = 0.5
# Edge id -> {point index: [tangent toward the piece before the index, tangent into the piece after]}.
Breaks = dict[int, dict[int, list[np.ndarray | None]]]


@dataclass
class OccludedEnd:
    """One visible contour piece ending where a front object begins."""

    pos: np.ndarray  # (2,) end point
    away: np.ndarray  # unit direction from the end point into the visible piece
    front: int  # label of the occluder
    left: int  # labels on both sides, seen walking toward the end point (into the occluder)
    right: int
    points: np.ndarray  # visible piece, starting at pos
    along: np.ndarray | None = None  # occluder outline direction at pos, for sliding
    ref: tuple = ()  # owner-specific reference
    width: float = 0.0  # strokes only
    color: np.ndarray | None = None  # strokes only


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def _label_at(labels: np.ndarray, pt: np.ndarray) -> int:
    h, w = labels.shape
    c, r = int(np.floor(pt[0])), int(np.floor(pt[1]))
    if not (0 <= r < h and 0 <= c < w):
        return -1
    return int(labels[r, c])


def probe_front(labels: np.ndarray, pos: np.ndarray, direction: np.ndarray, sides: set[int], reach: float) -> int | None:
    """First label beyond `pos` along `direction` that is not one of `sides` (crosses thin rims)."""
    for k in range(1, int(reach / PROBE_STEP) + 1):
        lab = _label_at(labels, pos + k * PROBE_STEP * direction)
        if lab not in sides:
            return lab if lab > 0 else None
    return None


def _inside_fraction(labels: np.ndarray, a: np.ndarray, b: np.ndarray, label: int, trim: float) -> float:
    """Share of the segment a-b inside `label`, ignoring `trim` px at each end: occluders often
    carry a rim there (the same allowance `probe_front` makes)."""
    gap = float(np.linalg.norm(b - a))
    n = max(int(np.ceil(gap)), 2)
    t = np.linspace(0.0, 1.0, n + 1)[1:-1]
    if gap > 2 * trim + 2:
        t = t[(t * gap > trim) & ((1.0 - t) * gap > trim)]
    return float(np.mean([_label_at(labels, a + s * (b - a)) == label for s in t]))


def _sides_match(e1: OccludedEnd, e2: OccludedEnd, medians: np.ndarray, p: Params) -> bool:
    """Walking from e1 through the occluder into e2 keeps each side on the same side.

    Both sides may differ slightly in color, or one side may be the very same region while the
    other changes (a rim line visible on one side of the occluder only).
    """
    pairs = ((e1.left, e2.right), (e1.right, e2.left))  # e2's sides are seen walking into the occluder
    des = [float(delta_e(medians[a], medians[b])) for a, b in pairs]
    if all(d < p.occlusion_side_de for d in des):
        return True
    return any(a == b for a, b in pairs) and max(des) < 3 * p.occlusion_side_de


def pair_occluded_ends(
    ends: list[OccludedEnd], labels: np.ndarray, medians: np.ndarray | None, p: Params
) -> list[tuple[int, int]]:
    """Greedy mutual pairing of occluded ends by alignment cost.

    `medians` (per label) is required for region contours; strokes carry `color`/`width` instead.
    """
    max_gap = p.px(p.occlusion_max_gap)
    by_front: dict[int, list[int]] = {}
    for i, e in enumerate(ends):
        by_front.setdefault(e.front, []).append(i)
    cands = []
    for idx in by_front.values():
        for a_i, i in enumerate(idx):
            for j in idx[a_i + 1 :]:
                a, b = ends[i], ends[j]
                v = b.pos - a.pos
                gap = float(np.linalg.norm(v))
                if gap < 1e-6 or gap > max_gap:
                    continue
                u = v / gap
                ang_a, ang_b = _angle(-a.away, u), _angle(-b.away, -u)
                if max(ang_a, ang_b) > p.occlusion_angle_deg:
                    continue
                if a.color is not None and b.color is not None:
                    if delta_e(a.color, b.color) >= p.occlusion_side_de:
                        continue
                    if not (0.6 <= a.width / max(b.width, 1e-6) <= 1.0 / 0.6):
                        continue
                elif medians is None or not _sides_match(a, b, medians, p):
                    continue
                if _inside_fraction(labels, a.pos, b.pos, a.front, p.px(p.occlusion_probe)) < p.occlusion_inside:
                    continue
                rms = _fit_rms(a, b, p)
                if rms > p.px(p.occlusion_fit_rms):
                    continue  # the two pieces do not lie on one smooth curve
                cands.append((ang_a + ang_b + gap / max_gap * 10.0 + rms * 10.0, i, j))
    used: set[int] = set()
    out = []
    for _, i, j in sorted(cands):
        if i in used or j in used:
            continue
        used.update((i, j))
        out.append((i, j))
    return out


@dataclass
class JointCurve:
    """Quadratic t = a + b s + c s^2 in a frame at `origin` with axis `u` and normal `perp`."""

    origin: np.ndarray
    u: np.ndarray
    perp: np.ndarray
    coef: np.ndarray  # (a, b, c)

    def lateral_error(self, pt: np.ndarray) -> float:
        d = pt - self.origin
        s, t = float(d @ self.u), float(d @ self.perp)
        a, b, c = self.coef
        return t - (a + b * s + c * s * s)

    def slope(self, pt: np.ndarray) -> float:
        s = float((pt - self.origin) @ self.u)
        return float(self.coef[1] + 2 * self.coef[2] * s)

    def tangent(self, pt: np.ndarray) -> np.ndarray:
        return unit(self.u + self.slope(pt) * self.perp)


def joint_curve(a: OccludedEnd, b: OccludedEnd, p: Params) -> JointCurve:
    """Least-squares curve through both visible pieces near their ends (straight if too few points)."""
    length = p.px(p.occlusion_fit_len)
    pts = np.vstack([e.points[arc_lengths(e.points) <= length] for e in (a, b)])
    u = unit(b.pos - a.pos)
    perp = np.array([u[1], -u[0]])
    d = pts - a.pos
    s, t = d @ u, d @ perp
    deg = 2 if len(pts) >= 6 and np.ptp(s) > 1e-6 else 1
    coef = np.polyfit(s, t, deg)[::-1]
    return JointCurve(a.pos.copy(), u, perp, np.pad(coef, (0, 3 - len(coef))))


def _fit_rms(a: OccludedEnd, b: OccludedEnd, p: Params) -> float:
    """RMS lateral residual of both pieces against their joint curve."""
    curve = joint_curve(a, b, p)
    length = p.px(p.occlusion_fit_len)
    pts = np.vstack([e.points[arc_lengths(e.points) <= length] for e in (a, b)])
    d = pts - curve.origin
    s, t = d @ curve.u, d @ curve.perp
    r = t - (curve.coef[0] + curve.coef[1] * s + curve.coef[2] * s * s)
    return float(np.sqrt(np.mean(r * r)))


def slide_onto(curve: JointCurve, pos: np.ndarray, along: np.ndarray | None, max_shift: float) -> np.ndarray:
    """Move `pos` along the occluder outline direction `along` onto the joint curve, clamped."""
    if along is None:
        return pos
    rate = float(along @ curve.perp) - curve.slope(pos) * float(along @ curve.u)
    if abs(rate) < 0.3:
        return pos
    lam = float(np.clip(-curve.lateral_error(pos) / rate, -max_shift, max_shift))
    return pos + lam * along


def _contour_ends(pm: PlanarMap, tangents: list[list[np.ndarray | None]], labels: np.ndarray, p: Params) -> list[OccludedEnd]:
    reach = p.px(p.occlusion_probe)
    r0, length = p.px(p.tangent_r0), p.px(p.tangent_len)
    ends: list[OccludedEnd] = []

    def add(pos, away, piece, away_left, away_right, along, ref):
        front = probe_front(labels, pos, -away, {away_left, away_right}, reach)
        if front is not None:
            ends.append(OccludedEnd(pos.copy(), away, front, away_right, away_left, piece, along, ref))

    for nid, node in enumerate(pm.nodes):
        for k, (eid, which) in enumerate(node.ends):
            e = pm.edges[eid]
            t = tangents[eid][which]
            if t is None or min(e.left, e.right) < 0:
                continue
            others = [tangents[o][w] for i, (o, w) in enumerate(node.ends) if i != k]
            if any(o is not None and np.array_equal(t, -o) for o in others):
                continue  # continues G1 through the node
            along = unit(others[0] - others[1]) if len(others) == 2 and all(o is not None for o in others) else None
            piece = e.points if which == 0 else e.points[::-1]
            al, ar = (e.left, e.right) if which == 0 else (e.right, e.left)
            add(node.pos, t, piece, al, ar, along, ("node", nid, k))

    window = max(int(round(p.px(p.corner_window))), 1)
    for eid, e in enumerate(pm.edges):
        if min(e.left, e.right) < 0:
            continue
        pts = e.points
        n = len(pts)
        # Lower than the fitting corner threshold: a contour meeting an occluder's rim at a shallow
        # angle bends only ~50 deg. Unpaired candidates never split anything.
        for c in corner_indices(pts, window, p.occlusion_corner_deg, closed=e.closed):
            if e.closed:
                after = np.roll(pts, -c, axis=0)
                before = np.vstack([after[:1], after[:0:-1]])
            else:
                after, before = pts[c:], pts[c::-1]
            if len(after) < 3 or len(before) < 3 or c in (0, n - 1):
                continue
            d_after = direction_from_end(after, r0, length)
            d_before = direction_from_end(before, r0, length)
            # Walking away along `after` the left label is e.left; along `before` the sides swap.
            add(pts[c], d_after, after, e.left, e.right, d_before, ("corner", eid, c, 1))
            add(pts[c], d_before, before, e.right, e.left, d_after, ("corner", eid, c, 0))
    return ends


def link_contours(
    pm: PlanarMap,
    tangents: list[list[np.ndarray | None]],
    labels: np.ndarray,
    medians: np.ndarray,
    p: Params,
) -> tuple[int, Breaks]:
    """Pair occluded region contours across their occluders.

    Updates node positions, edge points and `tangents` (per edge [t0, t1], fit_open convention) in
    place. Paired corner ends are returned as forced breaks for `fitting.fit_path`. Background
    (label 0) never occludes. Returns (pair count, breaks).
    """
    ends = _contour_ends(pm, tangents, labels, p)
    pairs = pair_occluded_ends(ends, labels, medians, p)
    max_shift = p.px(p.occlusion_max_slide)
    snap_len = p.px(p.tangent_r0 + p.tangent_len)
    breaks: Breaks = {}
    moved: set[tuple] = set()
    for i, j in pairs:
        curve = joint_curve(ends[i], ends[j], p)
        for idx in (i, j):
            end = ends[idx]
            kind = end.ref[0]
            key = end.ref[:3]
            if kind == "node":
                _, nid, k = end.ref
                node = pm.nodes[nid]
                if key not in moved:
                    node.pos = slide_onto(curve, node.pos, end.along, max_shift)
                    for eid, which in node.ends:
                        pm.edges[eid].points[0 if which == 0 else -1] = node.pos
                    moved.add(key)
                eid, which = node.ends[k]
                e = pm.edges[eid]
                start, step = (0, 1) if which == 0 else (len(e.points) - 1, -1)
                snap_toward(e.points, start, step, e.closed, curve, snap_len)
                t = curve.tangent(node.pos)
                tangents[eid][which] = t if float(t @ end.away) >= 0 else -t
            else:
                _, eid, c, side = end.ref
                pts = pm.edges[eid].points
                if key not in moved:
                    pts[c] = slide_onto(curve, pts[c], end.along, max_shift)
                    moved.add(key)
                snap_toward(pts, c, 1 if side == 1 else -1, pm.edges[eid].closed, curve, snap_len)
                t = curve.tangent(pts[c])
                slot = breaks.setdefault(eid, {}).setdefault(c, [None, None])
                slot[side] = t if float(t @ end.away) >= 0 else -t
    return len(pairs), breaks


def snap_toward(pts: np.ndarray, start: int, step: int, closed: bool, curve: JointCurve, length: float) -> None:
    """Pull the points next to an occluded end onto the joint curve, fully at the end and fading
    to nothing at `length`. Near the occluder anti-aliasing and rims bend the traced contour, the
    same reason tangent estimation skips its first r0 pixels. Open-edge end points stay put."""
    n = len(pts)
    i, s = start, 0.0
    prev = pts[start].copy()
    while True:
        i = (i + step) % n if closed else i + step
        if i == start or (not closed and not 0 < i < n - 1):
            return
        orig = pts[i].copy()
        s += float(np.linalg.norm(orig - prev))
        if s >= length:
            return
        pts[i] = orig - (1.0 - s / length) * curve.lateral_error(orig) * curve.perp
        prev = orig
