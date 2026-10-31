"""Polyline geometry shared by the boundary graph (design.md section 5) and strokes (section 3).

All points are float (N, 2) arrays in SVG user space: x = column, y = row, pixel (r, c)
covers [c, c+1] x [r, r+1].
"""

from __future__ import annotations

import numpy as np


def arc_lengths(points: np.ndarray) -> np.ndarray:
    """Cumulative arc length, s[0] = 0."""
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def resample(points: np.ndarray, step: float = 1.0, closed: bool = False) -> np.ndarray:
    """Uniform arc-length resampling. Open curves keep both endpoints; closed curves do not repeat the start."""
    pts = np.asarray(points, dtype=np.float64)
    if closed:
        pts = np.vstack([pts, pts[:1]])
    s = arc_lengths(pts)
    total = s[-1]
    if total < 1e-9 or len(pts) < 2:
        return pts[:1].copy()
    n = max(int(round(total / step)), 1)
    t = np.linspace(0.0, total, n + 1)
    if closed:
        t = t[:-1]
    return np.stack([np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])], axis=1)


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return np.asarray(v, dtype=np.float64) / n if n > 1e-12 else np.array([1.0, 0.0])


def direction_from_end(points: np.ndarray, r0: float, length: float) -> np.ndarray:
    """Unit direction in which a branch leaves its first point (design.md 5.2).

    `points` starts at the node. Points within `r0` of arc length are ignored because
    anti-aliasing and pixel staircase corrupt them; the direction is the principal axis of
    the points in [r0, r0 + length], oriented away from the node. Short branches shrink the
    window proportionally instead of failing.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return np.array([1.0, 0.0])
    s = arc_lengths(pts)
    total = s[-1]
    lo, hi = r0, r0 + length
    if total < hi:
        lo, hi = min(r0, total / 3.0), total
    sel = pts[(s >= lo) & (s <= hi)]
    if len(sel) < 2:
        return unit(pts[-1] - pts[0])
    centered = sel - sel.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    d = vt[0]
    if np.dot(d, sel.mean(axis=0) - pts[0]) < 0:
        d = -d
    return unit(d)


def continuity_cost(d_i: np.ndarray, d_j: np.ndarray) -> float:
    """Angle in degrees between d_i and -d_j; 0 means the two branches form a straight line."""
    c = float(np.clip(-np.dot(d_i, d_j), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def pair_branches(dirs: list[np.ndarray], max_angle_deg: float) -> list[tuple[int, int]]:
    """Pair branches at a junction that continue through it (design.md 5.3).

    `dirs` are unit directions leaving the junction. Returned pairs share a tangent
    (T bar or X crossing); unpaired branches end at the junction.
    Degree 4 prefers the pairing with minimum total cost when both of its pairs pass the
    threshold (X type); otherwise, and for every other degree, pairs are chosen greedily by cost.
    """
    n = len(dirs)
    cost = {(i, j): continuity_cost(dirs[i], dirs[j]) for i in range(n) for j in range(i + 1, n)}
    if n == 4:
        pairings = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]
        best = min(pairings, key=lambda pr: cost[pr[0]] + cost[pr[1]])
        if all(cost[p] < max_angle_deg for p in best):
            return list(best)
    used: set[int] = set()
    out: list[tuple[int, int]] = []
    for (i, j), c in sorted(cost.items(), key=lambda kv: kv[1]):
        if c >= max_angle_deg:
            break
        if i in used or j in used:
            continue
        out.append((i, j))
        used.update((i, j))
    return out


def turning_angles(points: np.ndarray, window: int, closed: bool) -> np.ndarray:
    """Turning angle in degrees at every point using neighbors `window` samples away.

    Open curves get 0 at the first/last `window` samples, where the angle is undefined.
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    ang = np.zeros(n)
    if n < 2 * window + 1:
        return ang
    idx = np.arange(n)
    if closed:
        prev, nxt, valid = (idx - window) % n, (idx + window) % n, np.ones(n, dtype=bool)
    else:
        prev, nxt = np.clip(idx - window, 0, n - 1), np.clip(idx + window, 0, n - 1)
        valid = (idx >= window) & (idx < n - window)
    a = pts - pts[prev]
    b = pts[nxt] - pts
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    ok = valid & (na > 1e-9) & (nb > 1e-9)
    cos = np.einsum("ij,ij->i", a[ok], b[ok]) / (na[ok] * nb[ok])
    ang[ok] = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    return ang


def corner_indices(points: np.ndarray, window: int, angle_deg: float, closed: bool) -> list[int]:
    """Indices of corners (design.md 5.4): local maxima of the turning angle above `angle_deg`.

    `points` should be resampled at ~1 px so `window` is in pixels. Maxima are suppressed
    within +-window so one sharp tip yields one corner.
    """
    ang = turning_angles(points, window, closed)
    n = len(ang)
    cand = np.nonzero(ang > angle_deg)[0]
    out: list[int] = []
    for i in cand[np.argsort(-ang[cand], kind="stable")]:
        if closed:
            near = any(min(abs(i - j), n - abs(i - j)) <= window for j in out)
        else:
            near = any(abs(i - j) <= window for j in out)
        if not near:
            out.append(int(i))
    return sorted(out)


def tangent_at(points: np.ndarray, i: int, half: int = 3, closed: bool = False) -> np.ndarray:
    """Central-difference unit tangent at sample i, pointing forward along the polyline."""
    n = len(points)
    if closed:
        a, b = points[(i - half) % n], points[(i + half) % n]
    else:
        a, b = points[max(i - half, 0)], points[min(i + half, n - 1)]
    return unit(b - a)


def signed_area(points: np.ndarray) -> float:
    """Shoelace area of a closed polygon; positive when counter-clockwise in a y-up frame."""
    x, y = points[:, 0], points[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))
