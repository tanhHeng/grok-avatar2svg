"""Cubic Bezier fitting (design.md 6.2-6.4), after Schneider, "An Algorithm for Automatically
Fitting Digitized Curves", Graphics Gems (1990).

Tangent convention: `t0` is the unit tangent at the first point pointing into the curve, `t1` the
unit tangent at the last point pointing back into the curve (a straight A->B has t0 = unit(B - A),
t1 = unit(A - B)).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .geometry import arc_lengths, corner_indices, direction_from_end, resample, tangent_at, unit

MAX_REPARAM = 8
# Newton reparameterization is attempted while the error is within this many eps. Schneider uses 4;
# 16 with 8 iterations gives ~5 % fewer segments on example.png at the same error, ~2x fitting time.
REPARAM_RANGE = 16.0
SPLIT_TANGENT_HALF = 3


@dataclass
class _Seg:
    bez: np.ndarray
    i0: int
    i1: int
    t0: np.ndarray
    t1: np.ndarray


def bezier_eval(bez: np.ndarray, u: np.ndarray) -> np.ndarray:
    mt = 1.0 - u
    return (
        (mt ** 3)[:, None] * bez[0]
        + (3 * mt * mt * u)[:, None] * bez[1]
        + (3 * mt * u * u)[:, None] * bez[2]
        + (u ** 3)[:, None] * bez[3]
    )


def _d1(bez: np.ndarray, u: np.ndarray) -> np.ndarray:
    mt = 1.0 - u
    return (
        (3 * mt * mt)[:, None] * (bez[1] - bez[0])
        + (6 * mt * u)[:, None] * (bez[2] - bez[1])
        + (3 * u * u)[:, None] * (bez[3] - bez[2])
    )


def _d2(bez: np.ndarray, u: np.ndarray) -> np.ndarray:
    return (6 * (1.0 - u))[:, None] * (bez[2] - 2 * bez[1] + bez[0]) + (6 * u)[:, None] * (
        bez[3] - 2 * bez[2] + bez[1]
    )


def _chord_params(pts: np.ndarray) -> np.ndarray:
    s = arc_lengths(pts)
    return s / s[-1] if s[-1] > 1e-12 else np.linspace(0.0, 1.0, len(pts))


def _straight(p0: np.ndarray, p3: np.ndarray, t0: np.ndarray, t1: np.ndarray) -> np.ndarray:
    d = float(np.linalg.norm(p3 - p0)) / 3.0
    return np.array([p0, p0 + t0 * d, p3 + t1 * d, p3])


def _generate(pts: np.ndarray, u: np.ndarray, t0: np.ndarray, t1: np.ndarray) -> np.ndarray:
    """Least-squares handle lengths for fixed end points and end tangents."""
    p0, p3 = pts[0], pts[-1]
    mt = 1.0 - u
    b0, b1, b2, b3 = mt ** 3, 3 * mt * mt * u, 3 * mt * u * u, u ** 3
    a1 = b1[:, None] * t0
    a2 = b2[:, None] * t1
    c00, c01, c11 = (a1 * a1).sum(), (a1 * a2).sum(), (a2 * a2).sum()
    rest = pts - ((b0 + b1)[:, None] * p0 + (b2 + b3)[:, None] * p3)
    x0, x1 = (a1 * rest).sum(), (a2 * rest).sum()
    det = c00 * c11 - c01 * c01
    chord = float(np.linalg.norm(p3 - p0))
    if abs(det) > 1e-12:
        al1 = (x0 * c11 - x1 * c01) / det
        al2 = (c00 * x1 - c01 * x0) / det
    else:
        al1 = al2 = -1.0
    arc = float(arc_lengths(pts)[-1])
    if not (1e-6 * chord <= al1 <= arc and 1e-6 * chord <= al2 <= arc):
        # Degenerate system, reversed handles, or handles longer than the data itself (an
        # ill-conditioned solve that would swing the curve far away between samples).
        al1 = al2 = max(chord, arc * 0.5) / 3.0
    return np.array([p0, p0 + al1 * t0, p3 + al2 * t1, p3])


def _max_error(pts: np.ndarray, bez: np.ndarray, u: np.ndarray) -> tuple[float, int]:
    d = np.linalg.norm(bezier_eval(bez, u) - pts, axis=1)
    i = int(np.argmax(d))
    return float(d[i]), i


def _reparameterize(pts: np.ndarray, bez: np.ndarray, u: np.ndarray) -> np.ndarray:
    q = bezier_eval(bez, u) - pts
    q1, q2 = _d1(bez, u), _d2(bez, u)
    num = (q * q1).sum(axis=1)
    den = (q1 * q1).sum(axis=1) + (q * q2).sum(axis=1)
    step = np.divide(num, den, out=np.zeros_like(num), where=np.abs(den) > 1e-12)
    out = np.clip(u - step, 0.0, 1.0)
    out[0], out[-1] = 0.0, 1.0
    return out


def _fit_single(pts: np.ndarray, t0: np.ndarray, t1: np.ndarray, eps: float) -> tuple[np.ndarray, float, int]:
    """One cubic with Newton refinement when close; returns (bezier, max error, index of max error)."""
    if len(pts) <= 3:
        # Too few points to determine two handle lengths; use a straight segment.
        bez = _straight(pts[0], pts[-1], t0, t1)
        if len(pts) == 3:
            err, i = _max_error(pts, bez, _chord_params(pts))
            return bez, err, i
        return bez, 0.0, 0
    u = _chord_params(pts)
    bez = _generate(pts, u, t0, t1)
    err, i = _max_error(pts, bez, u)
    if eps < err <= REPARAM_RANGE * eps:
        for _ in range(MAX_REPARAM):
            u = _reparameterize(pts, bez, u)
            bez = _generate(pts, u, t0, t1)
            err, i = _max_error(pts, bez, u)
            if err <= eps:
                break
    return bez, err, i


def _split_tangent(pts: np.ndarray, i: int) -> np.ndarray:
    a = pts[max(i - SPLIT_TANGENT_HALF, 0)]
    b = pts[min(i + SPLIT_TANGENT_HALF, len(pts) - 1)]
    return unit(b - a)


def _fit_segments(pts: np.ndarray, t0: np.ndarray, t1: np.ndarray, eps: float) -> list[_Seg]:
    out: list[_Seg] = []
    stack = [(0, len(pts) - 1, t0, t1)]
    while stack:
        i0, i1, a, b = stack.pop()
        sub = pts[i0 : i1 + 1]
        bez, err, i = _fit_single(sub, a, b, eps)
        if err <= eps or len(sub) <= 3:
            out.append(_Seg(bez, i0, i1, a, b))
            continue
        k = i0 + min(max(i, 1), len(sub) - 2)
        tc = _split_tangent(pts, k)
        # Pushed in reverse so the left half is emitted first.
        stack.append((k, i1, tc, b))
        stack.append((i0, k, a, -tc))
    return out


def _simplify(pts: np.ndarray, segs: list[_Seg], eps: float) -> list[_Seg]:
    """Merge adjacent G1 segments while one cubic still fits their combined points (design 6.4)."""
    changed = True
    while changed and len(segs) > 1:
        changed = False
        merged: list[_Seg] = []
        k = 0
        while k < len(segs):
            if k + 1 < len(segs):
                a, b = segs[k], segs[k + 1]
                bez, err, _ = _fit_single(pts[a.i0 : b.i1 + 1], a.t0, b.t1, eps)
                if err <= eps:
                    merged.append(_Seg(bez, a.i0, b.i1, a.t0, b.t1))
                    k += 2
                    changed = True
                    continue
            merged.append(segs[k])
            k += 1
        segs = merged
    return segs


def fit_open(points: np.ndarray, t0: np.ndarray, t1: np.ndarray, eps: float) -> list[np.ndarray]:
    """Fit a G1 chain of cubics through `points`, exact at both ends, max deviation <= eps."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 1:
        return [np.repeat(pts, 4, axis=0)]
    t0, t1 = unit(t0), unit(t1)
    segs = _simplify(pts, _fit_segments(pts, t0, t1, eps), eps)
    segs[0].bez[0] = pts[0]
    segs[-1].bez[3] = pts[-1]
    return [s.bez for s in segs]


def _end_tangents(seg: np.ndarray, window: float) -> tuple[np.ndarray, np.ndarray]:
    return direction_from_end(seg, 0.0, window), direction_from_end(seg[::-1], 0.0, window)


def fit_path(
    points: np.ndarray,
    eps: float,
    *,
    closed: bool,
    t0: np.ndarray | None = None,
    t1: np.ndarray | None = None,
    corner_window: int = 4,
    corner_angle: float = 60.0,
    breaks: dict[int, list[np.ndarray | None]] | None = None,
    smooth: float = 0.0,
) -> list[np.ndarray]:
    """Resample at 1 px, split at corners (design 5.4), fit each piece with `fit_open`.

    Open paths keep their exact end points; `t0`/`t1` override the end tangents. Closed paths
    return a loop whose last end point equals the first start point and that is G1 everywhere
    except at corners. `breaks` maps indices of `points` to forced split points, kept exactly,
    with optional tangents [toward the piece before, into the piece after]; detected corners
    within `corner_window` of a forced split are dropped in its favor. `smooth` > 0 Gaussian-
    smooths each piece (sigma in px) before fitting, so waves longer than the fit tolerance can
    follow are removed while corners, breaks and end points stay exact.
    """
    src = np.asarray(points, dtype=np.float64)
    pts = resample(src, 1.0, closed=closed)
    n = len(pts)
    if n == 1:
        return [np.repeat(pts, 4, axis=0)]
    window = float(corner_window)
    forced = _map_breaks(src, pts, breaks, closed)
    out: list[np.ndarray] = []

    def cyc_dist(a: int, b: int) -> int:
        d = abs(a - b)
        return min(d, n - d) if closed else d

    def overrides(a: int, b: int, s0: np.ndarray, s1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if a in forced and forced[a][1] is not None:
            s0 = forced[a][1]
        if b in forced and forced[b][0] is not None:
            s1 = forced[b][0]
        return s0, s1

    found = corner_indices(pts, corner_window, corner_angle, closed=closed)
    splits = set(forced) | {c for c in found if all(cyc_dist(c, f) > corner_window for f in forced)}

    if not closed:
        cuts = sorted({0, n - 1} | splits)
        for k, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
            seg = smooth_piece(pts[a : b + 1], smooth)
            s0, s1 = overrides(a, b, *_end_tangents(seg, window))
            if k == 0 and t0 is not None:
                s0 = t0
            if k == len(cuts) - 2 and t1 is not None:
                s1 = t1
            out.extend(fit_open(seg, s0, s1, eps))
        return out

    if n < 4:
        ring = np.vstack([pts, pts[:1]])
        return [_straight(ring[k], ring[k + 1], unit(ring[k + 1] - ring[k]), unit(ring[k] - ring[k + 1])) for k in range(n)]

    if splits:
        corners = sorted(splits)
        first = corners[0]
        ring = np.vstack([np.roll(pts, -first, axis=0), pts[first : first + 1]])
        cuts = sorted((c - first) % n for c in corners) + [n]
        for a, b in zip(cuts[:-1], cuts[1:]):
            seg = smooth_piece(ring[a : b + 1], smooth)
            s0, s1 = overrides((a + first) % n, (b + first) % n, *_end_tangents(seg, window))
            out.extend(fit_open(seg, s0, s1, eps))
    else:
        if smooth > 0:
            pts = ndimage.gaussian_filter1d(pts, smooth, axis=0, mode="wrap")
        ring = np.vstack([pts, pts[:1]])
        mid = n // 2
        ta = tangent_at(pts, 0, SPLIT_TANGENT_HALF, closed=True)
        tb = tangent_at(pts, mid, SPLIT_TANGENT_HALF, closed=True)
        out.extend(fit_open(ring[: mid + 1], ta, -tb, eps))
        out.extend(fit_open(ring[mid:], tb, -ta, eps))
    out[-1][3] = out[0][0]
    return out


def smooth_piece(seg: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing that keeps both end points and, to first order, the end directions:
    the piece is extended by point reflection through each end before filtering."""
    n = len(seg)
    if sigma <= 0 or n < 3:
        return seg
    k = min(n - 1, int(np.ceil(3 * sigma)))
    ext = np.vstack([2 * seg[0] - seg[k:0:-1], seg, 2 * seg[-1] - seg[-2 : -k - 2 : -1]])
    out = ndimage.gaussian_filter1d(ext, sigma, axis=0, mode="nearest")[k : k + n]
    out[0], out[-1] = seg[0], seg[-1]
    return out


def _map_breaks(
    src: np.ndarray, pts: np.ndarray, breaks: dict[int, list[np.ndarray | None]] | None, closed: bool
) -> dict[int, list[np.ndarray | None]]:
    """Map break indices of the input polyline onto the resampled one (by arc length) and pin the
    resampled point to the exact input position. Open-path end points cannot be breaks."""
    if not breaks:
        return {}
    ring = np.vstack([src, src[:1]]) if closed else src
    s = arc_lengths(ring)
    total = s[-1]
    n = len(pts)
    n_seg = n if closed else n - 1
    out: dict[int, list[np.ndarray | None]] = {}
    for idx, tans in breaks.items():
        ri = int(round(s[idx] / total * n_seg)) if total > 0 else 0
        if closed:
            ri %= n
        elif not 0 < ri < n - 1:
            continue
        pts[ri] = src[idx]
        out[ri] = tans
    return out
