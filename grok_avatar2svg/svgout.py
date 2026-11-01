"""SVG emission (design.md section 8).

Layers, bottom to top: underlay strokes on shared foreground edges, region fills, strokes.
Background (label 0) is not drawn; the canvas stays transparent there.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .coloring import Fill
from .colorspace import lab_to_hex
from .params import Params
from .planar import PlanarMap
from .strokes import Stroke

Curve = list[np.ndarray]


@dataclass
class _Paint:
    ref: str            # value for fill= / stroke=
    gradient: str = ""  # <linearGradient> markup when ref is url(#...)


class _PathWriter:
    """Builds compact path data from absolute control points.

    Coordinates are rounded in absolute space first and relative offsets are taken between
    rounded values, so rounding never accumulates along a path.
    """

    def __init__(self, precision: int):
        self.precision = precision
        self.parts: list[str] = []
        self.cur = np.zeros(2)

    def _num(self, v: float) -> str:
        s = f"{v:.{self.precision}f}"
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        return "0" if s in ("-0", "") else s

    def _nums(self, values) -> str:
        out = ""
        for v in values:
            s = self._num(v)
            out += s if (not out or s.startswith("-")) else " " + s
        return out

    def _round(self, pt: np.ndarray) -> np.ndarray:
        return np.round(np.asarray(pt, dtype=np.float64), self.precision)

    def move(self, pt: np.ndarray) -> None:
        q = self._round(pt)
        self.parts.append("M" + self._nums(q))
        self.cur = q

    def cubic(self, bez: np.ndarray) -> None:
        q = self._round(bez[1:])
        rel = (q - self.cur).ravel()
        if not np.any(rel):
            return
        self.parts.append("c" + self._nums(rel))
        self.cur = q[-1]

    def close(self) -> None:
        self.parts.append("z")

    def curve(self, curve: Curve, closed: bool) -> None:
        if not curve:
            return
        self.move(curve[0][0])
        for bez in curve:
            self.cubic(bez)
        if closed:
            self.close()

    def data(self) -> str:
        return "".join(self.parts)


def _reverse(curve: Curve) -> Curve:
    return [b[::-1] for b in reversed(curve)]


def cycle_curve(curves: list[Curve], cycle: list[tuple[int, bool]]) -> Curve:
    out: Curve = []
    for edge_id, forward in cycle:
        c = curves[edge_id]
        out.extend(c if forward else _reverse(c))
    return out


def _paint(fill: Fill, gid: int) -> _Paint:
    if fill.gradient is None:
        return _Paint(lab_to_hex(fill.lab))
    p0, p1, lab0, lab1 = fill.gradient
    name = f"g{gid}"
    markup = (
        f'<linearGradient id="{name}" gradientUnits="userSpaceOnUse" '
        f'x1="{p0[0]:.1f}" y1="{p0[1]:.1f}" x2="{p1[0]:.1f}" y2="{p1[1]:.1f}">'
        f'<stop offset="0" stop-color="{lab_to_hex(lab0)}"/>'
        f'<stop offset="1" stop-color="{lab_to_hex(lab1)}"/></linearGradient>'
    )
    return _Paint(f"url(#{name})", markup)


def build_svg(
    pm: PlanarMap,
    curves: list[Curve],
    fills: dict[int, Fill],
    areas: np.ndarray,
    strokes: list[Stroke],
    stroke_fills: list[Fill],
    p: Params,
) -> str:
    """Assemble the SVG document.

    `curves[i]` is the fitted curve of `pm.edges[i]` (ignored for edges not referenced by any cycle).
    `areas` is indexable by label and only orders fill groups; any order renders the same shapes.
    """
    w, h = pm.width, pm.height
    defs: list[str] = []
    gid = 0

    # Fill groups: flat fills of equal color share one path; gradient fills stay separate.
    groups: dict[str, list[int]] = {}
    for label, fill in fills.items():
        if label not in pm.cycles:
            continue
        paint = _paint(fill, gid)
        if paint.gradient:
            gid += 1
            defs.append(paint.gradient)
        groups.setdefault(paint.ref, []).append(label)
    order = sorted(groups, key=lambda ref: -int(sum(areas[l] for l in groups[ref])))
    rank_of_label = {label: rank for rank, ref in enumerate(order) for label in groups[ref]}
    ref_of_label = {label: ref for ref in order for label in groups[ref]}

    fill_elems: list[str] = []
    for ref in order:
        pw = _PathWriter(p.precision)
        for label in groups[ref]:
            for cycle in pm.cycles[label]:
                pw.curve(cycle_curve(curves, cycle), closed=True)
        fill_elems.append(f'<path fill="{ref}" fill-rule="evenodd" d="{pw.data()}"/>')

    # Underlay (design 8.2): each edge between two drawn regions, painted like the side drawn first.
    used_edges = {eid for cycles in pm.cycles.values() for cyc in cycles for eid, _ in cyc}
    underlay: dict[str, _PathWriter] = {}
    for eid in sorted(used_edges):
        e = pm.edges[eid]
        if e.left not in rank_of_label or e.right not in rank_of_label:
            continue
        first = e.left if rank_of_label[e.left] <= rank_of_label[e.right] else e.right
        ref = ref_of_label[first]
        pw = underlay.setdefault(ref, _PathWriter(p.precision))
        pw.curve(curves[eid], closed=e.closed)
    underlay_elems = [
        f'<path fill="none" stroke="{ref}" stroke-width="{p.underlay_width:g}" '
        f'stroke-linecap="butt" stroke-linejoin="round" d="{pw.data()}"/>'
        for ref, pw in underlay.items()
    ]

    # Clip the underlay to the foreground so it never shows on the transparent background.
    clip_attr = ""
    if underlay_elems and pm.cycles.get(0):
        pw = _PathWriter(p.precision)
        pw.parts.append(f"M0 0H{w}V{h}H0z")
        for cycle in pm.cycles[0]:
            pw.curve(cycle_curve(curves, cycle), closed=True)
        defs.append(f'<clipPath id="fg"><path clip-rule="evenodd" d="{pw.data()}"/></clipPath>')
        clip_attr = ' clip-path="url(#fg)"'

    # Strokes: nonzero rule so overlapping chain outlines of one stroke union instead of cancelling.
    stroke_groups: dict[str, _PathWriter] = {}
    for stroke, fill in zip(strokes, stroke_fills):
        ref = lab_to_hex(fill.lab)
        pw = stroke_groups.setdefault(ref, _PathWriter(p.precision))
        for sub in stroke.outline:
            pw.curve(sub, closed=True)
    stroke_elems = [f'<path fill="{ref}" d="{pw.data()}"/>' for ref, pw in stroke_groups.items()]

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    if defs:
        out.append("<defs>" + "".join(defs) + "</defs>")
    if underlay_elems:
        out.append(f"<g{clip_attr}>" + "".join(underlay_elems) + "</g>")
    out.append("<g>" + "".join(fill_elems) + "</g>")
    if stroke_elems:
        out.append("<g>" + "".join(stroke_elems) + "</g>")
    out.append("</svg>")
    return "\n".join(out) + "\n"
