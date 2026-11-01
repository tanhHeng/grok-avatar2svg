from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .background import drop_background, find_background
from .coloring import Fill, region_fills, unify_colors
from .colorspace import rgb_to_lab
from .fitting import fit_path
from .junctions import assign_tangents
from .occlusion import link_contours
from .params import Params
from .planar import build_planar_map
from .regions import region_areas, region_medians
from .segment import segment
from .strokes import extract_strokes
from .subpixel import refine_edges
from .svgout import build_svg


@dataclass
class Result:
    svg: str
    background_lab: np.ndarray | None
    stats: dict[str, object] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)


def load_image(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Returns (uint8 RGB, opaque mask). Pixels with alpha 0 are treated as background."""
    img = Image.open(path)
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        rgba = np.asarray(img.convert("RGBA"))
        return np.ascontiguousarray(rgba[..., :3]), rgba[..., 3] > 0
    rgb = np.asarray(img.convert("RGB"))
    return rgb, np.ones(rgb.shape[:2], dtype=bool)


def vectorize(rgb: np.ndarray, opaque: np.ndarray, params: Params | None = None) -> Result:
    h, w = rgb.shape[:2]
    p = (params or Params()).for_image(w, h)
    timings: dict[str, float] = {}

    def step(name, fn, *args):
        t = time.perf_counter()
        out = fn(*args)
        timings[name] = time.perf_counter() - t
        return out

    lab = step("lab", rgb_to_lab, rgb)
    labels = step("segment", segment, lab, opaque, p)
    n_segments = int(labels.max())
    background, c_bg = step("background", find_background, lab, labels, p)
    labels, strokes, stroke_mask = step("strokes", extract_strokes, lab, labels, p)
    labels = drop_background(labels, background)
    pm = step("planar", build_planar_map, labels, p)
    medians = region_medians(lab, labels)
    step("subpixel", refine_edges, pm, lab, medians, stroke_mask, p)
    tangents, node_types = step("junctions", assign_tangents, pm, p)
    occlusion_pairs, breaks = step("occlusion", link_contours, pm, tangents, labels, medians, p)

    t = time.perf_counter()
    used = {eid for cycles in pm.cycles.values() for cyc in cycles for eid, _ in cyc}
    window = max(int(round(p.px(p.corner_window))), 1)
    curves: list[list[np.ndarray]] = [[] for _ in pm.edges]
    for eid in used:
        e = pm.edges[eid]
        t0, t1 = tangents[eid]
        curves[eid] = fit_path(
            e.points, p.px(p.eps_region), closed=e.closed, t0=t0, t1=t1,
            corner_window=window, corner_angle=p.corner_angle_deg, breaks=breaks.get(eid),
            smooth=p.px(p.smooth_region),
        )
    timings["fitting"] = time.perf_counter() - t

    t = time.perf_counter()
    # Stroke pixels show the stroke color; masking them out keeps them from biasing region colors.
    color_labels = np.where(stroke_mask, 0, labels)
    fills = region_fills(lab, color_labels, p)
    stroke_fills = [Fill(lab=np.array(s.color, dtype=np.float64)) for s in strokes]
    unify_colors(list(fills.values()) + stroke_fills, p)
    timings["coloring"] = time.perf_counter() - t

    svg = step("svg", build_svg, pm, curves, fills, region_areas(labels), strokes, stroke_fills, p)
    stats = {
        "segments": n_segments,
        "regions": int(labels.max()),
        "strokes": len(strokes),
        "nodes": len(pm.nodes),
        "edges": len(used),
        "edge_beziers": sum(len(curves[e]) for e in used),
        "node_types": node_types,
        "occlusion_pairs": occlusion_pairs,
        "gradients": sum(f.gradient is not None for f in fills.values()),
    }
    return Result(svg=svg, background_lab=c_bg, stats=stats, timings=timings)
