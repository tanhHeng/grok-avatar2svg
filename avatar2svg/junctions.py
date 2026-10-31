"""Junction typing and end tangents (design.md 5.2-5.3)."""

from __future__ import annotations

import numpy as np

from .geometry import direction_from_end, pair_branches, unit
from .params import Params
from .planar import PlanarMap

Tangent = np.ndarray | None


def assign_tangents(pm: PlanarMap, p: Params) -> tuple[list[list[Tangent]], dict[str, int]]:
    """Per edge (t0, t1) in `fitting.fit_open` convention, plus node type counts.

    Paired branches (T bar, X crossing) get exactly opposite tangents so the curve passes the node
    G1. Nodes on the image border are left out of the counts.
    """
    tangents: list[list[Tangent]] = [[None, None] for _ in pm.edges]
    counts = {"T": 0, "Y": 0, "X": 0, "other": 0}
    r0, length = p.px(p.tangent_r0), p.px(p.tangent_len)
    for node in pm.nodes:
        if not node.ends:
            continue
        dirs = []
        for eid, which in node.ends:
            pts = pm.edges[eid].points
            dirs.append(direction_from_end(pts if which == 0 else pts[::-1], r0, length))
        pairs = pair_branches(dirs, p.pair_angle_deg)
        for i, j in pairs:
            u = unit(dirs[i] - dirs[j])
            dirs[i], dirs[j] = u, -u
        for (eid, which), d in zip(node.ends, dirs):
            tangents[eid][which] = d

        if any(min(pm.edges[eid].left, pm.edges[eid].right) < 0 for eid, _ in node.ends):
            continue
        deg = len(node.ends)
        if deg == 3:
            counts["T" if len(pairs) == 1 else "Y"] += 1
        elif deg == 4 and len(pairs) == 2:
            counts["X"] += 1
        else:
            counts["other"] += 1
    return tangents, counts
