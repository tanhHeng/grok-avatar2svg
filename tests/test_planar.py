import unittest
from collections import Counter

import numpy as np

from avatar2svg.junctions import assign_tangents
from avatar2svg.params import Params
from avatar2svg.planar import build_planar_map


def oriented(pm, eid, forward):
    pts = pm.edges[eid].points
    return pts if forward else pts[::-1]


class PlanarMapInvariants(unittest.TestCase):
    def check_invariants(self, labels, p=None):
        pm = build_planar_map(labels, p or Params())
        uses = Counter()
        for label, cycles in pm.cycles.items():
            for cyc in cycles:
                for eid, fwd in cyc:
                    e = pm.edges[eid]
                    self.assertEqual(label, e.left if fwd else e.right)
                    uses[(eid, fwd)] += 1
                if len(cyc) == 1 and pm.edges[cyc[0][0]].closed:
                    continue
                for (ea, fa), (eb, fb) in zip(cyc, cyc[1:] + cyc[:1]):
                    np.testing.assert_allclose(oriented(pm, ea, fa)[-1], oriented(pm, eb, fb)[0], atol=1e-9)
        for eid, e in enumerate(pm.edges):
            self.assertEqual(uses[(eid, True)], 1 if e.left >= 0 else 0, f"edge {eid} forward")
            self.assertEqual(uses[(eid, False)], 1 if e.right >= 0 else 0, f"edge {eid} backward")
        return pm

    def test_region_with_hole_has_outer_and_inner_cycle(self):
        labels = np.ones((30, 30), dtype=np.int32)
        labels[5:25, 5:25] = 2
        yy, xx = np.mgrid[:30, :30]
        labels[(yy - 15) ** 2 + (xx - 15) ** 2 < 25] = 3
        pm = self.check_invariants(labels)
        self.assertEqual(len(pm.cycles[2]), 2)

    def test_checkerboard_corner(self):
        labels = np.ones((20, 20), dtype=np.int32)
        labels[5:10, 5:10] = 2
        labels[10:15, 10:15] = 2
        self.check_invariants(labels)

    def test_close_nodes_are_merged_without_breaking_cycles(self):
        # A one-pixel sliver puts several nodes within the cluster radius.
        labels = np.ones((30, 30), dtype=np.int32)
        labels[:, 15:] = 2
        labels[10:20, 15:] = 3
        labels[14, 14:16] = 4
        self.check_invariants(labels)


class Junctions(unittest.TestCase):
    def test_t_junction_bar_edges_share_one_tangent(self):
        labels = np.ones((40, 40), dtype=np.int32)
        labels[:, 20:] = 2
        labels[20:, 20:] = 3
        pm = build_planar_map(labels, Params())
        tangents, counts = assign_tangents(pm, Params())
        self.assertEqual(counts["T"], 1)
        center = min(range(len(pm.nodes)), key=lambda n: np.linalg.norm(pm.nodes[n].pos - [20, 20]))
        bar = [tangents[eid][which] for eid, which in pm.nodes[center].ends
               if {pm.edges[eid].left, pm.edges[eid].right} in ({1, 2}, {1, 3})]
        self.assertEqual(len(bar), 2)
        np.testing.assert_allclose(bar[0], -bar[1], atol=1e-9)


if __name__ == "__main__":
    unittest.main()
