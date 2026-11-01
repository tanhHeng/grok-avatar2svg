import unittest

import cv2
import numpy as np

from grok_avatar2svg.colorspace import rgb_to_lab
from grok_avatar2svg.fitting import bezier_eval
from grok_avatar2svg.junctions import assign_tangents
from grok_avatar2svg.occlusion import link_contours
from grok_avatar2svg.params import Params
from grok_avatar2svg.planar import build_planar_map
from grok_avatar2svg.strokes import extract_strokes

BAR_TOP, BAR_BOTTOM = 26, 36  # rows covered by the occluding bar: [26, 36)


def cut_boundary_labels(lower_offset: int) -> np.ndarray:
    """Dark | light vertical boundary at x = 30, cut by a horizontal bar (label 3).

    Above the bar: 1 | 2, below: 4 | 5 (same colors as 1 | 2), shifted by `lower_offset` px.
    """
    labels = np.zeros((64, 60), dtype=np.int32)
    labels[:BAR_TOP, :30] = 1
    labels[:BAR_TOP, 30:] = 2
    labels[BAR_BOTTOM:, : 30 + lower_offset] = 4
    labels[BAR_BOTTOM:, 30 + lower_offset :] = 5
    labels[BAR_TOP:BAR_BOTTOM] = 3
    return labels


MEDIANS = np.array([[0, 0, 0], [40, 10, 0], [85, 5, 0], [99, 0, 0], [40, 10, 0], [85, 5, 0]], dtype=np.float32)


class ContourContinuity(unittest.TestCase):
    def link(self, labels):
        p = Params()
        pm = build_planar_map(labels, p)
        tangents, _ = assign_tangents(pm, p)
        pairs, _ = link_contours(pm, tangents, labels, MEDIANS, p)
        stems = {}
        for node in pm.nodes:
            if not BAR_TOP - 2 <= node.pos[1] <= BAR_BOTTOM + 2:
                continue  # only the ends where the boundary meets the bar
            for eid, which in node.ends:
                e = pm.edges[eid]
                if {e.left, e.right} in ({1, 2}, {4, 5}):
                    stems[frozenset((e.left, e.right))] = tangents[eid][which]
        return pairs, stems

    def test_boundary_cut_by_bar_continues_straight(self):
        pairs, stems = self.link(cut_boundary_labels(1))
        self.assertEqual(pairs, 1)
        top, bottom = stems[frozenset((1, 2))], stems[frozenset((4, 5))]
        # One curve through the bar: the tangent leaving upward is the reverse of the one leaving downward.
        self.assertLess(float(np.degrees(np.arccos(np.clip(-top @ bottom, -1, 1)))), 2.0)

    def test_offset_pieces_are_not_joined(self):
        pairs, _ = self.link(cut_boundary_labels(10))
        self.assertEqual(pairs, 0)


class StrokeCutByOccluder(unittest.TestCase):
    def test_line_ends_flat_at_occluder_and_lines_up_across_it(self):
        top, bottom = 34, 46  # rows covered by the occluding bar: [34, 46)
        labels = np.ones((80, 60), dtype=np.int32)
        labels[4:top, 29:32] = 2  # upper visible piece of a 3 px line
        labels[bottom:76, 29:32] = 4  # lower visible piece of the same line
        labels[top:bottom, :] = 3  # bar in front of it
        palette = np.array([[0, 0, 0], [240, 220, 220], [150, 100, 110], [255, 255, 255], [150, 100, 110]], dtype=np.uint8)
        # Blur stands in for anti-aliasing: real lines have no flat core, which is what makes them strokes.
        img = cv2.GaussianBlur(palette[labels], (0, 0), 0.7)
        _, strokes, _ = extract_strokes(rgb_to_lab(img), labels, Params())
        self.assertEqual(sorted(s.label for s in strokes), [2, 4])
        by_label = {s.label: s for s in strokes}

        def outline_points(s):
            return np.vstack([bezier_eval(b, np.linspace(0, 1, 50)) for sub in s.outline for b in sub])

        upper, lower = outline_points(by_label[2]), outline_points(by_label[4])
        # Both ends reach the bar edge at full width instead of tapering or rounding off before it.
        self.assertAlmostEqual(float(upper[:, 1].max()), top, delta=0.6)
        self.assertAlmostEqual(float(lower[:, 1].min()), bottom, delta=0.6)
        self.assertGreater(np.ptp(upper[upper[:, 1] > top - 1.0][:, 0]), 2.4)
        self.assertGreater(np.ptp(lower[lower[:, 1] < bottom + 1.0][:, 0]), 2.4)

        def end_dir(curve, y):
            first, last = curve[0], curve[-1]
            d = first[1] - first[0] if abs(first[0][1] - y) < abs(last[3][1] - y) else last[2] - last[3]
            return d / np.linalg.norm(d)

        up, down = end_dir(by_label[2].centerline[0], top), end_dir(by_label[4].centerline[0], bottom)
        self.assertLess(float(np.degrees(np.arccos(np.clip(-up @ down, -1, 1)))), 3.0)


if __name__ == "__main__":
    unittest.main()
