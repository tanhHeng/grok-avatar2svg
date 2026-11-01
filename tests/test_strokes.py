import unittest

import cv2
import numpy as np

from avatar2svg.colorspace import rgb_to_lab
from avatar2svg.fitting import bezier_eval
from avatar2svg.geometry import signed_area
from avatar2svg.params import Params
from avatar2svg.strokes import _Branch, _Graph, _Node, _outline_loops, _simplify_graph, extract_strokes

PALETTE = np.array([[0, 0, 0], [250, 240, 240], [200, 150, 160]], dtype=np.uint8)


def band(width: int) -> tuple[np.ndarray, np.ndarray]:
    """A long vertical band of label 2 on label 1, anti-aliased like real input."""
    labels = np.ones((120, 60), dtype=np.int32)
    labels[10:110, 30 - width // 2 : 30 - width // 2 + width] = 2
    img = cv2.GaussianBlur(PALETTE[labels], (0, 0), 0.7)
    return rgb_to_lab(img), labels


def line_image(x0: float, x1: float, fade: tuple[int, int] | None = None) -> np.ndarray:
    """Lab image of a vertical line covering x0 <= x < x1 (exact pixel coverage) over rows 10-109,
    optionally fading linearly to the background between the two `fade` rows; anti-aliased."""
    cols = np.arange(60)
    alpha = np.zeros((120, 60))
    alpha[10:110] = np.clip(np.minimum(cols + 1, x1) - np.maximum(cols, x0), 0.0, 1.0)
    if fade is not None:
        start, end = fade
        alpha *= np.clip((end - np.arange(120)) / (end - start), 0.0, 1.0)[:, None]
    alpha = cv2.GaussianBlur(alpha, (0, 0), 0.7)[..., None]
    return rgb_to_lab(np.round(PALETTE[1] * (1.0 - alpha) + PALETTE[2] * alpha).astype(np.uint8))


def outline_points(stroke) -> np.ndarray:
    return np.vstack([bezier_eval(b, np.linspace(0.0, 1.0, 200)) for curve in stroke.outline for b in curve])


def extent_at(points: np.ndarray, y: float) -> tuple[float, float]:
    xs = points[np.abs(points[:, 1] - y) < 0.3, 0]
    return float(xs.min()), float(xs.max())


class StrokeClassification(unittest.TestCase):
    def test_thin_line_is_a_stroke(self):
        lab, labels = band(3)
        _, strokes, _ = extract_strokes(lab, labels, Params())
        self.assertEqual([s.label for s in strokes], [2])

    def test_narrow_flat_band_stays_a_region(self):
        # 9 px wide and 100 px long passes the thin-and-long test, but its core is flat color:
        # a color block such as a hair strand, whose edges must be fitted as region boundaries.
        lab, labels = band(9)
        out, strokes, mask = extract_strokes(lab, labels, Params())
        self.assertEqual(strokes, [])
        self.assertFalse(mask.any())
        np.testing.assert_array_equal(out, labels)


class LineFromInk(unittest.TestCase):
    def test_stroke_follows_the_ink_not_the_region(self):
        # The line covers 28.5 <= x < 31.5, but segmentation gave it columns 29-32: one pixel too
        # far right and one too wide (2 x EDT of that region draws it 4 px wide).
        lab = line_image(28.5, 31.5)
        labels = np.ones((120, 60), dtype=np.int32)
        labels[10:110, 29:33] = 2
        _, strokes, _ = extract_strokes(lab, labels, Params())
        center = np.vstack([bezier_eval(b, np.linspace(0.0, 1.0, 200)) for b in strokes[0].centerline[0]])
        self.assertAlmostEqual(float(center[np.argmin(np.abs(center[:, 1] - 60.0)), 0]), 30.0, delta=0.15)
        # The drawn color is the line's blurred core, a little lighter than the line itself, so the
        # same ink takes a little more than 3 px.
        outline = outline_points(strokes[0])
        mean_width = abs(signed_area(outline)) / float(np.ptp(outline[:, 1]))
        self.assertGreater(mean_width, 3.0)
        self.assertLess(mean_width, 3.6)


class TracedTips(unittest.TestCase):
    def test_fading_tail_tapers_out_where_the_ink_ends(self):
        # The line fades out between rows 60 and 100, but its region stops at row 60: the faint
        # part went to the background region. Ink falls below tip_min_ink (0.3) around row 88.
        lab = line_image(28.5, 31.5, fade=(60, 100))
        labels = np.ones((120, 60), dtype=np.int32)
        labels[10:61, 29:32] = 2
        _, strokes, _ = extract_strokes(lab, labels, Params())
        pts = outline_points(strokes[0])
        self.assertGreater(pts[:, 1].max(), 82.0)
        self.assertLess(pts[:, 1].max(), 95.0)
        def width(y):
            left, right = extent_at(pts, y)
            return right - left
        self.assertLess(width(80.0), 0.75 * width(40.0))

    def test_gap_in_a_line_is_bridged(self):
        # One line in the image, split by segmentation into rows 10-49 and 62-109 with the gap given
        # to the background region.
        lab = line_image(28.5, 31.5)
        labels = np.ones((120, 60), dtype=np.int32)
        labels[10:50, 29:32] = 2
        labels[62:110, 29:32] = 3
        _, strokes, _ = extract_strokes(lab, labels, Params())
        by_label = {s.label: outline_points(s) for s in strokes}
        self.assertGreaterEqual(by_label[2][:, 1].max(), 61.0)
        self.assertLessEqual(by_label[3][:, 1].min(), 51.0)


class SpurPruning(unittest.TestCase):
    def fork(self, spur: tuple[float, float]) -> list[np.ndarray]:
        """Free end points left after simplifying a junction at (30, 30) joining a line down to
        (30, 60), a side branch to (40, 30) and a short spur along `spur`."""
        hub = np.array([30.0, 30.0])
        ends = [hub + np.array(spur), np.array([40.0, 30.0]), np.array([30.0, 60.0])]
        graph = _Graph(nodes=[_Node(hub, True)] + [_Node(e, False) for e in ends])
        for i, e in enumerate(ends):
            n = int(np.ceil(np.linalg.norm(e - hub))) + 1
            graph.branches.append(_Branch(np.linspace(hub, e, n), 0, i + 1))
        _simplify_graph(graph, 3.0, Params())
        return [pt for br in graph.branches if br.alive for pt in (br.pts[0], br.pts[-1])]

    def test_spur_straight_on_from_the_line_is_kept(self):
        # The start of a tip that segmentation cut short; pruning it would bend the line into the side branch.
        ends = self.fork((0.0, -1.5))
        self.assertTrue(any(np.allclose(e, [30.0, 28.5]) for e in ends))

    def test_spur_off_to_the_side_is_pruned(self):
        ends = self.fork((1.1, -1.1))
        self.assertFalse(any(np.allclose(e, [31.1, 28.9]) for e in ends))


class CutEnds(unittest.TestCase):
    def outline_end_x(self, cut_deg: float) -> float:
        """Rightmost outline x of a straight 4 px line along +x ending at x = 40, cut along a line
        `cut_deg` degrees from the stroke direction."""
        poly = np.stack([np.arange(0.0, 41.0), np.zeros(41)], axis=1)
        center = [np.array([[0.0, 0.0], [40 / 3, 0.0], [80 / 3, 0.0], [40.0, 0.0]])]
        a = np.radians(cut_deg)
        cut = np.array([np.cos(a), np.sin(a)])
        loop = _outline_loops(center, poly, np.full(41, 4.0), False, ["round", cut])[0]
        return float(loop[:, 0].max())

    def test_perpendicular_cut_is_flat(self):
        self.assertAlmostEqual(self.outline_end_x(90.0), 40.0, delta=0.05)

    def test_shallow_cut_never_becomes_a_spike(self):
        # Cutting a 4 px line along a 20 degree line would stretch one side 5.5 px past the end.
        self.assertLessEqual(self.outline_end_x(20.0), 40.0 + 2.0)


if __name__ == "__main__":
    unittest.main()
