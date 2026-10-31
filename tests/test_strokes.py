import unittest

import cv2
import numpy as np

from avatar2svg.colorspace import rgb_to_lab
from avatar2svg.params import Params
from avatar2svg.strokes import _outline_loops, extract_strokes

PALETTE = np.array([[0, 0, 0], [250, 240, 240], [200, 150, 160]], dtype=np.uint8)


def band(width: int) -> tuple[np.ndarray, np.ndarray]:
    """A long vertical band of label 2 on label 1, anti-aliased like real input."""
    labels = np.ones((120, 60), dtype=np.int32)
    labels[10:110, 30 - width // 2 : 30 - width // 2 + width] = 2
    img = cv2.GaussianBlur(PALETTE[labels], (0, 0), 0.7)
    return rgb_to_lab(img), labels


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
