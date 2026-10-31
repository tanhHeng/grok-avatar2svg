import unittest

import cv2
import numpy as np

from avatar2svg.colorspace import rgb_to_lab
from avatar2svg.params import Params
from avatar2svg.strokes import extract_strokes

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


if __name__ == "__main__":
    unittest.main()
