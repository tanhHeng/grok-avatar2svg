import importlib.util
import unittest

import numpy as np

from grok_avatar2svg.pipeline import vectorize

HAS_RESVG = importlib.util.find_spec("resvg_py") is not None
A = np.array([200, 60, 60])
B = np.array([60, 160, 60])


def two_blocks() -> np.ndarray:
    """Dark frame (background) around two color blocks sharing a slanted edge."""
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    img[:] = (30, 30, 35)
    yy, xx = np.mgrid[:300, :300]
    inner = (yy >= 40) & (yy < 260) & (xx >= 40) & (xx < 260)
    img[inner & (xx < 0.7 * yy + 60)] = A
    img[inner & (xx >= 0.7 * yy + 60)] = B
    return img


@unittest.skipUnless(HAS_RESVG, "needs the optional resvg-py package")
class Seams(unittest.TestCase):
    def test_shared_edge_shows_no_backdrop(self):
        from grok_avatar2svg.validate import render

        img = two_blocks()
        result = vectorize(img, np.ones(img.shape[:2], dtype=bool))
        out = render(result.svg, "#ff00ff").astype(float)
        yy, xx = np.mgrid[:300, :300]
        edge = (np.abs(xx - (0.7 * yy + 60)) < 2) & (yy > 60) & (yy < 240)
        # A pixel on the shared edge must be a blend of A and B; backdrop showing through moves it off that segment.
        ab = (B - A) / np.linalg.norm(B - A)
        rel = out[edge] - A
        off_segment = np.linalg.norm(rel - (rel @ ab)[:, None] * ab, axis=1)
        self.assertLess(float(off_segment.max()), 12.0)

    def test_background_stays_transparent(self):
        from grok_avatar2svg.validate import render

        img = two_blocks()
        result = vectorize(img, np.ones(img.shape[:2], dtype=bool))
        rgba = render(result.svg, None)
        self.assertEqual(int(rgba[:30, :, 3].max()), 0)
        # Anti-aliasing rounding may leave a single edge pixel at 254.
        self.assertGreaterEqual(int(rgba[60:240, 60:240, 3].min()), 250)


if __name__ == "__main__":
    unittest.main()
