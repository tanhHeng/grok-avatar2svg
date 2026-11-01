import unittest

import numpy as np

from grok_avatar2svg.params import Params
from grok_avatar2svg.segment import _reassign_offcolor

DARK = (15.0, 0.0, -2.0)
LIGHT = (89.0, 11.0, 3.0)
RIM = (72.0, 15.0, 1.0)


def scene() -> tuple[np.ndarray, np.ndarray]:
    """Dark region 1 | light region 2 above a rim band 3, like the outline of a hair lobe."""
    labels = np.ones((30, 30), dtype=np.int32)
    labels[:, 15:] = 2
    labels[20:] = 3
    lab = np.zeros((30, 30, 3), dtype=np.float32)
    lab[labels == 1] = DARK
    lab[labels == 2] = LIGHT
    lab[labels == 3] = RIM
    return lab, labels


class OffColorReassignment(unittest.TestCase):
    def test_sliver_of_neighbor_color_returns_to_the_neighbor(self):
        lab, labels = scene()
        labels[12:20, 14] = 3  # rim region reaching up as a 1 px sliver over dark pixels
        out = _reassign_offcolor(lab, labels, Params())
        np.testing.assert_array_equal(out[12:20, 14], 1)

    def test_half_blended_edge_pixels_stay(self):
        lab, labels = scene()
        lab[:20, 15] = (np.array(DARK) + np.array(LIGHT)) / 2
        out = _reassign_offcolor(lab, labels, Params())
        np.testing.assert_array_equal(out, labels)

    def test_gradient_region_is_not_eroded(self):
        lab, labels = scene()
        ramp = np.linspace(0.0, 8.0, 15, dtype=np.float32)
        lab[:20, 15:, 0] = LIGHT[0] + ramp  # region 2 brightens toward the right edge
        labels[:20, 29] = 4
        lab[:20, 29] = (LIGHT[0] + 8.0, LIGHT[1], LIGHT[2])  # neighbor matches the bright end
        out = _reassign_offcolor(lab, labels, Params())
        np.testing.assert_array_equal(out, labels)


if __name__ == "__main__":
    unittest.main()
