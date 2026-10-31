import unittest

import numpy as np

from avatar2svg.fitting import bezier_eval, fit_open, fit_path
from avatar2svg.geometry import unit


def max_distance(curve, points):
    dense = np.vstack([bezier_eval(b, np.linspace(0.0, 1.0, 400)) for b in curve])
    return max(float(np.min(np.linalg.norm(dense - q, axis=1))) for q in points)


class FitOpen(unittest.TestCase):
    def test_noisy_arc_stays_within_eps_with_exact_ends(self):
        rng = np.random.default_rng(0)
        th = np.linspace(0.0, np.pi / 2, 80)
        pts = np.stack([50 * np.cos(th), 50 * np.sin(th)], axis=1) + rng.uniform(-0.3, 0.3, (80, 2))
        curve = fit_open(pts, np.array([0.0, 1.0]), np.array([1.0, 0.0]), 0.6)
        np.testing.assert_array_equal(curve[0][0], pts[0])
        np.testing.assert_array_equal(curve[-1][3], pts[-1])
        self.assertLessEqual(max_distance(curve, pts), 0.6 + 1e-6)

    def test_single_cubic_refits_as_one_segment(self):
        bez = np.array([[0.0, 0.0], [30.0, 40.0], [70.0, 40.0], [100.0, 0.0]])
        pts = bezier_eval(bez, np.linspace(0.0, 1.0, 120))
        curve = fit_open(pts, unit(bez[1] - bez[0]), unit(bez[2] - bez[3]), 0.5)
        self.assertEqual(len(curve), 1)


class FitPath(unittest.TestCase):
    def test_closed_circle_is_g1_and_within_eps(self):
        th = np.linspace(0.0, 2 * np.pi, 200, endpoint=False)
        pts = np.stack([40 * np.cos(th), 40 * np.sin(th)], axis=1)
        curve = fit_path(pts, 0.6, closed=True)
        np.testing.assert_array_equal(curve[-1][3], curve[0][0])
        for a, b in zip(curve, curve[1:] + curve[:1]):
            u, v = unit(a[3] - a[2]), unit(b[1] - b[0])
            self.assertAlmostEqual(float(u[0] * v[1] - u[1] * v[0]), 0.0, places=6)
        self.assertLessEqual(max_distance(curve, pts), 0.6 + 1e-6)

    def test_right_angle_becomes_a_segment_boundary(self):
        pts = np.vstack([np.stack([np.arange(0, 30.0), np.zeros(30)], 1), np.stack([np.full(30, 30.0), np.arange(0, 30.0)], 1)])
        curve = fit_path(pts, 0.5, closed=False)
        joints = [b[3] for b in curve[:-1]]
        self.assertTrue(any(np.linalg.norm(j - [30.0, 0.0]) < 1.0 for j in joints))

    def test_forced_break_keeps_point_and_tangents(self):
        th = np.linspace(0.0, np.pi, 120)
        pts = np.stack([60 * np.cos(th), 60 * np.sin(th)], axis=1)
        t_before, t_after = unit(np.array([1.0, -0.2])), unit(np.array([-1.0, 0.3]))
        curve = fit_path(pts, 0.6, closed=False, breaks={60: [t_before, t_after]})
        k = next(i for i, b in enumerate(curve[:-1]) if np.allclose(b[3], pts[60]))
        np.testing.assert_allclose(unit(curve[k][2] - curve[k][3]), t_before, atol=1e-9)
        np.testing.assert_allclose(unit(curve[k + 1][1] - curve[k + 1][0]), t_after, atol=1e-9)

    def test_smoothing_removes_waves_but_keeps_corner_and_ends(self):
        s = np.arange(0.0, 60.0)
        wave = 0.8 * np.sin(2 * np.pi * s / 12.0)  # larger than eps, so a plain fit follows every bump
        pts = np.vstack([np.stack([s, wave], 1), np.stack([60.0 + wave, s], 1)[1:]])
        self.assertGreaterEqual(len(fit_path(pts, 0.6, closed=False)), 10)
        curve = fit_path(pts, 0.6, closed=False, smooth=3.0)
        self.assertLessEqual(len(curve), 4)  # about one cubic per arm
        np.testing.assert_array_equal(curve[0][0], pts[0])
        np.testing.assert_array_equal(curve[-1][3], pts[-1])
        self.assertTrue(any(np.linalg.norm(b[3] - [60.0, 0.0]) < 1.0 for b in curve[:-1]))


if __name__ == "__main__":
    unittest.main()
