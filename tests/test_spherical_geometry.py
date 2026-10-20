"""Analytical tests; synthetic fixtures are never benchmark measurements."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose
from PIL import Image
from scipy.spatial.transform import Rotation
from build_standard_gt import build_standard_gt
from experiments.spherical_geometry.geometry import (
    angle_deg, spherical_grid, compensated_motion, translation_parallax, select_pairs)
from experiments.spherical_geometry.run import Config, run
from experiments.spherical_geometry.plots import save_figures, erp_map


def poses(centres, rotations=None):
    out = np.tile(np.eye(4), (len(centres), 1, 1))
    out[:, :3, 3] = centres
    if rotations is not None:
        out[:, :3, :3] = rotations
    return out


class SphericalGeometryTests(unittest.TestCase):
    def test_cardinal_seam_poles_and_area(self):
        rays, lon, lat, area = spherical_grid(5, 9)
        assert_allclose(rays[2, 4], [0, 0, 1], atol=1e-14)
        assert_allclose(rays[2, 6], [1, 0, 0], atol=1e-14)
        assert_allclose(rays[0, 4], [0, -1, 0], atol=1e-14)
        assert_allclose(rays[:, 0], rays[:, -1], atol=1e-14)
        self.assertAlmostEqual(area.sum(), 4*np.pi, places=12)
        assert_allclose(area[:, 0]*2, area[:, 1])
        self.assertTrue((area > 0).all())
        assert_allclose(angle_deg([[1, 0, 0]], [[-1, 0, 0]]), [180.])

    def test_pure_rotation_compensation(self):
        r = Rotation.from_euler("xyz", [23, -61, 17], degrees=True).as_matrix()
        p = poses([[2, 3, 4], [2, 3, 4]], [np.eye(3), r])
        a = np.array([[0, 0, 1.], [1., 0, 0], [0, 1., 0]])
        b = a @ r
        raw, compensated = compensated_motion(a, b, *p)
        self.assertTrue(np.max(raw) > 10)
        assert_allclose(compensated, 0, atol=1e-12)

    def test_known_baseline_range_and_combined_rotation(self):
        p = poses([[0, 0, 0], [1, 0, 0]])
        expected = np.rad2deg(np.arctan(1/10))
        self.assertAlmostEqual(float(translation_parallax([0, 0, 1], 10, *p)), expected)
        self.assertAlmostEqual(float(translation_parallax([1, 0, 0], 10, *p)), 0)
        near = translation_parallax([0, 0, 1], 5, *p)
        far = translation_parallax([0, 0, 1], 20, *p)
        self.assertGreater(near, far)
        p[1, :3, :3] = Rotation.from_euler("y", 57, degrees=True).as_matrix()
        target = np.array([[-1., 0, 10.]]) @ p[1, :3, :3]
        _, residual = compensated_motion([[0, 0, 1]], target, *p)
        assert_allclose(residual, [expected], atol=1e-12)

    def test_metric_selection_boundaries_loop_and_determinism(self):
        p = poses([[0, 0, 0], [10, 0, 0], [1, 0, 0], [0, 0, 0]])
        selected, coverage = select_pairs(p, [0, .1, 2, 11], 20)
        self.assertEqual([(r["frame_i"], r["frame_j"]) for r in selected if r["bin"] == 0], [(0, 3)])
        self.assertEqual([r["eligible_pairs"] for r in coverage], [1, 2, 3])
        small, cov = select_pairs(p, [0, 1, 2], 20)
        self.assertEqual([r["eligible_pairs"] for r in cov], [1, 2])
        self.assertTrue(all(r["bin"] == 1 for r in small if r["baseline_m"] == 1))
        p = poses(np.c_[np.arange(30), np.zeros((30, 2))])
        a, _ = select_pairs(p, [0, 30], 5, seed=51)
        b, _ = select_pairs(p, [0, 30], 5, seed=51)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 5)
        one, coverage = select_pairs(p[:1], [0, 1])
        self.assertEqual(one, [])
        self.assertEqual(coverage[0]["eligible_pairs"], 0)

    def test_invalid_inputs_and_rotation_filter(self):
        p = poses([[0, 0, 0], [1, 0, 0]])
        for edges in ([0, 0, 1], [-1, 1], [0, np.nan]):
            with self.assertRaises(ValueError):
                select_pairs(p, edges)
        with self.assertRaises(ValueError):
            angle_deg([0, 0, 0], [1, 0, 0])
        with self.assertRaises(ValueError):
            translation_parallax([0, 0, 1], 0, *p)
        with self.assertRaises(ValueError):
            translation_parallax([1, 0, 0], 1, *p)  # Point at target camera.
        p[1, :3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
        selected, _ = select_pairs(p, [0, 2], max_rotation_deg=45)
        self.assertEqual(selected, [])

    def test_exporter_runner_and_no_feature_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            images = root/"input"
            images.mkdir()
            for i, x in enumerate([0, 1, 3]):
                Image.new("RGB", (32, 16)).save(images/f"equi_{i}_{x}_0_0_0_0_0.png")
            sequence = build_standard_gt(images, root/"gt", sequence_id="synthetic")
            output = root/"reports"
            result = run(Config(str(sequence), str(output), baseline_edges_m=(0, 2, 4),
                                pairs_per_bin=4, grid_height=5, grid_width=9,
                                static_formats=()))
            self.assertEqual(len(result["pairs"]), 3)
            self.assertTrue((result["diagnostics"].status == "inconclusive").all())
            self.assertEqual(json.loads((output/"run_status.json").read_text())["status"], "completed")
            self.assertEqual(len(json.loads((output/"frame_manifest.json").read_text())), 3)
            self.assertTrue((output/"figures/trajectory.html").is_file())
            self.assertFalse((output/"depth_errors.csv").exists())
            with self.assertRaises(FileExistsError):
                run(Config(str(sequence), str(output), static_formats=()))

    def test_measured_residual_tables_and_plots(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from experiments.spherical_geometry.run import diagnose
        from experiments.spherical_geometry.plots import motion_figures
        p = poses([[0, 0, 0], [1, 0, 0]])
        ds = SimpleNamespace(paths=["a.png", "b.png"], poses=p)
        a = np.array([[0., 0, 1], [1., 0, 0]])
        b = np.array([[-1., 0, 10], [9., 0, 0]])
        uv = np.array([[4., 2], [6., 2]])
        selected, _ = select_pairs(p, [0, 2])
        with patch("diagnostics.rotation_check.image_features", return_value=None), patch(
                "diagnostics.rotation_check.matched_bearings",
                return_value=(a, b, uv, uv, (5, 9), (5, 9))):
            summary, matches = diagnose(ds, selected, 1, 15.)
        self.assertEqual(summary.iloc[0]["status"], "measured")
        self.assertEqual(len(matches), 2)
        assert_allclose(matches.residual_deg, [np.rad2deg(np.arctan(.1)), 0.], atol=1e-12)
        figures = motion_figures(matches, summary)
        self.assertEqual(len(figures), 3)
        for fig in figures.values():
            self.assertGreater(len(fig.to_json()), 100)

    def test_static_export(self):
        with tempfile.TemporaryDirectory() as temp:
            _, lon, lat, area = spherical_grid(5, 9)
            status = save_figures({"synthetic_area": erp_map(area, lon, lat, "Synthetic test", "sr")}, temp)
            self.assertTrue(all(s["status"] == "saved" for s in status), status)
            for ext in ("html", "png", "pdf"):
                self.assertGreater((Path(temp)/("synthetic_area."+ext)).stat().st_size, 100)


if __name__ == "__main__":
    unittest.main()
