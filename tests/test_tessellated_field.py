"""Run with: venv/bin/python -m unittest discover -s tests -p test_tessellated_field.py"""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from plot_tessellated_waveguide_field import Window, assemble_field, interpolate, load_eigenvector, load_exports
from plot_tessellated_waveguide_field import plot_window, window_color_limits, window_disk_roles


class TessellatedFieldTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.csv = Path(self.temp.name) / "mode.csv"
        self.axis = np.array([-1., 0., 1.])

    def test_eigenvector_uses_labels_and_retains_phase(self):
        self.csv.write_text("# z=1.25, residual=0.0001\nell,basis,real,imag\n"
                            "1,1,2,-3\n-1,0,4,5\n0,1,0,0\n"
                            "1,0,-1,0\n0,0,0,2\n-1,1,3,4\n")
        coeff = load_eigenvector(self.csv, [{"ell": ell} for ell in (-1, 0, 1)], 2)
        np.testing.assert_array_equal(coeff, [[4+5j, 3+4j], [2j, 0], [-1, 2-3j]])

    def test_invalid_eigenvectors_are_rejected(self):
        for rows, error in (("0,0,1,0\n0,0,2,0\n", "duplicate"),
                            ("0,0,1,0\n", "expected 2 coefficients"),
                            ("0,0.5,1,0\n", "integer"),
                            ("1,0,1,0\n", "unexpected"),
                            ("0,0,1\n", "finite numeric"),
                            ("0,0,nan,0\n", "finite")):
            with self.subTest(rows=rows):
                self.csv.write_text("ell,basis,real,imag\n" + rows)
                with self.assertRaisesRegex(ValueError, error):
                    load_eigenvector(self.csv, [{"ell": 0}], 2)

    def test_monopole_has_one_basis(self):
        self.csv.write_text("ell,basis,real,imag\n0,0,0,2\n")
        np.testing.assert_array_equal(load_eigenvector(self.csv, [{"ell": 0}], 1), [[2j]])

    def test_export_layout_separates_window_and_template_radii(self):
        prefix = Path(self.temp.name) / "basis"
        Path(f"{prefix}_parameters.csv").write_text(
            "window_radius,patch_radius,m_ang,modes,bent,defect_radius,fringe,template_radius,template_count\n"
            "1,5,0,1,1,0.455,1,3,7\n")
        rows = ["ell,x,y,window_index,coefficient_start"]
        for ell in range(-5, 6):
            x, y = (-ell, 0) if ell <= 0 else (0, ell)
            rows.append(f"{ell},{x},{y},{int(np.clip(ell, -3, 3)) + 3},{ell + 5}")
        Path(f"{prefix}_sites.csv").write_text("\n".join(rows) + "\n")
        for k in range(7):
            np.savetxt(f"{prefix}_patch_{k}_axis.csv", self.axis[None, :], delimiter=",")
            for component in ("real", "imag"):
                np.savetxt(f"{prefix}_patch_{k}_basis_0_{component}.csv", np.full((3, 3), k), delimiter=",")
        params, sites, windows = load_exports(prefix)
        self.assertEqual(params["window_radius"], 1)
        self.assertEqual(len(windows), 7)
        self.assertEqual([s["window_index"] for s in sites], [0, 0, 0, 1, 2, 3, 4, 5, 6, 6, 6])
        self.assertEqual(windows[6].basis[0, 1, 1], 6+6j)

    def test_legacy_exports_require_regeneration(self):
        prefix = Path(self.temp.name) / "legacy"
        Path(f"{prefix}_parameters.csv").write_text(
            "window_radius,patch_radius,m_ang,modes,bent,defect_radius\n1,5,0,1,1,0.455\n")
        with self.assertRaisesRegex(ValueError, "Regenerate.*straight endpoint"):
            load_exports(prefix)

    def test_translation_and_xy_orientation(self):
        xx, yy = np.meshgrid(self.axis, self.axis)
        window = Window(self.axis, (xx + 10j * yy)[None, ...])
        sites = [{"ell": 0, "x": 2., "y": 3., "window_index": 0}]
        x, y, field, _ = assemble_field(sites, {0: window}, np.array([[1j]]), spacing=0.5)
        gx, gy = np.meshgrid(x, y)
        np.testing.assert_allclose(field, 1j * ((gx - 2) + 10j * (gy - 3)))

    def test_endpoint_template_uses_each_sites_own_coefficients(self):
        sites = [{"ell": ell, "x": float(ell), "y": 0.,
                  "window_index": int(np.clip(ell, -1, 1)) + 1} for ell in range(-2, 3)]
        axis = np.array([-0.25, 0., 0.25])
        windows = {k: Window(axis, np.array([np.full((3, 3), k+1+1j),
                                             np.full((3, 3), 2-1j)])) for k in range(3)}
        coeff = np.array([[1, 0], [2, 1j], [0, 3], [-1j, 4], [5, 6j]])
        x, y, field, coverage = assemble_field(sites, windows, coeff, spacing=0.25)
        for q, site in enumerate(sites):
            ix, iy = np.flatnonzero(x == site["x"])[0], np.flatnonzero(y == 0)[0]
            expected = coeff[q, 0] * (site["window_index"]+1+1j) + coeff[q, 1] * (2-1j)
            self.assertAlmostEqual(field[iy, ix], expected)
        self.assertTrue(np.isnan(field[coverage == 0]).all())

    def test_overlap_is_complex_sum_without_averaging(self):
        sites = [{"x": 0., "y": 0., "window_index": 0}] * 2
        window = Window(self.axis, np.ones((1, 3, 3), dtype=complex))
        _, _, field, coverage = assemble_field(sites, {0: window}, np.array([[1+2j], [-1+3j]]))
        np.testing.assert_allclose(field, 5j)
        np.testing.assert_array_equal(coverage, 2)

    def test_interpolation_keeps_masks_but_ignores_zero_weight_neighbors(self):
        field = np.ones((3, 3), dtype=complex)
        field[0, 0] = np.nan
        exact = interpolate(self.axis, field, np.array([0., 1.]), np.array([0., 1.]))
        np.testing.assert_allclose(exact, 1)
        between = interpolate(self.axis, field, np.array([-0.5]), np.array([-0.5]))
        self.assertTrue(np.isnan(between[0, 0]))

    def test_zero_coefficients_do_not_propagate_nan_masks(self):
        window = Window(self.axis, np.array([np.ones((3, 3)), np.full((3, 3), np.nan)]))
        sites = [{"x": 0., "y": 0., "window_index": 0}]
        _, _, field, _ = assemble_field(sites, {0: window}, np.array([[2j, 0]]))
        np.testing.assert_allclose(field, 2j)
        _, _, field, _ = assemble_field(sites, {0: window}, np.array([[2j, 1]]))
        self.assertTrue(np.isnan(field).all())

    def test_gallery_identifies_source_and_retained_disks_in_mesh_order(self):
        params = {"bent": 1, "window_radius": 1}
        disks = [{"x": x, "y": y, "is_source": source, "is_defect": defect}
                 for x, y, source, defect in [(1, 0, 0, 1), (0, 0, 1, 1), (0, 1, 0, 1),
                                               (0, 2, 0, 1), (1, 1, 0, 0)]]
        self.assertEqual(window_disk_roles(params, disks),
                         ["retained", "source", "retained", "fringe_defect", "cladding"])
        # A repeated incoming endpoint has local path offsets in the -x direction.
        disks[2].update(x=-1, y=0)
        self.assertEqual(window_disk_roles(params, disks),
                         ["retained", "source", "retained", "fringe_defect", "cladding"])

    def test_square_includes_opposite_arm_through_window_boundary(self):
        params = {"bent": 1, "window_radius": 6}
        disks = [{"x": x, "y": y, "is_source": 0, "is_defect": 1}
                 for x, y in [(-5, 1), (-5, 2), (-5, 6), (-5, 7), (7, 0), (6, 0)]]
        self.assertEqual(window_disk_roles(params, disks),
                         ["retained", "retained", "retained",
                          "fringe_defect", "fringe_defect", "retained"])

    def test_gallery_preserves_unweighted_complex_samples_and_masks(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        first = np.array([[1+2j, np.nan+1j*np.nan, 3-4j], [2j, 0, 1], [-3j, 2, 4]])
        window = Window(self.axis, np.array([first, 2j * first]))
        params = {"bent": 1, "window_radius": 1, "fringe": 1, "m_ang": 2, "template_count": 7}
        metadata = {"t": 0, "source_x": 0, "source_y": 0, "center_disk": 7, "is_straight": 0}
        disks = [{"disk": 7, "x": 0, "y": 0, "radius": 0.455, "is_source": 1, "is_defect": 1}]
        limits = window_color_limits({0: window, 1: Window(self.axis, 3 * window.basis)})
        fig = plot_window(0, window, params, metadata, disks, limits)
        try:
            for s in range(2):
                for column, expected in ((1, window.basis[s].real), (2, window.basis[s].imag)):
                    image = fig.axes[3*s+column].collections[0]
                    np.testing.assert_allclose(np.ma.filled(image.get_array(), np.nan), expected,
                                               equal_nan=True)
            # Both rows use the same color scale, set using ALL windows, not per-panel amplitudes.
            self.assertIs(fig.axes[1].collections[0].norm, fig.axes[4].collections[0].norm)
            self.assertEqual(fig.axes[1].collections[0].norm.vmax, limits[1])
            # Cell boundaries enclose centers at +/-1 (core) and +/-2 (full mesh).
            boundaries = [(patch.get_x(), patch.get_width()) for patch in fig.axes[0].patches
                          if not patch.get_fill()]
            self.assertIn((-1.5, 3), boundaries)
            self.assertIn((-2.5, 5), boundaries)
        finally:
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
