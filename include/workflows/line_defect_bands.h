#ifndef SUBWAVELENGTHRESONATORS_WORKFLOWS_LINE_DEFECT_BANDS_H
#define SUBWAVELENGTHRESONATORS_WORKFLOWS_LINE_DEFECT_BANDS_H

namespace workflows {

// SUBWAVELENGTH line-defect band diagram via the multipole defect operator M^eps
// (Ammari, Hiltunen, Liu, Miao & Zhu, arXiv:2512.05370 -- the tight-binding / quasi-periodic
// capacitance framework for line defects in the subwavelength regime).
//
// For a square-lattice crystal of high-contrast resonators with one defect ROW (period 1 in x,
// Bloch parameter alpha_x along the defect), this computes, for each alpha_x:
//   * the unperturbed crystal band edges  -- frequencies omega where the bulk multipole operator
//     A^alpha (Utils::multipole_crystal_A) is singular, i.e. sigma_min(A^alpha) is minimal. For
//     the Fig. 2 strip view, the plotted first-band edge is at alpha_y = pi, while the next
//     visible band edge is at alpha_y = 0;
//   * the line-defect band -- frequencies omega in the subwavelength bandgap where the line-defect
//     operator M^eps (Multipole::crystal_line_M_operator) is singular, sigma_min(M^eps) -> 0.
// A defect mode is a frequency in the bulk gap where M^eps is (near-)singular; the band is the
// locus of such frequencies as alpha_x sweeps the Brillouin zone.
//
// This is the "operator M" route and is reliable for the SUBWAVELENGTH (monopole) defect band.
// For a higher-frequency, non-subwavelength (e.g. dipole) defect band, use the frequency-dependent
// capacitance matrix instead -- see run_defect_capacitance_bands / docs/defect_capacitance_bands.md.
//
// Writes (repository root):
//   crystal_first_band.csv      alpha_x, omega   (upper edge of the first crystal band)
//   crystal_second_band.csv     alpha_x, omega   (next visible crystal band edge)
//   defect_bands.csv            alpha_x, omega   (accepted line-defect band, sigma_min(M) < tol)
//   defect_band_diagnostics.csv alpha_x, omega_1, sigma(A_1), omega_2, sigma(A_2), omega_M, sigma(M)
// Plot with src/plot.py (toggles `line_defect_bands`, `line_defect_diagnostics`).
//
// Defaults reproduce the deeply-subwavelength "Figure 2" geometry (R = 0.05, R_defect = 0.8 R,
// delta = 2e-4).
//
// Parameters:
//   radius         crystal resonator radius R
//   defect_radius  line-defect resonator radius R_defect
//   delta          material contrast parameter
//   first_band_*   search window for the upper edge of the first crystal band
//   second_band_*  search window for the next visible crystal band edge
//   defect_band_hi upper end of the defect-mode search window; the lower end is chosen from the
//                  first-band edge at each alpha_x
//   num_alpha      number of alpha_x subintervals; the CSVs contain num_alpha + 1 samples
//   n_gauss        Gauss quadrature order for the alpha_y integral in M^eps
//   v              wave speed of the background medium
//   v_b            interior wave speed of the crystal resonators
//   v_bd           interior wave speed of the defect resonators
void run_line_defect_bands_M(double radius = 0.05,
                             double defect_radius = 0.04,
                             double delta = 2e-4,
                             double first_band_lo = 0.238,
                             double first_band_hi = 0.265,
                             double second_band_lo = 1.5,
                             double second_band_hi = 3.5,
                             double defect_band_hi = 0.31,
                             int num_alpha = 30,
                             int n_gauss = 21,
                             double v = 1.0,
                             double v_b = 1.0,
                             double v_bd = 1.0);

// Higher-frequency line-defect bands from the multipole operator M^eps, searched AROUND the defect
// resonator's interior Neumann resonances omega_0 = v_b * j'_{m,1}/R_def (dipole, quadrupole, ...),
// for direct comparison with run_defect_capacitance_bands at the SAME geometry. Coarse sigma_min(M)
// scan + golden refine within +/- window_factor*delta of each omega_0, excluding the exterior
// Dirichlet flat-band frequencies, tagging projected-gap membership; multiplicity 2 for m>=1.
//
// It also searches the SUBWAVELENGTH (static-monopole) band over [sub_lo, sub_hi] (labelled m=-1),
// so one run gives the subwavelength + higher (dipole/quadrupole) bands at the same geometry; set
// sub_lo >= sub_hi to skip it.
//
// Writes "defect_neumann_bands.csv": m, alpha_x, omega, sigma_min(M), in_gap, near_dirichlet
//   (m = -1 subwavelength, 0 breathing, 1 dipole, 2 quadrupole, ...).
// n_multipole = 0 auto-selects a frequency-appropriate cutoff (avoids high-order J_n(kR) contamination).
// v is the background wave speed, v_b the crystal resonators' interior speed, and v_bd the
// defect resonators' interior speed; the searches recenter on omega_0 = v_bd * j'_{m,1}/R_def.
void run_line_defect_bands_neumann(double radius = 0.35,
                                   double defect_radius = 0.455,
                                   double delta = 0.001,
                                   int max_m = 2,
                                   double window_factor = 20.0,
                                   int num_alpha = 31,
                                   int n_gauss = 20,
                                   int n_multipole = 0,
                                   double sub_lo = 0.5,
                                   double sub_hi = 3.6,
                                   double v = 1.0,
                                   double v_b = 1.0,
                                   double v_bd = 1.0);

}  // namespace workflows

#endif  // SUBWAVELENGTHRESONATORS_WORKFLOWS_LINE_DEFECT_BANDS_H
