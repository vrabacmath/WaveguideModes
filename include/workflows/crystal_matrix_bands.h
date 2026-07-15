#ifndef SUBWAVELENGTHRESONATORS_WORKFLOWS_CRYSTAL_MATRIX_BANDS_H
#define SUBWAVELENGTHRESONATORS_WORKFLOWS_CRYSTAL_MATRIX_BANDS_H

namespace workflows {

// Bulk square-lattice band diagrams along M -> Gamma -> X -> M for the old R = 0.25 crystal.
//
// The same sweep is run twice:
//   * SpectralOperators::CrystalA, the nodal boundary-integral crystal matrix;
//   * Utils::multipole_crystal_A, the Fourier multipole crystal matrix.
//
// The workflow is intentionally just the raw unperturbed crystal diagnostic: for each
// (alpha, omega) point it records sigma_min(A^alpha(omega)) and log(abs(det(A^alpha(omega)))).
//
// Writes:
//   crystalA_m_gamma_x_m.csv    s, omega, sigma_min, log_abs_det, alpha_x, alpha_y
//   multipoleA_m_gamma_x_m.csv  s, omega, sigma_min, log_abs_det, alpha_x, alpha_y
void run_crystal_matrix_bands(double radius = 0.25,
                              double delta = 1.0e-2,
                              int points_per_disk = 32,
                              int n_multipole = 8,
                              int points_per_segment = 13,
                              double omega_lo = 0.05,
                              double omega_hi = 6.0,
                              int n_omega = 140,
                              double omega_imag = 1.0e-3);

// Projected bulk spectrum for the LINE-DEFECT band diagrams: for each alpha_x on [0, pi] and
// each omega on a grid, sweep the transverse Bloch parameter alpha_y over [0, pi] and record
// the minimum and median over alpha_y of sigma_min(multipole crystal A). A bulk band passes
// through (alpha_x, omega) for SOME alpha_y iff the minimum dips well below the
// (contamination-floor) median -- the same relative criterion as the in_gap test of
// run_line_defect_bands_neumann. Shade min < 0.25 * median as "projected bulk band" when
// overlaying the defect bands.
//
// Writes projected_bulk_bands.csv: alpha_x, omega, min_sigma, median_sigma.
void run_projected_bulk_bands(double radius = 0.35,
                              double delta = 0.05,
                              int n_multipole = 7,
                              int num_alpha_x = 31,
                              int n_alpha_y = 13,
                              double omega_lo = 3.8,
                              double omega_hi = 9.2,
                              int n_omega = 271,
                              double omega_imag = 1.0e-3);

}  // namespace workflows

#endif  // SUBWAVELENGTHRESONATORS_WORKFLOWS_CRYSTAL_MATRIX_BANDS_H
