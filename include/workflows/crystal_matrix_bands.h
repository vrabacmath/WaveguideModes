#ifndef SUBWAVELENGTHRESONATORS_WORKFLOWS_CRYSTAL_MATRIX_BANDS_H
#define SUBWAVELENGTHRESONATORS_WORKFLOWS_CRYSTAL_MATRIX_BANDS_H

namespace workflows {

/*Bulk square lattice band diagrams along M -> Gamma -> X -> M for the old R = 0.25 crystal.

The same sweep is run twice:
* SpectralOperators::CrystalA, the nodal boundary-integral crystal matrix;
* Utils::multipole_crystal_A, the Fourier multipole crystal matrix.

The workflow is intentionally just the unperturbed crystal diagnostic: for each
(alpha, omega) point it records sigma_min(A^alpha(omega)) and log(abs(det(A^alpha(omega)))).

parameters:
* radius = crystal resonator radius, default 0.25
* delta = inverse contrast parameter, default 1.0e-2
* points_per_disk = number of boundary discretization points, default 32
* n_multipole = number of multipoles in the multipole crystal matrix, default 8
* points_per_segment = number of k-points along each segment of the M-Gamma-X-M path, default 13
* omega_lo, omega_hi = frequency sweep range, default [0.05, 6.0]
* n_omega = number of frequency points, default 140
* omega_imag = small imaginary part of omega for the spectral operator, default 1.0e-3
* v = background wave speed, default 1.0
* v_b = crystal resonator interior wave speed, default 1.0

Writes:
  crystalA_m_gamma_x_m.csv    s, omega, sigma_min, log_abs_det, alpha_x, alpha_y
  multipoleA_m_gamma_x_m.csv  s, omega, sigma_min, log_abs_det, alpha_x, alpha_y*/
void run_crystal_matrix_bands(double radius = 0.25,
                              double delta = 1.0e-2,
                              int points_per_disk = 32,
                              int n_multipole = 8,
                              int points_per_segment = 13,
                              double omega_lo = 0.05,
                              double omega_hi = 6.0,
                              int n_omega = 140,
                              double omega_imag = 1.0e-3,
                              double v = 1.0,
                              double v_b = 1.0);

/* Projected bulk spectrum for the line defect band diagrams: for each alpha_x on [0, pi] and
each omega on a grid, sweep the transverse Bloch parameter alpha_y over [0, pi] and record
the minimum and median over alpha_y of sigma_min(multipole crystal A). A bulk band passes
through (alpha_x, omega) for some alpha_y iff the minimum dips well below the
(contamination-floor) median -- the same relative criterion as the in_gap test of
run_line_defect_neumann. Shade min < 0.25 * median as "projected bulk band" when
overlaying the defect bands.

parameters:
* radius = crystal resonator radius, default 0.35
* delta = inverse contrast parameter, default 0.05
* n_multipole = number of multipoles in the multipole crystal matrix, default 7
* num_alpha_x = number of alpha_x points in [0, pi], default 31
* n_alpha_y = number of alpha_y points in [0, pi], default 13
* omega_lo, omega_hi = frequency sweep range, default [3.8, 9.2]
* n_omega = number of frequency points, default 271
* omega_imag = small imaginary part of omega for the spectral operator, default 1.0e-3
* v = background wave speed, default 1.0
* v_b = crystal resonator interior wave speed, default 1.0

Writes projected_bulk_bands.csv: alpha_x, omega, min_sigma, median_sigma.*/
void run_projected_bulk_bands(double radius = 0.35,
                              double delta = 0.05,
                              int n_multipole = 7,
                              int num_alpha_x = 31,
                              int n_alpha_y = 13,
                              double omega_lo = 3.8,
                              double omega_hi = 9.2,
                              int n_omega = 271,
                              double omega_imag = 1.0e-3,
                              double v = 1.0,
                              double v_b = 1.0);

}  // namespace workflows

#endif  // SUBWAVELENGTHRESONATORS_WORKFLOWS_CRYSTAL_MATRIX_BANDS_H
