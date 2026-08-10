#ifndef SUBWAVELENGTHRESONATORS_WORKFLOWS_DEFECT_CAPACITANCE_BANDS_H
#define SUBWAVELENGTHRESONATORS_WORKFLOWS_DEFECT_CAPACITANCE_BANDS_H

namespace workflows {

/* Higher-frequency line defect band via the frequency-dependent capacitance matrix based
on Ammari et al. Frequency-Dependent Capacitance Matrix Formulation for Fabry-Perot 
Resonances in Two and Three Dimensional Systems. Version 1. 
(arXiv:2605.27572v1), built from boundary integral operators (S, K*).

parameters:
* radius = crystal resonator radius, default 0.35
* defect_radius = defect resonator radius, default 0.455
* delta = inverse contrast parameter, default 0.05
* max_m = maximum angular order (0 monopole, 1 dipole, 2 quadrupole, ...), default 2
* n_rows = number of crystal rows on each side of the defect row (supercell height = 2 n_rows + 1), default 4
* points_per_disk = number of boundary discretization points, default 24
* num_alpha = number of quasi-periodic alpha_x points in [alpha_lo_frac, alpha_hi_frac] * 2 pi, default 31
* alpha_lo_frac, alpha_hi_frac = lower/upper fractions of 2 pi for the quasi-periodic alpha_x range, default 0.0, 1.0
* v = background wave speed, default 1.0
* v_b = crystal resonator interior wave speed (does not enter the leading-order capacitance), default 1.0
* v_bd = defect resonator interior wave speed (enters the leading-order capacitance), default 1.0

  Lambda_ext = (1/2 I + (K^{alpha,k})*) (S^{alpha,k})^{-1},   k = omega_0 / v,
  C^reg_{pq} = -(v_b^2 / 2 omega_0) <Lambda_ext[g_q], g_p>_{dD},
  omega = omega_0 + delta * lambda,   lambda = eig(C^reg),

with omega_0 = v_b * j'_{m,1}/R_def the interior Neumann eigenvalue of the defect
resonator. 
m = multipole order and g_p is
the Neumann mode e^{+/- i m theta} on the DEFECT disk, zero on the crystal disks.

The exterior is the crystal, so it is modelled by a SUPERCELL: a column of disks at
y = -n_rows..n_rows, periodic in x (the n=0 disk is the defect of radius defect_radius, the
rest are crystal disks of radius radius). x-periodicity makes this a (2 n_rows + 1)-row
crystal slab with one defect row. omega_0 lies in the bulk gap (paper Case 1, eq. (4.7)), so
C^reg is Hermitian (Lemma 4.2) and the band is real/bound. The defect mode
is evanescent through the cladding with radiative width Im(omega) -> 0 as n_rows grows.

Writes "defect_capacitance_bands_leading.csv" (leading-order resonances): m, alpha_x, Re(omega),
  Im(omega), Re(lambda), Im(lambda), herm_residual.
Verified by the unit test CapacitanceMatrix.SingleDiskDipoleMatchesAnalytic.*/

void run_defect_capacitance_bands(double radius = 0.35,
                                  double defect_radius = 0.455,
                                  double delta = 0.05,
                                  int max_m = 2,
                                  int n_rows = 4,
                                  int points_per_disk = 24,
                                  int num_alpha = 31,
                                  double alpha_lo_frac = 0.0,
                                  double alpha_hi_frac = 1.0,
                                  double v = 1.0,
                                  double v_b = 1.0,
                                  double v_bd = 1.0);


}  // namespace workflows

#endif  // SUBWAVELENGTHRESONATORS_WORKFLOWS_DEFECT_CAPACITANCE_BANDS_H
