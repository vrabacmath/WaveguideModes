#ifndef SUBWAVELENGTHRESONATORS_WORKFLOWS_DEFECT_CAPACITANCE_BANDS_H
#define SUBWAVELENGTHRESONATORS_WORKFLOWS_DEFECT_CAPACITANCE_BANDS_H

namespace workflows {

// Higher-frequency line-defect band via the FREQUENCY-DEPENDENT CAPACITANCE MATRIX of
// Ammari et al. (arXiv:2605.27572v1), built from boundary integral operators (S, K*).
//
// Unlike the channel-projected multipole operator M^eps (which only exposes the *monopole*
// / subwavelength defect band), the higher (dipole, m=1) defect band is found here from the
// genuine capacitance matrix. The Fabry-Perot boundary integral is rewritten as an exterior
// Dirichlet-to-Neumann map, whose projection onto the resonators' interior Neumann eigenmodes
// is the capacitance matrix:
//   Lambda_ext = (1/2 I + (K^{alpha,k})*) (S^{alpha,k})^{-1},   k = omega_0 / v,
//   C^reg_{pq} = -(v_b^2 / 2 omega_0) <Lambda_ext[g_q], g_p>_{dD},
//   omega = omega_0 + delta * lambda,   lambda = eig(C^reg),
// with omega_0 = v_b * j'_{m,1}/R_def the interior Neumann (dipole) eigenvalue of the defect
// resonator. Per Thm 4.3 the matrix is m x m with m = dim of the interior Neumann eigenspace at
// omega_0; at omega_0 only the defect disk is at a Neumann eigenvalue (the crystal disks' dipole
// eigenvalue j'_{1,1}/R is off in the gap), so m = modes_per_disk (=2 for the dipole) and g_p is
// the Neumann mode e^{+/- i m theta} on the DEFECT disk, zero on the crystal disks.
//
// The exterior is the *crystal*, so it is modelled by a SUPERCELL: a column of disks at
// y = -n_rows..n_rows, periodic in x (the n=0 disk is the defect of radius defect_radius, the
// rest are crystal disks of radius radius). x-periodicity makes this a (2 n_rows + 1)-row
// crystal slab with one defect row. omega_0 lies in the bulk gap (paper Case 1, eq. (4.7)), so
// C^reg is Hermitian (Lemma 4.2) and the band is real/bound; for the finite slab the defect mode
// is evanescent through the cladding, with radiative width Im(omega) -> 0 as n_rows grows.
//
// The leading-order eigenvalues only SEED an exact solve: each band point is refined to a root
// of the exact supercell transmission condition
//     [ delta (1/2 I + K*(omega)) - B_int(omega) S(omega) ] phi = 0,
// with B_int the exact interior DtN map of every disk (diagonal in each disk's angular Fourier
// basis, d_m = k_b J_m'(k_b R_i)/J_m(k_b R_i)). The roots are exact in delta -- they agree with
// the multipole M^eps Neumann-search band (line-defect-neumann) to <= 2e-4 -- and remove the
// O(delta^2) error of the leading-order formula (~0.07 at Gamma for this geometry).
//
// Runs one mode family per angular order m = 0..max_m (m=0 breathing, 1 dipole, 2 quadrupole,
// 3 octupole), each searched around its own defect Neumann frequency omega_0 = v_b j'_{m,1}/R_def
// -- mirroring run_line_defect_bands_neumann so the two methods can be overlaid per m. Unlike
// the sigma_min(M^eps) search, the exact transmission solve needs no spurious-Dirichlet
// exclusions: with v_b = v the interior DtN pole of B_int at j_{p,q}/R_i cancels the spurious
// kernel of S at the same frequency analytically, so the roots pass through those lines.
//
// Writes "defect_capacitance_bands.csv" (EXACT roots): m, alpha_x, Re(omega), Im(omega),
//   Re(lambda_eff), Im(lambda_eff), last_secant_step, where lambda_eff = (omega - omega_0)/delta.
// Writes "defect_capacitance_bands_leading.csv" (leading-order seeds): m, alpha_x, Re(omega),
//   Im(omega), Re(lambda), Im(lambda), herm_residual.
// A near-real root (|Im omega| small) is a bound defect mode; a genuinely complex root is a
// leaky resonance (the family's window not fully inside a projected bulk gap).
// Verified by the unit test CapacitanceMatrix.SingleDiskDipoleMatchesAnalytic.
// alpha_lo_frac/alpha_hi_frac restrict the swept window to [lo, hi] * pi (useful to rerun a
// failed or interesting alpha segment at higher resolution).
void run_defect_capacitance_bands(double radius = 0.35,
                                  double defect_radius = 0.455,
                                  double delta = 0.05,
                                  int max_m = 2,
                                  int n_rows = 4,
                                  int points_per_disk = 24,
                                  int num_alpha = 31,
                                  double alpha_lo_frac = 0.0,
                                  double alpha_hi_frac = 1.0);

}  // namespace workflows

#endif  // SUBWAVELENGTHRESONATORS_WORKFLOWS_DEFECT_CAPACITANCE_BANDS_H
