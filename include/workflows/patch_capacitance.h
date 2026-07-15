#ifndef SUBWAVELENGTHRESONATORS_WORKFLOWS_PATCH_CAPACITANCE_H
#define SUBWAVELENGTHRESONATORS_WORKFLOWS_PATCH_CAPACITANCE_H

namespace workflows {

// Real-space capacitance matrix of a FINITE, CENTERED patch of the line-defect crystal -- the
// non-periodic counterpart of run_defect_capacitance_bands (no Floquet transform, one explicit
// finite cluster). The cluster is:
//   * defect resonators (radius defect_radius) at (m, 0) for m = -n_defect .. n_defect;
//   * cladding crystal resonators (radius radius) at (m, +/- n) for n = 1 .. n_clad.
// It is centered on x = 0 so the middle resonator (index n_defect) is a BULK site whose row of
// couplings is free of the patch-edge artefacts.
//
// The exterior is free space, so Lambda_ext = (1/2 I + K*) S^{-1} is built from the FREE-SPACE
// layer potentials ops.S / ops.Kstar on the whole cluster -- the same operator convention
// validated by the unit test CapacitanceMatrix.SingleDiskDipoleMatchesAnalytic. Projecting
// Lambda_ext onto the defect resonators' interior Neumann traces g_p (dipole e^{+/- i theta} by
// default) gives the finite capacitance matrix
//   C_{pq} = -(v_b^2 / 2 omega_0) <Lambda_ext[g_q], g_p>_{dD},   omega_0 = v_b j'_{m,1}/R_def.
// Assembled ONCE: a single S, one LU factorisation, all 2(2 n_defect + 1) defect modes as the
// columns of G, C = -(v_b^2/2 omega_0) G^H diag(sigma) (1/2 + K*) S^{-1} G.
//
// The center resonator's row of 2x2 blocks C_{center, center+l} is the finite-patch real-space
// coupling C_l -- directly comparable to the inverse-Floquet C_l obtained by ifft of the
// quasi-periodic C^alpha (capacitance_matrix_per_alpha.csv from run_defect_capacitance_bands,
// see src/fullC.py). Agreement in the bulk is the check that truncating the infinite defect
// line to a finite patch is justified (the couplings decay fast enough).
//
// Writes:
//   patch_capacitance_couplings.csv  l, |C_l|_F, then modes*modes complex entries (re,im)
//                                    row-major -- the center resonator's 2x2 coupling blocks.
//   patch_defect_resonances.csv      Re(lambda), Im(lambda), Re(omega), Im(omega) -- eig(C),
//                                    omega = omega_0 + delta*lambda (finite defect resonances).
void run_patch_capacitance(double radius = 0.35,
                           double defect_radius = 0.455,
                           double delta = 0.05,
                           int m_ang = 1,
                           int n_defect = 10,
                           int n_clad = 4,
                           int points_per_disk = 16);

}  // namespace workflows

#endif  // SUBWAVELENGTHRESONATORS_WORKFLOWS_PATCH_CAPACITANCE_H
