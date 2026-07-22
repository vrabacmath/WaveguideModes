#ifndef SUBWAVELENGTHRESONATORS_WORKFLOWS_BENT_WAVEGUIDE_H
#define SUBWAVELENGTHRESONATORS_WORKFLOWS_BENT_WAVEGUIDE_H

namespace workflows {

// Real-space capacitance matrix of a FINITE patch of a crystal with a 90-degree bent waveguide
// -- the non-periodic counterpart of run_defect_capacitance_bands (no Floquet transform, one explicit
// finite cluster). The cluster is:
//  * defect resonators (radius defect_radius) at (m, 0) and (0, m) for m = 0 .. L, where
//    L = n_defect + fringe - 1 -- an L-shaped chain with its corner at the origin;
//  * cladding crystal resonators (radius radius) at every other site of [-n_clad, n_clad]^2.
// Chain sites are addressed by signed path distance ell from the corner: site(ell) = (-ell, 0) on
// the +x arm (ell <= 0) and (0, ell) on the +y arm (ell >= 0). Couplings are reported as the corner
// resonator's row, C_l = C_{corner, site(l)}.
//
// The outer `fringe` layers of the chain (default 1) stay in the mesh -- so they still screen -- but
// are dropped from the reported row, leaving the 2*n_defect - 1 interior sites |ell| <= n_defect - 1
// free of patch-edge artefacts.
//
// The geometry is invariant under reflection about y = x, which maps site(l) <-> site(-l). That
// reflection is orientation-reversing, so it swaps the +m/-m modes with a phase: the 2x2 blocks at
// +l and -l are permuted and phase-rotated, and it is |C_l|_F -- not the raw entries -- that must
// match |C_{-l}|_F. Exact symmetry of the quadrature needs points_per_disk to be a multiple of 4.
//
// Writes:
//   bent_waveguide_couplings.csv  l, |C_l|_F, then modes*modes complex entries (re,im)
//                                    row-major -- the corner resonator's 2x2 coupling blocks.
//   bent_waveguide_resonances.csv      Re(lambda), Im(lambda), Re(omega), Im(omega) -- eig(C),
//                                    omega = omega_0 + delta*lambda (finite defect resonances).
void run_bent_waveguide(double radius = 0.35,
                        double defect_radius = 0.455,
                        double delta = 0.05,
                        int m_ang = 1,
                        int n_defect = 3,
                        int n_clad = 3,
                        int fringe = 1,
                        int points_per_disk = 16);

}  // namespace workflows

#endif  // SUBWAVELENGTHRESONATORS_WORKFLOWS_BENT_WAVEGUIDE_H
