#ifndef SUBWAVELENGTHRESONATORS_WORKFLOWS_BENT_WAVEGUIDE_H
#define SUBWAVELENGTHRESONATORS_WORKFLOWS_BENT_WAVEGUIDE_H

namespace workflows {

/* Real-space capacitance matrix of a finite patch of a crystal with a 90-degree bent waveguide.
 The cluster is an L-shaped chain with its corner at the origin, consisting of
 * defect resonators (radius defect_radius) at (x, 0) and (0, x) for x = 0 .. L, where
   L = n_defect + fringe - 1
 * cladding crystal resonators (radius radius) at every other site of [-n_clad, n_clad]^2.
Chain sites are addressed by signed path distance ell from the corner: site(ell) = (-ell, 0) on
the +x arm (ell <= 0) and (0, ell) on the +y arm (ell >= 0). Couplings are reported as the corner
resonator's row, C_l = C_{corner, site(l)}.

The outer `fringe` layers of the chain (default 1) stay in the main computations but
are dropped from the capacitance matrix.

The geometry is invariant under reflection about y = x, which maps site(l) <-> site(-l). That
reflection is orientation-reversing, so it swaps the +m/-m modes with a phase: the 2x2 blocks at
+l and -l are permuted and phase-rotated, and it is |C_l|_F - the Frobenius norm, not the
raw entries - that must match |C_{-l}|_F.

Parameters:
  radius = radius of the cladding disks
  defect_radius = radius of the defect disks
  delta = inverse contrast
  m_ang = order, monopole, dipole, quadrupole...
  n_defect = number of defect disks along each arm of the L (total = 2*n_defect + 1)
  n_clad = number of cladding disks along each axis (total = (2*n_clad + 1)^2 - (2*n_defect + 1))
  fringe = number of boundary defect disks to keep in the mesh but drop from C
  points_per_disk = number of quadrature points per disk boundary
  v = wave speed in free space
  v_b = wave speed in the cladding crystal interior
  v_bd = wave speed in the defect interior

Writes:
  bent_waveguide_couplings.csv  l, |C_l|_F, then modes*modes complex entries (re,im)
                                   row-major -- the corner resonator's 2x2 coupling blocks.
  bent_waveguide_resonances.csv      Re(lambda), Im(lambda), Re(omega), Im(omega) -- eig(C),
                                   omega = omega_0 + delta*lambda (finite defect resonances).*/
void run_bent_waveguide(double radius = 0.35,
                        double defect_radius = 0.455,
                        double delta = 0.05,
                        int m_ang = 1,
                        int n_defect = 3,
                        int n_clad = 3,
                        int fringe = 1,
                        int points_per_disk = 16,
                        double v = 1.0,
                        double v_b = 1.0,
                        double v_bd = 1.0);

}  // namespace workflows

#endif  // SUBWAVELENGTHRESONATORS_WORKFLOWS_BENT_WAVEGUIDE_H
