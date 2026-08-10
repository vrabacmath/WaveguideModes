#ifndef SUBWAVELENGTHRESONATORS_WORKFLOWS_PATCH_CAPACITANCE_H
#define SUBWAVELENGTHRESONATORS_WORKFLOWS_PATCH_CAPACITANCE_H

namespace workflows {

/*Real-space capacitance matrix of a finite, centered patch of a crystal with a line defect: the
non-periodic counterpart of run_defect_capacitance_bands (no Floquet transform, one explicit
finite cluster). The cluster is a "sandwiched" line of defect resonators::
  * defect resonators (radius defect_radius) at (m, 0) for m = -n_defect .. n_defect;
  * cladding crystal resonators (radius radius) at (m, +/- n) for n = 1 .. n_clad, and m as above.

The exterior is free space, so Lambda_ext = (1/2 I + K*) S^{-1} is built from the free space
layer potentials ops.S / ops.Kstar on the whole cluster.
Applying Lambda_ext to the defect resonators' interior Neumann traces g_p (dipole e^{+/- i theta}
on circles) gives the finite capacitance matrix
  C_{pq} = -(v_b^2 / 2 omega_0) <Lambda_ext[g_q], g_p>_{dD},   omega_0 = v_b j'_{m,1}/R_def.
Assembled once: a single S, one LU factorisation, all 2(2 n_defect - 2 fringe + 1) defect modes as the
columns of G:
    C = -(v_b^2/2 omega_0) G^H diag(sigma) (1/2 + K*) S^{-1} G.

The center resonator's row of 2x2 blocks C_{center, center+l} is the finite-patch real-space
coupling C_l directly comparable to the inverse Floquet C_l obtained by an IFFT of the
quasi-periodic C^alpha (capacitance_matrix_per_alpha.csv from run_defect_capacitance_bands,
see src/fullC.py). Agreement in the bulk is the check that truncating the infinite defect
line to a finite patch is justified (the couplings decay fast enough).

Parameters:
* radius = radius of the cladding disks
* defect_radius = radius of the defect disks
* delta = inverse contrast
* m_ang = order, monopole, dipole, quadrupole...
* n_defect = number of defect disks along the line (total = 2*n_defect + 1)
* n_clad = number of cladding disks along each axis (total = (2*n_clad + 1)^2 - (2*n_defect + 1))
* fringe = number of boundary defect disks to keep in the mesh but drop from C
* points_per_disk = number of quadrature points per disk boundary
* v = wave speed in free space
* v_b = wave speed in the cladding crystal interior
* v_bd = wave speed in the defect interior

Writes:
  patch_capacitance_couplings.csv  l, |C_l|_F, then modes*modes complex entries (re,im)
                                   row-major -- the center resonator's 2x2 coupling blocks.
  patch_defect_resonances.csv      Re(lambda), Im(lambda), Re(omega), Im(omega) -- eig(C),
                                   omega = omega_0 + delta*lambda (finite defect resonances).*/
void run_patch_capacitance(double radius = 0.35,
                           double defect_radius = 0.455,
                           double delta = 0.05,
                           int m_ang = 1,
                           int n_defect = 10,
                           int n_clad = 4,
                           int fringe = 1,
                           int points_per_disk = 16,
                           double v = 1.0,
                           double v_b = 1.0,
                           double v_bd = 1.0);

}  // namespace workflows

#endif  // SUBWAVELENGTHRESONATORS_WORKFLOWS_PATCH_CAPACITANCE_H
