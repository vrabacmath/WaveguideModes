#ifndef SUBWAVELENGTHRESONATORS_WORKFLOWS_BENT_WAVEGUIDE_FIELD_H
#define SUBWAVELENGTHRESONATORS_WORKFLOWS_BENT_WAVEGUIDE_FIELD_H

namespace workflows {

/* Reconstruct and save the physical field u_n(x) of one eigenmode of the bent-waveguide patch.

The leading-order (O(delta)) field needs no new solve: for an eigenpair (lambda, v) of the
capacitance matrix C, psi = X v is the single-layer density on the whole cluster, cladding
included, because X = S^{-1} G and S^{-1} is dense. Then

  exterior          u(r) = sum_j psi_j G_k(r, r_j) sigma_j
  defect disk p     u    = sum_sgn v[modes*p+sgn] * Anorm * J_m(beta rho/R_def)/J_m(beta) e^{+-i m phi}
  cladding disk     u    = 0            (non-resonant => sound-soft at leading order)

The two branches agree at rho = R_def by construction, so u is continuous across defect
boundaries; the normal-derivative jump is the O(delta) model error.

Parameters:
* radius = radius of the cladding disks
* defect_radius = radius of the defect disks
* delta = inverse contrast
* m_ang = order, monopole, dipole, quadrupole...
* n_defect = number of defect disks along each arm of the L (total = 2*n_defect + 1)
* n_clad = number of cladding disks along each axis (total = (2*n_clad + 1)^2 - (2*n_defect + 1))
* fringe = number of boundary defect disks to keep in the mesh but drop from C
* points_per_disk = number of quadrature points per disk boundary
* mode_index = selects the eigenmode after sorting by Re(lambda), negative means "most localized at the corner".
* grid_points = number of points along each axis of the square grid [-L-1, L+1]^2 on which to evaluate u
* v = wave speed in free space
* v_b = wave speed in the cladding crystal interior
* v_bd = wave speed in the defect interior

Writes:
  bent_waveguide_modes.csv   per-mode Re/Im(lambda), Re/Im(omega), corner weight, participation
  bent_waveguide_field_{real,imag,mag,phase}.csv   grid_points^2 field maps
  bent_waveguide_field_axis.csv                    the grid coordinate axis */
  void run_bent_waveguide_field(double radius = 0.35,
                              double defect_radius = 0.455,
                              double delta = 0.05,
                              int m_ang = 1,
                              int n_defect = 3,
                              int n_clad = 3,
                              int fringe = 1,
                              int points_per_disk = 64,
                              int mode_index = -1,
                              int grid_points = 300,
                              double v = 1.0,
                              double v_b = 1.0,
                              double v_bd = 1.0);

}  // namespace workflows

#endif  // SUBWAVELENGTHRESONATORS_WORKFLOWS_BENT_WAVEGUIDE_FIELD_H
