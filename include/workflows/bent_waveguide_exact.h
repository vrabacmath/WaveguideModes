#ifndef SUBWAVELENGTHRESONATORS_WORKFLOWS_BENT_WAVEGUIDE_EXACT_H
#define SUBWAVELENGTHRESONATORS_WORKFLOWS_BENT_WAVEGUIDE_EXACT_H

namespace workflows {

/* Exact resonances of the bent-waveguide patch, with no delta-asymptotics: the characteristic
values omega* where the full transmission operator A(omega) on the whole cluster is singular,

  A(omega) = [  S_b                  -S          ]  (interior/exterior single + adjoint-double
             [ -1/2 I + K*_b  -delta(1/2 I + K*) ]   layers, all disks, cladding included)

found by Muller's method on the analytic indicator mu(omega) = r^H A(omega)^{-1} r
the eigenvalue of A closest to 0 is estimated using inverse iterations with a probe vector r.

mode_index >= 0 refines and draws that capacitance mode (sorted by Re lambda). mode_index < 0
refines a spread set {smallest, middle, largest lambda} -- chosen to expose how the O(delta)
error grows with |lambda| -- plus the most corner-localized mode, which is the one drawn.

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
* seed_re, seed_im = complex frequency seed for Muller's method; if < 0, use the capacitance eigenvalue
* v = wave speed in free space
* v_b = wave speed in the cladding crystal interior
* v_bd = wave speed in the defect interior

Writes:
  bent_waveguide_exact_resonances.csv  j, Re/Im lambda, Re/Im omega_asym, Re/Im omega_exact,
                                       |domega|, iterations, ||A psi|| residual
  bent_waveguide_exact_field_{real,imag,mag,phase}.csv, bent_waveguide_exact_field_axis.csv
seed_re > 0 (with mode_index >= 0) overrides the capacitance seed with the complex frequency
(seed_re, seed_im) -- e.g. a Richardson-improved guess from a larger delta solve, when the
"miniband" is too dense for the raw asymptotic seed to stay in its own basin.

grid_points <= 0 skips the field evaluation entirely (resonance table + fingerprint only) and
leaves any existing field CSVs untouched -- use this for validation sweeps.*/
void run_bent_waveguide_exact(double radius = 0.35,
                              double defect_radius = 0.455,
                              double delta = 0.05,
                              int m_ang = 1,
                              int n_defect = 3,
                              int n_clad = 3,
                              int fringe = 1,
                              int points_per_disk = 16,
                              int mode_index = -1,
                              int grid_points = 200,
                              double seed_re = -1.0,
                              double seed_im = 0.0,
                              double v = 1.0,
                              double v_b = 1.0,
                              double v_bd = 1.0);

}  // namespace workflows

#endif  // SUBWAVELENGTHRESONATORS_WORKFLOWS_BENT_WAVEGUIDE_EXACT_H
