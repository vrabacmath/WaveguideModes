#ifndef SUBWAVELENGTHRESONATORS_WORKFLOWS_BENT_WAVEGUIDE_EXACT_H
#define SUBWAVELENGTHRESONATORS_WORKFLOWS_BENT_WAVEGUIDE_EXACT_H

namespace workflows {

// EXACT resonances of the bent-waveguide patch, with no delta-asymptotics: the characteristic
// values omega* where the full transmission operator A(omega) on the whole cluster is singular,
//
//   A(omega) = [  S_b           -S          ]      (interior/exterior single + adjoint-double
//              [ -1/2 I + K*_b  -delta(1/2 I + K*) ]  layers, all disks, cladding included)
//
// found by Muller's method on the analytic indicator g(omega) = 1 / (w^H A(omega)^{-1} r)
// (poles of the resolvent -> zeros of g; sigma_min itself is not analytic, so it cannot drive
// Muller). Seeds are the capacitance predictions omega_0 + delta*lambda_j -- that is what the
// whole capacitance apparatus is FOR in this workflow. The mode is the null vector of
// A(omega*), obtained by inverse iteration; there is no right-hand side because a resonance is
// a free oscillation.
//
// mode_index >= 0 refines and draws that capacitance mode (sorted by Re lambda). mode_index < 0
// refines a spread set {smallest, middle, largest lambda} -- chosen to expose how the O(delta)
// error grows with |lambda| -- plus the most corner-localized mode, which is the one drawn.
//
// Writes:
//   bent_waveguide_exact_resonances.csv  j, Re/Im lambda, Re/Im omega_asym, Re/Im omega_exact,
//                                        |domega|, iterations, ||A psi|| residual
//   bent_waveguide_exact_field_{real,imag,mag,phase}.csv, bent_waveguide_exact_field_axis.csv
// seed_re > 0 (with mode_index >= 0) overrides the capacitance seed with the complex frequency
// (seed_re, seed_im) -- e.g. a Richardson-improved guess from a larger-delta solve, when the
// miniband is too dense for the raw asymptotic seed to stay in its own basin.
//
// grid_points <= 0 skips the field evaluation entirely (resonance table + fingerprint only) and
// leaves any existing field CSVs untouched -- use this for validation sweeps.
//
// Wave speeds: v background, v_b crystal interior, v_bd defect interior. Interior fields are
// local to each disk, so A(omega) mixes them row-wise: boundary rows of defect disks carry the
// interior layers at k_bd = omega/v_bd, cladding rows at k_b = omega/v_b.
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
