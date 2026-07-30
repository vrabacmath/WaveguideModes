#ifndef SUBWAVELENGTHRESONATORS_WORKFLOWS_BENT_WAVEGUIDE_PATCH_H
#define SUBWAVELENGTHRESONATORS_WORKFLOWS_BENT_WAVEGUIDE_PATCH_H

#include <map>
#include <utility>
#include <vector>

#include "Eigen/Dense"
#include "boundary_mesh.h"
#include "spectral_operators.h"

namespace workflows {

using namespace Eigen;

// First positive zero of J_m' = interior Neumann eigenvalue index (radial order n = 1).
double first_neumann_zero(int m);

struct Disk {
    double r;
    Vector2d c;
};

// Everything the bent-waveguide workflows share: the L-shaped defect chain embedded in a
// cladding crystal, plus the exterior operators evaluated on it.
//
// The key field for field reconstruction is X = S^{-1} G. G is nonzero only on the defect
// disks, but S^{-1} is dense, so X is nonzero on EVERY boundary point -- the cladding rows
// carry the induced density that enforces the cladding's (sound-soft, at leading order)
// boundary condition. So the cladding is fully represented here even though it contributes
// no rows or columns to C.
struct BentPatch {
    BoundaryMesh mesh;
    std::vector<Disk> disks;
    std::map<std::pair<int, int>, int> defect_index;  // lattice site -> disk index
    std::vector<int> main_indices;                    // disk indices, path order ell = -L_main..+L_main

    MatrixXcd G;  // Ntot x (modes * n_main): Dirichlet traces of the interior Neumann modes
    MatrixXcd X;  // Ntot x (modes * n_main): S^{-1} G -- single-layer density on ALL disks
    MatrixXcd C;  // (modes*n_main)^2: exterior DtN projected onto the defect-mode subspace
    VectorXd sigma;

    cpxd k, kVb, kV, kVbd;  // kV: background speed, kVb: crystal interior, kVbd: defect interior
    double omega0, beta, Anorm, radius, defect_radius;
    int L, L_main, center, modes, N, m_ang, Ntot;

    // Lattice site of chain position `ell`: the +x arm for ell <= 0, the +y arm for ell >= 0.
    static std::pair<int, int> site_at(int ell) {
        return (ell <= 0) ? std::make_pair(-ell, 0) : std::make_pair(0, ell);
    }
};

// Assemble the patch and its exterior operators. `fringe` outer layers of the defect chain are
// kept in the mesh (so they still screen) but excluded from main_indices, leaving the
// 2*n_defect - 1 interior sites free of patch-edge artefacts.
// Wave speeds: v background, v_b crystal interior, v_bd defect interior. omega_0 = v_bd
// j'_{m,1}/R_def and C carries v_bd^2; v_b does not enter the leading-order capacitance but is
// stored on the patch (kVb) for the exact workflows.
BentPatch build_bent_patch(double radius, double defect_radius, int m_ang, int n_defect,
                           int n_clad, int fringe, int points_per_disk, double v = 1.0,
                           double v_b = 1.0, double v_bd = 1.0, bool verbose = true);

}  // namespace workflows

#endif  // SUBWAVELENGTHRESONATORS_WORKFLOWS_BENT_WAVEGUIDE_PATCH_H
