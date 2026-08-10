#ifndef SUBWAVELENGTHRESONATORS_WORKFLOWS_BENT_WAVEGUIDE_PATCH_H
#define SUBWAVELENGTHRESONATORS_WORKFLOWS_BENT_WAVEGUIDE_PATCH_H

#include <map>
#include <utility>
#include <vector>

#include "Eigen/Dense"
#include "boundary_mesh.h"
#include "spectral_operators.h"
#include "workflows/bessel_helpers.h"

namespace workflows {

using namespace Eigen;

struct Disk {
    double r;
    Vector2d c;
};

/* The the exterior operators evaluated on an L-shaped defect chain in a crystal
are shared by all workflows with a bent waveguide.

The key field for field reconstruction is X = S^{-1} G. G is nonzero only on the defect
disks, but S^{-1} is dense, so X is nonzero on every boundary point - the cladding rows
carry the induced density that enforces the cladding's (sound-soft, at leading order)
boundary condition. So the cladding is fully represented here up to order delta
even though it contributes no rows or columns to C. 

Members:
* mesh = The boundary mesh for the patch
* disks = List of all disks in the patch
* defect_index =  Mapping from lattice sites to disk indices
* main_indices = Indices of the main defect disks in path order
* G = The coupling matrix for the defect modes 
* X = The single-layer density on all disks
* C = The exterior DtN operator projected onto the defect-mode subspace
* sigma = The boundary condition values
* k, kVb, kV, kVbd = Wave numbers for the background, crystal interior, and defect interior
* omega0 = The resonant frequency of the defect
* beta = The first positive zero of the derivative of the Bessel function J_m
* Anorm = Normalization constant for the defect modes
* radius = Radius of the cladding disks
* defect_radius = Radius of the defect disks
* L = Half-length of the defect chain
* L_main = Half-length of the main defect chain (excluding fringe)
* center = Index of the center disk in the defect chain
* modes = Number of defect modes (1 for monopole, 2 for dipole, etc.)
* N = Number of quadrature points per disk boundary
* m_ang = Angular order of the defect modes
* Ntot = Total number of boundary segments in the patch
*/
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

/* Assemble the patch and its exterior operators. `fringe` outer layers of the defect chain are
kept in the mesh but excluded from main_indices.

Parameters:
* radius = radius of the cladding disks
* defect_radius = radius of the defect disks
* m_ang = order, monopole, dipole, quadrupole...
* n_defect = number of defect disks along each arm of the L (total = 2*n_defect + 1)
* n_clad = number of cladding disks along each axis (total = (2*n_clad + 1)^2 - (2*n_defect + 1))
* fringe = number of boundary defect disks to keep in the mesh but drop from C
* points_per_disk = number of quadrature points per disk boundary
* v = wave speed in free space
* v_b = wave speed in the cladding crystal interior
* v_bd = wave speed in the defect interior*/
BentPatch build_bent_patch(double radius, double defect_radius, int m_ang, int n_defect,
                           int n_clad, int fringe, int points_per_disk, double v = 1.0,
                           double v_b = 1.0, double v_bd = 1.0, bool verbose = true);

}  // namespace workflows

#endif  // SUBWAVELENGTHRESONATORS_WORKFLOWS_BENT_WAVEGUIDE_PATCH_H
