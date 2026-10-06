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
    MatrixXcd C;  // (modes*cb_main)^2: exterior DtN projected onto the defect-mode subspace
    MatrixXcd Ccb;// (modes*cb_main)x(modes*n_main): rectangular with extra Colbrook rows for the outer defect layers
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
                           int n_clad, int colbrook, int fringe, int points_per_disk, double v = 1.0,
                           double v_b = 1.0, double v_bd = 1.0, bool verbose = true);

struct TessellatedPatch {
    int window_radius, patch_radius, colbrook, fringe;
    bool bent;
    // All bend-intersecting meshes are distinct; the two outer templates are straight.
    int template_radius() const { return window_radius + fringe + 1; }
    int template_index(int ell) const {
        const int T = template_radius();
        return ell < -T ? 0 : (ell > T ? 2 * T : ell + T);
    }
    std::vector<BoundaryMesh> meshes;

    std::vector<std::map<std::pair<int, int>, int>> defect_indices;  // source-relative site -> disk index
    // All defect centers in [-window_radius, window_radius]^2, ordered along the path.
    std::vector<std::vector<int>> main_indices;
    // row_offsets[k][q] = target ell - source ell for main_indices[k][q].
    // Near the bend these offsets can exceed window_radius and the row count varies.
    std::vector<std::vector<int>> row_offsets;
    // disks[k][i] describes mesh disk i in window k, in source-relative coordinates.
    // The central disk is defect_indices[k].at({0,0}), not necessarily disk 0.
    std::vector<std::vector<Disk>> disks;

    // Columns: ell = -patch_radius..patch_radius; rows include colbrook extra sites at each end.
    MatrixXcd bigC;
    // Template t + template_radius() excites path site t; t=-T..T, T=window_radius+fringe+1.
    // The endpoint meshes are purely straight. Row block q has path offset row_offsets[k][q].
    std::vector<MatrixXcd> patch_colC;
    // Ntot x modes per window: density for unit central sources, columns +m then -m
    // (one column for m=0), with rows in the corresponding mesh's boundary-point order.
    std::vector<MatrixXcd> patch_X;

    cpxd k, kVb, kV, kVbd;  // kV: background speed, kVb: crystal interior, kVbd: defect interior
    double omega0, beta, Anorm, radius, defect_radius;

    int N, modes, m_ang;
};

// Reuse cached block columns without any boundary-integral solves. There are 2*T+1
// templates for source labels -T..T. Template k has modes*row_offsets[k].size() rows
// and modes columns; offsets must be strictly increasing and include the source (0).
// Repeat the two strictly straight endpoint columns outside -T..T.
// Place each block at global row ell + row_offsets[k][q], cropping only to the
// requested global row range [-patch_radius-colbrook, patch_radius+colbrook].
MatrixXcd assemble_tessellated_columns(const std::vector<MatrixXcd>& columns,
                                     const std::vector<std::vector<int>>& row_offsets,
                                     int patch_radius, int colbrook);

// Local square meshes have half-width window_radius + fringe, centered on the source disk.
// Build 2*(window_radius+fringe)+3 templates, including two strictly straight endpoints.
// Project every defect with a local center in [-window_radius, window_radius]^2.
// The surrounding fringe stays in the boundary solve but contributes no coupling rows.
// The bend uses BentPatch::site_at; bent=false uses the straight line (ell, 0).
TessellatedPatch build_tessellated_patch(int window_radius, int patch_radius, int colbrook, bool bent,
                                         double radius, double defect_radius, int m_ang,
                                         int fringe, int points_per_disk, double v = 1.0,
                                         double v_b = 1.0, double v_bd = 1.0, bool verbose = true);


} // namespace workflows

#endif  // SUBWAVELENGTHRESONATORS_WORKFLOWS_BENT_WAVEGUIDE_PATCH_H
