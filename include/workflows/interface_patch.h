#ifndef SUBWAVELENGTHRESONATORS_WORKFLOWS_INTERFACE_PATCH_H
#define SUBWAVELENGTHRESONATORS_WORKFLOWS_INTERFACE_PATCH_H

#include <map>
#include <utility>
#include <vector>

#include "Eigen/Dense"
#include "boundary_mesh.h"
#include "disk.h"
#include "spectral_operators.h"
#include "workflows/bessel_helpers.h"

namespace workflows {

using namespace Eigen;

using ::Disk;

struct LatticeVectors {
    // a1 is the common tangential period. a2 points away from the interface
    // into this material (upward for up, downward for down).
    Vector2d a1, a2;
};

struct RowTarget {
    int local_x, local_y;
    // Index within the target cell, not within the whole template mesh.
    int disk_number;
};

struct InterfaceLayout {
    using Cell = std::pair<int, int>;
    std::map<Cell, Eigen::Index> column_start, row_start;
    // source_rows[j] is the output row for the same physical DOF as column j.
    std::vector<Eigen::Index> source_rows;
    Eigen::Index rows = 0, cols = 0;
};

struct InterfaceSolveDiagnostics {
    // Local single-layer solve S*X=G0, not the global shifted capacitance matrix.
    double rcond, relative_residual, backward_error, density_norm;
};

struct InterfacePatch {
    int window_radius, patch_radius, colbrook, fringe;
    int Nmultipole = 10;  // multipole truncation order for the exterior operators
    std::vector<BoundaryMesh> meshes;

    std::vector<std::vector<RowTarget>> row_offsets;
    // disks[k][i] describes mesh disk i in window k, in source-relative coordinates.
    // disk_indices[k].equal_range({0,0}) identifies all central-cell source disks.
    std::vector<Disk> disks_up, disks_down;
    LatticeVectors lattice_up, lattice_down;
    std::vector<std::vector<Disk>> disks;  // disks[k] describes mesh disks in window k, in source-relative coordinates.
    std::vector<std::multimap<std::pair<int, int>, int>> disk_indices;  // maps source-relative coordinates to disk index in disks[k]

    // Columns: all source-cell DOFs in the square; rows include colbrook padding on every side.
    MatrixXcd bigC;
    // Template t has source height R-t, R=window_radius+fringe. Both bulk endpoints are cached.
    // Row block q describes row_offsets[t][q]; columns excite the central cell in disk/mode order.
    std::vector<MatrixXcd> patch_colC;
    // ((2*Nmultipole+1)*disk_count) x (modes*source_disk_count) Fourier density coefficients
    // for unit, unscaled G0 traces. Each source disk has +m then -m (one column for m=0).
    // Rows use mode-major ordering; fourier_basis(...) * patch_X gives nodal densities.
    std::vector<MatrixXcd> patch_X;
    std::vector<InterfaceSolveDiagnostics> solve_diagnostics;

    cpxd k, kV;
    double omega0, beta, Anorm;

    int N, modes, nm, m_ang;
};

// Source cells occupy [-patch_radius,patch_radius]^2; output cells include
// colbrook padding on all four sides. Cells at y >= 0 use disks_up.
InterfaceLayout make_interface_layout(const InterfacePatch& ip);

// First upper row is y=0; first lower row is y=-1. Basis disk centers are
// measured from these cell origins, in global Cartesian axes.
Vector2d interface_cell_origin(const InterfacePatch& ip, int x, int y);

// Reuse cached local columns, retaining targets inside the core window square.
MatrixXcd assemble_interface_matrix(const InterfacePatch& ip);

// Cell-index square windows have half-width R=window_radius+fringe.
// Build 2*R+2 templates, including homogeneous upper/lower endpoints.
// All mesh disks participate in the exterior solve; fringe rows are omitted in bigC.
// Geometry is generated globally, then translated to the source cell origin.
InterfacePatch build_interface_patch(int window_radius, int patch_radius, int colbrook, 
                                        LatticeVectors up, LatticeVectors down, int nm,
                                        vector<Disk> disks_up, vector<Disk> disks_down,
                                        int fringe, int points_per_disk,
                                        double omega0, double v = 1.0, int Nmultipole = 10,
                                        bool verbose = true);


} // namespace workflows

#endif  // SUBWAVELENGTHRESONATORS_WORKFLOWS_INTERFACE_PATCH_H
