#include "workflows/bent_waveguide_patch.h"
#include "workflows/defect_capacitance_bands.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <stdexcept>

#include "utils.h"

namespace workflows {

using namespace Eigen;

BentPatch build_bent_patch(double radius, double defect_radius, int m_ang, int n_defect,
                           int n_clad, int colbrook, int fringe, int points_per_disk, double v, double v_b,
                           double v_bd, bool verbose) {
    if (n_defect < 1 || n_clad < 0 || colbrook < 0 || fringe < 0)
        throw std::invalid_argument("Bent patch requires n_defect >= 1 and nonnegative n_clad, colbrook, and fringe");
    BentPatch p;
    p.N = std::max(points_per_disk, 8);
    p.m_ang = m_ang;
    p.modes = (m_ang == 0) ? 1 : 2;
    p.beta = first_neumann_zero(m_ang);
    p.radius = radius;
    p.defect_radius = defect_radius;
    p.kV = v;
    p.kVb = v_b;
    p.kVbd = v_bd;
    p.omega0 = v_bd * p.beta / defect_radius;
    p.k = cpxd(p.omega0, 1e-3) / p.kV;  // small Im(k) regularises the near-resonant solve

    // --- build the centered patch --------------------------------------------------------
    // The defect chain is an L: it runs in along +x, turns at the corner (0,0), and runs back
    // out along +y. Sites are addressed by signed path distance `ell` from the corner:
    //     site(ell) = (-ell, 0)  for ell <= 0   (the +x arm)
    //               = ( 0, ell)  for ell >= 0   (the +y arm)
    // Which can be simplified to only ell.
    // Everything else inside the [-n_clad, n_clad]^2 square is a cladding disk.
    //
    // The lattice sweep below pushes disks in mesh order, which is NOT path order (it emits the
    // whole +y arm before the +x arm). So record each defect site's disk index as it is pushed,
    // then build the path-ordered list by looking sites up. Neither trace placement nor `center`
    // then relies on arithmetic over the mesh layout.
    p.L = n_defect + colbrook + fringe - 1;  // colbrook half-length: ell runs -Lcb .. +Lcb
    p.L_main = p.L - fringe;  // outermost |ell| kept after discarding `fringe` layers

    auto is_defect = [](int mx, int ny) {
        return ((mx == 0) && (ny >= 0)) || ((ny == 0) && (mx >= 0));
    };

    for (int mx = -p.L; mx <= p.L; ++mx) {
        for (int ny = -p.L; ny <= p.L; ++ny) {
            if (is_defect(mx, ny)) {
                p.defect_index[{mx, ny}] = static_cast<int>(p.disks.size());
                p.disks.push_back({defect_radius, Vector2d(mx, ny)});
            } else if (std::abs(mx) <= n_clad && std::abs(ny) <= n_clad) {
                p.disks.push_back({radius, Vector2d(mx, ny)});
            }
        }
    }

    // Interior sites in path order, ell = -L_main .. +L_main. The outer `fringe` layers of the
    // chain are left out of the coupling row but stay in the mesh and the computation.
    for (int ell = -p.L_main; ell <= p.L_main; ++ell) {
        const std::pair<int, int> s = BentPatch::site_at(ell);
        p.main_indices.push_back(p.defect_index.at(s));
        if (verbose)
            std::cout << "location of defect disk: (" << s.first << ", " << s.second
                      << "), ell = " << ell << ", index = " << p.defect_index.at(s) << "\n";
    }

    for (std::size_t i = 0; i < p.disks.size(); ++i) {
        BoundaryMesh d(p.N);
        d.generate_circle(p.disks[i].r, p.disks[i].c);
        p.mesh.add_mesh(d);
    }
    p.Ntot = p.mesh.get_num_segments();
    p.center = p.L_main;  // main_indices[center] is the corner (0,0), by construction

    const int n_def = static_cast<int>(p.defect_index.size());  // = 2 L + 1
    if (verbose)
        std::cout << "Patch capacitance (finite, centered): R=" << radius << ", R_def="
                  << defect_radius << ", m=" << m_ang << ", omega_0=" << p.omega0 << ", "
                  << p.disks.size() << " disks (" << n_def << " defect + "
                  << (p.disks.size() - static_cast<std::size_t>(n_def)) << " cladding), "
                  << p.Ntot << " boundary points\n";

    // --- interior Neumann traces: one column-block per defect resonator ------------------
    p.sigma.resize(p.Ntot);
    for (int i = 0; i < p.Ntot; ++i) p.sigma(i) = p.mesh.get_vertex(i).sigma;
    // L^2(D) normalization of e^{i m theta} on a disk of radius R_def: A = 1/sqrt(pi R^2 (1 - m^2/beta^2)).
    p.Anorm = 1.0 / (std::sqrt(M_PI) * defect_radius *
                     std::sqrt(std::max(1.0 - double(m_ang * m_ang) / (p.beta * p.beta), 1e-12)));

    const int n_main = static_cast<int>(p.main_indices.size());
    p.G = MatrixXcd::Zero(p.Ntot, p.modes * n_main);
    for (int q = 0; q < n_main; ++q) {
        const int s0 = p.mesh.get_start_index(p.main_indices[q]);
        for (int loc = 0; loc < p.N; ++loc) {
            const double theta = 2.0 * M_PI * double(loc) / double(p.N);
            for (int sgn = 0; sgn < p.modes; ++sgn) {
                const double s = (sgn == 0) ? 1.0 : -1.0;
                p.G(s0 + loc, p.modes * q + sgn) =
                    p.Anorm * std::exp(cpxd(0, 1.0) * s * double(m_ang) * theta);
            }
        }
    }

    // --- exterior DtN, assembled ONCE ----------------------------------------------------
    // Lambda_ext = (1/2 I + K*) S^{-1} with the free-space single/adjoint-double layers on the
    // whole finite cluster; C = -(v_b^2 / 2 omega_0) G^H diag(sigma) Lambda_ext G.
    SpectralOperators ops(p.mesh);
    MatrixXcd S, Kstar;
    ops.S(S, p.k);
    ops.Kstar(Kstar, p.k);
    p.X = S.partialPivLu().solve(p.G);
    const MatrixXcd LamG = 0.5 * p.X + Kstar * p.X;
    p.C = -(p.kVbd * p.kVbd) / (2.0 * p.omega0) * (p.G.adjoint() * (p.sigma.asDiagonal() * LamG));
    // Keep all extended-patch rows and remove the extra layers from each end of the columns.
    const int trim = colbrook * p.modes;
    p.Ccb = p.C.middleCols(trim, p.C.cols() - 2 * trim);
    cout << "Patch capacitance matrix Ccb = " << p.Ccb << "\n\n";
    cout << "Patch capacitance matrix C = " << p.C << "\n\n";

    return p;
}

MatrixXcd assemble_tessellated_columns(const std::vector<MatrixXcd>& columns,
                                     int patch_radius, int colbrook) {
    if (columns.empty() || columns.size() % 2 == 0 || patch_radius < 0 || colbrook < 0)
        throw std::invalid_argument("Expected an odd number of templates and nonnegative patch/row radii");
    const int window_size = static_cast<int>(columns.size());
    const int window_radius = window_size / 2;
    const int modes = static_cast<int>(columns.front().cols());
    if (patch_radius < window_radius || modes == 0)
        throw std::invalid_argument("Patch radius must cover the window, and templates must have modes");
    for (const auto& column : columns)
        if (column.cols() != modes || column.rows() != modes * window_size)
            throw std::invalid_argument("Each template must contain one mode block per path offset");

    const int row_radius = patch_radius + colbrook;
    MatrixXcd bigC = MatrixXcd::Zero(modes * (2 * row_radius + 1),
                                    modes * (2 * patch_radius + 1));
    for (int j = -patch_radius; j <= patch_radius; ++j) {
        const int mesh = std::clamp(j, -window_radius, window_radius) + window_radius;
        for (int d = -window_radius; d <= window_radius; ++d) {
            const int row = j + d;
            if (row < -row_radius || row > row_radius) continue;
            bigC.block(modes * (row + row_radius), modes * (j + patch_radius), modes, modes) =
                columns[mesh].block(modes * (d + window_radius), 0, modes, modes);
        }
    }
    return bigC;
}

TessellatedPatch build_tessellated_patch(int window_radius, int patch_radius, int colbrook, bool bent,
                                         double radius, double defect_radius, int m_ang,
                                         int fringe, int points_per_disk, double v, double v_b,
                                         double v_bd, bool verbose) {
    if (window_radius < 0 || patch_radius < window_radius || colbrook < 0 || fringe < 0)
        throw std::invalid_argument("Require 0 <= window_radius <= patch_radius and nonnegative buffers");
    if (radius <= 0 || defect_radius <= 0 || v <= 0 || v_b <= 0 || v_bd <= 0 || m_ang < 0)
        throw std::invalid_argument("Radii and speeds must be positive and angular order nonnegative");

    TessellatedPatch tp;
    tp.window_radius = window_radius;
    tp.patch_radius = patch_radius;
    tp.colbrook = colbrook;
    tp.N = std::max(points_per_disk, 8);
    tp.m_ang = m_ang;
    tp.modes = (m_ang == 0) ? 1 : 2;
    tp.radius = radius;
    tp.defect_radius = defect_radius;
    tp.kV = v;
    tp.kVb = v_b;
    tp.kVbd = v_bd;
    tp.beta = first_neumann_zero(m_ang);
    tp.omega0 = v_bd * tp.beta / defect_radius;
    tp.k = cpxd(tp.omega0, 1e-3) / tp.kV;
    tp.Anorm = 1.0 / (std::sqrt(M_PI) * defect_radius *
                     std::sqrt(std::max(1.0 - double(m_ang * m_ang) / (tp.beta * tp.beta), 1e-12)));

    const int window_size = 2 * window_radius + 1;
    const int mesh_radius = window_radius + fringe;
    tp.meshes.resize(window_size);
    tp.defect_indices.resize(window_size);
    tp.main_indices.resize(window_size);
    tp.patch_colC.resize(window_size);
    const auto site_at = [bent](int ell) {
        return bent ? BentPatch::site_at(ell) : std::make_pair(ell, 0);
    };

    for (int t = -window_radius; t <= window_radius; ++t) {
        const int mesh = t + window_radius;
        const auto source = site_at(t);
        auto& patch_mesh = tp.meshes[mesh];
        auto& defects = tp.defect_indices[mesh];
        auto& main = tp.main_indices[mesh];

        // Translate a square window to each source site; store coordinates relative to it.
        // Keep all defect disks in the square (including fringe) in the exterior solve.
        for (int x = -mesh_radius; x <= mesh_radius; ++x) {
            for (int y = -mesh_radius; y <= mesh_radius; ++y) {
                const int gx = x + source.first, gy = y + source.second;
                const bool defect = bent ? ((gx == 0 && gy >= 0) || (gy == 0 && gx >= 0))
                                         : (gy == 0);
                if (defect) defects[{x, y}] = patch_mesh.get_num_meshes();
                BoundaryMesh disk(tp.N);
                disk.generate_circle(defect ? defect_radius : radius, Vector2d(x, y));
                patch_mesh.add_mesh(disk);
            }
        }
        // Matrix order follows path offsets, independently of the mesh's disk order.
        for (int d = -window_radius; d <= window_radius; ++d) {
            const auto target = site_at(t + d);
            main.push_back(defects.at({target.first - source.first, target.second - source.second}));
        }

        const int Ntot = patch_mesh.get_num_segments();
        VectorXd sigma(Ntot);
        for (int i = 0; i < Ntot; ++i) sigma(i) = patch_mesh.get_vertex(i).sigma;
        MatrixXcd G = MatrixXcd::Zero(Ntot, tp.modes * window_size);
        for (int q = 0; q < window_size; ++q) {
            const int s0 = patch_mesh.get_start_index(main[q]);
            for (int loc = 0; loc < tp.N; ++loc) {
                const double theta = 2.0 * M_PI * double(loc) / double(tp.N);
                for (int sgn = 0; sgn < tp.modes; ++sgn) {
                    const double sign = (sgn == 0) ? 1.0 : -1.0;
                    G(s0 + loc, tp.modes * q + sgn) =
                        tp.Anorm * std::exp(cpxd(0, sign * double(m_ang) * theta));
                }
            }
        }
        // d=0 is always block window_radius; a mesh disk index is NOT a G column index.
        const MatrixXcd g0 = G.middleCols(tp.modes * window_radius, tp.modes);
        SpectralOperators ops(patch_mesh);
        MatrixXcd S, Kstar;
        ops.S(S, tp.k);
        ops.Kstar(Kstar, tp.k);
        const MatrixXcd X = S.partialPivLu().solve(g0);
        const MatrixXcd LamG = 0.5 * X + Kstar * X;
        tp.patch_colC[mesh] = -(tp.kVbd * tp.kVbd) / (2.0 * tp.omega0) *
                              (G.adjoint() * (sigma.asDiagonal() * LamG));
        if (verbose)
            std::cout << "Tessellation template " << mesh << ": source ell=" << t
                      << ", " << patch_mesh.get_num_meshes() << " disks, " << Ntot << " boundary points\n";
    }

    tp.bigC = assemble_tessellated_columns(tp.patch_colC, patch_radius, colbrook);
    return tp;
}

}  // namespace workflows
