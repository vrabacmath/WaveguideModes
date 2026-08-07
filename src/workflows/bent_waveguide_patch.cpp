#include "workflows/bent_waveguide_patch.h"
#include "workflows/defect_capacitance_bands.h"

#include <cmath>
#include <complex>
#include <iostream>

#include "utils.h"

namespace workflows {

using namespace Eigen;

BentPatch build_bent_patch(double radius, double defect_radius, int m_ang, int n_defect,
                           int n_clad, int fringe, int points_per_disk, double v, double v_b,
                           double v_bd, bool verbose) {
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
    // Everything else inside the [-n_clad, n_clad]^2 square is a cladding disk.
    //
    // The lattice sweep below pushes disks in mesh order, which is NOT path order (it emits the
    // whole +y arm before the +x arm). So record each defect site's disk index as it is pushed,
    // then build the path-ordered list by looking sites up. Neither trace placement nor `center`
    // then relies on arithmetic over the mesh layout.
    p.L = n_defect + fringe - 1;   // chain half-length: ell runs -L .. +L
    p.L_main = p.L - fringe;       // outermost |ell| kept after discarding `fringe` layers

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
    // chain are left out of the coupling row but stay in the mesh, so they still screen.
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

    return p;
}

}  // namespace workflows
