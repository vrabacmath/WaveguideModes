#include "workflows/patch_capacitance.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
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

void run_patch_capacitance(double radius, double defect_radius, double delta,
                           int m_ang, int n_defect, int n_clad, int fringe,
                           int points_per_disk, double v, [[maybe_unused]] double v_b,
                           double v_bd) {
    // v_b (crystal interior speed) does not enter the leading-order capacitance: the cladding
    // disks are off-resonant at omega_0 and contribute only through the exterior operators.
    const int N = std::max(points_per_disk, 8);
    const int modes = (m_ang == 0) ? 1 : 2;
    const double beta = first_neumann_zero(m_ang);
    const double omega0 = v_bd * beta / defect_radius;
    const cpxd k = cpxd(omega0, 1e-3) / v;

    // Keep fringe defect disks in the mesh so they contribute, but omit them from the capacitance matrix.
    const int L = n_defect + fringe;
    const int main_lo = -n_defect;
    const int main_hi = n_defect;

    std::vector<Disk> disks;
    std::vector<int> defect_disk;  // disk index of each defect at (m,0), ordered m = -L..L
    for (int mx = -L; mx <= L; ++mx) {
        for (int ny = 1; ny <= n_clad; ++ny) {
            disks.push_back({radius, Vector2d(mx, ny)});
            disks.push_back({radius, Vector2d(mx, -ny)});
        }
        defect_disk.push_back(static_cast<int>(disks.size()));
        disks.push_back({defect_radius, Vector2d(mx, 0.0)});
    }

    BoundaryMesh mesh;
    for (const Disk& disk : disks) {
        BoundaryMesh d(N);
        d.generate_circle(disk.r, disk.c);
        mesh.add_mesh(d);
    }
    const int Ntot = mesh.get_num_segments();

    std::vector<int> main_indices;
    for (int mx = main_lo; mx <= main_hi; ++mx) main_indices.push_back(defect_disk[mx + L]);
    const int n_main = static_cast<int>(main_indices.size());
    const int center = n_defect;
    const int n_def = static_cast<int>(defect_disk.size());

    std::cout << "Patch capacitance (finite, centered): R=" << radius << ", R_def="
              << defect_radius << ", m=" << m_ang << ", omega_0=" << omega0 << ", "
              << disks.size() << " disks (" << n_def << " defect + "
              << (disks.size() - static_cast<std::size_t>(n_def)) << " cladding), " << Ntot
              << " boundary points\n";

    VectorXd sigma(Ntot);
    for (int i = 0; i < Ntot; ++i) sigma(i) = mesh.get_vertex(i).sigma;
    const double Anorm = 1.0 / (std::sqrt(M_PI) * defect_radius *
        std::sqrt(std::max(1.0 - double(m_ang * m_ang) / (beta * beta), 1e-12)));

    MatrixXcd G = MatrixXcd::Zero(Ntot, modes * n_main);
    for (int q = 0; q < n_main; ++q) {
        const int s0 = mesh.get_start_index(main_indices[q]);
        for (int loc = 0; loc < N; ++loc) {
            const double theta = 2.0 * M_PI * double(loc) / double(N);
            for (int sgn = 0; sgn < modes; ++sgn) {
                const double s = (sgn == 0) ? 1.0 : -1.0;
                G(s0 + loc, modes * q + sgn) =
                    Anorm * std::exp(cpxd(0, 1.0) * s * double(m_ang) * theta);
            }
        }
    }

    SpectralOperators ops(mesh);
    MatrixXcd S, Kstar;
    ops.S(S, k);
    ops.Kstar(Kstar, k);
    const MatrixXcd X = S.partialPivLu().solve(G);
    const MatrixXcd LamG = 0.5 * X + Kstar * X;
    const MatrixXcd C = -(v_bd * v_bd) / (2.0 * omega0) * (G.adjoint() * (sigma.asDiagonal() * LamG));

    const std::string filename = "patch_capacitance_couplings" + std::to_string(n_defect) + "_" +
                                 std::to_string(n_clad) + "_" + std::to_string(fringe) + ".csv";
    std::ofstream out_c(filename);
    out_c.setf(std::ios::scientific);
    out_c.precision(std::numeric_limits<double>::max_digits10);
    std::cout << "  center-row coupling decay |C_l|_F:\n";
    for (int q = 0; q < n_main; ++q) {
        const int ell = q - center;
        const MatrixXcd blk = C.block(modes * center, modes * q, modes, modes);
        out_c << ell << "," << blk.norm();
        for (int a = 0; a < modes; ++a)
            for (int b = 0; b < modes; ++b)
                out_c << "," << blk(a, b).real() << "," << blk(a, b).imag();
        out_c << "\n";
        std::cout << "    l=" << ell << "  |C_l|=" << blk.norm() << "\n";
    }
    out_c.close();

    ComplexEigenSolver<MatrixXcd> es(C);
    const VectorXcd lam = es.eigenvalues();
    std::vector<int> order(lam.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&lam](int a, int b) { return lam(a).real() < lam(b).real(); });

    std::ofstream out_r("patch_defect_resonances.csv");
    out_r.setf(std::ios::scientific);
    out_r.precision(std::numeric_limits<double>::max_digits10);
    for (int j = 0; j < static_cast<int>(order.size()); ++j) {
        const cpxd l = lam(order[j]);
        const cpxd omega = omega0 + delta * l;
        out_r << l.real() << "," << l.imag() << "," << omega.real() << "," << omega.imag() << "\n";
    }
    out_r.close();

    std::cout << "[wrote] " << filename << " (center-row C_l), "
              << "patch_defect_resonances.csv (eig(C) -> omega=omega_0+delta*lambda)\n";
}

}  // namespace workflows
