#include "workflows/patch_capacitance.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <fstream>
#include <iostream>
#include <vector>

#include "Eigen/Dense"
#include "boundary_mesh.h"
#include "spectral_operators.h"

namespace workflows {

using namespace Eigen;

namespace {

// First positive zero of J_m' = interior Neumann eigenvalue index (radial order n = 1).
double first_neumann_zero(int m) {
    switch (m) {
        case 0: return 3.8317059702;  // breathing (= j_{1,1})
        case 1: return 1.8411837813;  // dipole
        case 2: return 3.0542369282;  // quadrupole
        case 3: return 4.2011889412;  // octupole
        default: return 1.8411837813;
    }
}

}  // namespace

void run_patch_capacitance(double radius, double defect_radius, double delta,
                           int m_ang, int n_defect, int n_clad, int points_per_disk) {
    const int N = std::max(points_per_disk, 8);
    const int modes = (m_ang == 0) ? 1 : 2;
    const double beta = first_neumann_zero(m_ang);
    const cpxd kVb = 1.0, kV = 1.0;
    const double omega0 = std::abs(kVb) * beta / defect_radius;
    const cpxd k = cpxd(omega0, 1e-3) / kV;  // small Im(k) regularises the near-resonant solve

    // --- build the centered patch --------------------------------------------------------
    // Rows x = -n_defect .. n_defect; each row is a defect disk at (m, 0) plus cladding crystal
    // disks at (m, +/- n), n = 1 .. n_clad. Record each defect disk's index so trace placement
    // never relies on arithmetic over the mesh layout (the source of the earlier index bug).
    struct Disk { double r; Vector2d c; };
    std::vector<Disk> disks;
    std::vector<int> defect_disk;  // disk index of the defect at (m,0), ordered m = -Nd..Nd
    for (int mx = -n_defect; mx <= n_defect; ++mx) {
        for (int ny = 1; ny <= n_clad; ++ny) {
            disks.push_back({radius, Vector2d(mx, ny)});
            disks.push_back({radius, Vector2d(mx, -ny)});
        }
        defect_disk.push_back(static_cast<int>(disks.size()));
        disks.push_back({defect_radius, Vector2d(mx, 0.0)});
    }

    BoundaryMesh mesh(N);
    mesh.generate_circle(disks[0].r, disks[0].c);
    for (std::size_t i = 1; i < disks.size(); ++i) {
        BoundaryMesh d(N);
        d.generate_circle(disks[i].r, disks[i].c);
        mesh.add_mesh(d);
    }
    const int Ntot = mesh.get_num_segments();
    const int n_def = static_cast<int>(defect_disk.size());  // = 2 n_defect + 1
    const int center = n_defect;                             // index of the x = 0 resonator

    std::cout << "Patch capacitance (finite, centered): R=" << radius << ", R_def="
              << defect_radius << ", m=" << m_ang << ", omega_0=" << omega0 << ", "
              << disks.size() << " disks (" << n_def << " defect + "
              << (disks.size() - static_cast<std::size_t>(n_def)) << " cladding), " << Ntot
              << " boundary points\n";

    // --- interior Neumann traces: one column-block per defect resonator ------------------
    VectorXd sigma(Ntot);
    for (int i = 0; i < Ntot; ++i) sigma(i) = mesh.get_vertex(i).sigma;
    // L^2(D) normalization of e^{i m theta} on a disk of radius R_def: A = 1/sqrt(pi R^2 (1 - m^2/beta^2)).
    const double Anorm = 1.0 / (std::sqrt(M_PI) * defect_radius *
        std::sqrt(std::max(1.0 - double(m_ang * m_ang) / (beta * beta), 1e-12)));

    MatrixXcd G = MatrixXcd::Zero(Ntot, modes * n_def);
    for (int p = 0; p < n_def; ++p) {
        const int disk = defect_disk[p];
        const int s0 = mesh.get_start_index(disk);
        for (int loc = 0; loc < N; ++loc) {
            const double theta = 2.0 * M_PI * double(loc) / double(N);
            for (int sgn = 0; sgn < modes; ++sgn) {
                const double s = (sgn == 0) ? 1.0 : -1.0;
                G(s0 + loc, modes * p + sgn) =
                    Anorm * std::exp(cpxd(0, 1.0) * s * double(m_ang) * theta);
            }
        }
    }

    // --- exterior DtN, assembled ONCE ----------------------------------------------------
    // Lambda_ext = (1/2 I + K*) S^{-1} with the free-space single/adjoint-double layers on the
    // whole finite cluster; C = -(v_b^2 / 2 omega_0) G^H diag(sigma) Lambda_ext G.
    SpectralOperators ops(mesh);
    MatrixXcd S, Kstar;
    ops.S(S, k);
    ops.Kstar(Kstar, k);
    const PartialPivLU<MatrixXcd> lu = S.partialPivLu();
    const MatrixXcd X = lu.solve(G);
    const MatrixXcd LamG = 0.5 * X + Kstar * X;
    const MatrixXcd C = -(kVb * kVb) / (2.0 * omega0) * (G.adjoint() * (sigma.asDiagonal() * LamG));

    // --- center-row real-space couplings C_l = C_{center, center+l} ----------------------
    std::ofstream out_c("patch_capacitance_couplings.csv");
    out_c.setf(std::ios::scientific);
    out_c.precision(10);
    std::cout << "  center-row coupling decay |C_l|_F:\n";
    for (int q = 0; q < n_def; ++q) {
        const int ell = q - center;
        const MatrixXcd blk = C.block(modes * center, modes * q, modes, modes);
        out_c << ell << "," << blk.norm();
        for (int a = 0; a < modes; ++a)
            for (int b = 0; b < modes; ++b)
                out_c << "," << blk(a, b).real() << "," << blk(a, b).imag();
        out_c << "\n";
        if (ell >= 0)
            std::cout << "    l=" << ell << "  |C_l|=" << blk.norm() << "\n";
    }
    out_c.close();

    // --- finite defect resonances: eigenvalues of the full C matrix ----------------------
    ComplexEigenSolver<MatrixXcd> es(C);
    VectorXcd lam = es.eigenvalues();
    std::sort(lam.data(), lam.data() + lam.size(),
              [](const cpxd& x, const cpxd& y) { return x.real() < y.real(); });
    std::ofstream out_r("patch_defect_resonances.csv");
    out_r.setf(std::ios::scientific);
    out_r.precision(10);
    for (int j = 0; j < lam.size(); ++j) {
        const cpxd omega = omega0 + delta * lam(j);  // finite analog of omega = omega_0 + delta*lambda
        out_r << lam(j).real() << "," << lam(j).imag() << "," << omega.real() << ","
              << omega.imag() << "\n";
    }
    out_r.close();

    std::cout << "[wrote] patch_capacitance_couplings.csv (center-row C_l), "
                 "patch_defect_resonances.csv (eig(C) -> omega=omega_0+delta*lambda)\n";
}

}  // namespace workflows
