#include "workflows/defect_capacitance_bands.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>

#include "Eigen/Dense"
#include "boundary_mesh.h"
#include "spectral_operators.h"

namespace workflows {

using namespace Eigen;

namespace {

// First positive zero of J_m' = interior Neumann eigenvalue index (radial order n=1).
double first_neumann_zero(int m) {
    switch (m) {
        case 0: return 3.8317059702;  // breathing (= j_{1,1})
        case 1: return 1.8411837813;  // dipole (doubly degenerate)
        case 2: return 3.0542369282;  // quadrupole
        case 3: return 4.2011889412;  // octupole
        default: return 1.8411837813;
    }
}

cpxd cyl_j_prime(int n, cpxd z) {
    return (n == 0) ? -bessel::cyl_j(1, z)
                    : 0.5 * (bessel::cyl_j(n - 1, z) - bessel::cyl_j(n + 1, z));
}

}  // namespace

void run_defect_capacitance_bands(double radius, double defect_radius, double delta,
                                  int max_m, int n_rows, int points_per_disk, int num_alpha,
                                  double alpha_lo_frac, double alpha_hi_frac,
                                  double v, [[maybe_unused]] double v_b, double v_bd) {
    // v_b (crystal interior speed) does not enter the leading-order capacitance: the crystal
    // disks are off-resonant at omega_0 and contribute only through the exterior operators.
    // --- line-defect supercell, periodic in x (period 1) ---------------------------------
    // A column of disks at y = -K..K (K = n_rows). The n=0 disk is the defect (radius
    // defect_radius); all others are crystal disks (radius). x-quasi-periodicity replicates
    // the column into a (2K+1)-row crystal slab carrying a single defect ROW at y=0 -- the
    // line defect. As K grows the slab -> the infinite crystal and the defect mode becomes
    // exactly bound (its radiative width Im(omega) -> 0).
    const int N = points_per_disk;
    const int nd = 2 * n_rows + 1;        // disks in the supercell
    const int defect_disk = n_rows;       // index of the y=0 (defect) disk

    std::vector<double> disk_radius(nd);
    std::vector<Vector2d> disk_center(nd);
    for (int n = -n_rows; n <= n_rows; ++n) {
        disk_radius[n + n_rows] = (n == 0) ? defect_radius : radius;
        disk_center[n + n_rows] = Vector2d(0.0, n);
    }
    BoundaryMesh mesh(N);
    mesh.generate_circle(disk_radius[0], disk_center[0]);
    for (int i = 1; i < nd; ++i) {
        BoundaryMesh d(N);
        d.generate_circle(disk_radius[i], disk_center[i]);
        mesh.add_mesh(d);
    }
    SpectralOperators ops(mesh);
    const int Ntot = mesh.get_num_segments();

    VectorXd sigma(Ntot);
    for (int i = 0; i < Ntot; ++i) sigma(i) = mesh.get_vertex(i).sigma;

    std::cout << "Capacitance-matrix line-defect bands (BIE / DtN, supercell): R=" << radius
              << ", R_def=" << defect_radius << ", m=0.." << max_m << ", delta=" << delta
              << ", " << nd << " disks (" << n_rows << " crystal rows each side)\n";

    // One family per angular order m = 0..max_m, each with its own interior Neumann base
    // frequency omega_0 = v_b j'_{m,1}/R_def and eigenmode traces g_p (paper eq. (4.8),
    // Thm 4.3): the capacitance matrix is modes x modes with modes = dim of the interior
    // Neumann eigenspace at omega_0 (1 for m = 0, 2 for m >= 1). At omega_0 ONLY the defect
    // disk is at a Neumann eigenvalue, so g_p ~ A e^{+/- i m theta} on the DEFECT disk, zero
    // on the crystal disks; the crystal enters only through the exterior operators.
    // L^2(D)-normalization: A = 1/sqrt(pi R_def^2 (1 - m^2/beta^2)), beta = j'_{m,1}.
    struct ModeFamily {
        int m;          // angular order
        double omega0;  // interior Neumann base frequency of the defect disk
        int modes;      // multiplicity (size of C^reg)
        MatrixXcd G;    // eigenmode traces on the supercell boundary
    };
    std::vector<ModeFamily> families;
    for (int m_ang = 0; m_ang <= max_m; ++m_ang) {
        const double beta = first_neumann_zero(m_ang);
        ModeFamily fam{m_ang, v_bd * beta / defect_radius, (m_ang == 0) ? 1 : 2,
                       MatrixXcd::Zero(Ntot, (m_ang == 0) ? 1 : 2)};
        const double Anorm =
            1.0 / (std::sqrt(M_PI) * defect_radius
                   * std::sqrt(std::max(1.0 - double(m_ang * m_ang) / (beta * beta), 1e-12)));
        const int s0 = mesh.get_start_index(defect_disk);
        for (int loc = 0; loc < N; ++loc)
            for (int sgn = 0; sgn < fam.modes; ++sgn)
                fam.G(s0 + loc, sgn) = Anorm * std::exp(cpxd(0, 1.0) * ((sgn == 0) ? 1.0 : -1.0)
                                                        * double(m_ang) * 2. * M_PI / double(N) * double(loc));
        families.push_back(std::move(fam));
        std::cout << "   m=" << m_ang << ": omega_0=" << families.back().omega0
                  << " (= v_bd * " << beta << "/R_def), multiplicity " << families.back().modes << "\n";
    }


    MatrixXcd S, Kstar;  // workspace shared by all assemblies

    // std::ofstream out("defect_capacitance_bands.csv");
    std::ofstream out_lo("defect_capacitance_bands_leading.csv");
    std::ofstream out_cap("capacitance_matrix_per_alpha.csv");
    out_cap.setf(std::ios::scientific);
    out_cap.precision(10);
    // out.setf(std::ios::scientific);
    // out.precision(10);
    out_lo.setf(std::ios::scientific);
    out_lo.precision(10);

    // Continuation seeds from the previous alpha, one root set per mode family.
    std::vector<std::vector<cpxd>> prev_roots(families.size());
    for (int a = 0; a < num_alpha; ++a) {
        const double frac = (num_alpha > 1) ? static_cast<double>(a) / (num_alpha - 1) : 0.0;
        double alpha = 2. * M_PI * (alpha_lo_frac + (alpha_hi_frac - alpha_lo_frac) * frac);
        if (std::abs(alpha) < 1e-9) alpha = 1e-6;
        std::cout << "  alpha_x=" << alpha << "\n";

        for (std::size_t fi = 0; fi < families.size(); ++fi) {
            const ModeFamily& fam = families[fi];
            const double omega0 = fam.omega0;
            const int modes = fam.modes;
            const double W = 10.0 * delta;  // family window half-width (as line-defect-neumann)
            const cpxd k = cpxd(omega0, 1e-3) / v;  // small Im(k) regularises the periodic G^alpha

            // Frequency-dependent capacitance matrix (Fabry-Perot, arXiv:2605.27572 eq. 4.11):
            //   Lambda_ext = (1/2 I + K*) S^{-1},   k = omega_0 / v
            //   C^reg_{pq} = -(v_b^2 / 2 omega_0) <Lambda_ext[g_q], g_p>_{dD}
            // The boundary inner product <.,.>_{dD} is the quadrature sum (diag(sigma)).
            ops.PeriodicS(S, k, alpha, 1.0);
            ops.PeriodicKstar(Kstar, k, alpha, 1.0);
            const MatrixXcd X = S.partialPivLu().solve(fam.G);
            const MatrixXcd LamG = 0.5 * X + Kstar * X;   // Lambda_ext G = (1/2 + K*) S^{-1} G
            const MatrixXcd Creg =
                -(v_bd * v_bd) / (2.0 * omega0) * (fam.G.adjoint() * (sigma.asDiagonal() * LamG));

            // complex symmetry - should be symmetric
            const double herm = (Creg - Creg.adjoint()).norm();

            ComplexEigenSolver<MatrixXcd> es(Creg);
            VectorXcd lam = es.eigenvalues();
            std::sort(lam.data(), lam.data() + lam.size(),
                      [](const cpxd& x, const cpxd& y) { return x.real() < y.real(); });

            for (int j = 0; j < modes; ++j) {
                const cpxd omega = omega0 + delta * lam(j);  // leading order: Thm 4.3, eq (4.12)
                out_lo << fam.m << "," << alpha << "," << omega.real() << "," << omega.imag()
                       << "," << lam(j).real() << "," << lam(j).imag() << "," << herm << "\n";
                cout << "    m=" << fam.m << " leading-order seed: omega=" << omega
                          << " lambda=" << lam(j) << " herm=" << herm << "\n";
            }

            // header
            out_cap << "alpha,fi";
            for (int i = 0; i < modes; ++i) {
                for (int j = 0; j < modes; ++j) {
                    out_cap << ",C_" << i << "_" << j << "_re";
                    out_cap << ",C_" << i << "_" << j << "_im";
                }
            }
            out_cap << "\n";

            // inside alpha loop
            out_cap << alpha << "," << fi;
            for (int i = 0; i < modes; ++i) {
                for (int j = 0; j < modes; ++j) {
                    auto z = Creg(i, j);
                    out_cap << "," << z.real() << "," << z.imag();
                }
            }
            out_cap << "\n";
        }
    }
    // out.close();
    out_lo.close();
    out_cap.close();
    // std::cout << "[wrote] defect_capacitance_bands.csv (exact transmission roots), "
    //              "defect_capacitance_bands_leading.csv (leading-order seeds)\n";
    std::cout << "[wrote] defect_capacitance_bands_leading.csv (leading-order seeds), "
                 "capacitance_matrix_per_alpha.csv (C^reg per alpha)\n";
}

}  // namespace workflows
