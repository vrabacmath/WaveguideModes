#include "workflows/bent_waveguide.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "Eigen/Dense"
#include "utils.h"
#include "workflows/bent_waveguide_patch.h"

namespace workflows {

using namespace Eigen;

void run_bent_waveguide(double radius, double defect_radius, double delta,
                           int m_ang, int n_defect, int n_clad, int fringe, int points_per_disk) {
    const BentPatch p = build_bent_patch(radius, defect_radius, m_ang, n_defect, n_clad,
                                         fringe, points_per_disk);
    const int modes = p.modes;
    const int center = p.center;
    const int n_main = static_cast<int>(p.main_indices.size());

    Utils::draw_mesh(p.mesh, "bent_waveguide_mesh.csv");

    // --- corner-row real-space couplings C_l = C_{corner, site(l)} ----------------------
    const std::string filename = "bent_waveguide_couplings" + std::to_string(n_defect) + "_" +
                                 std::to_string(n_clad) + "_" + std::to_string(fringe) + ".csv";
    std::ofstream out_c(filename);
    out_c.setf(std::ios::scientific);
    // |C_0| converges with patch size extremely fast
    out_c.precision(std::numeric_limits<double>::max_digits10);
    std::cout << "  corner-row coupling decay |C_l|_F:\n";
    for (int q = 0; q < n_main; ++q) {
        const int ell = q - center;
        const MatrixXcd blk = p.C.block(modes * center, modes * q, modes, modes);
        out_c << ell << "," << blk.norm();
        for (int a = 0; a < modes; ++a)
            for (int b = 0; b < modes; ++b)
                out_c << "," << blk(a, b).real() << "," << blk(a, b).imag();
        out_c << "\n";
        std::cout << "    l=" << ell << "  |C_l|=" << blk.norm() << "\n";
    }
    out_c.close();

    // --- finite defect resonances: eigenvalues of the full C matrix ----------------------
    // Sort a permutation rather than the eigenvalue array in place: sorting `lam` directly would
    // silently break its correspondence with es.eigenvectors(), which run_bent_waveguide_field
    // relies on.
    ComplexEigenSolver<MatrixXcd> es(p.C);
    const VectorXcd lam = es.eigenvalues();
    std::vector<int> order(lam.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&lam](int a, int b) { return lam(a).real() < lam(b).real(); });

    std::ofstream out_r("bent_waveguide_resonances.csv");
    out_r.setf(std::ios::scientific);
    out_r.precision(std::numeric_limits<double>::max_digits10);
    for (int j = 0; j < static_cast<int>(order.size()); ++j) {
        const cpxd l = lam(order[j]);
        const cpxd omega = p.omega0 + delta * l;  // finite analog of omega = omega_0 + delta*lambda
        out_r << l.real() << "," << l.imag() << "," << omega.real() << "," << omega.imag() << "\n";
    }
    out_r.close();

    std::cout << "[wrote] " << filename << " (corner-row C_l), "
                 "bent_waveguide_resonances.csv (eig(C) -> omega=omega_0+delta*lambda)\n";
}

}  // namespace workflows
