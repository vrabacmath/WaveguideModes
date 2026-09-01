#include "hex_crystal.h"
#include "utils.h"

using namespace Eigen;
using namespace std;

std::vector<KPoint> Crystal::make_m_gamma_k_m_path(int points_per_segment) {
    points_per_segment = std::max(points_per_segment, 2);
    std::vector<KPoint> path;
    path.reserve(3 * (points_per_segment - 1) + 1);

    auto append_segment = [&](double& s0, Vector2d start, Vector2d end) {
        const double length = (end - start).norm();
        for (int i = 0; i < points_per_segment; ++i) {
            if (!path.empty() && i == 0) continue;
            const double t = static_cast<double>(i) / static_cast<double>(points_per_segment - 1);
            const Vector2d alpha = (1.0 - t) * start + t * end;
            path.push_back({s0 + t * length, alpha.x(), alpha.y()});
        }
        s0 += length;
    };

    // hexagonal Brillouin zone: M is an edge midpoint and K is an adjacent corner.
    double s = 0.0;
    append_segment(s, 0.5 * b1, Vector2d(0.0, 0.0));   // M -> Gamma
    append_segment(s, Vector2d(0.0, 0.0), (2.0 * b1 + b2) / 3.0);  // Gamma -> K
    append_segment(s, (2.0 * b1 + b2) / 3.0, 0.5 * b1);  // K -> M
    return path;
}


void Crystal::compute_high_symmetry_bands(double omega_lo, double omega_hi, int n_omega, double omega_imag,
                                            double v, double v_b, const std::string& filename, bool crystalA) {
    n_omega = std::max(n_omega, 3);
    if (!(omega_lo < omega_hi)) std::swap(omega_lo, omega_hi);

    BoundaryMesh mesh = get_mesh();
    double delta = get_delta();

    std::cout << "Crystal bands: R=" << mesh.get_num_meshes() << ", delta=" << delta
                << ", " << n_omega << " omega samples in [" << omega_lo << "," << omega_hi
                << "]\n";

    int n_multipole = 7;
    assert(circles_mesh && "multipole crystal A only works for circular disks!");

    if (crystalA) {
        SpectralOperators ops(mesh);
        run_one_sweep("CrystalA", filename, make_m_gamma_k_m_path(20), omega_lo, omega_hi, n_omega,
                        [&](double omega, double alpha_x, double alpha_y) {
                            MatrixXcd A;
                            const cpxd omega_c(omega, omega_imag);
                            ops.CrystalA(A, omega_c / v, omega_c / v_b, Vector2d(alpha_x, alpha_y),
                                        a1, a2, delta);
                            return measure_matrix(A);
                        });
    }

    run_one_sweep("multipoleA", "multipoleA_m_gamma_k_m_hex.csv", make_m_gamma_k_m_path(20), omega_lo, omega_hi, n_omega,
                    [&](double omega, double alpha_x, double alpha_y) {
                    MatrixXcd A;
                    const cpxd omega_c(omega, omega_imag);
                    Utils::multipole_crystal_A(A, n_multipole, Rs, shifts, omega_c / v, omega_c / v_b,
                                            Vector2d(alpha_x, alpha_y), a1, a2, delta);

                    Vector3d last_3_singular_values = last_3_sv(A);
                    Vector2d last_2_singular_values = last_3_singular_values.tail<2>();
                    if (last_2_singular_values.isZero(0.1 * last_3_singular_values(0))) {
                        std::cout << "There could be a Dirac point at omega=" << omega
                                    << ", alpha=(" << alpha_x << "," << alpha_y << ")\n";
                        std::cout << "  last 3 singular values: " << last_3_singular_values.transpose() << "\n";
                    }
                    return measure_matrix(A);
                });

    std::cout << "[wrote] " << filename
                << " (omega,sigma_min(log10),log_abs_det)\n";
}
