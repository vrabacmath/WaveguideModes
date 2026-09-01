#include "workflows/crystal_matrix_bands.h"

namespace workflows {

using namespace Eigen;

std::vector<KPoint> make_m_gamma_x_m_path(int points_per_segment) {
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

    double s = 0.0;
    append_segment(s, Vector2d(M_PI, M_PI), Vector2d(0.0, 0.0));   // M -> Gamma
    append_segment(s, Vector2d(0.0, 0.0), Vector2d(M_PI, 0.0));    // Gamma -> X
    append_segment(s, Vector2d(M_PI, 0.0), Vector2d(M_PI, M_PI));  // X -> M
    return path;
}

void run_crystal_matrix_bands(double radius,
                              double delta,
                              int points_per_disk,
                              int n_multipole,
                              int points_per_segment,
                              double omega_lo,
                              double omega_hi,
                              int n_omega,
                              double omega_imag,
                              double v,
                              double v_b) {
    points_per_disk = std::max(points_per_disk, 8);
    n_multipole = std::max(n_multipole, 1);
    n_omega = std::max(n_omega, 3);
    if (!(omega_lo < omega_hi)) {
        std::swap(omega_lo, omega_hi);
    }

    const std::vector<KPoint> path = make_m_gamma_x_m_path(points_per_segment);
    const Vector2d a1(1.0, 0.0), a2(0.0, 1.0);

    BoundaryMesh mesh(points_per_disk);
    mesh.generate_circle(radius);
    SpectralOperators ops(mesh);

    std::cout << "Crystal matrix band comparison (M-Gamma-X-M): R=" << radius
              << ", delta=" << delta << ", points/disk=" << points_per_disk
              << ", N_multipole=" << n_multipole << ", omega=[" << omega_lo << ", "
              << omega_hi << "], Im(omega)=" << omega_imag << "\n";

    auto crystalA_measure = [&](double omega, double alpha_x, double alpha_y) {
        MatrixXcd A;
        const cpxd omega_c(omega, omega_imag);
        ops.CrystalA(A, omega_c / v, omega_c / v_b, Vector2d(alpha_x, alpha_y), a1, a2, delta);
        return measure_matrix(A);
    };

    auto multipoleA_measure = [&](double omega, double alpha_x, double alpha_y) {
        MatrixXcd A;
        const cpxd omega_c(omega, omega_imag);
        Utils::multipole_crystal_A(A, n_multipole, radius, omega_c / v, omega_c / v_b,
                                   Vector2d(alpha_x, alpha_y), delta);
        return measure_matrix(A);
    };

    run_one_sweep("CrystalA", "crystalA_m_gamma_x_m.csv", path, omega_lo, omega_hi, n_omega,
                  crystalA_measure);
    run_one_sweep("multipoleA", "multipoleA_m_gamma_x_m.csv", path, omega_lo, omega_hi, n_omega,
                  multipoleA_measure);
}

void run_projected_bulk_bands(double radius, double delta, int n_multipole,
                              int num_alpha_x, int n_alpha_y,
                              double omega_lo, double omega_hi, int n_omega,
                              double omega_imag, double v, double v_b) {
    num_alpha_x = std::max(num_alpha_x, 2);
    n_alpha_y = std::max(n_alpha_y, 3);
    n_omega = std::max(n_omega, 3);
    if (!(omega_lo < omega_hi)) std::swap(omega_lo, omega_hi);

    std::cout << "Projected bulk spectrum (min over alpha_y of sigma_min(multipole crystal A)): R="
              << radius << ", delta=" << delta << ", N_multipole=" << n_multipole << ", "
              << num_alpha_x << " x " << n_omega << " grid, " << n_alpha_y << " alpha_y samples\n";

    std::ofstream out("projected_bulk_bands.csv");
    out.setf(std::ios::scientific);
    out.precision(10);

    for (int ia = 0; ia < num_alpha_x; ++ia) {
        double alpha_x = M_PI * ia / (num_alpha_x - 1);
        if (std::abs(alpha_x) < 1e-9) alpha_x = 1e-6;
        for (int io = 0; io < n_omega; ++io) {
            const double omega = omega_lo + (omega_hi - omega_lo) * io / (n_omega - 1);
            const cpxd omega_c(omega, omega_imag);
            std::vector<double> vals(n_alpha_y);
            for (int j = 0; j < n_alpha_y; ++j) {
                const double alpha_y = M_PI * j / (n_alpha_y - 1);
                MatrixXcd A;
                Utils::multipole_crystal_A(A, n_multipole, radius, omega_c / v, omega_c / v_b,
                                           Vector2d(alpha_x, alpha_y), delta);
                vals[j] = JacobiSVD<MatrixXcd>(A).singularValues().tail<1>()(0);
            }
            std::sort(vals.begin(), vals.end());
            out << alpha_x << "," << omega << "," << vals.front() << ","
                << vals[n_alpha_y / 2] << "\n";
        }
        std::cout << "  alpha_x " << (ia + 1) << "/" << num_alpha_x << " done\n";
    }
    out.close();
    std::cout << "[wrote] projected_bulk_bands.csv (alpha_x, omega, min_sigma, median_sigma)\n";
}

}  // namespace workflows
