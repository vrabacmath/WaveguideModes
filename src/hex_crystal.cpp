#include "hex_crystal.h"
#include "utils.h"
#include <stdexcept>

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
    cout << "Reciprocal lattice vectors: " << b1.transpose() << ", " << b2.transpose() << "\n";
    cout << "Gamma point: " << Vector2d(0.0, 0.0).transpose() << "\n";
    cout << "M point: " << (0.5 * b1).transpose() / M_PI << "pi\n";
    cout << "K point: " << ((2.0 * b1 + b2) / 3.0).transpose() / M_PI << "pi\n";

    append_segment(s, 0.5 * b1, Vector2d(0.0, 0.0));   // M -> Gamma
    append_segment(s, Vector2d(0.0, 0.0), (2.0 * b1 + b2) / 3.0);  // Gamma -> K
    append_segment(s, (2.0 * b1 + b2) / 3.0, 0.5 * b1);  // K -> M
    return path;
}


void Crystal::compute_high_symmetry_bands(double omega_lo, double omega_hi, int n_omega, double omega_imag,
                                        const std::string& filename, int Npath, bool crystalA) {
    n_omega = std::max(n_omega, 3);
    if (!(omega_lo < omega_hi)) std::swap(omega_lo, omega_hi);

    BoundaryMesh mesh = get_mesh();
    double delta = get_delta();

    std::cout << "Crystal bands: R=" << mesh.get_num_meshes() << ", delta=" << delta
                << ", " << n_omega << " omega samples in [" << omega_lo << "," << omega_hi
                << "]\n";

    // int n_multipole = 7;
    double Rs_max = 0.0;
    for (const auto& R : Rs) {
        Rs_max = std::max(Rs_max, R);
    }

    double v_lo = std::min(v, v_b);
    int n_multipole = std::max(7, static_cast<int>(std::ceil(omega_hi * Rs_max / v_lo)) + 5);
    assert(circles_mesh && "multipole crystal A only works for circular disks!");
    std::cout << "Using " << n_multipole << " multipole terms.\n";

    if (crystalA) {
        SpectralOperators ops(mesh);
        run_one_sweep("CrystalA", filename, make_m_gamma_k_m_path(Npath), omega_lo, omega_hi, n_omega,
                        [&](double omega, double alpha_x, double alpha_y) {
                            MatrixXcd A;
                            const cpxd omega_c(omega, omega_imag);
                            ops.CrystalA(A, omega_c / v, omega_c / v_b, Vector2d(alpha_x, alpha_y),
                                        a1, a2, delta);
                            return measure_matrix(A);
                        });
    }

    vector<Vector3d> candidates;
    bool collect_candidates = true;
    auto meas_mult_A = [&](double omega, double alpha_x, double alpha_y) {
        MatrixXcd A;
        const cpxd omega_c(omega, omega_imag);
        Utils::multipole_crystal_A(A, n_multipole, Rs, shifts, omega_c / v, omega_c / v_b,
                                  Vector2d(alpha_x, alpha_y), a1, a2, delta);

        if (collect_candidates) {
            Vector3d last_3_singular_values = last_3_sv(A);
            Vector2d last_2_singular_values = last_3_singular_values.tail<2>();
            if (last_2_singular_values.isZero(0.1 * last_3_singular_values(0)) && abs(last_2_singular_values(0) - last_2_singular_values(1)) < 1e-7) {
                std::cout << "There could be a Dirac point at omega=" << omega
                          << ", alpha=(" << alpha_x << "," << alpha_y << ")\n";
                std::cout << "  last 3 singular values: " << last_3_singular_values.transpose() << "\n";
                if (candidates.empty() || (Vector3d(alpha_x, alpha_y, omega) - candidates.back()).norm() > 1e-1) {
                    candidates.push_back(Vector3d(alpha_x, alpha_y, omega));
                }
            }
        }
        return measure_matrix(A);
    };

    run_one_sweep("multipoleA", "multipoleA_m_gamma_k_m_hex.csv", make_m_gamma_k_m_path(Npath), omega_lo, omega_hi, n_omega,
                    meas_mult_A);

    // The local scan must not modify the candidate vector being iterated.
    collect_candidates = false;
    const double radius = 0.15;
    const int Nalpha = 21; // Odd count includes the candidate itself.
    for (size_t candidate_i = 0; candidate_i < candidates.size(); ++candidate_i) {
        const Vector3d alpha = candidates[candidate_i];
        std::cout << "Candidate Dirac point at omega=" << alpha.z()
                  << ", alpha=(" << alpha.x() << "," << alpha.y() << ")\n";
        const std::string candidate_filename = "multipoleA_dirac_candidate_omega_"
            + std::to_string(alpha.z()) + "_" + std::to_string(candidate_i) + ".csv";
        std::ofstream out(candidate_filename);
        if (!out) throw std::runtime_error("Cannot open " + candidate_filename);
        out.precision(std::numeric_limits<double>::max_digits10);
        out << "alpha_x,alpha_y,omega,sigma_min\n";

        for (int alpha_i = 0; alpha_i < Nalpha; ++alpha_i) {
            double alpha_x = alpha.x() - radius + 2.0 * radius * alpha_i / (Nalpha - 1.0);
            for (int alpha_j = 0; alpha_j < Nalpha; ++alpha_j) {
                double alpha_y = alpha.y() - radius + 2.0 * radius * alpha_j / (Nalpha - 1.0);
                for (int omega_i = 0; omega_i < n_omega; ++omega_i) {
                    double omega = omega_lo + (omega_hi - omega_lo) * omega_i / (n_omega - 1.0);
                    MatrixMeasure measure = meas_mult_A(omega, alpha_x, alpha_y);
                    out << alpha_x << "," << alpha_y << "," << omega << "," << measure.sigma_min << "\n";
                }
            }
        }
        cout << "Wrote Dirac candidate data to " << candidate_filename << "\n";
    }

    std::cout << "[wrote] " << filename
                << " (omega,sigma_min(log10),log_abs_det)\n";
}

void Crystal::compute_local_bands(double omega_lo, double omega_hi, int n_omega,
                                  double omega_imag, const std::string& filename,
                                  Vector2d center, double radius, int n_alpha) {
    if (!(omega_lo > 0 && omega_hi > omega_lo && n_omega >= 3 && n_alpha >= 3
          && radius > 0 && omega_imag >= 0 && std::isfinite(omega_hi)
          && std::isfinite(radius) && std::isfinite(omega_imag) && center.allFinite())) {
        throw std::invalid_argument("Local scan needs 0 < omega_lo < omega_hi, at least 3 samples per axis, "
                                    "positive finite radius and nonnegative finite omega_imag");
    }
    if (!circles_mesh || Rs.empty() || !(v > 0 && v_b > 0)) {
        throw std::invalid_argument("Local multipole scan needs circular resonators and positive speeds");
    }
    const double max_radius = *std::max_element(Rs.begin(), Rs.end());
    const int n_multipole = std::max(7, static_cast<int>(std::ceil(omega_hi * max_radius / std::min(v, v_b))) + 5);
    std::ofstream out(filename);
    if (!out) throw std::runtime_error("Cannot open " + filename);
    out.precision(std::numeric_limits<double>::max_digits10);
    out << "alpha_x,alpha_y,omega,sigma_min,sigma_2,sigma_3\n";
    std::cout << "Local scan: center=(" << center.transpose() << "), radius=" << radius
              << ", " << n_alpha << " x " << n_alpha << " Bloch points, " << n_omega
              << " frequencies, step=" << (omega_hi - omega_lo) / (n_omega - 1)
              << ", Im(omega)=" << omega_imag << ", multipole order=" << n_multipole << "\n";

    // Each point is independent. Keep CSV writes ordered and outside the parallel loop.
    for (int ix = 0; ix < n_alpha; ++ix) {
        const double x = center.x() - radius + 2 * radius * ix / (n_alpha - 1);
        std::vector<Vector3d> values(static_cast<size_t>(n_alpha) * n_omega);
        #pragma omp parallel for schedule(static)
        for (int iy = 0; iy < n_alpha; ++iy) {
            const double y = center.y() - radius + 2 * radius * iy / (n_alpha - 1);
            for (int iw = 0; iw < n_omega; ++iw) {
                const double w = omega_lo + (omega_hi - omega_lo) * iw / (n_omega - 1);
                const cpxd wc(w, omega_imag);
                MatrixXcd A;
                Utils::multipole_crystal_A(A, n_multipole, Rs, shifts, wc / v, wc / v_b,
                                          Vector2d(x, y), a1, a2, get_delta());
                values[static_cast<size_t>(iy) * n_omega + iw] = last_3_sv(A);
            }
        }
        for (int iy = 0; iy < n_alpha; ++iy) {
            const double y = center.y() - radius + 2 * radius * iy / (n_alpha - 1);
            for (int iw = 0; iw < n_omega; ++iw) {
                const double w = omega_lo + (omega_hi - omega_lo) * iw / (n_omega - 1);
                const auto& sv = values[static_cast<size_t>(iy) * n_omega + iw];
                out << x << ',' << y << ',' << w << ',' << sv(2) << ',' << sv(1) << ',' << sv(0) << '\n';
            }
        }
        std::cout << "  Bloch row " << ix + 1 << '/' << n_alpha << std::endl;
    }
    if (!out) throw std::runtime_error("Failed writing " + filename);
    std::cout << "[wrote] " << filename << "\n";
}
