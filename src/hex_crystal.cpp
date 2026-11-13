#include "hex_crystal.h"
#include "multipole.h"
#include <cassert>
#include <stdexcept>

using namespace Eigen;
using namespace std;

Crystal::Crystal(std::vector<Disk> disk_data, double contrast, Vector2d lattice_a1,
                 Vector2d lattice_a2, int point_count, double background_speed)
    : mesh(std::max(point_count, 8)), disks(std::move(disk_data)), points_per_disk(point_count),
      v(background_speed), delta(contrast), a1(lattice_a1), a2(lattice_a2) {
    if (disks.empty() || point_count < 8 || !std::isfinite(v) || v <= 0 ||
        !std::isfinite(delta) || delta < 0 || !a1.allFinite() || !a2.allFinite())
        throw std::invalid_argument("Crystal requires disks, >=8 boundary points, positive speed, "
                                    "nonnegative contrast and finite lattice vectors");
    Matrix2d A;
    A.col(0) = a1;
    A.col(1) = a2;
    if (std::abs(A.determinant()) <= std::numeric_limits<double>::epsilon() * a1.norm() * a2.norm())
        throw std::invalid_argument("Crystal lattice vectors must be independent");
    const Matrix2d B = 2 * M_PI * A.inverse().transpose();
    b1 = B.col(0);
    b2 = B.col(1);
    for (std::size_t i = 0; i < disks.size(); ++i) {
        const auto& disk = disks[i];
        if (!disk.c.allFinite() || !std::isfinite(disk.r) || disk.r <= 0 ||
            !std::isfinite(disk.v_b) || disk.v_b <= 0)
            throw std::invalid_argument("Every crystal disk needs a finite center and positive radius/speed");
        BoundaryMesh component(point_count);
        component.generate_circle(disk.r, disk.c);
        if (i == 0) mesh = component;
        else mesh.add_mesh(component);
    }
}

int Crystal::suggested_multipole_order(double omega_hi) const {
    if (!std::isfinite(omega_hi) || omega_hi <= 0)
        throw std::invalid_argument("Multipole cutoff requires a positive finite upper frequency");
    double max_argument = 0;
    for (const auto& disk : disks)
        max_argument = std::max(max_argument, omega_hi * disk.r / std::min(v, disk.v_b));
    return std::max(7, static_cast<int>(std::ceil(max_argument)) + 5);
}

MatrixXcd Crystal::multipole_matrix(cpxd omega, Vector2d alpha, int order) const {
    std::vector<cpxd> interior_wavenumbers;
    for (const auto& disk : disks) interior_wavenumbers.push_back(omega / disk.v_b);
    MatrixXcd A;
    Multipole::crystal_A(A, order, disks, omega / v, interior_wavenumbers, alpha, a1, a2, delta);
    return A;
}

MatrixXcd Crystal::boundary_matrix(cpxd omega, Vector2d alpha) const {
    const int n = mesh.get_num_segments();
    SpectralOperators ops(mesh);
    MatrixXcd S, K;
    ops.CrystalS(S, omega / v, alpha, a1, a2);
    ops.CrystalKstar(K, omega / v, alpha, a1, a2);
    MatrixXcd A = MatrixXcd::Zero(2 * n, 2 * n);
    A.topRightCorner(n, n) = -S;
    A.bottomRightCorner(n, n) = -delta * (0.5 * MatrixXcd::Identity(n, n) + K);
    for (std::size_t i = 0; i < disks.size(); ++i) {
        BoundaryMesh component(points_per_disk);
        component.generate_circle(disks[i].r, disks[i].c);
        SpectralOperators interior(component);
        interior.S(S, omega / disks[i].v_b);
        interior.Kstar(K, omega / disks[i].v_b);
        const int start = mesh.get_start_index(i);
        A.block(start, start, points_per_disk, points_per_disk) = S;
        A.block(n + start, start, points_per_disk, points_per_disk) =
            -0.5 * MatrixXcd::Identity(points_per_disk, points_per_disk) + K;
    }
    return A;
}

std::vector<KPoint> Crystal::make_m_gamma_k_m_path(int points_per_segment) const {
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
                                        int mode_index, double radius, int Npath, bool crystalA, bool scan_dirac_candidates) {
    n_omega = std::max(n_omega, 3);
    if (!(omega_lo < omega_hi)) std::swap(omega_lo, omega_hi);

    BoundaryMesh mesh = get_mesh();
    double delta = get_delta();

    std::cout << "Crystal bands: R=" << mesh.get_num_meshes() << ", delta=" << delta
                << ", " << n_omega << " omega samples in [" << omega_lo << "," << omega_hi
                << "]\n";

    const int n_multipole = suggested_multipole_order(omega_hi);
    std::cout << "Using " << n_multipole << " multipole terms.\n";

    std::string filename = "_m_gamma_k_m_hex" + std::to_string(mode_index) + ".csv";

    if (crystalA) {
        run_one_sweep("CrystalA", "crystalA" + filename, make_m_gamma_k_m_path(Npath), omega_lo, omega_hi, n_omega,
                        [&](double omega, double alpha_x, double alpha_y) {
                            return measure_matrix(boundary_matrix(cpxd(omega, omega_imag),
                                                                  Vector2d(alpha_x, alpha_y)));
                        });
    }

    vector<Vector3d> candidates;
    bool collect_candidates = scan_dirac_candidates;
    auto meas_mult_A = [&](double omega, double alpha_x, double alpha_y) {
        const MatrixXcd A = multipole_matrix(cpxd(omega, omega_imag),
                                             Vector2d(alpha_x, alpha_y), n_multipole);

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

    run_one_sweep("multipoleA", "multipoleA" + filename, make_m_gamma_k_m_path(Npath), omega_lo, omega_hi, n_omega,
                    meas_mult_A);

    // The local scan must not modify the candidate vector being iterated.
    collect_candidates = false;
    // const double radius = 0.15 * pow(10., -(mode_index % 10));
    const int Nalpha = 21; // Odd count includes the candidate itself.
    for (size_t candidate_i = 0; candidate_i < candidates.size(); ++candidate_i) {
        const Vector3d alpha = candidates[candidate_i];
        std::cout << "Candidate Dirac point at omega=" << alpha.z()
                  << ", alpha=(" << alpha.x() << "," << alpha.y() << ")\n";
        if (abs(alpha.x() - 4.1888) > 1e-3 || abs(alpha.y() - 0.0) > 1e-3) {
            std::cout << "  Skipping candidate that is not at the expected K point.\n";
            continue;
        }
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
        break;
    }

    std::cout << "[wrote] multipoleA" << filename
              << " (s,omega,sigma_min,log_abs_det,alpha_x,alpha_y)\n";
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
    const int n_multipole = suggested_multipole_order(omega_hi);
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
                const MatrixXcd A = multipole_matrix(wc, Vector2d(x, y), n_multipole);
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
