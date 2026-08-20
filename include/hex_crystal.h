#ifndef HEX_CRYSTAL_H
#define HEX_CRYSTAL_H

#include <string>
#include <utility>
#include "workflows/crystal_matrix_bands.h"

#include <Eigen/Dense>

#include "boundary_mesh.h"


struct KPoint {
    double s;
    double alpha_x;
    double alpha_y;
};

struct MatrixMeasure {
    double sigma_min;
    double log_abs_det;
};

inline MatrixMeasure measure_matrix(const MatrixXcd& A) {
    const VectorXd singular_values = JacobiSVD<MatrixXcd>(A).singularValues();
    double log_abs_det = 0.0;
    for (int i = 0; i < singular_values.size(); ++i) {
        log_abs_det += std::log(std::max(singular_values(i), std::numeric_limits<double>::min()));
    }
    return {singular_values.tail<1>()(0), log_abs_det};
}

template <typename Measure>
void run_one_sweep(const std::string& label,
                   const std::string& filename,
                   const std::vector<KPoint>& path,
                   double omega_lo,
                   double omega_hi,
                   int n_omega,
                   const Measure& measure_at) {
    std::ofstream out(filename);
    out.setf(std::ios::scientific);
    out.precision(10);

    std::cout << "  " << label << ": " << path.size() << " k-points, " << n_omega
              << " omega samples\n";

    for (int p = 0; p < static_cast<int>(path.size()); ++p) {
        const KPoint& kp = path[p];
        for (int i = 0; i < n_omega; ++i) {
            const double omega = omega_lo + (omega_hi - omega_lo) * i / static_cast<double>(n_omega - 1);
            const MatrixMeasure measure = measure_at(omega, kp.alpha_x, kp.alpha_y);
            out << kp.s << "," << omega << "," << measure.sigma_min << "," << measure.log_abs_det
                << "," << kp.alpha_x << "," << kp.alpha_y << "\n";
        }

        std::cout << "    " << label << " k-point " << (p + 1) << "/" << path.size()
                  << "  s=" << kp.s << "\n";
    }

    std::cout << "  [wrote] " << filename << "\n";
}


class Crystal {
public:
    Crystal(BoundaryMesh mesh, double delta, Eigen::Vector2d a1, Eigen::Vector2d a2)
        : mesh(mesh), delta(delta), a1(a1), a2(a2) {
        // It would be good to check that a1 and a2 create a lattice compatible with the mesh.

        MatrixXd A(2, 2);
        A.col(0) = a1;
        A.col(1) = a2;
        MatrixXd B = 2 * M_PI * A.inverse().transpose().eval();
        b1 = B.col(0);
        b2 = B.col(1);
    }

    void compute_high_symmetry_bands(double omega_lo, double omega_hi, int n_omega,
                                     double omega_imag, double v, double v_b,
                                     const std::string& filename);

    BoundaryMesh get_mesh() const { return mesh; }
    std::pair<Eigen::Vector2d, Eigen::Vector2d> get_lattice_vectors() const { return {a1, a2}; }
    std::pair<Eigen::Vector2d, Eigen::Vector2d> get_reciprocal_vectors() const { return {b1, b2}; }
    double get_delta() const { return delta; }

private:
    BoundaryMesh mesh;
    double delta;
    Eigen::Vector2d a1;
    Eigen::Vector2d a2;
    Eigen::Vector2d b1;
    Eigen::Vector2d b2;

    std::vector<KPoint> make_m_gamma_x_m_path(int points_per_segment);
};

#endif // HEX_CRYSTAL_H
