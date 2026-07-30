#include "multipole.h"

Multipole::Multipole(int N_multipole, int N_gauss, double multipole_radius, double point_defect_radius, double line_defect_radius, cpxd v, cpxd v_b, cpxd v_bd, double delta)
    : N_multipole(N_multipole), N_gauss(N_gauss), multipole_radius(multipole_radius), point_defect_radius(point_defect_radius),
      line_defect_radius(line_defect_radius), v(v), v_b(v_b), v_bd(v_bd), delta(delta) {
    this->gauss_nodes = LegendreMatrix::getPositions(this->N_gauss);
    this->gauss_weights = LegendreMatrix::getWeights(this->N_gauss);
}

MatrixXcd Multipole::crystal_M_operator(cpxd omega) {
    const int N = this->N_multipole;
    const int mode_count = 2 * N + 1;
    // M^epsilon acts on source pairs (f, g), not on a single multipole vector.
    const int density_count = 2 * mode_count;
    MatrixXcd integrated_resolvent = MatrixXcd::Zero(density_count, density_count);
    MatrixXcd crystal_A, A_D_map, A_defect_map;
    const MatrixXcd identity = MatrixXcd::Identity(density_count, density_count);
    Vector2d alpha(0., 0.);
    Utils::multipole_defect_A(crystal_A, A_D_map, A_defect_map, N, this->multipole_radius, this->point_defect_radius, omega / this->v, omega / this->v_b, omega / this->v_bd, alpha, this->delta);
    for (int i = 0; i < this->N_gauss; i++) {
        for (int j = 0; j < this->N_gauss; j++) {
            alpha = Vector2d(this->gauss_nodes[i] + 1., this->gauss_nodes[j] + 1.) * M_PI;
            Utils::multipole_crystal_A(crystal_A, N, this->multipole_radius, omega / this->v, omega / this->v_b, alpha, this->delta);
            // [-1, 1]^2 -> [0, 2pi]^2 has Jacobian pi^2.
            const double weight = this->gauss_weights[i] * this->gauss_weights[j] * M_PI * M_PI;
            integrated_resolvent += weight * crystal_A.partialPivLu().solve(identity);
        }
    }
    integrated_resolvent /= (2. * M_PI) * (2. * M_PI);
    MatrixXcd defect_operator = identity + (A_defect_map - A_D_map) * integrated_resolvent;
    return defect_operator;
}

MatrixXcd Multipole::crystal_line_M_operator(cpxd omega, double alpha_x) {
    const int N = this->N_multipole;
    const int mode_count = 2 * N + 1;
    // M^epsilon acts on source pairs (f, g), not on a single multipole vector.
    const int density_count = 2 * mode_count;
    MatrixXcd integrated_resolvent = MatrixXcd::Zero(density_count, density_count);
    MatrixXcd crystal_A, A_D_map, A_defect_map;
    const MatrixXcd identity = MatrixXcd::Identity(density_count, density_count);
    Vector2d alpha(0., 0.);
    Utils::multipole_defect_A(crystal_A, A_D_map, A_defect_map, N, this->multipole_radius, this->line_defect_radius, omega / this->v, omega / this->v_b, omega / this->v_bd, alpha, this->delta);
    for (int j = 0; j < this->N_gauss; j++) {
        alpha = Vector2d(alpha_x, (this->gauss_nodes[j] + 1.) * M_PI);
        Utils::multipole_crystal_A(crystal_A, N, this->multipole_radius, omega / this->v, omega / this->v_b, alpha, this->delta);
        // [-1, 1] -> [0, 2pi] has Jacobian pi.
        const double weight = this->gauss_weights[j] * M_PI;
        integrated_resolvent += weight * crystal_A.partialPivLu().solve(identity);
    }
    integrated_resolvent /= (2. * M_PI);
    MatrixXcd defect_operator = identity + (A_defect_map - A_D_map) * integrated_resolvent;
    return defect_operator;
}
