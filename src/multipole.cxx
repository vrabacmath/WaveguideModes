#include "multipole.h"
#include "q_function.h"
#include "utils.h"

/**
 * Assemble the circular-inclusion crystal operator in the Fourier multipole basis.
 *
 * The basis is ordered by angular modes -N,...,N. The returned matrix has the
 * same block convention as SpectralOperators::CrystalA:
 *   [ S_b        -S_ext      ]
 *   [ dS_b       -delta*dS_ext ]
 * where the exterior blocks use the unit-square quasi-periodic lattice sums
 * Q_n(k, alpha) from QFunction::q_function. Complex wavenumbers are supported.
 */
void Multipole::crystal_A(MatrixXcd &A_matrix, int N, double R, cpxd k, cpxd k_b, Vector2d alpha, double delta) {
    int M = 2 * N + 1;
    MatrixXcd S_ext(M, M), dS_ext(M, M), S_b(M, M), dS_b(M, M);
    const cpxd c = -cpxd(0, 0.5) * M_PI * R;

    auto cyl_h1_prime = [](int n, cpxd z) {
        return 0.5 * (bessel::cyl_h1(n - 1, z) - bessel::cyl_h1(n + 1, z));
    };

    auto cyl_j_prime = [](int n, cpxd z) {
        return 0.5 * (bessel::cyl_j(n - 1, z) - bessel::cyl_j(n + 1, z));
    };

    #pragma omp parallel for schedule(static) default(none) shared(S_ext, dS_ext, S_b, dS_b, N, M, R, k, k_b, alpha, c, cyl_h1_prime, cyl_j_prime)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            int m = i - N;
            int n = j - N;
            const cpxd j_source = bessel::cyl_j(n, k * R);
            const cpxd j_target = bessel::cyl_j(m, k * R);
            const cpxd jp_target = cyl_j_prime(m, k * R);
            if (i == j) {
                S_ext(i, i) = c * j_source * bessel::cyl_h1(m, k * R) +
                              c * j_source * QFunction::q_function(0, k, alpha) * j_target;
                dS_ext(i, i) = c * k * j_source * cyl_h1_prime(m, k * R) +
                               c * k * j_source * QFunction::q_function(0, k, alpha) * jp_target;
                S_b(i, i) = c * bessel::cyl_h1(n, k_b * R) * bessel::cyl_j(m, k_b * R);
                dS_b(i, i) = c * k_b * bessel::cyl_h1(n, k_b * R) * cyl_j_prime(m, k_b * R);
            } else {
                const cpxd q = QFunction::q_function(n - m, k, alpha);
                S_ext(i, j) = c * j_source * q * j_target;
                dS_ext(i, j) = c * k * j_source * q * jp_target;
                S_b(i, j) = 0.0;
                dS_b(i, j) = 0.0;
            }
        }
    }

    A_matrix = MatrixXcd::Zero(2 * M, 2 * M);
    A_matrix.block(0, 0, M, M) = S_b;
    A_matrix.block(0, M, M, M) = -S_ext;
    A_matrix.block(M, 0, M, M) = dS_b;
    A_matrix.block(M, M, M, M) = -delta * dS_ext;
}

void Multipole::crystal_A(MatrixXcd &A_matrix, int N, const std::vector<Disk>& disks, cpxd k, cpxd k_b, Vector2d alpha, Vector2d a1, Vector2d a2, double delta) {
    int M = 2 * N + 1;
    int NR = disks.size();

    MatrixXcd S_ext(NR * M, NR * M), dS_ext(NR * M, NR * M), S_b(NR * M, NR * M), dS_b(NR * M, NR * M);

    auto cyl_h1_prime = [](int n, cpxd z) {
        return 0.5 * (bessel::cyl_h1(n - 1, z) - bessel::cyl_h1(n + 1, z));
    };

    auto cyl_j_prime = [](int n, cpxd z) {
        return 0.5 * (bessel::cyl_j(n - 1, z) - bessel::cyl_j(n + 1, z));
    };

    for (int ri = 0; ri < NR; ri++) {
        for (int rj = 0; rj < NR; rj++) {
            double R_i = disks[ri].r;
            double R_j = disks[rj].r;
            Vector2d shift_i = disks[ri].c;
            Vector2d shift_j = disks[rj].c;

            Vector2d d = shift_i - shift_j;

            const cpxd c = -cpxd(0, 0.5) * M_PI * R_j;

            #pragma omp parallel for schedule(static) default(none) shared(S_ext, dS_ext, S_b, dS_b, N, NR, M, ri, rj, R_i, R_j, d, k, k_b, alpha, a1, a2, c, cyl_h1_prime, cyl_j_prime)
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < M; j++) {
                    int m = i - N;
                    int n = j - N;
                    const cpxd j_source = bessel::cyl_j(n, k * R_j);
                    const cpxd j_target = bessel::cyl_j(m, k * R_i);
                    const cpxd jp_target = cyl_j_prime(m, k * R_i);
                    if (i == j && ri == rj) {
                        S_ext(i * NR + ri, i * NR + ri) = c * j_source * bessel::cyl_h1(m, k * R_i) +
                                    c * j_source * QFunction::q_function(0, k, alpha, a1, a2, d) * j_target;
                        dS_ext(i * NR + ri, i * NR + ri) = c * k * j_source * cyl_h1_prime(m, k * R_i) +
                                    c * k * j_source * QFunction::q_function(0, k, alpha, a1, a2, d) * jp_target;
                        S_b(i * NR + ri, i * NR + ri) = c * bessel::cyl_h1(n, k_b * R_j) * bessel::cyl_j(m, k_b * R_i);
                        dS_b(i * NR + ri, i * NR + ri) = c * k_b * bessel::cyl_h1(n, k_b * R_j) * cyl_j_prime(m, k_b * R_i);
                    } else {
                        const cpxd q = QFunction::q_function(n - m, k, alpha, a1, a2, d);
                        S_ext(i * NR + ri, j * NR + rj) = c * j_source * q * j_target;
                        dS_ext(i * NR + ri, j * NR + rj) = c * k * j_source * q * jp_target;
                        S_b(i * NR + ri, j * NR + rj) = 0.0;
                        dS_b(i * NR + ri, j * NR + rj) = 0.0;
                    }
                }
            }
        }
    }

    A_matrix = MatrixXcd::Zero(2 * M * NR, 2 * M * NR);
    A_matrix.block(0, 0, M * NR, M * NR) = S_b;
    A_matrix.block(0, M * NR, M * NR, M * NR) = -S_ext;
    A_matrix.block(M * NR, 0, M * NR, M * NR) = dS_b;
    A_matrix.block(M * NR, M * NR, M * NR, M * NR) = -delta * dS_ext;
}

void Multipole::S(MatrixXcd &S_matrix, int N, const std::vector<Disk>& disks, cpxd k) {
    int M = 2 * N + 1;
    int NR = disks.size();

    MatrixXcd S(NR * M, NR * M);

    for (int ri = 0; ri < NR; ri++) {
        for (int rj = 0; rj < NR; rj++) {
            double R_i = disks[ri].r;
            double R_j = disks[rj].r;
            Vector2d shift_i = disks[ri].c;
            Vector2d shift_j = disks[rj].c;

            Vector2d d = shift_i - shift_j;

            const cpxd c = -cpxd(0, 0.5) * M_PI * R_j;

            #pragma omp parallel for schedule(static) default(none) shared(S, N, NR, M, ri, rj, R_i, R_j, d, k, c)
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < M; j++) {
                    int m = i - N;
                    int n = j - N;
                    const cpxd j_source = bessel::cyl_j(n, k * R_j);
                    const cpxd j_target = bessel::cyl_j(m, k * R_i);
                    if (i == j && ri == rj) {
                        S(i * NR + ri, i * NR + ri) = c * j_source * bessel::cyl_h1(n, k * R_i);
                    } else if (ri == rj) {
                        // A circle's self-interaction is diagonal in Fourier modes.
                        S(i * NR + ri, j * NR + rj) = 0.0;
                    } else {
                        S(i * NR + ri, j * NR + rj) = c * j_source * bessel::cyl_h1(n - m, k * d.norm()) * exp(cpxd(0, n - m) * atan2(d.y(), d.x())) * j_target;
                    }
                }
            }
        }
    }

    S_matrix = S;
}

void Multipole::Kstar(MatrixXcd &Kstar_matrix, int N, const std::vector<Disk>& disks, cpxd k) {
    int M = 2 * N + 1;
    int NR = disks.size();

    MatrixXcd Kstar(NR * M, NR * M);

    auto cyl_h1_prime = [](int n, cpxd z) {
        return 0.5 * (bessel::cyl_h1(n - 1, z) - bessel::cyl_h1(n + 1, z));
    };

    for (int ri = 0; ri < NR; ri++) {
        for (int rj = 0; rj < NR; rj++) {
            double R_i = disks[ri].r;
            double R_j = disks[rj].r;
            Vector2d shift_i = disks[ri].c;
            Vector2d shift_j = disks[rj].c;

            Vector2d d = shift_i - shift_j;

            const cpxd c = -cpxd(0, 0.5) * M_PI * R_j;

            #pragma omp parallel for schedule(static) default(none) shared(Kstar, N, NR, M, ri, rj, R_i, R_j, d, k, c, cyl_h1_prime)
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < M; j++) {
                    int m = i - N;
                    int n = j - N;
                    const cpxd j_source = bessel::cyl_j(n, k * R_j);
                    const cpxd jp_target = 0.5 * (bessel::cyl_j(m - 1, k * R_i) -
                                                   bessel::cyl_j(m + 1, k * R_i));
                    if (i == j && ri == rj) {
                        // The exterior normal derivative is 1/2 I + Kstar.
                        Kstar(i * NR + ri, i * NR + ri) = c * k * j_source * cyl_h1_prime(m, k * R_i) - 0.5;
                    } else if (ri == rj) {
                        Kstar(i * NR + ri, j * NR + rj) = 0.0;
                    } else {
                        Kstar(i * NR + ri, j * NR + rj) = c * k * j_source * bessel::cyl_h1(n - m, k * d.norm()) * exp(cpxd(0, n - m) * atan2(d.y(), d.x())) * jp_target;
                    }
                }
            }
        }
    }

    Kstar_matrix = Kstar;
}

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
            Multipole::crystal_A(crystal_A, N, this->multipole_radius, omega / this->v, omega / this->v_b, alpha, this->delta);
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
        Multipole::crystal_A(crystal_A, N, this->multipole_radius, omega / this->v, omega / this->v_b, alpha, this->delta);
        // [-1, 1] -> [0, 2pi] has Jacobian pi.
        const double weight = this->gauss_weights[j] * M_PI;
        integrated_resolvent += weight * crystal_A.partialPivLu().solve(identity);
    }
    integrated_resolvent /= (2. * M_PI);
    MatrixXcd defect_operator = identity + (A_defect_map - A_D_map) * integrated_resolvent;
    return defect_operator;
}
