//
// created by Lara on 6/22/2026
//

#ifndef SUBWAVELENGTHRESONATORS_MULTIPOLE_H
#define SUBWAVELENGTHRESONATORS_MULTIPOLE_H

#include "Eigen/Dense"
#include <complex>
#include <vector>
#include "disk.h"
#include "tools.h"
#include "legendre_matrix.h"
#include "kernels.h"
#include "bessel-library.hpp"

using namespace Eigen;



// /** @brief Computes the multipole expansion of the 2D Helmholtz Green's function. */
// cpxd helmholtz_2D_multipole(cpxd k, const Vector2d &x, const Vector2d &y, int N) {
//     cpxd G = 0.0;
//     for (int n = -N; n <= N; n++) {
//         double r = (x - y).norm();
//         double theta = atan2(x.y() - y.y(), x.x() - y.x());
//         G += bessel::cyl_h1(n, k * r) * exp(cpxd(0, n * theta));
//     }
//     return -0.25 * cpxd(0, 1.0) * G;
// }


class Multipole {
public:
    Multipole(int N_multipole, int N_gauss, double multipole_radius, double point_defect_radius, double line_defect_radius, cpxd v, cpxd v_b, cpxd v_bd, double delta);
    // Assemble the crystal transmission matrix in Fourier modes -N,...,N.
    // These stateless assemblers do not require the defect-operator configuration.
    // Single disk in a unit-square lattice; blocks are [S_b, -S_ext; dS_b, -delta*dS_ext].
    static void crystal_A(MatrixXcd &A_matrix, int N, double R, cpxd k, cpxd k_b,
                          Vector2d alpha, double delta);
    // Multiple disks in a general lattice, with mode-major ordering within each block.
    static void crystal_A(MatrixXcd &A_matrix, int N, const std::vector<Disk>& disks,
                          cpxd k, cpxd k_b,
                          Vector2d alpha, Vector2d a1, Vector2d a2, double delta);

    // Free-space circle operators, with mode-major ordering as in crystal_A.
    static void S(MatrixXcd &S_matrix, int N, const std::vector<Disk>& disks, cpxd k);

    // Principal-value Kstar; add 1/2 I to obtain the exterior normal derivative.
    static void Kstar(MatrixXcd &Kstar_matrix, int N, const std::vector<Disk>& disks, cpxd k);

    MatrixXcd crystal_M_operator(cpxd omega);
    MatrixXcd crystal_line_M_operator(cpxd omega, double alpha_x);

private:
    int N_multipole;
    int N_gauss;
    const double* gauss_nodes;
    const double* gauss_weights;
    double multipole_radius;
    double point_defect_radius;
    double line_defect_radius;
    cpxd v;
    cpxd v_b;
    cpxd v_bd;  // interior wave speed of the defect resonators
    double delta;
};

#endif // SUBWAVELENGTHRESONATORS_MULTIPOLE_H
