//
// Created by lara on 10/29/25.
//

#include <gtest/gtest.h>
#include <algorithm>
#include <iostream>
#include "spectral_operators.h"
#include "tools.h"
#include "kernels.h"
#include "q_function.h"
#include "utils.h"
#include "multipole.h"
#include "workflows/bent_waveguide_patch.h"

using namespace std;
using namespace Eigen;
using namespace std::complex_literals;

namespace {

Vector2d reduce_alpha_for_reference(Vector2d alpha) {
    constexpr double two_pi = 2.0 * M_PI;
    for (int j = 0; j < 2; ++j) {
        alpha(j) -= two_pi * std::round(alpha(j) / two_pi);
    }
    return alpha;
}

cpxd ipow_for_reference(int n) {
    int r = n % 4;
    if (r < 0) {
        r += 4;
    }

    switch (r) {
        case 0: return {1.0, 0.0};
        case 1: return {0.0, 1.0};
        case 2: return {-1.0, 0.0};
        default: return {0.0, -1.0};
    }
}

cpxd q_function_dual_y_reference(int n, double k, Vector2d alpha, int cutoff, double r) {
    alpha = reduce_alpha_for_reference(alpha);
    cpxd sum = 0.0;
    const int abs_n = std::abs(n);
    const double bessel_sign = (n < 0 && abs_n % 2 == 1) ? -1.0 : 1.0;

    for (int p = -cutoff; p <= cutoff; ++p) {
        for (int q = -cutoff; q <= cutoff; ++q) {
            Vector2d beta_m = alpha + 2.0 * M_PI * Vector2d(p, q);
            const double beta = beta_m.norm();
            const double denominator = k * k - beta * beta;

            cpxd angular = 1.0;
            if (n != 0) {
                if (beta < 1e-14) {
                    angular = 0.0;
                } else {
                    const double phi = std::atan2(beta_m.y(), beta_m.x());
                    angular = std::exp(cpxd(0.0, n * phi));
                }
            }

            sum += bessel_sign * bessel::cyl_j(abs_n, beta * r) * angular / denominator;
        }
    }

    cpxd sigma_y_times_j = 4.0 * ipow_for_reference(n) * sum;
    if (n == 0) {
        sigma_y_times_j -= bessel::cyl_y(0, k * r);
    }

    const cpxd sigma_y = sigma_y_times_j / (bessel_sign * bessel::cyl_j(abs_n, k * r));
    return (n == 0 ? -1.0 : 0.0) + cpxd(0.0, 1.0) * sigma_y;
}

cpxd q_function_direct_reference(int n, cpxd k, Vector2d alpha, int cutoff) {
    alpha = reduce_alpha_for_reference(alpha);
    cpxd sum = 0.0;
    const int abs_n = std::abs(n);
    const double bessel_sign = (n < 0 && abs_n % 2 == 1) ? -1.0 : 1.0;

    for (int m1 = -cutoff; m1 <= cutoff; ++m1) {
        for (int m2 = -cutoff; m2 <= cutoff; ++m2) {
            if (m1 == 0 && m2 == 0) {
                continue;
            }

            const Vector2d lattice_point(m1, m2);
            const double radius = lattice_point.norm();
            const double theta = std::atan2(lattice_point.y(), lattice_point.x());
            const cpxd angular = std::exp(cpxd(0.0, static_cast<double>(n) * theta));
            const cpxd phase = std::exp(cpxd(0.0, alpha.dot(lattice_point)));
            sum += bessel_sign * bessel::cyl_h1(abs_n, k * radius) * angular * phase;
        }
    }

    return sum;
}

MatrixXcd fourier_mode_matrix(int point_count, int order_cutoff) {
    const int mode_count = 2 * order_cutoff + 1;
    MatrixXcd modes(point_count, mode_count);

    for (int point = 0; point < point_count; ++point) {
        const double theta = 2.0 * M_PI * static_cast<double>(point) / static_cast<double>(point_count);
        for (int col = 0; col < mode_count; ++col) {
            const int order = col - order_cutoff;
            modes(point, col) = std::exp(cpxd(0.0, static_cast<double>(order) * theta));
        }
    }

    return modes;
}

MatrixXcd project_crystal_matrix_to_fourier(const MatrixXcd& nodal_matrix, int point_count, int order_cutoff) {
    const int mode_count = 2 * order_cutoff + 1;
    const MatrixXcd modes = fourier_mode_matrix(point_count, order_cutoff);
    MatrixXcd projected = MatrixXcd::Zero(2 * mode_count, 2 * mode_count);

    for (int row_block = 0; row_block < 2; ++row_block) {
        for (int col_block = 0; col_block < 2; ++col_block) {
            projected.block(row_block * mode_count, col_block * mode_count, mode_count, mode_count) =
                    modes.adjoint() *
                    nodal_matrix.block(row_block * point_count, col_block * point_count, point_count, point_count) *
                    modes / static_cast<double>(point_count);
        }
    }

    return projected;
}

cpxd h1_prime_reference(int n, cpxd z) {
    return 0.5 * (bessel::cyl_h1(n - 1, z) - bessel::cyl_h1(n + 1, z));
}

cpxd j_prime_reference(int n, cpxd z) {
    return 0.5 * (bessel::cyl_j(n - 1, z) - bessel::cyl_j(n + 1, z));
}

MatrixXcd local_effective_source_operator(int order_cutoff, double radius, cpxd k, cpxd k_b, double delta) {
    const int mode_count = 2 * order_cutoff + 1;
    const cpxd c = -cpxd(0.0, 0.5) * M_PI * radius;
    MatrixXcd local = MatrixXcd::Zero(2 * mode_count, 2 * mode_count);

    for (int i = 0; i < mode_count; ++i) {
        const int n = i - order_cutoff;
        const cpxd j_interior = bessel::cyl_j(n, k_b * radius);
        const cpxd h_interior = bessel::cyl_h1(n, k_b * radius);
        const cpxd jp_interior = j_prime_reference(n, k_b * radius);
        const cpxd j_exterior = bessel::cyl_j(n, k * radius);
        const cpxd h_exterior = bessel::cyl_h1(n, k * radius);
        const cpxd hp_exterior = h1_prime_reference(n, k * radius);

        local(i, i) = c * j_interior * h_interior;
        local(i, i + mode_count) = -c * j_exterior * h_exterior;
        local(i + mode_count, i) = c * k_b * jp_interior * h_interior;
        local(i + mode_count, i + mode_count) = -c * delta * k * j_exterior * hp_exterior;
    }

    return local;
}

MatrixXcd defect_projector1(int order_cutoff, double radius, double defect_radius, cpxd k, cpxd k_b) {
    const int mode_count = 2 * order_cutoff + 1;
    const double radius_ratio = radius / defect_radius;
    MatrixXcd projector = MatrixXcd::Zero(2 * mode_count, 2 * mode_count);

    for (int i = 0; i < mode_count; ++i) {
        const int n = i - order_cutoff;
        projector(i, i) = radius_ratio *
                          bessel::cyl_h1(n, k_b * radius) /
                          bessel::cyl_h1(n, k_b * defect_radius);
        projector(i + mode_count, i + mode_count) = radius_ratio *
                                                     bessel::cyl_j(n, k * radius) /
                                                     bessel::cyl_j(n, k * defect_radius);
    }

    return projector;
}

MatrixXcd defect_projector2(int order_cutoff, double radius, double defect_radius, cpxd k) {
    const int mode_count = 2 * order_cutoff + 1;
    MatrixXcd projector = MatrixXcd::Zero(2 * mode_count, 2 * mode_count);

    for (int i = 0; i < mode_count; ++i) {
        const int n = i - order_cutoff;
        projector(i, i) = bessel::cyl_j(n, k * defect_radius) / bessel::cyl_j(n, k * radius);
        projector(i + mode_count, i + mode_count) =
                j_prime_reference(n, k * defect_radius) / j_prime_reference(n, k * radius);
    }

    return projector;
}

} // namespace

inline Eigen::MatrixXcd makeA() {
    // Each array is the first ROW (k = 0..9) of a 10×10 Toeplitz block,
    // i.e. T(i,j) = a[|i-j|].
    // Top-left block (rows 1..10, cols 1..10)
    const std::array<std::complex<double>,10> a11 = {
            -0.2274-0.1571i, -0.0449-0.1424i,  0.0334-0.1073i,  0.0670-0.0699i,  0.0781-0.0441i,
            0.0802-0.0352i,  0.0781-0.0441i,  0.0670-0.0699i,  0.0334-0.1073i, -0.0449-0.1424i
    };

    // Top-right block (rows 1..10, cols 11..20)
    const std::array<std::complex<double>,10> a12 = {
            0.1581+0.1571i, -0.0393+0.1026i, -0.0809+0.0044i, -0.0461-0.0517i, -0.0099-0.0632i,
            0.0027-0.0624i, -0.0099-0.0632i, -0.0461-0.0517i, -0.0809+0.0044i, -0.0393+0.1026i
    };

    // Bottom-left block (rows 11..20, cols 1..10)
    const std::array<std::complex<double>,10> a21 = {
            -0.4500+0.0000i,  0.0597+0.0143i,  0.0590+0.0454i,  0.0427+0.0726i,  0.0244+0.0868i,
            0.0168+0.0906i,  0.0244+0.0868i,  0.0427+0.0726i,  0.0590+0.0454i,  0.0597+0.0143i
    };

    // Bottom-right block (rows 11..20, cols 11..20)
    const std::array<std::complex<double>,10> a22 = {
            -0.0550+0.0000i, -0.0058-0.0049i,  0.0014-0.0098i,  0.0096-0.0063i,  0.0124-0.0003i,
            0.0125+0.0021i,  0.0124-0.0003i,  0.0096-0.0063i,  0.0014-0.0098i, -0.0058-0.0049i
    };

    Eigen::MatrixXcd A(20,20);
    // fill 4 blocks using the Toeplitz rule with |i-j|
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            int k = std::abs(i - j);
            A(i,    j   ) = a11[k]; // top-left
            A(i,    j+10) = a12[k]; // top-right
            A(i+10, j   ) = a21[k]; // bottom-left
            A(i+10, j+10) = a22[k]; // bottom-right
        }
    }
    return A;
}

TEST(SampleTest, BasicAssertions) {
    // Expect two strings to be equal.
    EXPECT_STRNE("hello", "world");
    // Expect equality.
    EXPECT_EQ(7 * 6, 42);
}

TEST(DoubleLayerTest, Adjointness) {
    MatrixXcd Kstar, K;
    BoundaryMesh mesh(100);
    mesh.generate_ellipse(1.0, 10.0);

    // Helmholtz K and K* adjointness test
    SpectralOperators ops(mesh);
    ops.Kstar_diagonal(Kstar, 1.0);
    ops.K_diagonal(K, 1.0);

    Eigen::MatrixXcd W(100, 100);
    W.setZero();
    for (int i = 0; i < 100; i++) {
        W(i,i) = mesh.get_vertex(i).sigma;
    }

    double norm_diff = (W * Kstar - K.transpose() * W).norm();
    double norm_Kstar = (W * Kstar).norm();

    EXPECT_LT(norm_diff / norm_Kstar, 1e-10);

    // Laplace K and K* adjointness test
    ops.Kstar_diagonal(Kstar, 0.0);
    ops.K_diagonal(K, 0.0);

    norm_diff = (W * Kstar - K.transpose() * W).norm();
    norm_Kstar = (W * Kstar).norm();

    EXPECT_LT(norm_diff / norm_Kstar, 1e-10);
}

TEST(ReferenceComplexErrorFunctionTests, Value) {
    using namespace Tools;

    // Test values against known results
    complex<double> z1(0.0, 1.0);
    complex<double> result1 = complex<double>(1.0, -1.65042575879754);
    EXPECT_NEAR(abs(erfc_complex(z1) - result1), 0.0, 1e-6);

    complex<double> z2(1.0, 1.0);
    complex<double> result2 = complex<double>(1.31615128169795, 0.190453469237835);
    EXPECT_NEAR(abs(erf_complex(z2) - result2), 0.0, 1e-6);
}

TEST(OperatorATest, MatrixValues) {
    int N = 100; cpxd k = 2.0;
    Eigen::MatrixXcd A, Kstar(N, N);
    BoundaryMesh mesh(N);
    mesh.generate_circle(1.0);
    SpectralOperators ops(mesh);
    ops.Kstar_diagonal(A, k);

    if (abs(k) < 1e-6) { // Use Laplace kernel for small k
        #pragma omp parallel for schedule(static) default(none) shared(Kstar, mesh, N)
        for (int i = 0; i < N; i++) {
            Vector2d point_i = mesh.get_vertex(i).point;
            Vector2d normal_i = mesh.get_vertex(i).normal;
            for (int j = 0; j < N; j++) {
                if (i != j) {
                    Vector2d point_j = mesh.get_vertex(j).point;
                    Kstar(i, j) = 0.5 * M_1_PI * ((point_i - point_j).dot(normal_i)) /
                                  (point_i - point_j).squaredNorm() * mesh.get_vertex(j).sigma;
                } else {
                    // self-interaction term
                    Kstar(i, i) = mesh.get_vertex(i).curvature * 0.25 * M_1_PI * mesh.get_vertex(i).sigma;
                }
            }
        }
    } else {
        #pragma omp parallel for schedule(static) default(none) shared(Kstar, mesh, N, k)
        for (int i = 0; i < N; i++) {
            Vector2d point_i = mesh.get_vertex(i).point;
            Vector2d normal_i = mesh.get_vertex(i).normal;
            for (int j = 0; j < N; j++) {
                if (i != j) {
                    Vector2d point_j = mesh.get_vertex(j).point;
                    Vector2d r_ij = point_i - point_j;

                    double r_norm = r_ij.norm();
                    Kstar(i, j) = cpxd(0, 1.0) * 0.25 * k *
                                  bessel::cyl_h1(1, k * r_norm)
                                  * r_ij.dot(normal_i) / r_norm * mesh.get_vertex(j).sigma;
                } else {
                    // self-interaction term
                    Kstar(i, i) = 0.25 * M_1_PI
                                  * mesh.get_vertex(i).curvature * mesh.get_vertex(i).sigma;
                }
            }
        }
    }

    double norm_diff = (A - Kstar).norm();
    double norm_Aref = Kstar.norm();

    EXPECT_LT(norm_diff / norm_Aref, 4e-4);
}

TEST(GPeriodicTest, KernelValues) {
    using namespace std::complex_literals;
    std::array<complex<double>, 10> ref = {
        -0.157787986457114 + 0.055945613811098i,
        -0.159860483859888 + 0.049602253483013i,
        -0.161706182681814 + 0.043177169322235i,
        -0.163324231403706 + 0.036691367379490i,
        -0.164693679115019 + 0.030160911601187i,
        -0.165786410898441 + 0.023593796510772i,
        -0.166585561436668 + 0.016991532494431i,
        -0.167096287394980 + 0.010354362985495i,
        -0.167341194765235 + 0.003687204509985i,
        -0.167343695620199 - 0.002996952875490i
    };

    Vector2d a(0.0, -1.3);
    Vector2d src(1.1, 1.0);
    double period = 1.5;
    for (int j = 0; j < 10; j++) {
        double x = 4.22 * period * (double(j) / 30.); // [0,d]
        Vector2d r = Vector2d(x, 0.0) + a;
        Vector2d r1 = Vector2d(r.x(),  r.y());
        Vector2d r2 = Vector2d(r.x(), -r.y());
        auto G = Kernels::helmholtz_2D_periodic(2.0, -4.0, src, r1, period, sqrt(M_PI), 5);
        EXPECT_NEAR(abs(G - ref[j]), 0.0, 1e-10);
    }
}

TEST(GradGPeriodicTest, KernelValues) {
    using namespace std::complex_literals;

    // reference values for the gradient of the periodic Helmholtz kernel
    std::array<Vector2cd, 10> ref = {
        Vector2cd(0.069321594150835 - 0.180659637558201i,
                  0.119926995574832 - 0.310224126307529i),
        Vector2cd(0.031144289685568 - 0.191329706757032i,
                  0.052163218077043 - 0.329396476864701i),
        Vector2cd(-0.008575532228087 - 0.192985870446229i,
                  -0.017502604780295 - 0.333822145502856i),
        Vector2cd(-0.048372445863422 - 0.185779053873665i,
                  -0.086146375598964 - 0.323080606863239i),
        Vector2cd(-0.086579756893052 - 0.170381578357120i,
                  -0.150999849263327 - 0.297582369712031i),
        Vector2cd(-0.121356309283761 - 0.147804434206746i,
                  -0.209452334686518 - 0.258575768114624i),
        Vector2cd(-0.150830618401543 - 0.119213389991498i,
                  -0.259046842739146 - 0.208039386789508i),
        Vector2cd(-0.173330278234556 - 0.085808491726187i,
                  -0.297527088600956 - 0.148488711397857i),
        Vector2cd(-0.187619338096587 - 0.048799293570982i,
                  -0.322963424856258 - 0.082748734260221i),
        Vector2cd(-0.193057468102440 - 0.009465816356180i,
                  -0.333939461497825 - 0.013749305302845i)
    };

    Vector2d a(0.0, -1.3);
    Vector2d src(1.1, 1.0);
    double period = 1.5;
    for (int j = 0; j < 10; j++) {
        double x = 4.22 * period * (double(j) / 30.); // [0,d]
        Vector2d r = Vector2d(x, 0.0) + a;
        Vector2d r1 = Vector2d(r.x(),  r.y());
        Vector2d r2 = Vector2d(r.x(), -r.y());
        auto G_grad = Kernels::grad_helmholtz_2D_periodic(2.0, -1.0, src, r1, period, sqrt(M_PI), 5);
        EXPECT_NEAR((G_grad - ref[j]).norm(), 0.0, 1e-10);
    }
}

TEST(DirichletVsDirectGPeriodic, KernelValues) {
    using namespace std::complex_literals;

    Vector2d a(0.0, -1.3);
    Vector2d src(1.1, 1.0);
    double period = 1.5;
    for (int j = 0; j < 10; j++) {
        double x = 4.22 * period * (double(j) / 30.); // [0,d]
        Vector2d r = Vector2d(x, 0.0) + a;
        Vector2d r1 = Vector2d(r.x(), r.y());
        Vector2d r2 = Vector2d(r.x(), -r.y());
        auto G1 = Kernels::helmholtz_2D_periodic(2.0, -1.0, src, r1, period, sqrt(M_PI), 10);
        auto G2 = Kernels::helmholtz_2D_periodic(2.0, -1.0, src, r2, period, sqrt(M_PI), 10);

        auto G_direct = Kernels::dir_helmholtz_2D_periodic(2.0, -1.0, src, r, period, sqrt(M_PI), 10);

        EXPECT_NEAR(abs((G1 - G2) - G_direct), 0.0, 1e-12);

        auto G_grad1 = Kernels::grad_helmholtz_2D_periodic(2.0, -1.0, src, r1, period, sqrt(M_PI), 10);
        auto G_grad2 = Kernels::grad_helmholtz_2D_periodic(2.0, -1.0, src, r2, period, sqrt(M_PI), 10);
        auto G_grad_direct = Kernels::dir_grad_helmholtz_2D_periodic(2.0, -1.0, src, r, period, sqrt(M_PI), 10);

        EXPECT_NEAR((G_grad1 - G_grad2 - G_grad_direct).norm(), 0.0, 1e-12);
    }
}

//TEST(E1functionTest, Value) {
//    for (int i = 0; i < 100; i++) {
//        double x = 0.1 * i + 0.01;
//        double E1_val = E1(x);
//        double E1_ref = E1f(x);
//        EXPECT_NEAR(abs(E1_val - E1_ref), 0.0, 1e-7);
//    }
//}

TEST(ComplexHankelTest, Value) {
    using namespace std::complex_literals;
    complex<double> z(1.0, 1.0);
    complex<double> H1 = bessel::cyl_h1(0, z);
    complex<double> H1_ref(0.227449894802295 - 0.051055458673090i);
    EXPECT_NEAR(abs(H1 - H1_ref), 0.0, 1e-15);

    complex<double> H1_1 = bessel::cyl_h1(1, z);
    complex<double> H1_1_ref(-0.015640669069981 - 0.292666506764257i);
    EXPECT_NEAR(abs(H1_1 - H1_1_ref), 0.0, 1e-15);

    z = -1.0 - 1.0i;
    H1 = bessel::cyl_h1(0, z);
    H1_ref = 2.102666848414353 - 1.044115353891334i;
    EXPECT_NEAR(abs(H1 - H1_ref), 0.0, 1e-15);

    H1_1 = bessel::cyl_h1(1, z);
    H1_1_ref = -1.212680000775826 - 0.437389550889918i;
    EXPECT_NEAR(abs(H1_1 - H1_1_ref), 0.0, 1e-15);

    for (int j = 0; j < 100; j++) {
        double r = 1.08 * j + 0.1;
        complex<double> z = complex<double>(r, 0.0);

        complex<double> H1 = bessel::cyl_h1(0, z);
        complex<double> H1_ref = complex<double>(cyl_bessel_j(0, r), cyl_neumann(0, r));
        EXPECT_NEAR(abs(H1 - H1_ref), 0.0, 1e-12);
    }
}

TEST(SlowGPeriodicTestHelmholtz, CompareWithDirectSum) {
    using namespace std::complex_literals;

    Vector2cd gt = Vector2cd::Zero();
    double period = 1.5;
    double kbar = 1.0; cpxd k = 0.5 + 0.7; cpxd kd = sqrt(k * k - kbar * kbar);
    complex<double> t = 0.0,
//    ctrl = cpxd(0, -0.5) * exp(+cpxd(0, kbar) * 0.3 + cpxd(0, 1.) * kd * 0.3) / kd / period;
    ctrl = 0.0;

    for (int n = -1000000; n <= 1000000; n++) {
        Vector2d x1(0, 0);
        Vector2d x2(0.3 + period * n, 0.3);
        gt += exp(complex<double>(0.,-kbar) * (double)n * period) * Kernels::grad_helmholtz_2D(k, x1, x2);
        t += exp(complex<double>(0.,-kbar) * (double)n * period) * Kernels::helmholtz_2D(k, x1, x2);

        double l = 2. * M_PI * (double)n / period;
        cpxd denom = sqrt(-(l + kbar) * (l + kbar) + k * k);
//        if (n != 0) ctrl -= 0.5 / period * exp(+cpxd(0, kbar) * 0.3 -
//                            denom * 0.3 - cpxd(0, 1.) * l * 0.3)/ denom;
        ctrl += exp(cpxd(0, kbar + l)*0.3 + cpxd(0, 1.) * denom * 0.3)
                / (cpxd(0, 2.) * denom * period);
    }

    // 0 picks epsilon = sqrt(pi)/period
    auto kernel = Kernels::grad_helmholtz_2D_periodic(k, kbar, Vector2d(0.1, 0.2), Vector2d(0.4, 0.5), period, 0, 10);
    auto kernel2 = Kernels::helmholtz_2D_periodic(k, kbar, Vector2d(0.1, 0.2), Vector2d(0.4, 0.5), period, 0, 10);

    EXPECT_NEAR((gt - kernel).norm(), 0.0, 1e-3);
    EXPECT_NEAR(abs(t - kernel2), 0.0, 1e-3);
    EXPECT_NEAR(abs(ctrl - t), 0.0, 1e-3);
    EXPECT_NEAR(abs(ctrl - kernel2), 0.0, 1e-6);
}

TEST(SlowGPeriodicTestLaplace, CompareWithDirectSum) {
    using namespace std::complex_literals;
    double period = 1.5;

    double L = period;
    Vector2d x1(0.0, 0.0);
    Vector2d x2(0.3, 0.3);

    Vector2d r = x1 - x2;
    double rx = r.x();
    double ry = r.y();
    double ay = std::abs(ry);
    double s  = (ry >= 0.0) ? 1.0 : -1.0;

    std::complex<double> t = 0.0;
    Vector2cd gt = Vector2cd::Zero();

    t += ay / (2.0 * L) - std::log(2.0) * 0.5 * M_1_PI;

    int Nmax = 1000000;
    for (int n = 1; n <= Nmax; ++n) {
        double k = 2.0 * M_PI * n / L;
        double e = std::exp(-k * ay);
        double c = std::cos(k * rx);
        double s2 = std::sin(k * rx);

        t -= 0.5 * M_1_PI * e * c / n;

        gt.x() += (1.0 / L) * e * s2;
        gt.y() += (1.0 / L) * s * e * c;
    }

    gt.y() += s / L * 0.5;

    auto kernel = Kernels::laplace_2D_periodic(Vector2d(0., 0.), Vector2d(0.3, 0.3), period);
    auto kernel2 = Kernels::grad_laplace_2D_periodic(Vector2d(0., 0.), Vector2d(0.3, 0.3), period);

    EXPECT_NEAR(abs(t - kernel), 0.0, 1e-10);
    EXPECT_NEAR((gt - kernel2).norm(), 0.0, 1e-10);
}

TEST(OperatorLimitsKstar, CompareValues) {
    int N = 100; cpxd k = 1e-6;
    Eigen::MatrixXcd A_helmholtz, A_laplace;
    BoundaryMesh mesh(N);
    mesh.generate_circle(0.8, Vector2d(0.0, 2.0));
    SpectralOperators ops(mesh);
    ops.Kstar_diagonal(A_helmholtz, k);
    ops.Kstar_diagonal(A_laplace, 0.0);

    double norm_diff_Kstar = (A_helmholtz - A_laplace).norm();
    double norm_Aref_Kstar = A_laplace.norm();

    EXPECT_LT(norm_diff_Kstar / norm_Aref_Kstar, 1e-6);

    ops.DirPeriodicKstar_diagonal(A_helmholtz, k, -1.0, 1.0);
    ops.DirPeriodicKstar_diagonal(A_laplace, 0.0, -1.0, 1.0);
    double norm_diff_KstarP = (A_helmholtz - A_laplace).norm();
    double norm_Aref_KstarP = A_laplace.norm();

    EXPECT_LT(norm_diff_KstarP / norm_Aref_KstarP, 1e-6);

    ops.K_diagonal(A_helmholtz, k);
    ops.K_diagonal(A_laplace, 0.0);
    double norm_diff_K = (A_helmholtz - A_laplace).norm();
    double norm_Aref_K = A_laplace.norm();

    EXPECT_LT(norm_diff_K / norm_Aref_K, 1e-6);
}

TEST(FreeSpaceOperators, MultiMeshSingleLayerOffDiagonalMatchesKernel) {
    const int N = 24;
    const cpxd k(0.8, 0.2);

    BoundaryMesh mesh(N), mesh2(N);
    mesh.generate_circle(0.25, Vector2d(-0.7, 0.1));
    mesh2.generate_circle(0.2, Vector2d(0.75, -0.15));
    mesh.add_mesh(mesh2);

    MatrixXcd S;
    SpectralOperators ops(mesh);
    ops.S(S, k);

    const int start_i = mesh.get_start_index(0);
    const int end_i = mesh.get_end_index(0);
    const int start_j = mesh.get_start_index(1);
    const int end_j = mesh.get_end_index(1);

    for (int i = start_i; i <= end_i; ++i) {
        for (int j = start_j; j <= end_j; ++j) {
            const cpxd expected = Kernels::helmholtz_2D(k, mesh.get_vertex(i).point,
                                                        mesh.get_vertex(j).point)
                                  * mesh.get_vertex(j).sigma;
            EXPECT_NEAR(std::abs(S(i, j) - expected), 0.0, 1e-13);
        }
    }
}

TEST(FreeSpaceOperators, FullDoubleLayerAdjointnessOnMultipleComponents) {
    const int N = 48;
    const cpxd k(0.7, 0.1);

    BoundaryMesh mesh(N), mesh2(N);
    mesh.generate_ellipse(0.7, 0.35, Vector2d(-1.0, 0.0));
    mesh2.generate_circle(0.3, Vector2d(0.8, 0.15));
    mesh.add_mesh(mesh2);

    MatrixXcd Kstar, K;
    SpectralOperators ops(mesh);
    ops.Kstar(Kstar, k);
    ops.K(K, k);

    MatrixXcd W = MatrixXcd::Zero(mesh.get_num_segments(), mesh.get_num_segments());
    for (int i = 0; i < mesh.get_num_segments(); ++i) {
        W(i, i) = mesh.get_vertex(i).sigma;
    }

    const double rel = (W * Kstar - K.transpose() * W).norm() / (W * Kstar).norm();
    EXPECT_LT(rel, 1e-10);
}

TEST(PeriodicOperators, DynamicSingleLayerFiniteAndYTranslationInvariant) {
    const int N = 32;
    const double period = 1.0;
    const double kbar = 1.3;
    const cpxd k(0.6, 0.05);

    auto assemble = [&](double yc) {
        BoundaryMesh mesh(N);
        mesh.generate_circle(0.3, Vector2d(0.0, yc));
        SpectralOperators ops(mesh);
        MatrixXcd S;
        ops.PeriodicS(S, k, kbar, period);
        return S;
    };

    MatrixXcd S1 = assemble(0.5);
    MatrixXcd S2 = assemble(4.5);

    EXPECT_TRUE(S1.allFinite());
    EXPECT_TRUE(S2.allFinite());
    EXPECT_LT((S1 - S2).norm() / S1.norm(), 1e-10);
}

TEST(CrystalOperators, FullAndDiagonalAgreeForSingleComponent) {
    const int N = 32;
    const cpxd k(0.9, 0.15);
    const Vector2d alpha(0.3, -0.2);
    const Vector2d a1(1.0, 0.0);
    const Vector2d a2(0.2, 0.5 * sqrt(3.0));

    BoundaryMesh mesh(N);
    mesh.generate_circle(0.18);
    SpectralOperators ops(mesh);

    MatrixXcd S_full, S_diag, Kstar_full, Kstar_diag, K_full, K_diag;
    ops.CrystalS(S_full, k, alpha, a1, a2);
    ops.CrystalS_diagonal(S_diag, k, alpha, a1, a2);
    ops.CrystalKstar(Kstar_full, k, alpha, a1, a2);
    ops.CrystalKstar_diagonal(Kstar_diag, k, alpha, a1, a2);
    ops.CrystalK(K_full, k, alpha, a1, a2);
    ops.CrystalK_diagonal(K_diag, k, alpha, a1, a2);

    EXPECT_NEAR((S_full - S_diag).norm(), 0.0, 1e-13);
    EXPECT_NEAR((Kstar_full - Kstar_diag).norm(), 0.0, 1e-13);
    EXPECT_NEAR((K_full - K_diag).norm(), 0.0, 1e-13);
}

//TEST(FaddeevaVsErfcComplex, CompareValues) {
//    for (int j = -100; j <= 100; j++) {
//        double y = 0.1 * (double) j;
//        for (int i = 0; i <= 100; i++) {
//            double x = 0.1 * (double) i;
//            complex<double> z(x, y);
//            complex<double> erfc_ref = Tools::erfc_complex_ref(z);
//            complex<double> erfc_c = Tools::erfc_complex(z);
//            EXPECT_NEAR(abs((erfc_c - erfc_ref) / erfc_ref), 0.0, 1e-10);
//        }
//    }
//}

TEST(CapacitanceMatrixCOnvergence, CompareValues) {
    double eigenvalue = 0.0;

    for (int N = 32; N <= 128; N += 4) {
        BoundaryMesh mesh(N), mesh2(N);
        mesh.generate_circle(0.01, Vector2d(0.0, 0.04));
        mesh2.generate_circle(0.01, Vector2d(0.0, 0.04 + 0.03));
        mesh.add_mesh(mesh2);

        MatrixXcd capacity_matrix, S;
        SpectralOperators ops(mesh);
        ops.DirPeriodicS(S, 0.0, 0.0, 0.1);
        ops.makeCapacitanceMatrix(capacity_matrix, S);

        VectorXcd eigenvals = capacity_matrix.eigenvalues();
        if (eigenvalue) EXPECT_LT(abs(eigenvals(1) - eigenvalue), 1e-10);

        eigenvalue = eigenvals(1).real();
    }
}

// --- Quasi-periodic static Green's function / capacitance matrix (arXiv:2512.05370v2) ---

// At alpha = 0 the smooth correction collapses to the constant offset log(2)/(2 pi)
// between laplace_2D_periodic and the paper's G^{0,0} normalisation.
TEST(QuasiPeriodicStaticGreen, AlphaZeroCorrectionIsConstant) {
    const double expected = 0.5 * M_1_PI * std::log(2.0);
    Vector2d x(0.13, 0.42), y(-0.27, 0.05);
    cpxd c1 = Kernels::laplace_2D_quasiperiodic_correction(0.0, x, y, 200);
    cpxd c2 = Kernels::laplace_2D_quasiperiodic_correction(0.0, Vector2d(0.4, -0.3), Vector2d(0.1, 0.2), 200);
    EXPECT_NEAR(c1.real(), expected, 1e-14);
    EXPECT_NEAR(c1.imag(), 0.0, 1e-14);
    EXPECT_NEAR(c2.real(), expected, 1e-14);
}

// The assembled kernel  G^{alpha,0} = laplace_2D_periodic + correction  must be
// quasi-periodic in x1 with Bloch phase exp(i*alpha): G(x+(1,0),y) = e^{i alpha} G(x,y).
// This is the defining property of the lattice Green's function and validates the
// spectral construction. Tested away from x2 = y2 where the series converges quickly.
TEST(QuasiPeriodicStaticGreen, SatisfiesBlochQuasiPeriodicity) {
    const int M = 600;
    auto G = [&](double alpha, const Vector2d& x, const Vector2d& y) {
        return Kernels::laplace_2D_periodic(x, y, 1.0)
             + Kernels::laplace_2D_quasiperiodic_correction(alpha, x, y, M);
    };
    Vector2d y(0.05, 0.0);
    Vector2d x(0.17, 0.6);                     // |x2 - y2| = 0.6, fast convergence
    for (double alpha : {0.4, 1.3, -2.1, 2.9}) {
        cpxd lhs = G(alpha, x + Vector2d(1.0, 0.0), y);
        cpxd rhs = std::exp(cpxd(0, 1) * alpha) * G(alpha, x, y);
        EXPECT_NEAR(std::abs(lhs - rhs), 0.0, 1e-4) << "alpha = " << alpha;
    }
}

// The quasi-periodic capacitance matrix must be Hermitian with real, positive
// generalised eigenvalues, and (Gamma-point) becomes singular as alpha -> 0: the
// lowest eigenvalue collapses to ~0, while at the zone edge it stays O(1).
TEST(QuasiPeriodicStaticGreen, CapacitanceSpectrumIsPhysical) {
    const int N = 16;
    const double r = 0.3;
    BoundaryMesh mesh(N), m2(N), m3(N), m4(N);
    mesh.generate_circle(r, Vector2d(0.0, -1.5));
    m2.generate_circle(r, Vector2d(0.0, -0.5));
    m3.generate_circle(r, Vector2d(0.0, 0.5));
    m4.generate_circle(r, Vector2d(0.0, 1.5));
    mesh.add_mesh(m2); mesh.add_mesh(m3); mesh.add_mesh(m4);
    SpectralOperators ops(mesh);

    auto min_eig = [&](double alpha) {
        MatrixXcd S, C;
        ops.QuasiPeriodicStaticS(S, alpha, 1.0, 200);
        S = 0.5 * (S + S.adjoint().eval());
        ops.makeCapacitanceMatrix(C, S);
        C = 0.5 * (C + C.adjoint().eval());
        SelfAdjointEigenSolver<MatrixXcd> es(C);
        // Hermitian symmetry already enforced; eigenvalues are real.
        EXPECT_LT(es.eigenvalues()(0), es.eigenvalues()(3) + 1e-12);  // ascending
        EXPECT_GT(es.eigenvalues()(0), -1e-8);                       // positive (semi-)definite
        return es.eigenvalues()(0);
    };

    double lam_gamma = min_eig(1e-4);   // near Gamma point
    double lam_edge  = min_eig(M_PI);   // zone edge
    EXPECT_LT(lam_gamma, 1e-3);         // capacitance matrix singular at alpha -> 0
    EXPECT_GT(lam_edge, 1e-2);          // gapped away from Gamma
}

// A genuine (topologically protected) interface mode is a bound state: its eigenvalue
// is insensitive to the supercell size and its eigenvector stays localized at the
// interface. We build two SSH chains of different half-lengths, pick the most
// center-localized eigenvalue at a fixed alpha, and check (i) the interface eigenvalue
// barely changes with supercell size, and (ii) the mode is concentrated near the origin
// (small r.m.s. localization length) and independent of supercell size. This is the
// supercell-convergence test that distinguishes real interface modes from artifacts.
TEST(QuasiPeriodicStaticGreen, InterfaceModeConvergesWithSupercell) {
    const int N = 12;
    const double r = 0.3, alpha = 0.6 * M_PI, mass = M_PI * r * r;

    auto centres = [](int half) {
        std::vector<double> ys;
        for (int n = 1; n <= half; ++n) {
            ys.push_back(2.0 * n - 1.65);  ys.push_back(2.0 * n - 0.35);
            ys.push_back(-2.0 * n + 0.65); ys.push_back(-2.0 * n + 1.35);
        }
        std::sort(ys.begin(), ys.end());
        return ys;
    };

    auto interface_mode = [&](int half, double& lambda, double& xi_rms) {
        std::vector<double> ys = centres(half);
        const int nd = static_cast<int>(ys.size());
        BoundaryMesh mesh(N);
        mesh.generate_circle(r, Vector2d(0.0, ys[0]));
        for (int n = 1; n < nd; ++n) {
            BoundaryMesh d(N); d.generate_circle(r, Vector2d(0.0, ys[n])); mesh.add_mesh(d);
        }
        SpectralOperators ops(mesh);
        MatrixXcd S, C;
        ops.QuasiPeriodicStaticS(S, alpha, 1.0, 100);
        S = 0.5 * (S + S.adjoint().eval());
        ops.makeCapacitanceMatrix(C, S);
        C = 0.5 * (C + C.adjoint().eval());
        SelfAdjointEigenSolver<MatrixXcd> es(C);

        std::vector<int> by_abs(nd);
        std::iota(by_abs.begin(), by_abs.end(), 0);
        std::sort(by_abs.begin(), by_abs.end(),
                  [&](int a, int b) { return std::abs(ys[a]) < std::abs(ys[b]); });
        int best = 0; double best_c = -1.0;
        for (int m = 0; m < nd; ++m) {
            VectorXd w = es.eigenvectors().col(m).cwiseAbs2(); w /= w.sum();
            double c = 0.0; for (int j = 0; j < 6; ++j) c += w(by_abs[j]);
            if (c > best_c) { best_c = c; best = m; }
        }
        VectorXd w = es.eigenvectors().col(best).cwiseAbs2(); w /= w.sum();
        double ybar = 0.0; for (int n = 0; n < nd; ++n) ybar += w(n) * ys[n];
        double var = 0.0;  for (int n = 0; n < nd; ++n) var += w(n) * (ys[n]-ybar)*(ys[n]-ybar);
        lambda = es.eigenvalues()(best) / mass;
        xi_rms = std::sqrt(var);
    };

    double lam_small, xi_small, lam_large, xi_large;
    interface_mode(5, lam_small, xi_small);
    interface_mode(8, lam_large, xi_large);

    EXPECT_NEAR(lam_small, lam_large, 1e-3);  // eigenvalue converges with supercell size
    EXPECT_NEAR(xi_small, xi_large, 0.2);     // localization length is supercell-independent
    EXPECT_LT(xi_large, 3.0);                 // tightly bound to the interface (gaps ~ 0.7-1.3)
}

// The SSH interface mode is topologically protected: the two oppositely-dimerized
// half-crystals carry Zak phases differing by pi (a quantized Z2 index), which by
// bulk-edge correspondence guarantees an interface state. We extract the strong/weak
// bond capacitances and on-site energy from the bulk capacitance matrix, build the
// 2-band SSH Bloch model, and compute the Wilson-loop Zak phase for both unit-cell
// conventions. The strong bond (gap 0.7) must couple much more strongly than the weak
// bond (gap 1.3) -- the capacitance decay -- and the two Zak phases must differ by pi.
TEST(QuasiPeriodicStaticGreen, InterfaceModeIsTopological) {
    const int N = 14, half = 6;
    const double r = 0.3, alpha = 0.6 * M_PI;

    std::vector<double> ys;
    for (int n = 1; n <= half; ++n) {
        ys.push_back(2.0 * n - 1.65);  ys.push_back(2.0 * n - 0.35);
        ys.push_back(-2.0 * n + 0.65); ys.push_back(-2.0 * n + 1.35);
    }
    std::sort(ys.begin(), ys.end());
    const int nd = static_cast<int>(ys.size());
    BoundaryMesh mesh(N);
    mesh.generate_circle(r, Vector2d(0.0, ys[0]));
    for (int n = 1; n < nd; ++n) { BoundaryMesh d(N); d.generate_circle(r, Vector2d(0.0, ys[n])); mesh.add_mesh(d); }
    SpectralOperators ops(mesh);
    MatrixXcd S, C;
    ops.QuasiPeriodicStaticS(S, alpha, 1.0, 100);
    S = 0.5 * (S + S.adjoint().eval());
    ops.makeCapacitanceMatrix(C, S);
    C = 0.5 * (C + C.adjoint().eval());

    // bulk dimer in the upper half: strong bond (gap 0.7) then weak bond (gap 1.3)
    int n0 = -1;
    for (int n = nd / 2 + 2; n + 2 < nd; ++n)
        if (std::abs((ys[n+1]-ys[n]) - 0.7) < 0.05 && std::abs((ys[n+2]-ys[n+1]) - 1.3) < 0.05) { n0 = n; break; }
    ASSERT_GE(n0, 0);
    const double eps = 0.5 * (C(n0, n0).real() + C(n0 + 1, n0 + 1).real());
    const cpxd t_strong = C(n0, n0 + 1), t_weak = C(n0 + 1, n0 + 2);

    EXPECT_GT(std::abs(t_strong), 2.0 * std::abs(t_weak));  // capacitance decays with bond length

    auto zak = [&](cpxd t_intra, cpxd t_inter) {
        const int nb = 400;
        std::vector<Vector2cd> u(nb);
        for (int i = 0; i < nb; ++i) {
            double b = 2.0 * M_PI * i / nb;
            cpxd f = t_intra + t_inter * std::exp(cpxd(0, -1) * b);
            Matrix2cd h; h << cpxd(eps,0), f, std::conj(f), cpxd(eps,0);
            SelfAdjointEigenSolver<Matrix2cd> es(h);
            u[i] = es.eigenvectors().col(0);
        }
        cpxd prod(1, 0);
        for (int i = 0; i < nb; ++i) prod *= u[i].dot(u[(i + 1) % nb]);
        double z = -std::arg(prod); if (z < 0) z += 2 * M_PI; return z;
    };
    double z_strong = zak(t_strong, t_weak);  // strong bond as intra-cell
    double z_weak = zak(t_weak, t_strong);    // weak bond as intra-cell
    // each is quantized to {0, pi}; the difference is pi (topological distinction)
    EXPECT_NEAR(std::min(z_weak, 2 * M_PI - z_weak), M_PI, 1e-2);  // weak-intra is topological (pi)
    EXPECT_LT(std::min(z_strong, 2 * M_PI - z_strong), 1e-2);      // strong-intra is trivial (0)
}

// The multipole defect operator M^epsilon must be convergent in its two truncations
// (Brillouin-zone Gauss-Legendre quadrature N_gauss and multipole order N_multipole)
// for any candidate defect frequency to be trustworthy. We check that sigma_min(M) at a
// fixed (omega, alpha_x) is stable when both are refined -- the convergence half of the
// verification protocol behind the higher-frequency defect-band scan.
TEST(MultipoleDefectOperator, ConvergesUnderRefinement) {
    const double R = 0.05, Rd = 0.8 * R, Rp = 0.97 * R, delta = 2e-4;
    const cpxd v = 1.0, vb = 1.0, vbd = 1.0;
    const cpxd omega(4.2, 1e-3);
    const double alpha_x = 0.5 * M_PI;
    auto sigma_min = [](const MatrixXcd& A) {
        return Eigen::JacobiSVD<MatrixXcd>(A).singularValues().tail<1>()(0);
    };
    Multipole base(8, 35, R, Rp, Rd, v, vb, vbd, delta);
    Multipole fine_bz(8, 50, R, Rp, Rd, v, vb, vbd, delta);
    Multipole fine_N(12, 35, R, Rp, Rd, v, vb, vbd, delta);
    double s0 = sigma_min(base.crystal_line_M_operator(omega, alpha_x));
    double s_bz = sigma_min(fine_bz.crystal_line_M_operator(omega, alpha_x));
    double s_N = sigma_min(fine_N.crystal_line_M_operator(omega, alpha_x));
    EXPECT_TRUE(std::isfinite(s0));
    EXPECT_NEAR(s0, s_bz, 3e-3);  // converged in BZ quadrature (N_gauss 35 vs 50)
    EXPECT_NEAR(s0, s_N, 1e-6);   // converged in multipole order (N 8 vs 12)
}

// Full-wave cross-check: the topological interface mode of the capacitance model
// survives in the finite-frequency Helmholtz scattering problem. For a small free-space
// SSH cluster we take the interface eigenvalue of the static capacitance matrix, predict
// omega = mu_1 sqrt(delta * lambda_iface), and confirm the full-wave system operator
// A(omega) has a sharp resonance (sigma_min dip) near that frequency, far below its
// off-resonance value.
TEST(QuasiPeriodicStaticGreen, FullWaveResonanceNearCapacitancePrediction) {
    const int N = 10, half = 3;
    const double r = 0.3, delta = 1e-3, vb = 1.0;
    std::vector<double> ys;
    for (int n = 1; n <= half; ++n) {
        ys.push_back(2.0 * n - 1.65);  ys.push_back(2.0 * n - 0.35);
        ys.push_back(-2.0 * n + 0.65); ys.push_back(-2.0 * n + 1.35);
    }
    std::sort(ys.begin(), ys.end());
    const int nd = static_cast<int>(ys.size());
    BoundaryMesh mesh(N);
    mesh.generate_circle(r, Vector2d(0.0, ys[0]));
    for (int n = 1; n < nd; ++n) { BoundaryMesh d(N); d.generate_circle(r, Vector2d(0.0, ys[n])); mesh.add_mesh(d); }
    SpectralOperators ops(mesh);

    MatrixXcd S0, C;
    ops.S(S0, cpxd(0.0, 0.0));
    ops.makeCapacitanceMatrix(C, S0);
    C = 0.5 * (C + C.adjoint().eval());
    SelfAdjointEigenSolver<MatrixXcd> es(C);
    std::vector<int> by_abs(nd); std::iota(by_abs.begin(), by_abs.end(), 0);
    std::sort(by_abs.begin(), by_abs.end(), [&](int a, int b){ return std::abs(ys[a]) < std::abs(ys[b]); });
    int iface = 0; double best = -1;
    for (int m = 0; m < nd; ++m) {
        VectorXd w = es.eigenvectors().col(m).cwiseAbs2(); w /= w.sum();
        double c = w(by_abs[0]) + w(by_abs[1]);
        if (c > best) { best = c; iface = m; }
    }
    const double lam_iface = es.eigenvalues()(iface) / (M_PI * r * r);
    const double omega_pred = vb * std::sqrt(delta * lam_iface);

    auto smin = [&](double omega) {
        MatrixXcd A; ops.A(A, omega / vb, omega / vb, delta);
        return Eigen::BDCSVD<MatrixXcd>(A).singularValues().tail<1>()(0);
    };
    double on_res = 1e300;
    for (int i = 0; i < 9; ++i) on_res = std::min(on_res, smin((0.8 + 0.5 * i / 8.0) * omega_pred));
    const double off_res = smin(0.3 * omega_pred);
    EXPECT_LT(on_res, 1e-2);              // a sharp resonance near the prediction
    EXPECT_LT(on_res, 0.1 * off_res);     // far deeper than off-resonance
}

// The dynamic 1D-quasi-periodic operators must be (i) finite everywhere (no H0(0) blow-up
// on the diagonal), (ii) translation invariant in y (no spurious Dirichlet image tying them
// to the y=0 axis), and (iii) spectrally convergent. We check a disk placed at several y
// offsets gives identical operators, and that a fixed K* eigenvalue is N-converged.
TEST(PeriodicOperators, FiniteYInvariantConvergent) {
    const double a = 1.0, kbar = 1.3;
    const cpxd k(0.6, 0.0);

    auto build = [&](int N, double yc, MatrixXcd& S, MatrixXcd& K) {
        BoundaryMesh mesh(N); mesh.generate_circle(0.3, Vector2d(0.0, yc));
        SpectralOperators ops(mesh);
        ops.PeriodicS(S, k, kbar, a);
        ops.PeriodicKstar(K, k, kbar, a);
    };

    // (i) finite, including far from the y=0 axis (the old Dirichlet self-term overflowed there)
    MatrixXcd S1, K1, S2, K2;
    build(32, 0.5, S1, K1);
    build(32, 9.5, S2, K2);
    EXPECT_TRUE(S1.allFinite() && K1.allFinite());
    EXPECT_TRUE(S2.allFinite() && K2.allFinite());

    // (ii) y-translation invariance: the operators must not depend on the absolute y-position
    EXPECT_LT((S1 - S2).norm() / S1.norm(), 1e-10);
    EXPECT_LT((K1 - K2).norm() / K1.norm(), 1e-10);

    // (iii) spectral convergence: a fixed K* eigenvalue stabilizes (geometric in N)
    auto kstar_eig_near_half = [&](int N) {
        MatrixXcd S, K; build(N, 0.5, S, K);
        ComplexEigenSolver<MatrixXcd> es(K);
        auto ev = es.eigenvalues();
        cpxd best; double bd = 1e9;
        for (int i = 0; i < ev.size(); ++i) { double d = std::abs(ev(i) - cpxd(0.5, 0)); if (d < bd) { bd = d; best = ev(i); } }
        return best;
    };
    cpxd e48 = kstar_eig_near_half(48), e96 = kstar_eig_near_half(96);
    EXPECT_LT(std::abs(e48 - e96), 1e-6);  // converged to ~6 digits by N=48
}

TEST(SpectralEigenvaluesKstar, Value) {
    MatrixXcd Kstar, Kstar_periodic;
    double truth = 0.5;

    int N = 20;
    BoundaryMesh mesh(N);
    mesh.generate_circle(1.0, Vector2d(0.0, 3.0));

    SpectralOperators ops(mesh);
    ops.Kstar_diagonal(Kstar, 0.0);
    ops.DirPeriodicKstar_diagonal(Kstar_periodic, 0.0, 0.0, 3.6);

    cpxd eigenval = Kstar.eigenvalues()(N - 1);
    cpxd eigenval_periodic = Kstar_periodic.eigenvalues()(N - 1);

    EXPECT_NEAR(abs(eigenval.real() - truth), 0.0, 1e-10);
    EXPECT_NEAR(abs(eigenval_periodic.real() - truth), 0.0, 1e-10);
}

TEST(BiperiodicHelmholtzKernel, Value) {
    Vector2d alpha(0.3, 0.2);
    Vector2d a1(1.0, 0.0);
    Vector2d a2(0.2, 0.5 * sqrt(3.0));
    double h = 0.1;
    cpxd k = 1.0 + 0.2i;
    cpxd G1 = Kernels::helmholtz_2D_biperiodic(k, alpha, Vector2d(0.1, 0.2), Vector2d(0.3, 0.4), a1, a2);
    auto gradG1 = Kernels::grad_helmholtz_2D_biperiodic(k, alpha, Vector2d(0.1, 0.2), Vector2d(0.3, 0.4), a1, a2);

    for (int N : {4, 6, 8, 10, 12, 16, 20}) {
        auto G = Kernels::grad_helmholtz_2D_biperiodic(k, alpha, Vector2d(0.1, 0.2), Vector2d(0.3, 0.4), a1, a2, 0.0, N);
        cout << N << " " << G << endl;
    }

    cpxd G2 = 0.;
    Vector2cd gradG2 = Vector2cd::Zero();
    for (int i = -100; i < 100; i++) {
        for (int j = -100; j < 100; j++) {
            Vector2d shift = a1 * double(i) + a2 * double(j);
            G2 += exp(-cpxd(0,1.) * alpha.dot(shift)) * Kernels::helmholtz_2D(k, Vector2d(0.1, 0.2) - shift, Vector2d(0.3, 0.4));
            gradG2 += exp(-cpxd(0,1.) * alpha.dot(shift)) * Kernels::grad_helmholtz_2D(k, Vector2d(0.1, 0.2) - shift, Vector2d(0.3, 0.4));
        }
        if (i % 100 == 0) {
            cout << "i: " << i << ", partial G: " << G2 << endl;
        }
    }

    EXPECT_NEAR(abs(G1 - G2), 0.0, 1e-6);
    EXPECT_NEAR((gradG1 - gradG2).norm(), 0.0, 1e-6);
}

TEST(QFunctionTest, MatchesLintonDualRepresentationForSignedOrders) {
    QFunction::Options options;
    options.reciprocal_cutoff = 10;
    options.spatial_cutoff = 10;

    const double k = 0.85;
    const Vector2d alpha(0.72, 1.04);
    const double r = 0.37;

    for (int n : {-3, -2, -1, 0, 1, 2, 3}) {
        const cpxd ewald = QFunction::q_function(n, k, alpha, options);
        const cpxd dual_reference = q_function_dual_y_reference(n, k, alpha, 240, r);

        EXPECT_NEAR(std::abs(ewald - dual_reference), 0.0, 2e-3);
    }
}

TEST(QFunctionTest, ComplexWavenumberMatchesDirectDefinition) {
    QFunction::Options options;
    options.reciprocal_cutoff = 12;
    options.spatial_cutoff = 12;
    options.spatial_series_terms = 100;

    const cpxd k(0.9, 0.6);
    const Vector2d alpha(0.4, -0.2);

    for (int n : {-2, -1, 0, 1, 2}) {
        const cpxd ewald = QFunction::q_function(n, k, alpha, options);
        const cpxd direct_reference = q_function_direct_reference(n, k, alpha, 70);

        EXPECT_NEAR(std::abs(ewald - direct_reference), 0.0, 2e-7);
    }
}

TEST(QFunctionTest, RealAndComplexOverloadsAgree) {
    QFunction::Options options;
    options.reciprocal_cutoff = 10;
    options.spatial_cutoff = 10;

    const double k = 0.85;
    const Vector2d alpha(0.72, 1.04);

    for (int n : {-3, -2, -1, 0, 1, 2, 3}) {
        const cpxd real_overload = QFunction::q_function(n, k, alpha, options);
        const cpxd complex_overload = QFunction::q_function(n, cpxd(k, 0.0), alpha, options);

        EXPECT_NEAR(std::abs(real_overload - complex_overload), 0.0, 1e-13);
    }
}

TEST(QFunctionTest, RespectsBlochSymmetryAndPeriodicity) {
    QFunction::Options options;
    options.reciprocal_cutoff = 9;
    options.spatial_cutoff = 9;

    const double k = 0.9;
    const Vector2d alpha(0.37, 1.21);
    const Vector2d shifted = alpha + 2.0 * M_PI * Vector2d(2.0, -1.0);

    for (int n : {-3, -2, -1, 0, 1, 2, 3}) {
        const cpxd value = QFunction::q_function(n, k, alpha, options);
        const cpxd neg_alpha_value = QFunction::q_function(n, k, -alpha, options);
        const cpxd shifted_value = QFunction::q_function(n, k, shifted, options);
        const double sign = (std::abs(n) % 2 == 0) ? 1.0 : -1.0;

        EXPECT_NEAR(std::abs(value - sign * neg_alpha_value), 0.0, 1e-10);
        EXPECT_NEAR(std::abs(value - shifted_value), 0.0, 1e-10);
    }
}

TEST(QFunctionTest, StableAcrossEwaldParameters) {
    QFunction::Options reference_options;
    reference_options.reciprocal_cutoff = 10;
    reference_options.spatial_cutoff = 10;
    reference_options.eta = std::sqrt(M_PI);

    QFunction::Options shifted_options;
    shifted_options.reciprocal_cutoff = 12;
    shifted_options.spatial_cutoff = 12;
    shifted_options.eta = 1.35;

    const double k = 1.05;
    const Vector2d alpha(0.6, -0.4);

    for (int n : {-2, -1, 0, 1, 2}) {
        const cpxd reference = QFunction::q_function(n, k, alpha, reference_options);
        const cpxd shifted = QFunction::q_function(n, k, alpha, shifted_options);

        EXPECT_NEAR(std::abs(reference - shifted), 0.0, 1e-10);
    }
}

TEST(QFunctionTest, GivesFiniteValuesNearSmallWavenumber) {
    QFunction::Options options;
    options.reciprocal_cutoff = 10;
    options.spatial_cutoff = 10;

    const double k = 0.08;
    const Vector2d alpha(0.5, -0.25);

    for (int n : {-2, -1, 0, 1, 2}) {
        const cpxd value = QFunction::q_function(n, k, alpha, options);

        EXPECT_TRUE(std::isfinite(value.real()));
        EXPECT_TRUE(std::isfinite(value.imag()));
    }
}

TEST(MultipoleCrystalATest, MatchesBoundaryIntegralProjectionForCircle) {
    const int order_cutoff = 2;
    const int point_count = 96;
    const double radius = 0.16;
    const cpxd k(0.75, 0.08);
    const cpxd k_b(1.35, 0.04);
    const Vector2d alpha(0.35, -0.25);
    const double delta = 0.3;

    MatrixXcd multipole;
    Utils::multipole_crystal_A(multipole, order_cutoff, radius, k, k_b, alpha, delta);

    BoundaryMesh mesh(point_count);
    mesh.generate_circle(radius);
    SpectralOperators ops(mesh);
    MatrixXcd nodal;
    ops.CrystalA(nodal, k, k_b, alpha, Vector2d(1.0, 0.0), Vector2d(0.0, 1.0), delta);
    const MatrixXcd projected = project_crystal_matrix_to_fourier(nodal, point_count, order_cutoff);

    EXPECT_LT((multipole - projected).norm() / projected.norm(), 1e-10);
}

TEST(MultipoleDefectATest, LocalMapsMatchEffectiveSourceFormula) {
    const int order_cutoff = 3;
    const double radius = 0.22;
    const double defect_radius = 0.18;
    const cpxd k(0.72, 0.04);
    const cpxd k_b(1.08, 0.03);
    // A defect interior wavenumber distinct from k_b: the defect blocks (A_{D_d} and the interior
    // block of P1) must follow k_bd while A_D and everything exterior stay at k_b / k.
    const cpxd k_bd(0.91, 0.05);
    const Vector2d alpha(0.4, -0.35);
    const double delta = 2.0e-4;

    MatrixXcd alpha_operator, local_map, defect_map;
    Utils::multipole_defect_A(alpha_operator, local_map, defect_map, order_cutoff,
                              radius, defect_radius, k, k_b, k_bd, alpha, delta);

    MatrixXcd expected_alpha_operator;
    Utils::multipole_crystal_A(expected_alpha_operator, order_cutoff, radius, k, k_b, alpha, delta);
    const MatrixXcd expected_local =
            local_effective_source_operator(order_cutoff, radius, k, k_b, delta);
    const MatrixXcd defect_local =
            local_effective_source_operator(order_cutoff, defect_radius, k, k_bd, delta);
    const MatrixXcd p1 = defect_projector1(order_cutoff, radius, defect_radius, k, k_bd);
    const MatrixXcd p2 = defect_projector2(order_cutoff, radius, defect_radius, k);
    const MatrixXcd expected_defect = p2.partialPivLu().solve(defect_local * p1);

    EXPECT_LT((alpha_operator - expected_alpha_operator).norm() / expected_alpha_operator.norm(), 1e-13);
    EXPECT_LT((local_map - expected_local).norm() / expected_local.norm(), 1e-13);
    EXPECT_LT((defect_map - expected_defect).norm() / expected_defect.norm(), 1e-13);

    // Same-material sanity: with k_bd == k_b the operator must reduce to the old behavior.
    MatrixXcd alpha_operator_same, local_map_same, defect_map_same;
    Utils::multipole_defect_A(alpha_operator_same, local_map_same, defect_map_same, order_cutoff,
                              radius, defect_radius, k, k_b, k_b, alpha, delta);
    const MatrixXcd defect_local_same =
            local_effective_source_operator(order_cutoff, defect_radius, k, k_b, delta);
    const MatrixXcd p1_same = defect_projector1(order_cutoff, radius, defect_radius, k, k_b);
    const MatrixXcd expected_defect_same = p2.partialPivLu().solve(defect_local_same * p1_same);
    EXPECT_LT((defect_map_same - expected_defect_same).norm() / expected_defect_same.norm(), 1e-13);
}

// Validate the frequency-dependent capacitance matrix used by run_defect_capacitance_bands
// (Ammari, Li, Liu, Shao, Uhlmann, arXiv:2605.27572v1, Definition 3.9 / eq. (4.11)):
//   Lambda_ext = (1/2 I + K_D^*) S_D^{-1},  C^reg = -(v_b^2 / 2 omega_0) <Lambda_ext g_q, g_p>,
// against the closed-form dipole Fabry-Perot eigenvalue of a SINGLE isolated disk. The exact
// single-disk dipole resonance solves  delta H_1'(omega R)/H_1(omega R) = J_1'(omega R)/J_1(omega R);
// linearising at the Neumann base frequency omega_0 = j'_{1,1}/R (where J_1'(omega_0 R)=0) and using
// Bessel's equation  J_1''(beta) = -(1 - 1/beta^2) J_1(beta)  at beta = j'_{1,1} gives the leading
// eigenvalue lambda = -(1/(R(1-1/beta^2))) H_1'(omega_0 R)/H_1(omega_0 R), i.e. omega = omega_0 + delta*lambda.
TEST(CapacitanceMatrix, SingleDiskDipoleMatchesAnalytic) {
    const double R = 0.3;
    const int N = 256;
    const int m_ang = 1;
    const double beta = 1.8411837813;     // j'_{1,1}
    const double omega0 = beta / R;        // v_b = 1
    const cpxd k = omega0;                 // v = 1

    BoundaryMesh mesh(N);
    mesh.generate_circle(R, Vector2d(0.0, 0.0));
    SpectralOperators ops(mesh);
    const int Ntot = mesh.get_num_segments();

    auto h1p = [](int n, cpxd z) { return 0.5 * (bessel::cyl_h1(n - 1, z) - bessel::cyl_h1(n + 1, z)); };

    // L^2(D)-normalised dipole Neumann eigenmode traces g_+ ~ e^{i theta}, g_- ~ e^{-i theta}.
    const double A = 1.0 / (std::sqrt(M_PI) * R * std::sqrt(1.0 - double(m_ang * m_ang) / (beta * beta)));
    MatrixXcd G = MatrixXcd::Zero(Ntot, 2);
    VectorXd sigma(Ntot);
    for (int i = 0; i < Ntot; ++i) sigma(i) = mesh.get_vertex(i).sigma;
    for (int loc = 0; loc < N; ++loc) {
        const double theta = 2.0 * M_PI * loc / N;
        G(loc, 0) = A * std::exp(cpxd(0, 1.0) * (double) m_ang * theta);
        G(loc, 1) = A * std::exp(cpxd(0, 1.0) * (-(double) m_ang) * theta);
    }

    MatrixXcd S, Kstar;
    ops.S(S, k);
    ops.Kstar(Kstar, k);
    const MatrixXcd X = S.partialPivLu().solve(G);
    const MatrixXcd LamG = 0.5 * X + Kstar * X;                          // Lambda_ext G
    const MatrixXcd Creg = -(1.0 / (2.0 * omega0)) * (G.adjoint() * (sigma.asDiagonal() * LamG));

    ComplexEigenSolver<MatrixXcd> es(Creg);
    const VectorXcd lam = es.eigenvalues();

    const cpxd lam_exact =
            -(1.0 / (R * (1.0 - 1.0 / (beta * beta)))) * h1p(1, k * R) / bessel::cyl_h1(1, k * R);

    // both dipole eigenvalues equal the analytic value (rotational symmetry -> degenerate)
    const double tol = 0.02 * std::abs(lam_exact);
    for (int j = 0; j < 2; ++j) {
        EXPECT_NEAR(lam(j).real(), lam_exact.real(), tol);
        EXPECT_NEAR(lam(j).imag(), lam_exact.imag(), tol);
    }
}

TEST(BentWaveguidePatch, EndpointDefectsAreScreenersNotMatrixSites) {
    const int n_defect = 3;
    const int fringe = 1;
    const auto p = workflows::build_bent_patch(0.35, 0.455, 1, n_defect, 2, fringe, 16,
                                               /*v=*/1.0, /*v_b=*/1.0, /*v_bd=*/1.0,
                                               /*verbose=*/false);

    ASSERT_EQ(p.L, n_defect + fringe - 1);
    ASSERT_EQ(p.L_main, n_defect - 1);
    ASSERT_EQ(static_cast<int>(p.defect_index.size()), 2 * p.L + 1);
    ASSERT_EQ(static_cast<int>(p.main_indices.size()), 2 * p.L_main + 1);
    ASSERT_EQ(p.center, p.L_main);
    EXPECT_EQ(p.defect_index.count(workflows::BentPatch::site_at(-p.L)), 1);
    EXPECT_EQ(p.defect_index.count(workflows::BentPatch::site_at(p.L)), 1);
    EXPECT_EQ(std::find(p.main_indices.begin(), p.main_indices.end(),
                        p.defect_index.at(workflows::BentPatch::site_at(-p.L))),
              p.main_indices.end())
        << "negative endpoint is still a projected matrix site";
    EXPECT_EQ(std::find(p.main_indices.begin(), p.main_indices.end(),
                        p.defect_index.at(workflows::BentPatch::site_at(p.L))),
              p.main_indices.end())
        << "positive endpoint is still a projected matrix site";

    const int modes = p.modes;
    const int n_main = static_cast<int>(p.main_indices.size());
    double row_sym = 0.0;
    double row_scale = 0.0;
    for (int q = 0; q < n_main; ++q) {
        const int r = 2 * p.center - q;
        ASSERT_GE(r, 0);
        ASSERT_LT(r, n_main);
        const double left = p.C.block(modes * p.center, modes * q, modes, modes).norm();
        const double right = p.C.block(modes * p.center, modes * r, modes, modes).norm();
        row_sym = std::max(row_sym, std::abs(left - right));
        row_scale = std::max(row_scale, std::max(left, right));
    }
    EXPECT_LT(row_sym / row_scale, 1e-10);
}

// erfc_complex must satisfy the reflection identity erfc(z) + erfc(-z) = 2 everywhere. Before the
// reflection was applied for Re(z) < 0, arguments with Re(z) < 0 and |z| > 8 lost the dominant
// term (erfc(-large) ~ 2 came out as ~0), which annihilated the propagating far-field of the
// periodic kernels for any Im(k) != 0 (the Ewald spectral terms call erfc at -eps*dz +/- i kz/2eps).
TEST(ErfcComplex, ReflectionIdentityHoldsForLargeArguments) {
    const std::vector<cpxd> zs = {
        {10.46, 0.78}, {10.46, -0.78}, {0.4, 6.0}, {12.0, 1.4}, {3.0, 0.1}, {25.0, 1.0}};
    for (const cpxd& z : zs) {
        const cpxd sum = Tools::erfc_complex(z) + Tools::erfc_complex(-z);
        EXPECT_NEAR(abs(sum - 2.0), 0.0, 1e-12) << "z = " << z;
    }
    // spot value against the real erfc
    EXPECT_NEAR(abs(Tools::erfc_complex(cpxd(-10.5, 0.0)) - cpxd(std::erfc(-10.5), 0.0)), 0.0, 1e-12);
}

// The quasi-periodic Helmholtz kernel must be ANALYTIC across the real k-axis (one function,
// analytically continued), not conjugate-reflected: resonance root searches step across the axis
// and a branch jump there traps them in limit cycles. Check that a small imaginary step +/-ih
// around a real k changes the kernel by O(h) (analytic), for a far pair where the propagating
// (open-order) terms dominate -- exactly the terms whose branch used to flip with sign(Im k).
TEST(PeriodicHelmholtzKernel, AnalyticAcrossRealAxis) {
    const double kbar = M_PI;
    const Vector2d x(0.3, -3.7), y(0.1, 2.2);
    const double h = 1e-3;
    const cpxd k0(4.1750, 0.0);
    const cpxd g0 = Kernels::helmholtz_2D_periodic(k0, kbar, x, y, 1.0);
    const cpxd gp = Kernels::helmholtz_2D_periodic(k0 + cpxd(0, h), kbar, x, y, 1.0);
    const cpxd gm = Kernels::helmholtz_2D_periodic(k0 - cpxd(0, h), kbar, x, y, 1.0);
    ASSERT_GT(abs(g0), 0.1);                       // the propagating far-field must be present
    EXPECT_NEAR(abs(gp - g0), 0.0, 10.0 * h * abs(g0)) << "upper step not O(h)";
    EXPECT_NEAR(abs(gm - g0), 0.0, 10.0 * h * abs(g0)) << "lower step not O(h)";
    // second difference: for an analytic function (gp + gm - 2 g0) is O(h^2), for a conjugate
    // reflection it is O(Im g0) which is O(1) here
    EXPECT_NEAR(abs(gp + gm - 2.0 * g0), 0.0, 100.0 * h * h * abs(g0));
    // same for the gradient kernel
    const Vector2cd d0 = Kernels::grad_helmholtz_2D_periodic(k0, kbar, x, y, 1.0);
    const Vector2cd dp = Kernels::grad_helmholtz_2D_periodic(k0 + cpxd(0, h), kbar, x, y, 1.0);
    const Vector2cd dm = Kernels::grad_helmholtz_2D_periodic(k0 - cpxd(0, h), kbar, x, y, 1.0);
    EXPECT_NEAR((dp + dm - 2.0 * d0).norm(), 0.0, 100.0 * h * h * d0.norm());
}
