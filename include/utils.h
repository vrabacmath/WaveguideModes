//
// Created by lara on 11/24/25.
//

#ifndef SUBWAVELENGTHRESONATORS_UTILS_H
#define SUBWAVELENGTHRESONATORS_UTILS_H

#include "spectral_operators.h"
#include "tools.h"
#include <chrono>
#include "bessel-library.hpp"
#include "q_function.h"

using namespace std::chrono;
using namespace std;

namespace ConvergenceTests{
    inline void kernel_convergence_test() {
        cout << "Kernel convergence test:\n"; //--------------------------------
        vector<double> N_values;
        vector<double> errors;
        vector<double> grad_errors;

        double k = 10.0;
        double kbar = 2.0;
        Vector2d x(0.1, 0.2);
        Vector2d y(0.3, 0.4);
        double period = 0.4;

        cpxd G_ref = Kernels::dir_helmholtz_2D_periodic(k, kbar, x, y, period, 0.0, 50);
        auto grad_G_ref = Kernels::dir_grad_helmholtz_2D_periodic(k, kbar, x, y, period, 0.0, 50);
        cout << "Reference G: " << G_ref << endl;
        cout << "Reference grad G: " << grad_G_ref.transpose() << endl;

        for (int N = 1; N <= 30; N++) {
            cpxd G_approx = Kernels::dir_helmholtz_2D_periodic(k, kbar, x, y, period, 0.0, N);
            auto grad_G_approx = Kernels::dir_grad_helmholtz_2D_periodic(k, kbar, x, y, period, 0.0, N);

            double error = abs(G_approx - G_ref);
            N_values.push_back(N);
            errors.push_back(error);

            double grad_error = (grad_G_approx - grad_G_ref).norm();
            grad_errors.push_back(grad_error);

            cout << "N = " << N << ", G_approx = " << G_approx << ", error = " << error << endl;
            cout << "N = " << N << ", grad_G_approx = " << grad_G_approx.transpose()
                 << ", grad_error = " << grad_error << endl;
        }

        Tools::dump_csv("convergence_kernel.csv",
                        {N_values, errors, grad_errors});
    }

    inline void rayleigh_and_FD_test() {
        vector<double> fd_test;
        vector<double> ry_test;
        for (int i = 0; i < 32; i++) {
            double h = 0.1 * pow(0.5, i);
            cpxd dGx = (Kernels::helmholtz_2D_periodic(1.0 + 1i, 0.1, Vector2d(0.1 + h, 0.2), Vector2d(0.3, 0.4), 1.0) -
                        Kernels::helmholtz_2D_periodic(1.0 + 1i, 0.1, Vector2d(0.1 - h, 0.2), Vector2d(0.3, 0.4),
                                                       1.0)) / (2. * h);
            cpxd dGy = (Kernels::helmholtz_2D_periodic(1.0 + 1i, 0.1, Vector2d(0.1, 0.2 + h), Vector2d(0.3, 0.4), 1.0) -
                        Kernels::helmholtz_2D_periodic(1.0 + 1i, 0.1, Vector2d(0.1, 0.2 - h), Vector2d(0.3, 0.4),
                                                       1.0)) / (2. * h);

            Vector2cd gradG = Kernels::grad_helmholtz_2D_periodic(1.0 + 1i, 0.1, Vector2d(0.1, 0.2), Vector2d(0.3, 0.4),
                                                                  1.0);

//            cout << dGx << endl;
//            cout << dGy << endl;
//            cout << gradG << endl;

            fd_test.push_back((Vector2cd(dGx, dGy) - gradG).norm());
        }

        for (int i = 0; i < 20; i++) {
            int N = pow(2, i);
            double period = 1.5;
            double kbar = 1.0; cpxd k = 0.5 + 0.7; cpxd kd = sqrt(k * k - kbar * kbar);
//    ctrl = cpxd(0, -0.5) * exp(+cpxd(0, kbar) * 0.3 + cpxd(0, 1.) * kd * 0.3) / kd / period;
            cpxd ctrl = 0.0;

            for (int n = -N; n <= N; n++) {
                Vector2d x1(0, 0);
                Vector2d x2(0.3 + period * n, 0.3);

                double l = 2. * M_PI * (double)n / period;
                cpxd denom = sqrt(-(l + kbar) * (l + kbar) + k * k);
//        if (n != 0) ctrl -= 0.5 / period * exp(+cpxd(0, kbar) * 0.3 -
//                            denom * 0.3 - cpxd(0, 1.) * l * 0.3)/ denom;
                ctrl += exp(cpxd(0, kbar + l)*0.3 + cpxd(0, 1.) * denom * 0.3)
                        / (cpxd(0, 2.) * denom * period);
            }

            ry_test.push_back(abs(ctrl - Kernels::helmholtz_2D_periodic(k, kbar, Vector2d(0., 0.), Vector2d(0.3, 0.3), period)));
//            cout << ry_test[i] << endl;
        }
        Tools::dump_csv("FD_test_gradG.csv", {fd_test});
        Tools::dump_csv("test_rayleigh.csv", {ry_test});
    }

    inline void direct_sum_vs_ewald_S() {
        BoundaryMesh mesh(16);
        BoundaryMesh *local_mesh = &mesh;
        mesh.generate_circle(0.001, Vector2d(0.0, 0.002));

        int start_idx = 0;
        int end_idx = mesh.get_num_segments() - 1;
        int N = end_idx - start_idx + 1;

        MatrixXcd S(N,N), S2(N,N);

        VectorXcd R_hat_j_Kress = VectorXcd::Zero(N);
        R_hat_j_Kress(0) = 0.0;
        R_hat_j_Kress(N / 2) = -4.0 / N;
        for (int j = 1; j < N / 2; j++) {
            R_hat_j_Kress(j) = -2.0 / j;
            R_hat_j_Kress(N - j) = R_hat_j_Kress(j);
        }

        // Inverse FFT to get R_j in physical space
        VectorXcd R_j_Kress(N);
        Eigen::FFT<double> fft;
        R_j_Kress.setZero();
        fft.inv(R_j_Kress, R_hat_j_Kress);

        double k = 3.0, a = 0.01, kbar = 0.0;
        cpxd C = 0; double epsilon = sqrt(M_PI) / a;
        cpxd ratio = k / (2.0 * epsilon);
        for (int q = 1; q <= 20; q++) {
            C += pow(ratio, 2 * q) / (tgamma(q + 1) * q);
        }
        C *= M_1_PI * 0.25;

#pragma omp parallel for schedule(static) default(none) shared(S, S2, R_j_Kress, local_mesh, N, start_idx, end_idx, k, kbar, a, C)
        for (int i = start_idx; i <= end_idx; i++) {
            Vector2d point_i = local_mesh->get_vertex(i).point;
            Vector2d normal_i = local_mesh->get_vertex(i).normal;
            for (int j = start_idx; j <= end_idx; j++) {
                Vector2d point_j = local_mesh->get_vertex(j).point;
                double rij = (point_i - point_j).norm();

                if (i != j) {
                    cpxd alpha = 0.25 * M_1_PI * bessel::cyl_j(0, k * rij);
                    S(i, j) = alpha * local_mesh->get_vertex(j).tnorm * R_j_Kress((i - j + N) % N);
                    double sinterm = sin(M_PI * (i - j) / N);
                    S(i, j) += local_mesh->get_vertex(j).sigma *
                               (Kernels::dir_helmholtz_2D_periodic(k, kbar, point_i, point_j, a) -
                                alpha * log(4. * sinterm * sinterm));

                    S2(i, j) = alpha * local_mesh->get_vertex(j).tnorm * R_j_Kress((i - j + N) % N);
                    S2(i, j) += local_mesh->get_vertex(j).sigma *
                                (Kernels::sum_dir_helmholtz_2D_periodic(k, kbar, point_i, point_j, a) -
                                 alpha * log(4. * sinterm * sinterm));
                } else {
                    // self-interaction term
                    S(i, i) = 0.25 * M_1_PI * local_mesh->get_vertex(j).tnorm * R_j_Kress(0);
                    S(i, i) += local_mesh->get_vertex(i).sigma * Kernels::dir_helmholtz_2D_periodic(k, kbar, point_i, point_j, a);
                    S(i, i) += local_mesh->get_vertex(i).sigma * 0.25 * M_1_PI * (-C + EULER_GAMMA +
                                                                           log(M_PI / a / a * local_mesh->get_vertex(i).tnorm * local_mesh->get_vertex(i).tnorm));

                    S2(i, i) = 0.25 * M_1_PI * local_mesh->get_vertex(j).tnorm * R_j_Kress(0);
                    S2(i, i) += local_mesh->get_vertex(i).sigma * Kernels::sum_dir_helmholtz_2D_periodic(k, kbar, point_i, point_j, a);
                    S2(i, i) += local_mesh->get_vertex(i).sigma * (cpxd(0, -0.25) + 0.5 * M_1_PI *
                                                                             (EULER_GAMMA + log(0.5 * k * local_mesh->get_vertex(i).tnorm)));
                }
            }
        }

        cout << "Norm difference: " << (S - S2).norm() / S.norm() << endl << endl;
        cout << S - S2 << endl;
    }

    inline void direct_sum_vs_ewald_Kstar() {
        BoundaryMesh mesh(16);
        BoundaryMesh *local_mesh = &mesh;
        mesh.generate_circle(0.001, Vector2d(0.0, 0.002));

        int start_idx = 0;
        int end_idx = mesh.get_num_segments() - 1;
        int N = end_idx - start_idx + 1;

        MatrixXcd Kstar(N,N), Kstar2(N,N);

        VectorXcd R_hat_j_Kress = VectorXcd::Zero(N);
        R_hat_j_Kress(0) = 0.0;
        R_hat_j_Kress(N / 2) = -4.0 / N;
        for (int j = 1; j < N / 2; j++) {
            R_hat_j_Kress(j) = -2.0 / j;
            R_hat_j_Kress(N - j) = R_hat_j_Kress(j);
        }

        // Inverse FFT to get R_j in physical space
        VectorXcd R_j_Kress(N);
        Eigen::FFT<double> fft;
        R_j_Kress.setZero();
        fft.inv(R_j_Kress, R_hat_j_Kress);

        double k = 3.0, a = 0.01, kbar = 0.0;
        cpxd C = 0; double epsilon = sqrt(M_PI) / a;
        cpxd ratio = k / (2.0 * epsilon);
        for (int q = 1; q <= 20; q++) {
            C += pow(ratio, 2 * q) / (tgamma(q + 1) * q);
        }
        C *= M_1_PI * 0.25;

#pragma omp parallel for schedule(static) default(none) shared(Kstar, Kstar2, R_j_Kress, local_mesh, N, start_idx, end_idx, k, kbar, a, C)
        for (int i = start_idx; i <= end_idx; i++) {
            Vector2d point_i = local_mesh->get_vertex(i).point;
            Vector2d normal_i = local_mesh->get_vertex(i).normal;
            for (int j = start_idx; j <= end_idx; j++) {
                Vector2d point_j = local_mesh->get_vertex(j).point;
                double rij = (point_i - point_j).norm();

                if (i != j) {
                    Vector2d point_j = local_mesh->get_vertex(j).point;
                    Vector2d r_ij = point_i - point_j;
                    double r_norm = r_ij.norm();

                    cpxd alpha = k * 0.25 * M_1_PI * bessel::cyl_j(1, k * r_norm)
                                 * r_ij.dot(normal_i) / r_norm;

                    Kstar(i, j) = alpha * local_mesh->get_vertex(j).tnorm * R_j_Kress((i - j + N) % N);

                    double sinterm = sin(M_PI * (i - j) / N);
                    Kstar(i, j) += local_mesh->get_vertex(j).sigma *
                                   (Kernels::dir_grad_helmholtz_2D_periodic(k, kbar, point_i, point_j, a).dot(
                                           normal_i) -
                                    alpha * log(4. * sinterm * sinterm));

                    Kstar2(i, j) = alpha * local_mesh->get_vertex(j).tnorm * R_j_Kress((i - j + N) % N);
                    Kstar2(i, j) += local_mesh->get_vertex(j).sigma *
                                   (Kernels::sum_dir_grad_helmholtz_2D_periodic(k, kbar, point_i, point_j, a).dot(
                                           normal_i) -
                                    alpha * log(4. * sinterm * sinterm));
                } else {
                    // self-interaction is 0 for alpha
                    // the difference of the m = 0 term and the log sin term
                    Kstar(i, i) = 0.25 * M_1_PI
                                  * local_mesh->get_vertex(i).curvature * local_mesh->get_vertex(i).sigma;
                    Kstar(i, i) += local_mesh->get_vertex(i).sigma *
                                   Kernels::dir_grad_helmholtz_2D_periodic(k, kbar, point_i, point_i, a).dot(
                                           normal_i);
                    Kstar2(i, i) = 0.25 * M_1_PI
                                  * local_mesh->get_vertex(i).curvature * local_mesh->get_vertex(i).sigma;
                    Kstar2(i, i) += local_mesh->get_vertex(i).sigma *
                                   Kernels::sum_dir_grad_helmholtz_2D_periodic(k, kbar, point_i, point_i, a).dot(
                                           normal_i);
                }
            }
        }

        cout << "Norm difference: " << (Kstar - Kstar2).norm() / Kstar.norm() << endl << endl;
        cout << Kstar - Kstar2 << endl;
    }

    inline void convergence_with_period() {
        int NN = 100;
        vector<double> errorsK, errorsS;
        for (int n = 0; n < 12; n++) {
            double radius = 0.1 * pow(2.0, -n);

            BoundaryMesh testmesh(NN);
            testmesh.generate_circle(radius, Vector2d(0.0, 0.5));
            SpectralOperators testops(testmesh);
            MatrixXcd testKstar(NN, NN), testKstarperiodic, testS(NN, NN), testSperiodic;
            testops.DirPeriodicKstar(testKstarperiodic, 3.0, 0.0, 2.0);
            testops.DirPeriodicS(testSperiodic, 3.0, 0., 2.0);

//    S(i, j) = -0.25 * cpxd(0, 1.0)
//              * bessel::cyl_h1(0, k * r_norm)
//              * mesh.get_vertex(j).sigma;
//    } else {
//    // self-interaction term
//    S(i, i) = (cpxd(0, -0.25) + 0.5 * M_1_PI * (std::numbers::egamma + log(0.5 * k)
//    + log(0.5 * mesh.get_vertex(i).sigma) - 1.0)) * mesh.get_vertex(i).sigma;
//    }

            for (int i = 0; i < NN; i++) {
                Vector2d point_i = testmesh.get_vertex(i).point;
                Vector2d normal_i = testmesh.get_vertex(i).normal;
                for (int j = 0; j < NN; j++) {
                    if (i == j) {
                        // self-interaction term
                        testKstar(i, i) = testmesh.get_vertex(i).sigma * (0.25 * M_1_PI * testmesh.get_vertex(i).curvature
                                                                          - Kernels::grad_helmholtz_2D(3.0, point_i,
                                                                                                       Vector2d(point_i.x(),
                                                                                                                -point_i.y())).dot(
                                normal_i));
                        testS(i, i) = (cpxd(0, -0.25) + 0.5 * M_1_PI * (EULER_GAMMA + log(0.5 * 0.3)
                                                                        + log(0.5 * testmesh.get_vertex(i).sigma) - 1.0)
                                       - Kernels::helmholtz_2D(3.0, point_i, Vector2d(point_i.x(), -point_i.y()))) *
                                      testmesh.get_vertex(i).sigma;
                        continue;
                    }
                    Vector2d point_j = testmesh.get_vertex(j).point;
                    testKstar(i, j) = Kernels::dir_grad_helmholtz_2D(3.0, point_i, point_j).dot(normal_i)
                                      * testmesh.get_vertex(j).sigma;

                    testS(i, j) = Kernels::dir_helmholtz_2D(3.0, point_i, point_j) * testmesh.get_vertex(j).sigma;
                }
            }

//    cout << testKstar << endl << endl;
//    cout << testKstarperiodic << endl << endl;
//    cout << testKstar - testKstarperiodic << endl;
            cout << "Norm difference: " << (testKstar - testKstarperiodic).norm() << endl;

//    cout << testS << endl << endl;
//    cout << testSperiodic << endl << endl;
//    cout << testS - testSperiodic << endl;
            cout << "Norm difference: " << (testS - testSperiodic).norm() << endl << endl;

            errorsK.push_back((testKstar - testKstarperiodic).norm());
            errorsS.push_back((testS - testSperiodic).norm());
        }
        Tools::dump_csv("periodic_operator_convergence.csv",
                        {errorsK, errorsS});
    }

    inline void s_capmat_test() {
        MatrixXcd Sref, Smat; int N = 200;

        Smat.resize(2*N, 2*N);
        Smat.setZero();
        BoundaryMesh mesh(N); double a = 0.1;
        BoundaryMesh mesh2(N);
        mesh.generate_circle(0.01, Vector2d(0.04, 0.04));
        mesh2.generate_circle(0.01, Vector2d(0.0, 0.04));
        mesh.add_mesh(mesh2);

#pragma omp parallel for schedule(static) default(none) shared(Smat, mesh, N, a)
        for (int i = 0; i < 2*N; i++) {
            Vector2d point_i = mesh.get_vertex(i).point;
            Vector2d normal_i = mesh.get_vertex(i).normal;
            for (int j = 0; j < 2*N; j++) {
                if (i != j) {
                    Vector2d point_j = mesh.get_vertex(j).point;
                    Smat(i, j) = Kernels::dir_laplace_2D_periodic(point_i, point_j, a) * mesh.get_vertex(j).sigma;
                } else {
                    // self-interaction term
                    Smat(i, i) = 0.5 * M_1_PI * (log(mesh.get_vertex(i).sigma * 0.5) - 1.0
                                                 + log(M_PI / a)) * mesh.get_vertex(i).sigma;

                    // don't forget the mirrored part
                    Vector2d point_j = Vector2d(point_i.x(), -point_i.y());
                    Smat(i, j) -= Kernels::laplace_2D_periodic(point_i, point_j, a) * mesh.get_vertex(j).sigma;
                }
            }
        }

        SpectralOperators opsr(mesh);
        opsr.DirPeriodicS(Sref, 0.0, 0.0, 0.1);

        cout << "Difference S - Sref:\n" << Smat - Sref << endl;
        cout << "norm difference: " << (Smat - Sref).norm() << endl;
        cout << Smat.norm() << " " << Sref.norm() << endl;

        MatrixXcd c1, c2;
        opsr.makeCapacitanceMatrix(c1, Sref);
        opsr.makeCapacitanceMatrix(c2, Smat);

        cout << "Difference Capacitance matrices:\n" << c1 + c2 << endl;
        cout << "norm difference: " << (c1 - c2).norm() << endl;
        cout << c1 << endl;
        cout << c2 << endl;
    }

   inline void Kstar_cvg() {
       vector<double> eigenvaluesdb, eigenvaluesdb3;
       MatrixXcd Kstar, Kstar3;

       for (int n = 4; n <= 1024; n+=2) {
           BoundaryMesh dmesh;
           BoundaryMesh mesh(n); mesh.generate_circle(0.02, Vector2d(-0.0215, 0.040));
           BoundaryMesh mesh2(n); mesh2.generate_circle(0.02, Vector2d(0.0215, 0.040));
           BoundaryMesh mesh33(n); mesh33.generate_circle(0.02, Vector2d(0., 0.11));

           dmesh.add_mesh(mesh);
           dmesh.add_mesh(mesh2);
           dmesh.add_mesh(mesh33);

           MatrixXcd Kstar_diagonal;
           SpectralOperators ops(dmesh);
           ops.DirPeriodicKstar(Kstar_diagonal, 0.0, 0.0, 0.14);
           cout << "eigenvalues Kstar double bubble:\n" << Kstar_diagonal.eigenvalues()(2*n - 1) << endl;

           eigenvaluesdb.push_back(abs(Kstar_diagonal.eigenvalues()(2*n - 1)));

           BoundaryMesh dmesh3;
           BoundaryMesh mesh3(n); mesh3.generate_circle(0.02, Vector2d(-0.035, 0.040));
           BoundaryMesh mesh23(n); mesh23.generate_circle(0.02, Vector2d(0.035, 0.040));
           dmesh3.add_mesh(mesh3);
           dmesh3.add_mesh(mesh23);
           dmesh3.add_mesh(mesh33);

           MatrixXcd Kstar3;
           SpectralOperators ops3(dmesh3);
           ops3.DirPeriodicKstar(Kstar3, 0.0, 0.0, 0.14);
//        ops3.Kstar_diagonal(Kstar3, 10.0);
           cout << "eigenvalues Kstar_diagonal double bubble:\n" << Kstar3.eigenvalues()(2*n - 1) << endl;
           eigenvaluesdb3.push_back(abs(Kstar3.eigenvalues()(2*n - 1)));

           if (n == 256) n*=4;
       }

       Tools::dump_csv("eigenvalues_double_bubble2.csv", {eigenvaluesdb, eigenvaluesdb3});
    }
} // namespace ConvergenceTests

namespace Utils {

    inline void draw_kernel() {
        cout << "2D periodic Green's function test:\n"; //--------------------------------

        double period = 1.;
        Vector2d src(0.0, 1.0);
        const int Nx = 121*4, Nz = 161;
        const double Z = 2.0 * period;

        std::vector<std::vector<double>> mag(Nz, std::vector<double>(Nx)),
                realA(Nz, std::vector<double>(Nx)),
                imagA(Nz, std::vector<double>(Nx)),
                phase(Nz, std::vector<double>(Nx));

        Vector2d a1(1., 0.), a2(0.5, -sqrt(3.) * 0.5);

        for (int iz=0; iz<Nz; ++iz) {
            double z = -Z + (2.0*Z) * (double(iz) / (Nz-1));
            for (int ix=0; ix<Nx; ++ix) {
                double x = 4 * period * (double(ix) / (Nx-1)); // [0,d]
                Vector2d r(x, z);
//                auto G = Kernels::dir_helmholtz_2D_periodic(2.0, -1.0, r, src, period, sqrt(M_PI), 10);
                auto G = Kernels::helmholtz_2D_biperiodic(2.0, Vector2d(0.1, 0.1), r, src, a1, a2);
                realA[iz][ix] = std::real(G);
                imagA[iz][ix] = std::imag(G);
                mag[iz][ix]   = std::abs(G);
                phase[iz][ix] = std::atan2(std::imag(G), std::real(G));

//                cout << G * exp(-cpxd(0,1.) * Vector2d(0.1, 0.1).dot(a2 + a1)) << " "
//                << Kernels::helmholtz_2D_biperiodic(2.0, Vector2d(0.1, 0.1), r + a2 + a1, src, a1, a2) << endl;
            }
            Tools::print_stage_progress("Drawing solution rows", iz + 1, Nz);
        }
        Tools::finish_progress_line();

        dump_csv("G_mag.csv",   mag);
        dump_csv("G_real.csv",  realA);
        dump_csv("G_imag.csv",  imagA);
        dump_csv("G_phase.csv", phase);
    }

    inline void draw_solution(VectorXcd solution, cpxd omega, BoundaryMesh &mesh, double period, cpxd k, cpxd k_b) {
        cout << "2D periodic Green's function test:\n"; //--------------------------------

        assert(solution.size() == 2 * mesh.get_num_segments());

        Vector2d src(0.0, 1.0);
        const int Nx = 242, Nz = 242;

        std::vector<std::vector<double>> mag(Nz, std::vector<double>(Nx)),
                realA(Nz, std::vector<double>(Nx)),
                imagA(Nz, std::vector<double>(Nx)),
                phase(Nz, std::vector<double>(Nx));

        for (int iz=0; iz<Nz; ++iz) {
            double z = -period + 2. * period * (double(iz) / (Nz-1));
            for (int ix=0; ix<Nx; ++ix) {
                double x = -period/2. + period * (double(ix) / (Nx-1)); // [0,d]
                Vector2d r(x, z);
                cpxd G = 0.0;
                double Gre = 0.0, Gim = 0.0;
                const int N = mesh.get_num_segments();
                if ((x * x + (z - 3.) * (z - 3.) < 1.) || (x * x + (z - 6.) * (z - 6.) < 1.) || (x * x + (z - 9.) * (z - 9.) < 1.) ||
                        (x * x + (z + 3.) * (z + 3.) < 1.) || (x * x + (z + 6.) * (z + 6.) < 1.) || (x * x + (z + 9.) * (z + 9.) < 1.)) {
#pragma omp parallel for reduction(+:Gre, Gim)
                    for (int j = 0; j < N; j++) {
                        Vector2d r_j = mesh.get_vertex(j).point;
                        cpxd contrib = solution(j) * Kernels::dir_helmholtz_2D_periodic(k_b, 0.0, r, r_j, period) * mesh.get_vertex(j).sigma;
                        Gre += std::real(contrib);
                        Gim += std::imag(contrib);
                    }
                } else {
#pragma omp parallel for reduction(+:Gre, Gim)
                    for (int j = N; j < 2 * N; j++) {
                        int local_idx = j - N;
                        Vector2d r_j = mesh.get_vertex(local_idx).point;
                        cpxd contrib = solution(j) * Kernels::dir_helmholtz_2D_periodic(k, 0.0, r, r_j, period)
                                       * mesh.get_vertex(local_idx).sigma;
                        Gre += std::real(contrib);
                        Gim += std::imag(contrib);
                    }

                }
                G = cpxd(Gre, Gim);
                realA[iz][ix] = std::real(G);
                imagA[iz][ix] = std::imag(G);
                mag[iz][ix]   = std::abs(G);
                phase[iz][ix] = std::atan2(std::imag(G), std::real(G));
            }
            Tools::print_stage_progress("Drawing solution rows", iz + 1, Nz);
        }
        Tools::finish_progress_line();

        dump_csv("G_mag.csv",   mag);
        dump_csv("G_real.csv",  realA);
        dump_csv("G_imag.csv",  imagA);
        dump_csv("G_phase.csv", phase);
    }

    inline void draw_mesh(const BoundaryMesh& mesh, const std::string& filename) {
        std::vector<double> pointsx, pointsy, tanx, tany, nx, ny;
        for (int i = 0; i < mesh.get_num_segments(); i++) {
            Vector2d p = mesh.get_vertex(i).point;
            pointsx.push_back(p.x());
            pointsy.push_back(p.y());
            Vector2d t = mesh.get_vertex(i).tangent;
            tanx.push_back(t.x() * mesh.get_vertex(i).tnorm);
            tany.push_back(t.y() * mesh.get_vertex(i).tnorm);
            Vector2d n = mesh.get_vertex(i).normal;
            nx.push_back(n.x());
            ny.push_back(n.y());
        }
        Tools::dump_csv(filename, {pointsx, pointsy, tanx, tany, nx, ny});
    }

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
    inline void multipole_crystal_A(MatrixXcd &A_matrix, int N, double R, cpxd k, cpxd k_b, Vector2d alpha, double delta) {
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

    /**
     * Assemble the Fourier multipole matrices used for the defect-mode operator.
     *
     * A_matrix is the quasi-periodic unperturbed operator A^alpha from equation
     * (22). A_D_map is the local unperturbed disk map A_D from equation (17).
     * A_defect_map is A_D^epsilon = P2^{-1} A_{D_d} P1 from equation (19).
     * The defect operator in equation (25) is then
     *   I + (A_defect_map - A_D_map) * integral_BZ((A^alpha)^{-1}) d alpha / (2pi)^2.
     *
     * k_bd is the INTERIOR wavenumber of the defect resonator (omega / v_bd); it may differ
     * from the crystal interior wavenumber k_b when the defect is a different material. The
     * defect site's interior field is local to that site, so only A_{D_d} and the interior
     * block of P1 (which re-expresses the same interior field as a density on the defect
     * boundary) see k_bd; A_D and everything exterior keep k_b / k.
     */
    inline void multipole_defect_A(MatrixXcd &A_matrix, MatrixXcd &A_D_map, MatrixXcd &A_defect_map, int N, double R, double R_defect, cpxd k, cpxd k_b, cpxd k_bd, Vector2d alpha, double delta) {
        int M = 2 * N + 1;
        MatrixXcd S_ext(M, M), dS_ext(M, M), S_b(M, M), dS_b(M, M);
        const cpxd c = -cpxd(0, 0.5) * M_PI * R;
        const cpxd c_defect = -cpxd(0, 0.5) * M_PI * R_defect;
        const double r_ratio = R / R_defect;

        auto cyl_h1_prime = [](int n, cpxd z) {
            return 0.5 * (bessel::cyl_h1(n - 1, z) - bessel::cyl_h1(n + 1, z));
        };

        auto cyl_j_prime = [](int n, cpxd z) {
            return 0.5 * (bessel::cyl_j(n - 1, z) - bessel::cyl_j(n + 1, z));
        };

        A_D_map = MatrixXcd::Zero(2 * M, 2 * M);
        A_defect_map = MatrixXcd::Zero(2 * M, 2 * M);
        MatrixXcd A_defect = MatrixXcd::Zero(2 * M, 2 * M);
        MatrixXcd projector1 = MatrixXcd::Zero(2 * M, 2 * M),
                  projector2 = MatrixXcd::Zero(2 * M, 2 * M);

#pragma omp parallel for schedule(static) default(none) shared(S_ext, dS_ext, S_b, dS_b, A_D_map, A_defect_map, A_defect, r_ratio, N, M, R, R_defect, k, k_b, k_bd, alpha, c, c_defect, projector1, projector2, delta, cyl_h1_prime, cyl_j_prime)
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

                    const cpxd j_interior = bessel::cyl_j(n, k_b * R);
                    const cpxd h_interior = bessel::cyl_h1(n, k_b * R);
                    const cpxd jp_interior = cyl_j_prime(n, k_b * R);
                    const cpxd h_exterior = bessel::cyl_h1(n, k * R);
                    const cpxd hp_exterior = cyl_h1_prime(n, k * R);

                    A_D_map(i, i) = c * j_interior * h_interior;
                    A_D_map(i, i + M) = -c * j_source * h_exterior;
                    A_D_map(i + M, i) = c * k_b * jp_interior * h_interior;
                    A_D_map(M + i, M + i) = -c * k * delta * j_source * hp_exterior;

                    const cpxd j_defect_interior = bessel::cyl_j(n, k_bd * R_defect);
                    const cpxd h_defect_interior = bessel::cyl_h1(n, k_bd * R_defect);
                    const cpxd jp_defect_interior = cyl_j_prime(n, k_bd * R_defect);
                    const cpxd j_defect_exterior = bessel::cyl_j(n, k * R_defect);
                    const cpxd h_defect_exterior = bessel::cyl_h1(n, k * R_defect);
                    const cpxd hp_defect_exterior = cyl_h1_prime(n, k * R_defect);

                    A_defect(i, i) = c_defect * j_defect_interior * h_defect_interior;
                    A_defect(i, i + M) = -c_defect * j_defect_exterior * h_defect_exterior;
                    A_defect(i + M, i) = c_defect * k_bd * jp_defect_interior * h_defect_interior;
                    A_defect(M + i, M + i) = -c_defect * k * delta * j_defect_exterior * hp_defect_exterior;

                    // interior block of P1 at the DEFECT interior wavenumber: it maps the density
                    // generating a given interior field on dD to the one generating the same
                    // field on dD_d, and at the defect site that field lives at k_bd.
                    projector1(i, i) = r_ratio * bessel::cyl_h1(n, k_bd * R) / bessel::cyl_h1(n, k_bd * R_defect);
                    projector1(i + M, i + M) = r_ratio * bessel::cyl_j(n, k * R) / bessel::cyl_j(n, k * R_defect);
                    projector2(i, i) = bessel::cyl_j(n, k * R_defect) / bessel::cyl_j(n, k * R);
                    projector2(i + M, i + M) = cyl_j_prime(n, k * R_defect) / cyl_j_prime(n, k * R);
                } else {
                    const cpxd q = QFunction::q_function(n - m, k, alpha);
                    S_ext(i, j) = c * j_source * q * j_target;
                    dS_ext(i, j) = c * k * j_source * q * jp_target;
                    S_b(i, j) = 0.0;
                    dS_b(i, j) = 0.0;
                }
            }
        }

        A_defect_map = projector2.partialPivLu().solve(A_defect * projector1);

        A_matrix = MatrixXcd::Zero(2 * M, 2 * M);
        A_matrix.block(0, 0, M, M) = S_b;
        A_matrix.block(0, M, M, M) = -S_ext;
        A_matrix.block(M, 0, M, M) = dS_b;
        A_matrix.block(M, M, M, M) = -delta * dS_ext;
    }
}

#endif //SUBWAVELENGTHRESONATORS_UTILS_H
