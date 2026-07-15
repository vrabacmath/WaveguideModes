//
// Created by lara on 10/26/25.
//

#include "spectral_operators.h"

#define EPS 1e-12

namespace {

template <GreenKernels::BoundaryIntegralKernel Kernel>
void assemble_single_layer_diagonal(MatrixXcd& S,
                                    const BoundaryMesh& mesh,
                                    const VectorXcd& R_j_Kress,
                                    const Kernel& kernel) {
    const int Ntotal = mesh.get_num_segments();
    S = MatrixXcd::Zero(Ntotal, Ntotal);

    for (int m = 0; m < mesh.get_num_meshes(); m++) {
        const int start_idx = mesh.get_start_index(m);
        const int end_idx = mesh.get_end_index(m);
        const int N = end_idx - start_idx + 1;

#pragma omp parallel for schedule(static) default(none) shared(S, mesh, R_j_Kress, kernel, start_idx, end_idx, N)
        for (int i = start_idx; i <= end_idx; i++) {
            const Vertex& vertex_i = mesh.get_vertex(i);
            const Vector2d point_i = vertex_i.point;
            for (int j = start_idx; j <= end_idx; j++) {
                const Vertex& vertex_j = mesh.get_vertex(j);
                if (i != j) {
                    const Vector2d point_j = vertex_j.point;
                    const double rij = (point_i - point_j).norm();
                    const cpxd log_coefficient =
                            kernel.single_layer_log_coefficient(point_i, point_j, rij);
                    const double sinterm = std::sin(M_PI * (i - j) / N);
                    S(i, j) = log_coefficient * vertex_j.tnorm * R_j_Kress((i - j + N) % N);
                    S(i, j) += vertex_j.sigma *
                               (kernel.single_layer_value(point_i, point_j) -
                                log_coefficient * std::log(4.0 * sinterm * sinterm));
                } else {
                    S(i, i) = kernel.single_layer_self_log_coefficient(vertex_i)
                              * vertex_i.tnorm * R_j_Kress(0);
                    S(i, i) += kernel.single_layer_self_regular(vertex_i);
                }
            }
        }
    }
}

template <GreenKernels::BoundaryIntegralKernel Kernel>
void assemble_single_layer(MatrixXcd& S,
                           const BoundaryMesh& mesh,
                           const VectorXcd& R_j_Kress,
                           const Kernel& kernel) {
    assemble_single_layer_diagonal(S, mesh, R_j_Kress, kernel);

    for (int mesh_i = 0; mesh_i < mesh.get_num_meshes(); mesh_i++) {
        const int start_idx = mesh.get_start_index(mesh_i);
        const int end_idx = mesh.get_end_index(mesh_i);

        for (int mesh_j = 0; mesh_j < mesh.get_num_meshes(); mesh_j++) {
            const int start_jdx = mesh.get_start_index(mesh_j);
            const int end_jdx = mesh.get_end_index(mesh_j);

            if (mesh_i != mesh_j) {
#pragma omp parallel for schedule(static) default(none) shared(S, mesh, kernel, start_idx, end_idx, start_jdx, end_jdx)
                for (int i = start_idx; i <= end_idx; i++) {
                    const Vector2d point_i = mesh.get_vertex(i).point;
                    for (int j = start_jdx; j <= end_jdx; j++) {
                        const Vertex& vertex_j = mesh.get_vertex(j);
                        S(i, j) = kernel.single_layer_value(point_i, vertex_j.point) * vertex_j.sigma;
                    }
                }
            }
        }
    }
}

template <GreenKernels::BoundaryIntegralKernel Kernel>
void assemble_target_normal_diagonal(MatrixXcd& Kstar,
                                     const BoundaryMesh& mesh,
                                     const VectorXcd& R_j_Kress,
                                     const Kernel& kernel) {
    const int Ntotal = mesh.get_num_segments();
    Kstar = MatrixXcd::Zero(Ntotal, Ntotal);

    for (int m = 0; m < mesh.get_num_meshes(); m++) {
        const int start_idx = mesh.get_start_index(m);
        const int end_idx = mesh.get_end_index(m);
        const int N = end_idx - start_idx + 1;

#pragma omp parallel for schedule(static) default(none) shared(Kstar, mesh, R_j_Kress, kernel, start_idx, end_idx, N)
        for (int i = start_idx; i <= end_idx; i++) {
            const Vertex& vertex_i = mesh.get_vertex(i);
            const Vector2d point_i = vertex_i.point;
            const Vector2d normal_i = vertex_i.normal;
            for (int j = start_idx; j <= end_idx; j++) {
                const Vertex& vertex_j = mesh.get_vertex(j);
                if (i != j) {
                    const Vector2d point_j = vertex_j.point;
                    const double rij = (point_i - point_j).norm();
                    const cpxd log_coefficient =
                            kernel.target_normal_log_coefficient(point_i, point_j, normal_i, rij);
                    const double sinterm = std::sin(M_PI * (i - j) / N);
                    Kstar(i, j) = log_coefficient * vertex_j.tnorm * R_j_Kress((i - j + N) % N);
                    Kstar(i, j) += vertex_j.sigma *
                                   (kernel.target_normal_value(point_i, point_j, normal_i) -
                                    log_coefficient * std::log(4.0 * sinterm * sinterm));
                } else {
                    Kstar(i, i) = kernel.target_normal_self(vertex_i);
                }
            }
        }
    }
}

template <GreenKernels::BoundaryIntegralKernel Kernel>
void assemble_target_normal(MatrixXcd& Kstar,
                            const BoundaryMesh& mesh,
                            const VectorXcd& R_j_Kress,
                            const Kernel& kernel) {
    assemble_target_normal_diagonal(Kstar, mesh, R_j_Kress, kernel);

    for (int mesh_i = 0; mesh_i < mesh.get_num_meshes(); mesh_i++) {
        const int start_idx = mesh.get_start_index(mesh_i);
        const int end_idx = mesh.get_end_index(mesh_i);

        for (int mesh_j = 0; mesh_j < mesh.get_num_meshes(); mesh_j++) {
            const int start_jdx = mesh.get_start_index(mesh_j);
            const int end_jdx = mesh.get_end_index(mesh_j);

            if (mesh_i != mesh_j) {
#pragma omp parallel for schedule(static) default(none) shared(Kstar, mesh, kernel, start_idx, end_idx, start_jdx, end_jdx)
                for (int i = start_idx; i <= end_idx; i++) {
                    const Vertex& vertex_i = mesh.get_vertex(i);
                    for (int j = start_jdx; j <= end_jdx; j++) {
                        const Vertex& vertex_j = mesh.get_vertex(j);
                        Kstar(i, j) = kernel.target_normal_value(vertex_i.point,
                                                                 vertex_j.point,
                                                                 vertex_i.normal)
                                      * vertex_j.sigma;
                    }
                }
            }
        }
    }
}

template <GreenKernels::BoundaryIntegralKernel Kernel>
void assemble_source_normal_diagonal(MatrixXcd& K,
                                     const BoundaryMesh& mesh,
                                     const VectorXcd& R_j_Kress,
                                     const Kernel& kernel) {
    const int Ntotal = mesh.get_num_segments();
    K = MatrixXcd::Zero(Ntotal, Ntotal);

    for (int m = 0; m < mesh.get_num_meshes(); m++) {
        const int start_idx = mesh.get_start_index(m);
        const int end_idx = mesh.get_end_index(m);
        const int N = end_idx - start_idx + 1;

#pragma omp parallel for schedule(static) default(none) shared(K, mesh, R_j_Kress, kernel, start_idx, end_idx, N)
        for (int i = start_idx; i <= end_idx; i++) {
            const Vector2d point_i = mesh.get_vertex(i).point;
            for (int j = start_idx; j <= end_idx; j++) {
                const Vertex& vertex_j = mesh.get_vertex(j);
                if (i != j) {
                    const Vector2d point_j = vertex_j.point;
                    const double rij = (point_i - point_j).norm();
                    const cpxd log_coefficient =
                            kernel.source_normal_log_coefficient(point_i, point_j,
                                                                 vertex_j.normal, rij);
                    const double sinterm = std::sin(M_PI * (i - j) / N);
                    K(i, j) = log_coefficient * vertex_j.tnorm * R_j_Kress((i - j + N) % N);
                    K(i, j) += vertex_j.sigma *
                               (kernel.source_normal_value(point_i, point_j, vertex_j.normal) -
                                log_coefficient * std::log(4.0 * sinterm * sinterm));
                } else {
                    K(i, i) = kernel.source_normal_self(vertex_j);
                }
            }
        }
    }
}

template <GreenKernels::BoundaryIntegralKernel Kernel>
void assemble_source_normal(MatrixXcd& K,
                            const BoundaryMesh& mesh,
                            const VectorXcd& R_j_Kress,
                            const Kernel& kernel) {
    assemble_source_normal_diagonal(K, mesh, R_j_Kress, kernel);

    for (int mesh_i = 0; mesh_i < mesh.get_num_meshes(); mesh_i++) {
        const int start_idx = mesh.get_start_index(mesh_i);
        const int end_idx = mesh.get_end_index(mesh_i);

        for (int mesh_j = 0; mesh_j < mesh.get_num_meshes(); mesh_j++) {
            const int start_jdx = mesh.get_start_index(mesh_j);
            const int end_jdx = mesh.get_end_index(mesh_j);

            if (mesh_i != mesh_j) {
#pragma omp parallel for schedule(static) default(none) shared(K, mesh, kernel, start_idx, end_idx, start_jdx, end_jdx)
                for (int i = start_idx; i <= end_idx; i++) {
                    const Vector2d point_i = mesh.get_vertex(i).point;
                    for (int j = start_jdx; j <= end_jdx; j++) {
                        const Vertex& vertex_j = mesh.get_vertex(j);
                        K(i, j) = kernel.source_normal_value(point_i, vertex_j.point,
                                                             vertex_j.normal)
                                  * vertex_j.sigma;
                    }
                }
            }
        }
    }
}

} // namespace

/**
 * @brief Construct a SpectralOperators object and precompute the Kress quadrature weights.
 *
 * Initialises the Kress quadrature weights R_j via an inverse FFT of the
 * analytically known Fourier coefficients. Each submesh must have an even
 * number of segments.
 *
 * @param mesh  BoundaryMesh describing the geometry (one or more closed curves).
 */
SpectralOperators::SpectralOperators(BoundaryMesh mesh) {
    this->mesh = mesh;
    int totalN = mesh.get_num_segments();
    R_j_Kress.resize(totalN);

    for (int m = 0; m < mesh.get_num_meshes(); m++) {
        int N = mesh.get_end_index(m) - mesh.get_start_index(m) + 1;

        if (N % 2 != 0) {
            cerr << "Error: Number of vertices N in each submesh must be even for Kress quadrature." << endl;
            exit(EXIT_FAILURE);
        }

        VectorXcd R_hat_j_Kress = VectorXcd::Zero(N);
        R_hat_j_Kress(0) = 0.0;
        R_hat_j_Kress(N / 2) = -4.0 / N;
        for (int j = 1; j < N / 2; j++) {
            R_hat_j_Kress(j) = -2.0 / j;
            R_hat_j_Kress(N - j) = R_hat_j_Kress(j);
        }

        // Inverse FFT to get R_j in physical space
        VectorXcd R_j_Kress_sub(N);
        Eigen::FFT<double> fft;
        R_j_Kress_sub.setZero();
        fft.inv(R_j_Kress_sub, R_hat_j_Kress);

        R_j_Kress.segment(mesh.get_start_index(m), N) = R_j_Kress_sub;
    }

    // scale by pi to get final weights
    R_j_Kress *= M_PI;
//    cout << R_j_Kress.size() << endl;
//    cout << "Kress quadrature weights: " << R_j_Kress.transpose() << endl;
//    cout << R_j_Kress.sum() << endl;
}

/**
 * @brief Assemble the diagonal blocks of the single-layer operator for the 2D Helmholtz equation.
 *
 * Fills the block-diagonal part of the matrix S using Kress quadrature for
 * the logarithmic singularity splitting.  When |k| < 1e-6 the Laplace
 * kernel is used; otherwise the Hankel-function-based Helmholtz kernel is
 * used.
 *
 * @param[out] S  Complex matrix (Ntotal × Ntotal) to be filled.
 * @param      k  Wavenumber (complex).
 */
void SpectralOperators::S_diagonal(MatrixXcd &S, cpxd k) const {
    assemble_single_layer_diagonal(S, mesh, R_j_Kress, GreenKernels::FreeSpaceKernel{k});
}


/**
 * @brief Assemble the full single-layer operator matrix for the 2D Helmholtz equation.
 *
 * First fills the diagonal blocks via S_diagonal(), then adds the smooth
 * off-diagonal blocks (inter-mesh interactions) using the standard Green's
 * function without singularity splitting.
 *
 * @param[out] S  Complex matrix (Ntotal × Ntotal) to be filled.
 * @param      k  Wavenumber (complex).
 */
void SpectralOperators::S(MatrixXcd &S, cpxd k) const {
    assemble_single_layer(S, mesh, R_j_Kress, GreenKernels::FreeSpaceKernel{k});
}

/**
 * @brief Assemble the diagonal blocks of the adjoint double-layer operator (K*) for the 2D Helmholtz equation.
 *
 * Fills the block-diagonal part of the matrix using Kress quadrature.
 * When |k| < 1e-6 the Laplace kernel is used; otherwise the
 * Helmholtz kernel with singularity splitting is used.
 *
 * @param[out] Kstar  Complex matrix (Ntotal × Ntotal) to be filled.
 * @param      k      Wavenumber (complex).
 */
void SpectralOperators::Kstar_diagonal(MatrixXcd &Kstar, cpxd k) const {
    assemble_target_normal_diagonal(Kstar, mesh, R_j_Kress, GreenKernels::FreeSpaceKernel{k});
}

/**
 * @brief Assemble the full adjoint double-layer operator (K*) for the 2D Helmholtz equation.
 *
 * First fills the diagonal blocks via Kstar_diagonal(), then adds the
 * smooth off-diagonal blocks.
 *
 * @param[out] Kstar  Complex matrix (Ntotal × Ntotal) to be filled.
 * @param      k      Wavenumber (complex).
 */
void SpectralOperators::Kstar(Eigen::MatrixXcd &Kstar, cpxd k) const {
    assemble_target_normal(Kstar, mesh, R_j_Kress, GreenKernels::FreeSpaceKernel{k});
}

/**
 * @brief Assemble the diagonal blocks of the double-layer operator (K) for the 2D Helmholtz equation.
 *
 * Fills the block-diagonal part of the matrix using Kress quadrature.
 * When |k| < 1e-6 the Laplace kernel is used; otherwise the
 * Helmholtz kernel with singularity splitting is used.
 *
 * @param[out] K  Complex matrix (Ntotal × Ntotal) to be filled.
 * @param      k  Wavenumber (complex).
 */
void SpectralOperators::K_diagonal(MatrixXcd &K, cpxd k) {
    assemble_source_normal_diagonal(K, mesh, R_j_Kress, GreenKernels::FreeSpaceKernel{k});
}

/**
 * @brief Assemble the full double-layer operator (K) for the 2D Helmholtz equation.
 *
 * First fills the diagonal blocks via K_diagonal(), then adds the smooth
 * off-diagonal blocks (inter-mesh interactions).
 *
 * @param[out] K  Complex matrix (Ntotal × Ntotal) to be filled.
 * @param      k  Wavenumber (complex).
 */
void SpectralOperators::K(Eigen::MatrixXcd &K, cpxd k) {
    assemble_source_normal(K, mesh, R_j_Kress, GreenKernels::FreeSpaceKernel{k});
}

/**
 * @brief Assemble the diagonal blocks of the quasi-periodic single-layer operator.
 *
 * Uses the quasi-periodic Green's function for the 2D Helmholtz equation
 * with Kress quadrature for the logarithmic singularity.
 *
 * @param[out] S     Complex matrix (Ntotal × Ntotal) to be filled.
 * @param      k     Wavenumber (complex).
 * @param      kbar  Bloch wavenumber (quasi-periodicity parameter).
 * @param      a     Period in the x-direction.
 */
void SpectralOperators::PeriodicS_diagonal(MatrixXcd &S, cpxd k, double kbar, double a) {
    const BoundaryMesh *local_mesh = &mesh;
    int Ntotal = mesh.get_num_segments();
    S = MatrixXcd::Zero(Ntotal, Ntotal);

    for (int m = 0; m < mesh.get_num_meshes(); m++) {
        int start_idx = mesh.get_start_index(m);
        int end_idx = mesh.get_end_index(m);
        int N = end_idx - start_idx + 1;

        if (abs(k) < 1e-6 && abs(kbar) < 1e-6) { // Use Laplace kernel for small k
#pragma omp parallel for schedule(static) default(none) shared(S, local_mesh, N, a, start_idx, end_idx)
            for (int i = start_idx; i <= end_idx; i++) {
                Vector2d point_i = mesh.get_vertex(i).point;
                Vector2d normal_i = mesh.get_vertex(i).normal;
                for (int j = start_idx; j <= end_idx; j++) {
                    if (i != j) {
                        Vector2d point_j = mesh.get_vertex(j).point;
                        cpxd alpha = 0.25 * M_1_PI;
                        S(i, j) = alpha * mesh.get_vertex(j).tnorm * R_j_Kress((i - j + N) % N);

                        double sinterm = sin(M_PI * (i - j) / N);
                        S(i, j) += mesh.get_vertex(j).sigma * (Kernels::laplace_2D_periodic(point_i, point_j, a) -
                                                               alpha * log(4. * sinterm * sinterm));
                    } else {
                        // self-interaction term
                        S(i, i) = 0.25 * M_1_PI * mesh.get_vertex(i).tnorm * R_j_Kress(0);
                        S(i, i) += 0.5 * M_1_PI * log(M_PI / a * mesh.get_vertex(i).tnorm) * mesh.get_vertex(i).sigma;
                    }
                }
            }
        } else {
            cpxd C = 0; double epsilon = sqrt(M_PI) / a;
            cpxd ratio = k / (2.0 * epsilon);
            for (int q = 1; q <= 20; q++) {
                C += pow(ratio, 2 * q) / (tgamma(q + 1) * q);
            }
            // NOTE: C is the bare Ewald near-field series; the 1/(4 pi) factor is applied
            // once in the self-term below (do not pre-scale C here).
#pragma omp parallel for schedule(static) default(none) shared(S, local_mesh, N, start_idx, end_idx, k, kbar, a, C)
            for (int i = start_idx; i <= end_idx; i++) {
                Vector2d point_i = mesh.get_vertex(i).point;
                Vector2d normal_i = mesh.get_vertex(i).normal;
                for (int j = start_idx; j <= end_idx; j++) {
                    Vector2d point_j = mesh.get_vertex(j).point;
                    double rij = (point_i - point_j).norm();

                    if (i != j) {
                        cpxd alpha = 0.25 * M_1_PI * bessel::cyl_j(0, k * rij);
                        S(i, j) = alpha * mesh.get_vertex(j).tnorm * R_j_Kress((i - j + N) % N);
                        double sinterm = sin(M_PI * (i - j) / N);
                        S(i, j) += mesh.get_vertex(j).sigma *
                                  (Kernels::helmholtz_2D_periodic(k, kbar, point_i, point_j, a) -
                                   alpha * log(4. * sinterm * sinterm));
                    } else {
                        // self-interaction term. The Ewald kernel helmholtz_2D_periodic already
                        // skips its own p=0 spatial singularity, so at coincidence it returns the
                        // regular lattice part; Kress(0) carries the log-singular self weight and
                        // the (-C + gamma + log) term is the Ewald near-field constant of the p=0
                        // free-space cell. (Weighted by sigma like the off-diagonal entries.)
                        const double tnorm = mesh.get_vertex(i).tnorm, sig = mesh.get_vertex(i).sigma;
                        S(i, i) = 0.25 * M_1_PI * tnorm * R_j_Kress(0);
                        S(i, i) += sig * (Kernels::helmholtz_2D_periodic(k, kbar, point_i, point_i, a)
                                          + 0.25 * M_1_PI * (-C + EULER_GAMMA + log(M_PI / a / a * tnorm * tnorm)));
                    }
                }
            }
        }
    }
}

/* @brief Assemble the diagonal blocks of the quasi-periodic adjoint double-layer operator (K*) for the 2D Helmholtz equation.
 *
 * Uses the quasi-periodic Green's function for the 2D Helmholtz equation
 * with Kress quadrature for the logarithmic singularity.
 *
 * @param[out] Kstar  Complex matrix (Ntotal × Ntotal) to be filled.
 * @param      k      Wavenumber (complex).
 * @param      kbar   Bloch wavenumber (quasi-periodicity parameter).
 * @param      a      Period in the x-direction.
 */
void SpectralOperators::PeriodicKstar_diagonal(MatrixXcd &Kstar, cpxd k, double kbar, double a) {
    const BoundaryMesh *local_mesh = &mesh;
    int Ntotal = mesh.get_num_segments();
    Kstar = MatrixXcd::Zero(Ntotal, Ntotal);

    for (int m = 0; m < mesh.get_num_meshes(); m++) {
        int start_idx = mesh.get_start_index(m);
        int end_idx = mesh.get_end_index(m);
        int N = end_idx - start_idx + 1;

        if (abs(k) < 1e-6 && abs(kbar) < 1e-6) { // Use Laplace kernel for small k
#pragma omp parallel for schedule(static) default(none) shared(Kstar, local_mesh, N, a, start_idx, end_idx)
            for (int i = start_idx; i <= end_idx; i++) {
                Vector2d point_i = mesh.get_vertex(i).point;
                Vector2d normal_i = mesh.get_vertex(i).normal;
                for (int j = start_idx; j <= end_idx; j++) {
                    if (i != j) {
                        Vector2d point_j = mesh.get_vertex(j).point;
                        Kstar(i, j) = Kernels::grad_laplace_2D_periodic(point_i, point_j, a).dot(normal_i)
                                      * mesh.get_vertex(j).sigma;
                    } else {
                        // self-interaction term
                        Kstar(i, i) = mesh.get_vertex(i).curvature * 0.25 * M_1_PI * mesh.get_vertex(i).sigma;
                    }
                }
            }
        } else {
#pragma omp parallel for schedule(static) default(none) shared(Kstar, local_mesh, N, k, kbar, a, start_idx, end_idx)
            for (int i = start_idx; i <= end_idx; i++) {
                Vector2d point_i = mesh.get_vertex(i).point;
                Vector2d normal_i = mesh.get_vertex(i).normal;
                for (int j = start_idx; j <= end_idx; j++) {
                    if (i != j) {
                        Vector2d point_j = mesh.get_vertex(j).point;
                        Vector2d r_ij = point_i - point_j;
                        double r_norm = r_ij.norm();

                        cpxd alpha = -k * 0.25 * M_1_PI * bessel::cyl_j(1, k * r_norm)
                                     * r_ij.dot(normal_i) / r_norm;

                        Kstar(i, j) = alpha * mesh.get_vertex(j).tnorm * R_j_Kress((i - j + N) % N);

                        double sinterm = sin(M_PI * (i - j) / N);
                        Kstar(i, j) += mesh.get_vertex(j).sigma *
                                       (Kernels::grad_helmholtz_2D_periodic(k, kbar, point_i, point_j, a).cwiseProduct(
                                               normal_i).sum() - alpha * log(4. * sinterm * sinterm));
                    } else {
                        // self-interaction term. Use the NON-Dirichlet periodic gradient kernel
                        // (translation invariant in y); it skips its own p=0 singularity, so at
                        // coincidence it returns the regular lattice-image gradient. The curvature
                        // term is the principal value of the free-space p=0 self.
                        Kstar(i, i) = 0.25 * M_1_PI * mesh.get_vertex(i).curvature * mesh.get_vertex(i).sigma;
                        Kstar(i, i) += mesh.get_vertex(i).sigma *
                                       Kernels::grad_helmholtz_2D_periodic(k, kbar, point_i, point_i, a)
                                       .cwiseProduct(normal_i).sum();
                    }
                }
            }
        }
    }
}

/**
 * @brief Assemble the full quasi-periodic single-layer operator matrix.
 * 
 * Fills diagonal blocks via PeriodicKstar_diagonal(), then adds the smooth
 * off-diagonal inter-mesh blocks using the periodic Green's function.
 * 
 * @param[out] Kstar  Complex matrix (Ntotal × Ntotal) to be filled.
 * @param      k      Wavenumber (complex).
 * @param      kbar   Bloch wavenumber (quasi-periodicity parameter).
 * @param      a      Period in the x-direction.
 */
void SpectralOperators::PeriodicKstar(MatrixXcd &Kstar, cpxd k, double kbar, double a) {
    const BoundaryMesh *local_mesh = &mesh;
    int Ntotal = mesh.get_num_segments();
    PeriodicKstar_diagonal(Kstar, k, kbar, a); // fill the diagonal blocks

    for (int mesh_i = 0; mesh_i < mesh.get_num_meshes(); mesh_i++) {
        int start_idx = mesh.get_start_index(mesh_i);
        int end_idx = mesh.get_end_index(mesh_i);

        for (int mesh_j = 0; mesh_j < mesh.get_num_meshes(); mesh_j++) {
            int start_jdx = mesh.get_start_index(mesh_j);
            int end_jdx = mesh.get_end_index(mesh_j);

            if (mesh_i != mesh_j) {
                if (abs(k) < 1e-6 && abs(kbar) < 1e-6) { // Use Laplace kernel for small k
#pragma omp parallel for schedule(static) default(none) shared(Kstar, local_mesh, start_idx, end_idx, start_jdx, end_jdx, a)
                    for (int i = start_idx; i <= end_idx; i++) {
                        Vector2d point_i = mesh.get_vertex(i).point;
                        Vector2d normal_i = mesh.get_vertex(i).normal;
                        for (int j = start_jdx; j <= end_jdx; j++) {
                            Vector2d point_j = mesh.get_vertex(j).point;
                            Kstar(i, j) = Kernels::grad_laplace_2D_periodic(point_i, point_j, a).dot(normal_i)
                                          * mesh.get_vertex(j).sigma;
                        }
                    }
                } else {
#pragma omp parallel for schedule(static) default(none) shared(Kstar, local_mesh, start_idx, end_idx, start_jdx, end_jdx, k, kbar, a)
                    for (int i = start_idx; i <= end_idx; i++) {
                        Vector2d point_i = mesh.get_vertex(i).point;
                        Vector2d normal_i = mesh.get_vertex(i).normal;
                        for (int j = start_jdx; j <= end_jdx; j++) {
                            Vector2d point_j = mesh.get_vertex(j).point;
                            Kstar(i, j) = Kernels::grad_helmholtz_2D_periodic(k, kbar, point_i, point_j, a).cwiseProduct(
                                    normal_i).sum() * mesh.get_vertex(j).sigma;
                        }
                    }
                }
            }
        }
    }
}

/**
 * @brief Assemble the full quasi-periodic single-layer operator matrix.
 *
 * Fills diagonal blocks via PeriodicS_diagonal(), then adds the smooth
 * off-diagonal inter-mesh blocks using the periodic Green's function.
 *
 * @param[out] S     Complex matrix (Ntotal × Ntotal) to be filled.
 * @param      k     Wavenumber (complex).
 * @param      kbar  Bloch wavenumber.
 * @param      a     Period in the x-direction.
 */
void SpectralOperators::PeriodicS(MatrixXcd &S, cpxd k, double kbar, double a) {
    const BoundaryMesh *local_mesh = &mesh;
    int Ntotal = mesh.get_num_segments();
    PeriodicS_diagonal(S, k, kbar, a); // fill the diagonal blocks

    for (int mesh_i = 0; mesh_i < mesh.get_num_meshes(); mesh_i++) {
        int start_idx = mesh.get_start_index(mesh_i);
        int end_idx = mesh.get_end_index(mesh_i);

        for (int mesh_j = 0; mesh_j < mesh.get_num_meshes(); mesh_j++) {
            int start_jdx = mesh.get_start_index(mesh_j);
            int end_jdx = mesh.get_end_index(mesh_j);

            if (mesh_i != mesh_j) {
                if (abs(k) < 1e-6 && abs(kbar) < 1e-6) { // Use Laplace kernel for small k
#pragma omp parallel for schedule(static) default(none) shared(S, local_mesh, start_idx, end_idx, start_jdx, end_jdx, a)
                    for (int i = start_idx; i <= end_idx; i++) {
                        Vector2d point_i = mesh.get_vertex(i).point;
                        for (int j = start_jdx; j <= end_jdx; j++) {
                            Vector2d point_j = mesh.get_vertex(j).point;
                            S(i, j) = Kernels::laplace_2D_periodic(point_i, point_j, a)
                                      * mesh.get_vertex(j).sigma;
                        }
                    }
                } else {
#pragma omp parallel for schedule(static) default(none) shared(S, local_mesh, start_idx, end_idx, start_jdx, end_jdx, k, kbar, a)
                    for (int i = start_idx; i <= end_idx; i++) {
                        Vector2d point_i = mesh.get_vertex(i).point;
                        for (int j = start_jdx; j <= end_jdx; j++) {
                            Vector2d point_j = mesh.get_vertex(j).point;
                            S(i, j) = Kernels::helmholtz_2D_periodic(k, kbar, point_i, point_j, a)
                                      * mesh.get_vertex(j).sigma;
                        }
                    }
                }
            }
        }
    }
}

/**
 * @brief Assemble the static, 1D-quasi-periodic single-layer operator S^{alpha,0}.
 *
 * Builds the single-layer operator for the static (omega = 0) Green's function that
 * is quasi-periodic in the x-direction with Bloch phase @p alpha and free/decaying in
 * the y-direction, following Ammari et al. (arXiv:2512.05370v2, eqs (A.3)/(A.7)).
 *
 * Implementation strategy: the quasi-periodic kernel shares the SAME logarithmic
 * singularity as the periodic static kernel `laplace_2D_periodic`, so we
 *   (1) assemble the periodic static single-layer via PeriodicS(S, 0, 0, a)
 *       (Kress quadrature handles the singularity exactly), and
 *   (2) add the smooth correction C^alpha (Kernels::laplace_2D_quasiperiodic_correction),
 *       integrated with the trapezoidal weight sigma over every boundary pair.
 *
 * @param[out] S      Complex matrix (Ntotal x Ntotal).
 * @param      alpha  Quasi-momentum (Bloch phase) in [-pi, pi].
 * @param      a      Period in the x-direction (only a = 1 is physically supported here).
 * @param      M      Reciprocal-lattice truncation passed to the correction kernel.
 */
void SpectralOperators::QuasiPeriodicStaticS(MatrixXcd &S, double alpha, double a, int M) {
    const BoundaryMesh *local_mesh = &mesh;
    int Ntotal = mesh.get_num_segments();

    // (1) periodic static single-layer (carries the log singularity, alpha = 0 phase).
    PeriodicS(S, cpxd(0.0, 0.0), 0.0, a);

    // (2) add the smooth quasi-periodic correction, trapezoidal rule (weight sigma_j).
#pragma omp parallel for schedule(static) default(none) shared(S, local_mesh, Ntotal, alpha, M)
    for (int i = 0; i < Ntotal; i++) {
        Vector2d point_i = mesh.get_vertex(i).point;
        for (int j = 0; j < Ntotal; j++) {
            Vector2d point_j = mesh.get_vertex(j).point;
            S(i, j) += Kernels::laplace_2D_quasiperiodic_correction(alpha, point_i, point_j, M)
                       * mesh.get_vertex(j).sigma;
        }
    }
}

/**
 * @brief Assemble the diagonal blocks of the Dirichlet wall quasi-periodic single-layer operator.
 *
 * Uses the Dirichlet periodic Green's function (image-subtracted) for the
 * 2D Helmholtz equation with Kress quadrature for the logarithmic
 * singularity.  The Dirichlet condition is imposed by subtracting the
 * contribution of a mirror point reflected across the x-axis.
 *
 * @param[out] S     Complex matrix (Ntotal × Ntotal) to be filled.
 * @param      k     Wavenumber (complex).
 * @param      kbar  Bloch wavenumber.
 * @param      a     Period in the x-direction.
 */
void SpectralOperators::DirPeriodicS_diagonal(MatrixXcd &S, cpxd k, double kbar, double a) {
    assemble_single_layer_diagonal(S, mesh, R_j_Kress,
                                   GreenKernels::DirichletPeriodicKernel{k, kbar, a});
}

/**
 * @brief Assemble the full Dirichlet wall quasi-periodic single-layer operator matrix.
 *
 * Fills diagonal blocks via DirPeriodicS_diagonal(), then adds the smooth
 * off-diagonal inter-mesh blocks using the Dirichlet periodic Green's function.
 *
 * @param[out] S     Complex matrix (Ntotal × Ntotal) to be filled.
 * @param      k     Wavenumber (complex).
 * @param      kbar  Bloch wavenumber.
 * @param      a     Period in the x-direction.
 */
void SpectralOperators::DirPeriodicS(MatrixXcd &S, cpxd k, double kbar, double a) {
    assemble_single_layer(S, mesh, R_j_Kress,
                          GreenKernels::DirichletPeriodicKernel{k, kbar, a});
}

/**
 * @brief Assemble the diagonal blocks of the Dirichlet wall quasi-periodic adjoint double-layer operator (K*).
 *
 * Uses the Dirichlet periodic Green's function gradient with Kress
 * quadrature.  The self-interaction includes a mirror-point correction
 * to enforce the Dirichlet boundary condition on the axis of symmetry.
 *
 * @param[out] Kstar  Complex matrix (Ntotal × Ntotal) to be filled.
 * @param      k      Wavenumber (complex).
 * @param      kbar   Bloch wavenumber.
 * @param      a      Period in the x-direction.
 */
void SpectralOperators::DirPeriodicKstar_diagonal(MatrixXcd &Kstar, cpxd k, double kbar, double a) {
    assemble_target_normal_diagonal(Kstar, mesh, R_j_Kress,
                                    GreenKernels::DirichletPeriodicKernel{k, kbar, a});
}

/**
 * @brief Assemble the full Dirichlet wall quasi-periodic adjoint double-layer operator (K*).
 *
 * Fills diagonal blocks via DirPeriodicKstar_diagonal(), then adds the
 * smooth off-diagonal inter-mesh blocks.
 *
 * @param[out] Kstar  Complex matrix (Ntotal × Ntotal) to be filled.
 * @param      k      Wavenumber (complex).
 * @param      kbar   Bloch wavenumber.
 * @param      a      Period in the x-direction.
 */
void SpectralOperators::DirPeriodicKstar(MatrixXcd &Kstar, cpxd k, double kbar, double a) {
    assemble_target_normal(Kstar, mesh, R_j_Kress,
                           GreenKernels::DirichletPeriodicKernel{k, kbar, a});
}

/**
 * @brief Assemble the full coupled interior-exterior integral operator A for the 2D Helmholtz equation.
 *
 * Builds the 2N × 2N block system coupling interior (k_b) and exterior (k)
 * single-layer and adjoint double-layer operators.  A small imaginary part
 * is added to both wavenumbers (limiting absorption principle).
 *
 * @param[out] A_matrix  Complex matrix (2N × 2N) to be filled.
 * @param      k         Exterior wavenumber (complex).
 * @param      k_b       Interior (bubble) wavenumber (complex).
 * @param      delta     Inverse contrast parameter ρ_b / ρ.
 */
void SpectralOperators::A(MatrixXcd &A_matrix, cpxd k, cpxd k_b, double delta) const {
    int N = mesh.get_num_segments();
    k   *= cpxd(1., -EPS); // limiting absorption principle
    k_b *= cpxd(1., -EPS);

    A_matrix = MatrixXcd::Zero(2 * N, 2 * N);
    MatrixXcd S_matrix, Kstar_matrix,
            S_matrix_b, Kstar_matrix_b;

    SpectralOperators::S(S_matrix, k);
    SpectralOperators::S(S_matrix_b, k_b);
    SpectralOperators::Kstar(Kstar_matrix, k);
    SpectralOperators::Kstar(Kstar_matrix_b, k_b);

    A_matrix.block(0, 0, N, N) = S_matrix_b;
    A_matrix.block(0, N, N, N) = -S_matrix;
    A_matrix.block(N, 0, N, N) = -0.5 * MatrixXcd::Identity(N, N) + Kstar_matrix_b;
    A_matrix.block(N, N, N, N) = -(0.5 * MatrixXcd::Identity(N, N) + Kstar_matrix) * delta;
}

/**
 * @brief Assemble the Dirichlet wall periodic coupled interior-exterior integral operator.
 *
 * - mesh: BoundaryMesh object containing the geometry of the boundary.
 * Periodic analogue of A() using the Dirichlet periodic single-layer and
 * adjoint double-layer operators.  A small imaginary part is added to the
 * wavenumbers (limiting absorption principle).
 *
 * Fills the matrix A_matrix with the combined field operator values.
 * @param[out] A_matrix  Complex matrix (2N × 2N) to be filled.
 * @param      k         Exterior wavenumber (complex).
 * @param      k_b       Interior (bubble) wavenumber (complex).
 * @param      kbar      Bloch wavenumber.
 * @param      delta     Inverse contrast parameter.
 * @param      a         Period in the x-direction.
 */
void SpectralOperators::DirPeriodicA(MatrixXcd &A_matrix, cpxd k, cpxd k_b, double kbar, double delta, double a) {
    int N = mesh.get_num_segments();
//    k   *= cpxd(1., -EPS); // limiting absorption principle
//    k_b *= cpxd(1., -EPS);

    A_matrix = MatrixXcd::Zero(2 * N, 2 * N);
    MatrixXcd S_matrix, Kstar_matrix,
            S_matrix_b, Kstar_matrix_b;

    SpectralOperators::DirPeriodicS(S_matrix, k, kbar, a);
    SpectralOperators::DirPeriodicS_diagonal(S_matrix_b, k_b, kbar, a);
    SpectralOperators::DirPeriodicKstar(Kstar_matrix, k, kbar, a);
    SpectralOperators::DirPeriodicKstar_diagonal(Kstar_matrix_b, k_b, kbar, a);

    A_matrix.block(0, 0, N, N) = S_matrix_b;
    A_matrix.block(0, N, N, N) = -S_matrix;
    A_matrix.block(N, 0, N, N) = -0.5 * MatrixXcd::Identity(N, N) + Kstar_matrix_b;
    A_matrix.block(N, N, N, N) = -(0.5 * MatrixXcd::Identity(N, N) + Kstar_matrix) * delta;

//    cout << S_matrix << endl << endl;
//    cout << "alpha = " << k * sin(0.) << endl << "L = " << a << endl << "k = " << k << endl;
//
//    cout << Kstar_matrix << endl << endl;
//    cout << S_matrix_b << endl << endl;
//    cout << Kstar_matrix_b << endl << endl;
}

/**
 * @brief Build the right-hand side vector for the Dirichlet wall quasi-periodic scattering problem.
 *
 * Evaluates the incident plane wave and its normal derivative on every
 * boundary node and assembles the 2N-vector used as the RHS of the
 * combined-field system.
 *
 * @param[out] rhs    Complex vector (2N) to be filled.
 * @param      k      Exterior wavenumber (complex).
 * @param      kbar   Bloch wavenumber.
 * @param      delta  Inverse contrast parameter.
 */
void SpectralOperators::makeDirPeriodicRHS(VectorXcd &rhs, cpxd k, double kbar, double delta) {
    const BoundaryMesh *local_mesh = &mesh;
    int N = mesh.get_num_segments();
    rhs = VectorXcd::Zero(2 * N);

#pragma omp parallel for schedule(static) default(none) shared(rhs, local_mesh, N, k, kbar, delta)
    for (int i = 0; i < N; i++) {
        Vector2d pt_i = mesh.get_vertex(i).point;
        Vector2d normal_i = mesh.get_vertex(i).normal;
//        rhs(i) = sin(k * pt_i.y()) * exp(-cpxd(0, 1.0) * kbar * pt_i.x());
//        rhs(i + N) =               (k * normal_i.y() * cos(k * pt_i.y())
//                                    - cpxd(0, 1.0) * kbar * normal_i.x() * sin(k * pt_i.y()))
//                                   * delta * exp(-cpxd(0, 1.0) * kbar * pt_i.x());
        cpxd alpha = k * sin(0.), beta = k * cos(0.);
        cpxd u_i = exp(cpxd(0, 1.0) * (alpha * pt_i.x() - beta * pt_i.y()));
        cpxd u_r = exp(cpxd(0, 1.0) * (alpha * pt_i.x() + beta * pt_i.y()));

        rhs(i) = u_i - u_r;
        cpxd du_dn = (alpha * normal_i.x() - beta * normal_i.y()) * u_i - (alpha * normal_i.x() + beta * normal_i.y()) * u_r;
        rhs(i + N) = cpxd(0., 1.) * du_dn * delta;
    }

//    rhs *= -2.0 * cpxd(0, 1.0);
}


VectorXcd SpectralOperators::PeriodicDensityStabilized(cpxd k, cpxd k_b, double kbar, double delta, double a, cpxd eta) {
//    k   *= cpxd(1., -EPS); // limiting absorption principle
//    k_b *= cpxd(1., -EPS);
    int N = mesh.get_num_segments();

    MatrixXcd A_matrix = MatrixXcd::Zero(2 * N, 2 * N);
    MatrixXcd S_matrix, Kstar_matrix,
            S_matrix_b, Kstar_matrix_b;

    SpectralOperators::DirPeriodicS(S_matrix, k, kbar, a);
    SpectralOperators::DirPeriodicS_diagonal(S_matrix_b, k_b, kbar, a);
    SpectralOperators::DirPeriodicKstar(Kstar_matrix, k, kbar, a);
    SpectralOperators::DirPeriodicKstar_diagonal(Kstar_matrix_b, k_b, kbar, a);

    A_matrix.block(0, 0, N, N) = S_matrix_b;
    A_matrix.block(0, N, N, N) = -S_matrix / delta;
    A_matrix.block(N, 0, N, N) = -0.5 * MatrixXcd::Identity(N, N) + Kstar_matrix_b;
    A_matrix.block(N, N, N, N) = -(0.5 * MatrixXcd::Identity(N, N) + Kstar_matrix);

    VectorXcd F;
    makeDirPeriodicRHS(F, k, kbar, delta);

    VectorXcd psi = A_matrix.completeOrthogonalDecomposition().pseudoInverse() * F;

//    //residual
//    Eigen::VectorXcd u_min = svd.matrixU().col(2*N-1);
//    Eigen::VectorXcd v_min = svd.matrixV().col(2*N-1);
//
//    cpxd alpha = u_min.dot(F) / sigmamin;
//    Eigen::VectorXcd v_y = v_min.tail(N);
//    Eigen::VectorXcd y_res = alpha * v_y;
//    MatrixXcd A_matrix_2 = MatrixXcd::Zero(2 * N, 2 * N);
//    A_matrix_2.block(0, 0, N, N) = S_matrix_b;
//    A_matrix_2.block(0, N, N, N) = -S_matrix / delta;
//    A_matrix_2.block(N, 0, N, N) = -0.5 * MatrixXcd::Identity(N, N) + Kstar_matrix_b;
//    A_matrix_2.block(N, N, N, N) = -(0.5 * MatrixXcd::Identity(N, N) + Kstar_matrix) - 0.*cpxd(0, 1.0) * eta * S_matrix;
//    cout << "Residual norm: " << (A_matrix_2 * psi - F).norm() / F.norm() << endl;
//
//    psi.tail(N) += y_res;
    psi.tail(N) /= delta;

//    cout << (S_matrix_b * psi.head(N)).norm() / (Kstar_matrix_b * psi.head(N)).norm() << endl;

    return psi;

    /*MatrixXcd A_matrix(N, N), S_matrix, Kstar_matrix,
              S_matrix_b, Kstar_matrix_b;

    SpectralOperators::DirPeriodicS(S_matrix, k, kbar, a);
    SpectralOperators::DirPeriodicS(S_matrix_b, k_b, kbar, a);
    SpectralOperators::DirPeriodicKstar(Kstar_matrix, k, kbar, a);
    SpectralOperators::DirPeriodicKstar(Kstar_matrix_b, k_b, kbar, a);

    Eigen::PartialPivLU<MatrixXcd> S_b_lu(S_matrix_b);
//    auto S_b_lu = S_matrix_b.ldlt();

    A_matrix = (-0.5 * MatrixXcd::Identity(N, N) + Kstar_matrix_b) *
               S_b_lu.solve(S_matrix) - delta * (0.5 * MatrixXcd::Identity(N, N) + Kstar_matrix);

    VectorXcd rhs(2*N);
    SpectralOperators::makeDirPeriodicRHS(rhs, k, kbar, delta);

    VectorXcd F(N);
    VectorXcd F1 = rhs.head(N);
    VectorXcd F2 = rhs.tail(N);
    F = F2 - (-0.5 * MatrixXcd::Identity(N,N) + Kstar_matrix_b) * S_b_lu.solve(F1);

    Eigen::PartialPivLU<MatrixXcd> A_lu(A_matrix);
    VectorXcd density = A_lu.solve(F);
    return density;*/
}


/**
 * @brief Compute the capacitance matrix for the periodic geometry.
 *
 * Solves S ψ = 1 on each mesh to find the potential ψ corresponding to
 * a constant density, then integrates ψ against the density to find the
 * capacitance between each pair of meshes.  This is used for the low-frequency
 * asymptotics of the periodic scattering problem.
 *
 * @param[out] C     Complex matrix (N_mesh × N_mesh) to be filled.
 * @param      S     Single-layer operator matrix (Ntotal × Ntotal).
 * @return           Matrix of potentials ψ for each mesh.
 */
MatrixXcd SpectralOperators::makeCapacitanceMatrix(MatrixXcd &C, const MatrixXcd &S) {
    int N_mesh = mesh.get_num_meshes();
    MatrixXcd psi_matrix(mesh.get_num_segments(), N_mesh);

    C = MatrixXcd::Zero(N_mesh, N_mesh);
    auto temp = S.partialPivLu();

    for (int mesh_j = 0; mesh_j < N_mesh; mesh_j++) {
        int start_jdx = mesh.get_start_index(mesh_j);
        int end_jdx = mesh.get_end_index(mesh_j);

        VectorXcd ones = VectorXcd::Zero(S.rows());
        for (int idx = start_jdx; idx <= end_jdx; idx++) {
            ones(idx) = 1.0;
        }
        VectorXcd psi = temp.solve(ones);
        psi_matrix.col(mesh_j) = psi;

        for (int mesh_i = 0; mesh_i < N_mesh; mesh_i++) {
            int start_idx = mesh.get_start_index(mesh_i);
            int end_idx = mesh.get_end_index(mesh_i);

            for (int idx = start_idx; idx <= end_idx; idx++) {
                C(mesh_i, mesh_j) -= psi(idx) * mesh.get_vertex(idx).sigma;
            }
        }
    }
    return psi_matrix;
}

void SpectralOperators::CrystalS_diagonal(MatrixXcd &S, cpxd k, Vector2d alpha, Vector2d a1, Vector2d a2) {
    assemble_single_layer_diagonal(S, mesh, R_j_Kress,
                                   GreenKernels::BiperiodicKernel{k, alpha, a1, a2});
}

void SpectralOperators::CrystalS(MatrixXcd &S, cpxd k, Vector2d alpha, Vector2d a1, Vector2d a2) {
    assemble_single_layer(S, mesh, R_j_Kress,
                          GreenKernels::BiperiodicKernel{k, alpha, a1, a2});
}


void SpectralOperators::CrystalKstar_diagonal(MatrixXcd &Kstar, cpxd k, Vector2d alpha, Vector2d a1, Vector2d a2) {
    assemble_target_normal_diagonal(Kstar, mesh, R_j_Kress,
                                    GreenKernels::BiperiodicKernel{k, alpha, a1, a2});
}

void SpectralOperators::CrystalKstar(MatrixXcd &Kstar, cpxd k, Vector2d alpha, Vector2d a1, Vector2d a2) {
    assemble_target_normal(Kstar, mesh, R_j_Kress,
                           GreenKernels::BiperiodicKernel{k, alpha, a1, a2});
}

void SpectralOperators::CrystalK_diagonal(MatrixXcd &K, cpxd k, Vector2d alpha, Vector2d a1, Vector2d a2) {
    assemble_source_normal_diagonal(K, mesh, R_j_Kress,
                                    GreenKernels::BiperiodicKernel{k, alpha, a1, a2});
}

void SpectralOperators::CrystalK(MatrixXcd &K, cpxd k, Vector2d alpha, Vector2d a1, Vector2d a2) {
    assemble_source_normal(K, mesh, R_j_Kress,
                           GreenKernels::BiperiodicKernel{k, alpha, a1, a2});
}

void SpectralOperators::CrystalA(MatrixXcd &A_matrix, cpxd k, cpxd k_b, Vector2d alpha, Vector2d a1, Vector2d a2, double delta) {
    int N = mesh.get_num_segments();

    A_matrix = MatrixXcd::Zero(2 * N, 2 * N);
    MatrixXcd S_matrix, Kstar_matrix,
            S_matrix_b, Kstar_matrix_b;

    SpectralOperators::CrystalS(S_matrix, k, alpha, a1, a2);
//    SpectralOperators::CrystalS(S_matrix_b, k_b, alpha, a1, a2);
    SpectralOperators::S(S_matrix_b, k_b);
    SpectralOperators::CrystalKstar(Kstar_matrix, k, alpha, a1, a2);
//    SpectralOperators::CrystalKstar(Kstar_matrix_b, k_b, alpha, a1, a2);
    SpectralOperators::Kstar(Kstar_matrix_b, k_b);

    A_matrix.block(0, 0, N, N) = S_matrix_b;
    A_matrix.block(0, N, N, N) = -S_matrix;
    A_matrix.block(N, 0, N, N) = -0.5 * MatrixXcd::Identity(N, N) + Kstar_matrix_b;
    A_matrix.block(N, N, N, N) = -(0.5 * MatrixXcd::Identity(N, N) + Kstar_matrix) * delta;

//    cout << "S norm = " << S_matrix.norm() << endl;
//    cout << "K* norm = " << Kstar_matrix.norm() << endl;
//    cout << "S_b norm = " << S_matrix_b.norm() << endl;
//    cout << "K*_b norm = " << Kstar_matrix_b.norm() << endl;
}

/*void SpectralOperators::DefectCrystalA(MatrixXcd &A_matrix, BoundaryMesh &phantom_mesh, cpxd k, cpxd k_b, Vector2d alpha, Vector2d a1, Vector2d a2, double delta) {
    int N = mesh.get_num_segments();

    A_matrix = MatrixXcd::Zero(2 * N, 2 * N);
    MatrixXcd S_matrix, Kstar_matrix,
            S_matrix_b, Kstar_matrix_b;

    SpectralOperators::S(S_matrix, k);
    SpectralOperators::S(S_matrix_b, k_b);
    SpectralOperators::Kstar(Kstar_matrix, k);
    SpectralOperators::Kstar(Kstar_matrix_b, k_b);

    A_matrix.block(0, 0, N, N) = S_matrix_b;
    A_matrix.block(0, N, N, N) = -S_matrix;
    A_matrix.block(N, 0, N, N) = -0.5 * MatrixXcd::Identity(N, N) + Kstar_matrix_b;
    A_matrix.block(N, N, N, N) = -(0.5 * MatrixXcd::Identity(N, N) + Kstar_matrix) * delta;

    auto helmholtz_2D_phantom = [&phantom_mesh](cpxd k, Vector2d alpha, const Vector2d &x,
                                        Vector2d a1, Vector2d a2, double epsilon = 0.0, int N = 20) {
                                            
                                            return Kernels::helmholtz_2D_biperiodic(k, alpha, x, y, a1, a2) - Kernels::helmholtz_2D_periodic(k, alpha.x(), x, y, a1.x());
                                        };
    auto grad_helmholtz_2D_phantom = [&phantom_mesh](cpxd k, Vector2d alpha, const Vector2d &x,
                                        Vector2d a1, Vector2d a2, double epsilon = 0.0, int N = 20) {
                                            return Kernels::grad_helmholtz_2D_biperiodic(k, alpha, x, y, a1, a2) - Kernels::grad_helmholtz_2D_periodic(k, alpha.x(), x, y, a1.x());
                                        };

#pragma omp parallel for schedule(static) default(none) shared(A_matrix, phantom_mesh, k, k_b, alpha, a1, a2, delta)
    for (int i = 0; i < N; i++) {
        Vector2d point_i = mesh.get_vertex(i).point;
        Vector2d normal_i = mesh.get_vertex(i).normal;
        for (int j = 0; j < N; j++) {
            Vector2d point_j = mesh.get_vertex(j).point;
            A_matrix(i, j + N) -= helmholtz_2D_phantom(k, alpha, point_i, a1, a2) * mesh.get_vertex(j).sigma;
            A_matrix(i + N, j + N) -= delta *grad_helmholtz_2D_phantom(k, alpha, point_i, a1, a2).dot(normal_i) * mesh.get_vertex(j).sigma;
        }
    }
}*/
