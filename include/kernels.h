//
// Created by lara on 10/29/25.
//

#ifndef SUBWAVELENGTHRESONATORS_KERNELS_H
#define SUBWAVELENGTHRESONATORS_KERNELS_H

#include "Eigen/Dense"
#include <concepts>
#include <complex>
#include <cmath>
#include "tools.h"
#include <iostream>
#if __has_include(<omp.h>)
#include <omp.h>
#endif

// Include bessel library directly here since kernels.h functions are inline
// This is safe because each translation unit will get its own copy of inline functions
#include "bessel-library.hpp"

using namespace Eigen;
using namespace Tools;

//template <typename K>
//concept kernel_type = requires(const K kernel,
//                               const double k,
//                               const Vector2d &x,
//                               const Vector2d &y) {
//
//};

namespace Kernels {

    /** @brief 2D Laplace free-space Green's function. */
    inline double laplace_2D(const Vector2d &x, const Vector2d &y) {
        return -0.5 * M_1_PI * log((x - y).norm());
    }

    /** @brief 2D Helmholtz free-space Green's function. */
    inline cpxd helmholtz_2D(cpxd k, const Vector2d &x, const Vector2d &y) {
        double r = (x - y).norm();
        return -0.25 * cpxd(0, 1.0) * bessel::cyl_h1(0, k * r);
    }

    /** @brief Dirichlet image-corrected 2D Helmholtz Green's function. */
    inline cpxd dir_helmholtz_2D(cpxd k, const Vector2d &x, const Vector2d &y) {
        double r = (x - y).norm();
        double r2 = Vector2d(x.x() - y.x(), x.y() + y.y()).norm();
        return -0.25 * cpxd(0, 1.0) * bessel::cyl_h1(0, k * r) +
                0.25 * cpxd(0, 1.0) * bessel::cyl_h1(0, k * r2);
    }

    /** @brief Gradient of the 2D Helmholtz Green's function with respect to x. */
    inline Vector2cd grad_helmholtz_2D(cpxd k, const Vector2d &x, const Vector2d &y) {
        Vector2d r_vec = x - y;
        double r = r_vec.norm();
        cpxd factor = 0.25 * cpxd(0, 1.0) * k * bessel::cyl_h1(1, k * r) / r;
        return factor * r_vec;
    }

    /** @brief Gradient of Dirichlet image-corrected Helmholtz Green's function. */
    inline Vector2cd dir_grad_helmholtz_2D(cpxd k, const Vector2d &x, const Vector2d &y) {
        Vector2d r_vec = x - y;
        Vector2d r_vec2 = Vector2d(x.x() - y.x(), x.y() + y.y());
        double r = r_vec.norm();
        double r2 = r_vec2.norm();
        cpxd factor = 0.25 * cpxd(0, 1.0) * k * bessel::cyl_h1(1, k * r) / r;
        cpxd factor2 = 0.25 * cpxd(0, 1.0) * k * bessel::cyl_h1(1, k * r2) / r2;
        return factor * r_vec - factor2 * r_vec2;
    }

    // default period = 1 in the x direction
    /** @brief 1D-periodic 2D Laplace Green's function in x direction. */
    inline cpxd laplace_2D_periodic(const Vector2d &x, const Vector2d &y, double period = 1.0) {
        double rep_period = 1.0 / period;
        double sinhr = sinh(rep_period * M_PI * (x.y() - y.y()));
        double sinr = sin(rep_period * M_PI * (x.x() - y.x()));

        return 0.25 * M_1_PI * log(sinhr * sinhr + sinr * sinr);
    }

    /** @brief Dirichlet image-corrected periodic Laplace Green's function. */
    inline cpxd dir_laplace_2D_periodic(const Vector2d &x, const Vector2d &y, double period = 1.0) {
        double rep_period = 1.0 / period;
        double sinhr = sinh(rep_period * M_PI * (x.y() - y.y()));
        double sinr = sin(rep_period * M_PI * (x.x() - y.x()));
        double sinhs = sinh(rep_period * M_PI * (y.y() + x.y()));

        return 0.25 * M_1_PI * log(sinhr * sinhr + sinr * sinr)
             - 0.25 * M_1_PI * log(sinhs * sinhs + sinr * sinr);
    }

    /** @brief Gradient of periodic Laplace Green's function. */
    inline Vector2cd grad_laplace_2D_periodic(const Vector2d &x, const Vector2d &y, double period = 1.0) {
        double rep_period = 1.0 / period;
        double delta_x = x.x() - y.x();
        double delta_y = x.y() - y.y();

        double sinr = sin(rep_period * M_PI * delta_x);
        double cosr = cos(rep_period * M_PI * delta_x);
        double sinhr = sinh(rep_period * M_PI * delta_y);
        double coshr = cosh(rep_period * M_PI * delta_y);

        double denom = sinhr * sinhr + sinr * sinr;

        cpxd dG_dx = 0.5 * rep_period * (sinr * cosr) / denom;
        cpxd dG_dy = 0.5 * rep_period * (sinhr * coshr) / denom;

        return {dG_dx, dG_dy};
    }

    /** @brief Gradient of Dirichlet periodic Laplace Green's function. */
    inline Vector2cd dir_grad_laplace_2D_periodic(const Vector2d &x, const Vector2d &y, double period = 1.0) {
        double rep_period = 1.0 / period;
        double delta_x = x.x() - y.x();
        double delta_y = x.y() - y.y();
        double sum_y = x.y() + y.y();

        double sinr = sin(rep_period * M_PI * delta_x);
        double cosr = cos(rep_period * M_PI * delta_x);
        double sinhr = sinh(rep_period * M_PI * delta_y);
        double coshr = cosh(rep_period * M_PI * delta_y);
        double sinhs = sinh(rep_period * M_PI * sum_y);
        double coshs = cosh(rep_period * M_PI * sum_y);

        double denom = sinhr * sinhr + sinr * sinr;
        double denom2 = sinhs * sinhs + sinr * sinr;

        cpxd dG_dx = (sinr * cosr) / denom;
        cpxd dG_dy = (sinhr * coshr) / denom;

        dG_dx -= (sinr * cosr) / denom2;
        dG_dy -= (sinhs * coshs) / denom2;

        return {0.5 * rep_period * dG_dx, 0.5 * rep_period * dG_dy};
    }

    /**
     * @brief Smooth (non-singular) correction turning the periodic static Laplace
     *        kernel into the quasi-periodic one of Ammari et al. (arXiv:2512.05370v2).
     *
     * The 1D-quasi-periodic, static (omega = 0) Green's function used in that paper
     * (their eqs (A.3) and (A.7), with unit period a = 1 and quasi-momentum alpha) is
     *
     *   G^{alpha,0}(x,y) = -sum_{k in Z} e^{i(2 pi k + alpha)(x1-y1)}
     *                                    e^{-|2 pi k + alpha| |x2-y2|} / (2 |2 pi k + alpha|)   (alpha != 0),
     *   G^{0,0}(x,y)     =  |x2-y2|/2 - sum_{k != 0} e^{i 2 pi k (x1-y1)}
     *                                    e^{-|2 pi k| |x2-y2|} / (2 |2 pi k|).
     *
     * Both carry the SAME logarithmic singularity (1/2pi) log|x-y| as `laplace_2D_periodic`,
     * so their difference is smooth. This routine returns
     *
     *   C^alpha(x,y) = G^{alpha,0}(x,y) - laplace_2D_periodic(x,y)
     *               (= G^{alpha,0}(x,y) - G^{0,0}(x,y) + log(2)/(2 pi)),
     *
     * i.e. the smooth term such that  laplace_2D_periodic(x,y) + C^alpha(x,y) = G^{alpha,0}(x,y).
     * The series converges like O(1/k^2) (even at x2 = y2), so a moderate cutoff M suffices.
     * The +log(2)/(2 pi) accounts for the constant offset between `laplace_2D_periodic`
     * and the paper's G^{0,0} normalisation. Only unit period is supported (a = 1).
     *
     * @param alpha  Quasi-momentum (Bloch phase) in [-pi, pi].
     * @param x      Observation point.
     * @param y      Source point.
     * @param M      Reciprocal-lattice truncation (number of positive k).
     */
    inline cpxd laplace_2D_quasiperiodic_correction(double alpha, const Vector2d &x,
                                                    const Vector2d &y, int M = 200) {
        const double log2_over_2pi = 0.5 * M_1_PI * std::log(2.0);
        if (std::abs(alpha) < 1e-12) {
            // C^0 is the constant offset between the two normalisations.
            return cpxd(log2_over_2pi, 0.0);
        }
        const double x1 = x.x() - y.x();
        const double x2 = std::abs(x.y() - y.y());
        const cpxd I(0.0, 1.0);

        cpxd corr = cpxd(log2_over_2pi, 0.0);
        // Isolated k = 0 term of the quasi-periodic sum (no periodic counterpart).
        corr -= std::exp(I * alpha * x1) * std::exp(-std::abs(alpha) * x2)
                / (2.0 * std::abs(alpha));
        // -|x2|/2 term coming from G^{0,0}.
        corr -= 0.5 * x2;
        // Remaining k != 0 terms: quasi-periodic minus periodic (telescoping, O(1/k^2)).
        for (int k = 1; k <= M; ++k) {
            for (int s = -1; s <= 1; s += 2) {
                const double bk = 2.0 * M_PI * s * k + alpha; // 2 pi k + alpha
                const double b0 = 2.0 * M_PI * s * k;         // 2 pi k
                corr -= std::exp(I * bk * x1) * std::exp(-std::abs(bk) * x2) / (2.0 * std::abs(bk));
                corr += std::exp(I * b0 * x1) * std::exp(-std::abs(b0) * x2) / (2.0 * std::abs(b0));
            }
        }
        return corr;
    }

    /* complex double 2D Helmholtz periodic Green's function
     * using Ewald's method for a 1D periodic array in the x direction.
     * Default period = 1 in the x direction.
     * Based on:
     * F. Capolino, D.R. Wilton, W.A. Johnson,
     * Efficient computation of the 2-D Green's function for
     * 1-D periodic structures using the Ewald method
     *
     * Parameters:
     * - k: wavenumber
     * - kbar: wavenumber component along the periodic direction
     * - x: observation point (2D vector)
     * - y: source point (2D vector)
     * - period: periodicity in the x direction (default 1.0)
     * - epsilon: Ewald splitting parameter (default 1.0)
     * - N: number of terms in the spectral and spatial sums (default 10)
     */
    inline cpxd helmholtz_2D_periodic(cpxd k, double kbar, const Vector2d &x, const Vector2d &y,
                                          double period = 1.0, double epsilon = 0.0, int N = 10) {
        if (epsilon < 1e-12) {
            epsilon = sqrt(M_PI) / period;
        }
        double rep_period = 1.0 / period;
        cpxd I = cpxd(0, 1.0);
        double kx0 = kbar;
        cpxd kxp, Ikzp;

        cpxd Gspectral = 0.0,
                         Gspatial = 0.0;

        double delta_z = abs(x.y() - y.y());
        // Spectral sum
        for (int p = -N; p <= N; p++) {
            kxp = kx0 + 2.0 * M_PI * p * rep_period;
            // kzp = sqrt(k^2 - kxp^2) with the branch cut moved to the negative imaginary axis
            // of the argument: on and above the real k-axis this is the usual outgoing choice
            // (Im(kzp) <= 0 after Ikzp = -i kzp), and for Im(k) < 0 it is the ANALYTIC
            // continuation of that choice. Selecting the outgoing branch pointwise instead
            // (old code: flip on sign of Im(sqrt)) conjugate-reflects the kernel across the
            // real axis, which breaks root searches for resonances just below/above it.
            cpxd sqrt_arg2 = k * k - kxp * kxp;
            cpxd sqrt_arg = sqrt(sqrt_arg2);
            if (sqrt_arg2.real() < 0 && sqrt_arg2.imag() < 0) sqrt_arg = -sqrt_arg;
            Ikzp = -I * sqrt_arg;
            // High evanescent orders at large transverse separation make exp(Ikzp*delta_z)
            // overflow while the true (exp(-eps^2 delta_z^2)-suppressed) contribution is
            // negligible; skip them to avoid Inf*0 = NaN.
            if (abs(Ikzp.real()) * delta_z > 300.0) continue;
            if (abs(Ikzp.imag()) < 1e-12) {
                Gspectral += exp(-I * kxp * (x.x() - y.x())) / Ikzp *
                              (exp(Ikzp * delta_z) * erfc(Ikzp.real() / (2.0 * epsilon) + epsilon * delta_z) +
                              exp(-Ikzp * delta_z) * erfc(Ikzp.real() / (2.0 * epsilon) - epsilon * delta_z));
            } else if (abs(Ikzp.real() ) < 1e-12) {
                cpxd erfc1 = erfc_complex(Ikzp / (2.0 * epsilon) + epsilon * delta_z);
                cpxd erfc1_conj = 2.0 - conj(erfc1);
                Gspectral += exp(-I * kxp * (x.x() - y.x())) / Ikzp *
                              (exp(Ikzp * delta_z) * erfc1 + exp(-Ikzp * delta_z) * erfc1_conj);
            } else {
                Gspectral += exp(-I * kxp * (x.x() - y.x())) / Ikzp *
                              (exp(Ikzp * delta_z) * erfc_complex(Ikzp / (2.0 * epsilon) + epsilon * delta_z) +
                              exp(-Ikzp * delta_z) * erfc_complex(Ikzp / (2.0 * epsilon) - epsilon * delta_z));
            }
        }
        Gspectral *= 0.25 * rep_period;

        cpxd ratio = k / (2.0 * epsilon);
//        cout << "ratio: " << ratio << endl;

        // Spatial sum
        for (int p = -N; p <= N; p++) {
            Vector2d y_n = y + Vector2d(p * period, 0.0);
            double R_p = (x - y_n).norm();
            if (R_p < 1e-12) {
                continue; // skip singularity
            }
            double Repsilon_2 = R_p * R_p * epsilon * epsilon;
            double E = Tools::E1(Repsilon_2);
            cpxd term = exp(-I * kx0 * period * (double) p);
            cpxd term2 = 0;
            for (int q = 0; q <= 2 * N; q++) {
                term2 += pow(ratio, 2 * q) / tgamma(q + 1) * E;
                E = (exp(-Repsilon_2) - Repsilon_2 * E) / (double) (q + 1);
            }
            Gspatial += term * term2;
//            cout << Gspatial << endl;
        }

        Gspatial *= 0.25 * M_1_PI;

        return -(Gspatial + Gspectral);
    }

    /** @brief Gradient of Ewald-split 1D-periodic Helmholtz Green's function. */
    inline Vector2cd grad_helmholtz_2D_periodic(cpxd k, double kbar, const Vector2d &x, const Vector2d &y,
                                         double period = 1.0, double epsilon = 0.0, int N = 10) {
        if (epsilon < 1e-12) {
            epsilon = sqrt(M_PI) / period;
        }

        double rep_period = 1.0 / period;
        cpxd I = cpxd(0, 1.0);
        double kx0 = kbar;
        cpxd kxp, Ikzp;

        Vector2cd Gspectral = Vector2cd::Zero(),
                   Gspatial = Vector2cd::Zero();

        double diff_x = x.x() - y.x();
        double delta_z = x.y() - y.y();
        double sign_z = (delta_z > 0) ? 1.0 : (delta_z < 0 ? -1.0 : 0.0);
        delta_z = abs(delta_z);

        // Spectral sum
        for (int p = -N; p <= N; p++) {
            kxp = kx0 + 2.0 * M_PI * p * rep_period;
            // Analytic-continuation branch of kzp across the real k-axis; see
            // helmholtz_2D_periodic.
            cpxd sqrt_arg2 = k * k - kxp * kxp;
            cpxd sqrt_arg = sqrt(sqrt_arg2);
            if (sqrt_arg2.real() < 0 && sqrt_arg2.imag() < 0) sqrt_arg = -sqrt_arg;
            Ikzp = -I * sqrt_arg;
            // Skip high evanescent orders at large transverse separation (negligible but
            // exp(Ikzp*delta_z) would overflow); see helmholtz_2D_periodic.
            if (abs(Ikzp.real()) * delta_z > 300.0) continue;
            cpxd exp_term = exp(-I * kxp * diff_x) / Ikzp;
            cpxd exp_z = exp(Ikzp * delta_z);

            if (abs(Ikzp.imag()) < 1.e-12) {
                cpxd scalar_term1 = 1.0/exp_z * erfc(Ikzp.real() / (2.0 * epsilon) - epsilon * delta_z);
                cpxd scalar_term2 =     exp_z * erfc(Ikzp.real() / (2.0 * epsilon) + epsilon * delta_z);

                cpxd sum_x = -I * kxp * (scalar_term1 + scalar_term2);
                cpxd sum_z = Ikzp * sign_z * (scalar_term2 - scalar_term1);
                sum_z += sign_z * epsilon *
                         (exp_z * derfc_complex(Ikzp / (2.0 * epsilon) + epsilon * delta_z)
                          - (1.0/exp_z) * derfc_complex(Ikzp / (2.0 * epsilon) - epsilon * delta_z));

                Gspectral.x() += exp_term * sum_x;
                Gspectral.y() += exp_term * sum_z;
            } else if (abs(Ikzp.real()) < 1.e-12) {
                cpxd erfc1 = erfc_complex(Ikzp / (2.0 * epsilon) + epsilon * delta_z);
                cpxd erfc1_conj = 2.0 - conj(erfc1);

                cpxd sum_x = -I * kxp * (1.0/exp_z * erfc1_conj + exp_z * erfc1);
                cpxd sum_z = Ikzp * sign_z * (exp_z * erfc1 - 1.0/exp_z * erfc1_conj);

                sum_z += sign_z * epsilon *
                         (exp_z * derfc_complex(Ikzp / (2.0 * epsilon) + epsilon * delta_z)
                          - (1.0/exp_z) * derfc_complex(Ikzp / (2.0 * epsilon) - epsilon * delta_z));
                Gspectral.x() += exp_term * sum_x;
                Gspectral.y() += exp_term * sum_z;
            } else {
                cpxd scalar_term1 = 1.0 / exp_z * erfc_complex(Ikzp / (2.0 * epsilon) - epsilon * delta_z);
                cpxd scalar_term2 = exp_z * erfc_complex(Ikzp / (2.0 * epsilon) + epsilon * delta_z);

                cpxd sum_x = -I * kxp * (scalar_term1 + scalar_term2);
                cpxd sum_z = Ikzp * sign_z * (scalar_term2 - scalar_term1);
                sum_z += sign_z * epsilon *
                         (exp_z * derfc_complex(Ikzp / (2.0 * epsilon) + epsilon * delta_z)
                          - (1.0 / exp_z) * derfc_complex(Ikzp / (2.0 * epsilon) - epsilon * delta_z));

                Gspectral.x() += exp_term * sum_x;
                Gspectral.y() += exp_term * sum_z;
            }
        }
        Gspectral *= 0.25 * rep_period;

        cpxd ratio = k / (2.0 * epsilon);

        // Spatial sum
        for (int p = -N; p <= N; p++) {
            Vector2d y_n = y + Vector2d(p * period, 0.0);
            Vector2d r_vec = x - y_n;
            double R_p = r_vec.norm();
            if (R_p < 1e-12) {
                continue; // skip singularity
            }
            double Repsilon_2 = R_p * R_p * epsilon * epsilon;
            double E = Tools::E1(Repsilon_2);
            cpxd term1 = exp(-I * kx0 * period * (double) p);
            cpxd term2 = Tools::E0(Repsilon_2);
            for (int q = 1; q <= 2 * N; q++) {
                term2 += pow(ratio, 2 * q) / tgamma(q + 1) * E;
                E = (exp(-Repsilon_2) - Repsilon_2 * E) / (double) q;
            }
            Gspatial += term1 * term2 * -r_vec;
        }
        Gspatial *= 0.5 * M_1_PI * epsilon * epsilon;

        return -(Gspatial + Gspectral);
    }

    /** @brief Reference implementation via image subtraction of periodic Helmholtz kernels. */
    inline cpxd dir_helmholtz_2D_periodic_ref(double k, double kbar,
                                  const Vector2d &x, const Vector2d &y,
                                  double period = 1.0, double epsilon = 1.0, int N = 10)
    {
        Vector2d y_ref(y.x(), -y.y());
        auto G1 = helmholtz_2D_periodic(k, kbar, x, y,     period, epsilon, N);
        auto G2 = helmholtz_2D_periodic(k, kbar, x, y_ref, period, epsilon, N);
        return G1 - G2;
    }

    /** @brief Dirichlet periodic Helmholtz Green's function computed by Ewald splitting. */
    inline cpxd dir_helmholtz_2D_periodic(cpxd k, double kbar, const Vector2d &x, const Vector2d &y,
                                                 double period = 1.0, double epsilon = 0.0, int N = 10) {
        if (epsilon < 1e-12) {
            epsilon = sqrt(M_PI) / period;
        }

        double rep_period = 1.0 / period;
        cpxd I = cpxd(0, 1.0);
        double kx0 = kbar;
        cpxd kxp, Ikzp;

        cpxd Gspectral = 0.0,
                        Gspatial = 0.0;

        double diff_z = abs(x.y() - y.y());
        double sum_z = abs(x.y() + y.y());
        // Spectral sum
        for (int p = -N; p <= N; p++) {
            // classical x - y
            kxp = kx0 + 2.0 * M_PI * p * rep_period;
            cpxd sqrt_arg = sqrt(k * k - kxp * kxp);
            // choose branch with Im(kzp) <= 0
            Ikzp = (sqrt_arg.imag() < 0) ? I * sqrt_arg : -I * sqrt_arg;
            cpxd term1 = exp(-I * kxp * (x.x() - y.x())) / Ikzp;
            cpxd exp_z = exp(Ikzp * diff_z);
            cpxd exp_zs = exp(Ikzp * sum_z);

            if (abs(Ikzp.imag()) < 1.e-12) {
                Gspectral += term1 *
                             (exp(Ikzp * diff_z) * erfc(Ikzp.real() / (2.0 * epsilon) + epsilon * diff_z) +
                              exp(-Ikzp * diff_z) * erfc(Ikzp.real() / (2.0 * epsilon) - epsilon * diff_z));

                // Dirichlet x.x() - y.x(), y.y() + x.y()
                Gspectral -= term1 *
                             (exp_zs * erfc(Ikzp.real() / (2.0 * epsilon) + epsilon * sum_z) +
                              1./exp_zs * erfc(Ikzp.real() / (2.0 * epsilon) - epsilon * sum_z));
            } else if (abs(Ikzp.real()) < 1.e-12) {
                cpxd erfc1 = erfc_complex(Ikzp / (2.0 * epsilon) + epsilon * diff_z);
                cpxd erfc1_conj = 2.0 - conj(erfc1); // erfc_complex(Ikzp / (2.0 * epsilon) - epsilon * diff_z);
                Gspectral += term1 * (exp_z * erfc1 + 1. / exp_z * erfc1_conj);
//            cout << Gspectral << endl;

                // Dirichlet x.x() - y.x(), y.y() + x.y()
                cpxd erfc2 = erfc_complex(Ikzp / (2.0 * epsilon) + epsilon * sum_z);
                cpxd erfc2_conj = 2.0 - conj(erfc2); // erfc_complex(Ikzp / (2.0 * epsilon) - epsilon * sum_z);
                Gspectral -= term1 * (exp_zs * erfc2 + 1. / exp_zs * erfc2_conj);
            } else {
                Gspectral += term1 *
                                (exp_z * erfc_complex(Ikzp / (2.0 * epsilon) + epsilon * diff_z) +
                              1./exp_z * erfc_complex(Ikzp / (2.0 * epsilon) - epsilon * diff_z));

                // Dirichlet x.x() - y.x(), y.y() + x.y()
                Gspectral -= term1 *
                                (exp_zs * erfc_complex(Ikzp / (2.0 * epsilon) + epsilon * sum_z) +
                              1./exp_zs * erfc_complex(Ikzp / (2.0 * epsilon) - epsilon * sum_z));
            }
        }
        Gspectral *= 0.25 * rep_period;

        cpxd ratio = k / (2.0 * epsilon);
//        cout << "ratio: " << ratio << endl;

        double delta_x = x.x() - y.x();
        // Spatial sum
        for (int p = -N; p <= N; p++) {
            Vector2d y_n = y + Vector2d(p * period, 0.0);
            double R_p1 = (delta_x - p * period) * (delta_x - p * period) + diff_z * diff_z;
            double R_p2 = (delta_x - p * period) * (delta_x - p * period) + sum_z * sum_z;
            double Repsilon_2A = R_p1 * epsilon * epsilon;
            double Repsilon_2B = R_p2 * epsilon * epsilon;
            if (R_p1 < 1e-12) {
                double EB = Tools::E1(Repsilon_2B);
                cpxd term = exp(-I * kx0 * period * (double) p);
                cpxd term2 = 0;
                for (int q = 0; q <= 2 * N; q++) {
                    term2 += pow(ratio, 2 * q) / tgamma(q + 1) * -EB;
                    EB = (exp(-Repsilon_2B) - Repsilon_2B * EB) / (double) (q + 1);
                }
                Gspatial += term * term2;
                continue; // skip singularity
            }
            double EA = Tools::E1(Repsilon_2A);
            double EB = Tools::E1(Repsilon_2B);
            cpxd term = exp(-I * kx0 * period * (double) p);
            cpxd term2 = 0;
            for (int q = 0; q <= 2 * N; q++) {
                term2 += pow(ratio, 2 * q) / tgamma(q + 1) * (EA - EB);
                EA = (exp(-Repsilon_2A) - Repsilon_2A * EA) / (double) (q + 1);
                EB = (exp(-Repsilon_2B) - Repsilon_2B * EB) / (double) (q + 1);
            }
            Gspatial += term * term2;
//            cout << Gspatial << endl;
        }

        Gspatial *= 0.25 * M_1_PI;

        return -(Gspatial + Gspectral);
    }


    /** @brief Gradient of the Dirichlet periodic Helmholtz Green's function. */
    inline Vector2cd dir_grad_helmholtz_2D_periodic(cpxd k, double kbar, const Vector2d &x, const Vector2d &y,
                                                double period = 1.0, double epsilon = 0.0, int N = 10) {
        if (epsilon < 1e-12) {
            epsilon = sqrt(M_PI) / period;
        }

        double rep_period = 1.0 / period;
        cpxd I = cpxd(0, 1.0);
        double kx0 = kbar;
        cpxd kxp, Ikzp;

        Vector2cd Gspectral = Vector2cd::Zero(),
                Gspatial = Vector2cd::Zero();

        double diff_x = x.x() - y.x();
        double diff_z = x.y() - y.y();
        double sum_z = x.y() + y.y();
        double sign_dz = (diff_z > 0) ? 1.0 : (diff_z < 0 ? -1.0 : 0.0);
        double sign_sz = (sum_z > 0) ? 1.0 : (sum_z < 0 ? -1.0 : 0.0);
        diff_z = abs(diff_z);
        sum_z = abs(sum_z);

        // Spectral sum
        for (int p = -N; p <= N; p++) {
            kxp = kx0 + 2.0 * M_PI * p * rep_period;
            cpxd sqrt_arg = sqrt(k * k - kxp * kxp);
            // choose branch with Im(kzp) <= 0
            Ikzp = (sqrt_arg.imag() < 0) ? I * sqrt_arg : -I * sqrt_arg;
            cpxd exp_term = exp(-I * kxp * diff_x) / Ikzp;

            // classical x - y
            if (abs(Ikzp.imag()) < 1.e-12) {
                cpxd exp_z = exp(Ikzp * diff_z);

                cpxd scalar_term1 = 1.0 / exp_z * erfc(Ikzp.real() / (2.0 * epsilon) - epsilon * diff_z);
                cpxd scalar_term2 = exp_z * erfc(Ikzp.real() / (2.0 * epsilon) + epsilon * diff_z);

                cpxd sum_x = -I * kxp * (scalar_term1 + scalar_term2);
                cpxd sum_z1 = Ikzp * sign_dz * (scalar_term2 - scalar_term1);
                sum_z1 += sign_dz * epsilon *
                          (exp_z * derfc_complex(Ikzp / (2.0 * epsilon) + epsilon * diff_z)
                           - (1.0 / exp_z) * derfc_complex(Ikzp / (2.0 * epsilon) - epsilon * diff_z));

                Gspectral.x() += exp_term * sum_x;
                Gspectral.y() += exp_term * sum_z1;

                // Dirichlet x.x() - y.x(), y.y() + x.y()
                exp_z = exp(Ikzp * sum_z);
                scalar_term1 = 1.0 / exp_z * erfc(Ikzp.real() / (2.0 * epsilon) - epsilon * sum_z);
                scalar_term2 = exp_z * erfc(Ikzp.real() / (2.0 * epsilon) + epsilon * sum_z);

                sum_x = -I * kxp * (scalar_term1 + scalar_term2);
                sum_z1 = Ikzp * sign_sz * (scalar_term2 - scalar_term1);
                sum_z1 += sign_sz * epsilon *
                          (exp_z * derfc_complex(Ikzp / (2.0 * epsilon) + epsilon * sum_z)
                           - (1.0 / exp_z) * derfc_complex(Ikzp / (2.0 * epsilon) - epsilon * sum_z));

                Gspectral.x() -= exp_term * sum_x;
                Gspectral.y() -= exp_term * sum_z1;
            } else if (abs(Ikzp.real()) < 1.e-12) {
                cpxd exp_z = exp(Ikzp * diff_z);

                cpxd erfc1 = erfc_complex(Ikzp / (2.0 * epsilon) + epsilon * diff_z);
                cpxd erfc1_conj = 2.0 - conj(erfc1); // erfc_complex(Ikzp / (2.0 * epsilon) - epsilon * diff_z);

                cpxd sum_x = -I * kxp * (1.0 / exp_z * erfc1_conj + exp_z * erfc1);
                cpxd sum_z1 = Ikzp * sign_dz * (exp_z * erfc1 - 1.0 / exp_z * erfc1_conj);
                sum_z1 += sign_dz * epsilon *
                          (exp_z * derfc_complex(Ikzp / (2.0 * epsilon) + epsilon * diff_z)
                           - (1.0 / exp_z) * derfc_complex(Ikzp / (2.0 * epsilon) - epsilon * diff_z));

                Gspectral.x() += exp_term * sum_x;
                Gspectral.y() += exp_term * sum_z1;

                // Dirichlet x.x() - y.x(), y.y() + x.y()
                exp_z = exp(Ikzp * sum_z);

                cpxd erfc2 = erfc_complex(Ikzp / (2.0 * epsilon) + epsilon * sum_z);
                cpxd erfc2_conj = 2.0 - conj(erfc2); // erfc_complex(Ikzp / (2.0 * epsilon) - epsilon * sum_z);

                sum_x = -I * kxp * (1.0 / exp_z * erfc2_conj + exp_z * erfc2);
                sum_z1 = Ikzp * sign_sz * (exp_z * erfc2 - 1.0 / exp_z * erfc2_conj);
                sum_z1 += sign_sz * epsilon *
                          (exp_z * derfc_complex(Ikzp / (2.0 * epsilon) + epsilon * sum_z)
                           - (1.0 / exp_z) * derfc_complex(Ikzp / (2.0 * epsilon) - epsilon * sum_z));

                Gspectral.x() -= exp_term * sum_x;
                Gspectral.y() -= exp_term * sum_z1;
            } else {
                cpxd exp_z = exp(Ikzp * diff_z);

                cpxd scalar_term1 = 1.0 / exp_z * erfc_complex(Ikzp / (2.0 * epsilon) - epsilon * diff_z);
                cpxd scalar_term2 = exp_z * erfc_complex(Ikzp / (2.0 * epsilon) + epsilon * diff_z);

                cpxd sum_x = -I * kxp * (scalar_term1 + scalar_term2);
                cpxd sum_z1 = Ikzp * sign_dz * (scalar_term2 - scalar_term1);
                sum_z1 += sign_dz * epsilon *
                          (exp_z * derfc_complex(Ikzp / (2.0 * epsilon) + epsilon * diff_z)
                           - (1.0 / exp_z) * derfc_complex(Ikzp / (2.0 * epsilon) - epsilon * diff_z));

                Gspectral.x() += exp_term * sum_x;
                Gspectral.y() += exp_term * sum_z1;

                // Dirichlet x.x() - y.x(), y.y() + x.y()
                exp_z = exp(Ikzp * sum_z);
                scalar_term1 = 1.0 / exp_z * erfc_complex(Ikzp / (2.0 * epsilon) - epsilon * sum_z);
                scalar_term2 = exp_z * erfc_complex(Ikzp / (2.0 * epsilon) + epsilon * sum_z);

                sum_x = -I * kxp * (scalar_term1 + scalar_term2);
                sum_z1 = Ikzp * sign_sz * (scalar_term2 - scalar_term1);
                sum_z1 += sign_sz * epsilon *
                          (exp_z * derfc_complex(Ikzp / (2.0 * epsilon) + epsilon * sum_z)
                           - (1.0 / exp_z) * derfc_complex(Ikzp / (2.0 * epsilon) - epsilon * sum_z));

                Gspectral.x() -= exp_term * sum_x;
                Gspectral.y() -= exp_term * sum_z1;
            }
        }
        Gspectral *= 0.25 * rep_period;
//        cout << "Gspectral: " << Gspectral << endl;

        cpxd ratio = k / (2.0 * epsilon);

        // Spatial sum
        for (int p = -N; p <= N; p++) {
            Vector2d y_n = y + Vector2d(p * period, 0.0);
            Vector2d r_vec = x - y_n;
            double R_p1 = (diff_x - p * period) * (diff_x - p * period) + diff_z * diff_z;
            double R_p2 = (diff_x - p * period) * (diff_x - p * period) + sum_z * sum_z;
            double Repsilon_2A = R_p1 * epsilon * epsilon;
            double Repsilon_2B = R_p2 * epsilon * epsilon;
            if (R_p1 < 1e-12) {
                double R_p2 = (diff_x - p * period) * (diff_x - p * period) + sum_z * sum_z;
                double Repsilon_2B = R_p2 * epsilon * epsilon;
                double EB = Tools::E1(Repsilon_2B);
                cpxd term1 = exp(-I * kx0 * period * (double) p);
                cpxd term2B = Tools::E0(Repsilon_2B);
                for (int q = 1; q <= 2 * N; q++) {
                    term2B += pow(ratio, 2 * q) / tgamma(q + 1) * EB;
                    EB = (exp(-Repsilon_2B) - Repsilon_2B * EB) / (double) q;
                }
                Gspatial -= term1 * term2B * -Vector2d(r_vec.x(), x.y() + y.y());
                continue; // skip singularity
            }

//            if (R_p1 < 1e-12 || R_p2 < 1e-12) {
//                cout << "Singularity at p = " << p << " for R_p1" << endl;
//                cout << y_n << endl;
//                cout << x << endl;
//            }

            double EA = Tools::E1(Repsilon_2A);
            double EB = Tools::E1(Repsilon_2B);

            cpxd term1 = exp(-I * kx0 * period * (double) p);
            cpxd term2A = Tools::E0(Repsilon_2A);
            cpxd term2B = Tools::E0(Repsilon_2B);
            for (int q = 1; q <= 2 * N; q++) {
                term2A += pow(ratio, 2 * q) / tgamma(q + 1) * EA;
                term2B += pow(ratio, 2 * q) / tgamma(q + 1) * EB;
                EA = (exp(-Repsilon_2A) - Repsilon_2A * EA) / (double) q;
                EB = (exp(-Repsilon_2B) - Repsilon_2B * EB) / (double) q;
            }
            Gspatial += term1 * term2A * -r_vec;
            Gspatial -= term1 * term2B * -Vector2d(r_vec.x(), x.y() + y.y());
        }
        Gspatial *= 0.5 * M_1_PI * epsilon * epsilon;
//        cout << "Gspatial: " << Gspatial << endl;

        return -(Gspatial + Gspectral);
    }

    /** @brief Propagating-mode approximation for Dirichlet periodic Helmholtz kernel. */
    inline cpxd prop_dir_helmholtz_2D_periodic(cpxd k, double kbar,
                                        const Vector2d &x, const Vector2d &y,
                                        double period = 1.0) {
        return -sin(k * y.y()) * exp(cpxd(0, 1) * (k * x.y() - kbar * x.x())) / (k * period);
    }

    /** @brief Direct lattice-sum reference for Dirichlet periodic Helmholtz kernel. */
    inline cpxd sum_dir_helmholtz_2D_periodic(cpxd k, double kbar,
                                        const Vector2d &x, const Vector2d &y,
                                        double period = 1.0) {
        cpxd ctrla = 0.0, ctrlb = 0.0;
        Vector2cd gt = Vector2cd::Zero(), gt2 = Vector2cd::Zero();
        cpxd kd = sqrt(k * k - kbar * kbar);
        complex<double> t = 0.0, t2 = 0.0;

        for (int n = -1000000; n <= 1000000; n++) {
            Vector2d x2(y.x() + period * n, y.y());
            Vector2d x3(y.x() + period * n, -y.y());
//            gt2 += exp(complex<double>(0.,-kbar) * (double)n * period) * Kernels::grad_helmholtz_2D(k, x, x3);
            t2 += exp(complex<double>(0.,-kbar) * (double)n * period) * Kernels::helmholtz_2D(k, x, x3);

            double l = 2. * M_PI * (double)n / period;
            cpxd denom = sqrt(-(l + kbar) * (l + kbar) + k * k);
//        if (n != 0) ctrl -= 0.5 / period * exp(+cpxd(0, kbar) * 0.3 -
//                            denom * 0.3 - cpxd(0, 1.) * l * 0.3)/ denom;
            ctrla += exp(cpxd(0, kbar + l)*(x.x() - y.x()) + cpxd(0, 1.) * denom * (x.y() - y.y()))
                    / (cpxd(0, 2.) * denom * period);
            ctrlb += exp(cpxd(0, kbar + l)*(x.x() - y.x()) + cpxd(0, 1.) * denom * (x.y() + y.y()))
                     / (cpxd(0, 2.) * denom * period);

            if ((x - y).norm() < 1e-12 && n == 0) {
                continue;
            }
//            gt += exp(complex<double>(0.,-kbar) * (double)n * period) * Kernels::grad_helmholtz_2D(k, x, x2);
            t += exp(complex<double>(0.,-kbar) * (double)n * period) * Kernels::helmholtz_2D(k, x, x2);
        }

//        cout << "Direct sum G: " << t - t2 << endl;
//        cout << "Direct sum grad G: " << gt - gt2 << endl;
//        cout << "Rayleigh sum G: " << ctrla - ctrlb << endl;
        return t - t2;
    }

    /** @brief Direct lattice-sum reference for gradient of Dirichlet periodic kernel. */
    inline Vector2cd sum_dir_grad_helmholtz_2D_periodic(cpxd k, double kbar,
                                              const Vector2d &x, const Vector2d &y,
                                              double period = 1.0) {
        cpxd ctrla = 0.0, ctrlb = 0.0;
        Vector2cd gt = Vector2cd::Zero(), gt2 = Vector2cd::Zero();
        cpxd kd = sqrt(k * k - kbar * kbar);
        complex<double> t = 0.0, t2 = 0.0;

        for (int n = -1000000; n <= 1000000; n++) {
            Vector2d x2(y.x() + period * n, y.y());
            Vector2d x3(y.x() + period * n, -y.y());
            gt2 += exp(complex<double>(0.,-kbar) * (double)n * period) * Kernels::grad_helmholtz_2D(k, x, x3);
//            t2 += exp(complex<double>(0.,-kbar) * (double)n * period) * Kernels::helmholtz_2D(k, x, x3);

//            double l = 2. * M_PI * (double)n / period;
//            cpxd denom = sqrt(-(l + kbar) * (l + kbar) + k * k);
////        if (n != 0) ctrl -= 0.5 / period * exp(+cpxd(0, kbar) * 0.3 -
////                            denom * 0.3 - cpxd(0, 1.) * l * 0.3)/ denom;
//            ctrla += exp(cpxd(0, kbar + l)*(x.x() - y.x()) + cpxd(0, 1.) * denom * (x.y() - y.y()))
//                     / (cpxd(0, 2.) * denom * period);
//            ctrlb += exp(cpxd(0, kbar + l)*(x.x() - y.x()) + cpxd(0, 1.) * denom * (x.y() + y.y()))
//                     / (cpxd(0, 2.) * denom * period);

            if ((x - y).norm() < 1e-12 && n == 0) {
                continue;
            }
            gt += exp(complex<double>(0.,-kbar) * (double)n * period) * Kernels::grad_helmholtz_2D(k, x, x2);
//            t += exp(complex<double>(0.,-kbar) * (double)n * period) * Kernels::helmholtz_2D(k, x, x2);
        }

//        cout << "Direct sum G: " << t - t2 << endl;
//        cout << "Direct sum grad G: " << gt - gt2 << endl;
//        cout << "Rayleigh sum G: " << ctrla - ctrlb << endl;
        return gt - gt2;
    }

    inline cpxd helmholtz_2D_biperiodic(cpxd k, Vector2d alpha, const Vector2d &x, const Vector2d &y,
                                        Vector2d a1, Vector2d a2, double epsilon = 0.0, int N = 20) {
        cpxd I = cpxd(0, 1.0);
        cpxd Gspectral = 0.0,
              Gspatial = 0.0;

        Matrix2d A;
        A.col(0) = a1;
        A.col(1) = a2;
        double area = abs(a1.x() * a2.y() - a1.y() * a2.x());
        if (epsilon < 1e-12) {
            epsilon = M_PI / area;
        }

        // reciprocal lattice vectors
        Matrix2d B = 2. * M_PI * A.inverse().transpose();
        Vector2d b1 = B.col(0);
        Vector2d b2 = B.col(1);

//        cout << "a1 . b1: " << a1.dot(b1) << endl;
//        cout << "a1 . b2: " << a1.dot(b2) << endl;
//        cout << "a2 . b1: " << a2.dot(b1) << endl;
//        cout << "a2 . b2: " << a2.dot(b2) << endl;

        // Spectral sum
        for (int p1 = -N; p1 <= N; p1++) {
            for (int p2 = -N; p2 <= N; p2++) {
                Vector2d k_vec = alpha + p1 * b1 + p2 * b2;
                cpxd denom = k_vec.squaredNorm() - k * k;
                Gspectral += exp(-denom / (4. * epsilon) - I * k_vec.dot(x - y)) / denom;
            }
        }
        Gspectral /= area;

        // Spatial sum
        cpxd ratio = k * k / (4. * epsilon);
        for (int p1 = -N; p1 <= N; p1++) {
            for (int p2 = -N; p2 <= N; p2++) {
                Vector2d R_vec = p1 * a1 + p2 * a2;
                double R2 = (x - y - R_vec).squaredNorm();
                if (R2 < 1e-12) {
                    continue; // skip singularity
                }

                double R2epsilon = R2 * epsilon;

                double E = Tools::E1(R2epsilon);
                cpxd term = exp(-I * alpha.dot(R_vec));
                cpxd term2 = 0;
                for (int q = 0; q <= 2 * N; q++) {
                    term2 += pow(ratio, q) / tgamma(q + 1) * E;
                    E = (exp(-R2epsilon) - R2epsilon * E) / (double) (q + 1);
                }
                Gspatial += term * term2;
            }
        }
        Gspatial *= 0.25 * M_1_PI;

        return -(Gspatial + Gspectral);
    }

    inline Vector2cd grad_helmholtz_2D_biperiodic(cpxd k, Vector2d alpha, const Vector2d &x, const Vector2d &y,
                                                Vector2d a1, Vector2d a2, double epsilon = 0.0, int N = 20) {
        cpxd I = cpxd(0, 1.0);
        Vector2cd Gspectral = Vector2cd::Zero(),
                   Gspatial = Vector2cd::Zero();

        Matrix2d A;
        A.col(0) = a1;
        A.col(1) = a2;
        double area = abs(a1.x() * a2.y() - a1.y() * a2.x());
        if (epsilon < 1e-12) {
            epsilon = M_PI / area;
        }

        // reciprocal lattice vectors
        Matrix2d B = 2. * M_PI * A.inverse().transpose();
        Vector2d b1 = B.col(0);
        Vector2d b2 = B.col(1);

//        cout << "a1 . b1: " << a1.dot(b1) << endl;
//        cout << "a1 . b2: " << a1.dot(b2) << endl;
//        cout << "a2 . b1: " << a2.dot(b1) << endl;
//        cout << "a2 . b2: " << a2.dot(b2) << endl;

        // Spectral sum
        for (int p1 = -N; p1 <= N; p1++) {
            for (int p2 = -N; p2 <= N; p2++) {
                Vector2d k_vec = alpha + p1 * b1 + p2 * b2;
                cpxd denom = k_vec.squaredNorm() - k * k;
                Gspectral += k_vec * exp(-denom / (4. * epsilon) - I * k_vec.dot(x - y)) / denom;
            }
        }
        Gspectral *= -I / area;

        // Spatial sum
        cpxd ratio = k * k / (4. * epsilon);
        for (int p1 = -N; p1 <= N; p1++) {
            for (int p2 = -N; p2 <= N; p2++) {
                Vector2d R_vec = p1 * a1 + p2 * a2;
                double R2 = (x - y - R_vec).squaredNorm();
                if (R2 < 1e-12) {
                    continue; // skip singularity
                }

                double R2epsilon = R2 * epsilon;

                double E = Tools::E1(R2epsilon);
                cpxd term = exp(-I * alpha.dot(R_vec));
                cpxd term2 = Tools::E0(R2epsilon);
                for (int q = 1; q <= 2 * N; q++) {
                    term2 += pow(ratio, q) / tgamma(q + 1) * E;
                    E = (exp(-R2epsilon) - R2epsilon * E) / (double) (q);
                }
                Gspatial += term * term2 * -2.0 * (x - y - R_vec) * epsilon;
            }
        }
        Gspatial *= 0.25 * M_1_PI;

        return -(Gspatial + Gspectral);
    }
}

#endif //SUBWAVELENGTHRESONATORS_KERNELS_H
