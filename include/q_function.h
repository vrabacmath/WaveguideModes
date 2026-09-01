#ifndef SUBWAVELENGTHRESONATORS_Q_FUNCTION_H
#define SUBWAVELENGTHRESONATORS_Q_FUNCTION_H

#include "Eigen/Dense"
#include "bessel-library.hpp"
#include "math_constants.h"
#include "tools.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>

namespace QFunction {

    struct Options {
        int reciprocal_cutoff = 8;
        int spatial_cutoff = 8;
        int spatial_series_terms = 80;
        double eta = 0.0;
        double spatial_series_tolerance = 1e-15;
        double wood_tolerance = 1e-12;
    };

    namespace detail {

        inline double ewald_eta(const Options& options) {
            return options.eta > 0.0 ? options.eta : std::sqrt(M_PI);
        }

        inline int parity(int n) {
            return (std::abs(n) % 2 == 0) ? 1 : -1;
        }

        inline cpxd i_power(int n) {
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

        inline double expint_E_integer(int n, double x) {
            if (x <= 0.0) {
                throw std::domain_error("E_n(x) requires x > 0");
            }

            if (n == 0) {
                return std::exp(-x) / x;
            }
            if (n > 0) {
                double value = -std::expint(-x);
                for (int q = 1; q < n; ++q) {
                    value = (std::exp(-x) - x * value) / static_cast<double>(q);
                }
                return value;
            }

            double value = std::exp(-x) / x;
            for (int q = -1; q >= n; --q) {
                value = (std::exp(-x) - static_cast<double>(q) * value) / x;
            }
            return value;
        }

        inline cpxd expint_Ei(cpxd z) {
            const double scale = std::max(1.0, std::abs(z));
            if (std::abs(z) <= std::numeric_limits<double>::epsilon()) {
                throw std::domain_error("Ei(z) is singular at z = 0");
            }
            if (std::abs(z.imag()) <= 10.0 * std::numeric_limits<double>::epsilon() * scale &&
                z.real() > 0.0) {
                return std::expint(z.real());
            }

            cpxd term = z;
            cpxd sum = term;
            for (int m = 2; m <= 300; ++m) {
                term *= z / static_cast<double>(m);
                const cpxd update = term / static_cast<double>(m);
                sum += update;

                if (std::abs(update) <= 1e-15 * std::max(1.0, std::abs(sum))) {
                    break;
                }
            }

            return cpxd(EULER_GAMMA, 0.0) + std::log(z) + sum;
        }

        inline Vector2d reduce_reciprocal_cell(const Vector2d& alpha, const Vector2d& b1, const Vector2d& b2) {
            Matrix2d B;
            B.col(0) = b1;
            B.col(1) = b2;

            Vector2d coeff = B.inverse() * alpha;
            coeff(0) -= std::round(coeff(0));
            coeff(1) -= std::round(coeff(1));

            return B * coeff;
        }

        inline cpxd signed_power(cpxd z, int exponent) {
            cpxd result = 1.0;
            for (int j = 0; j < exponent; ++j) {
                result *= z;
            }
            return result;
        }

        inline cpxd angular_power_over_k(const Vector2d& v, cpxd k, int order) {
            const int p = std::abs(order);
            if (p == 0) {
                return 1.0;
            }

            cpxd z = order >= 0 ? cpxd(v.x(), -v.y()) : cpxd(v.x(), v.y());
            return pow(z / k, p);
        }

        inline cpxd spatial_integral(int order, cpxd k, double radius, double eta,
                                     const Options& options) {
            const double x = radius * radius * eta * eta;
            const cpxd c = 0.25 * k * k * radius * radius;
            const double radius_factor = 0.5 * std::pow(radius, -2.0 * order);

            cpxd series = 0.0;
            cpxd c_power_over_factorial = 1.0;
            int current_expint_order = 1 - order;
            double E = expint_E_integer(current_expint_order, x);
            for (int j = 0; j < options.spatial_series_terms; ++j) {
                if (j > 0) {
                    c_power_over_factorial *= c / static_cast<double>(j);
                }

                const cpxd term = c_power_over_factorial *
                                  std::pow(x, order - j) *
                                  E;
                series += term;

                if (std::abs(term) <= options.spatial_series_tolerance * std::max(1.0, std::abs(series))) {
                    break;
                }

                if (current_expint_order == 0) {
                    E = -std::expint(-x);
                } else {
                    E = (std::exp(-x) - x * E) / static_cast<double>(current_expint_order);
                }
                ++current_expint_order;
            }

            return radius_factor * series;
        }

        inline cpxd tau_lattice_sum(int n, cpxd k, const Vector2d& alpha, Vector2d a1, Vector2d a2, Vector2d d, const Options& options) {
            if (std::abs(k) <= std::numeric_limits<double>::epsilon()) {
                throw std::domain_error("q_function requires a nonzero wavenumber k");
            }
            if (options.reciprocal_cutoff < 0 || options.spatial_cutoff < 0) {
                throw std::domain_error("q_function cutoffs must be non-negative");
            }

            Matrix2d A;
            A.col(0) = a1;
            A.col(1) = a2;
            double area = abs(a1.x() * a2.y() - a1.y() * a2.x());

            // reciprocal lattice vectors
            Matrix2d B = 2. * M_PI * A.inverse().transpose();
            Vector2d b1 = B.col(0);
            Vector2d b2 = B.col(1);

            const Vector2d beta = reduce_reciprocal_cell(alpha, b1, b2);
            const double eta = ewald_eta(options) / sqrt(area);
            const int p = std::abs(n);
            const cpxd I(0.0, 1.0);

            cpxd tau = 0.0;
            if (n == 0 && d.isZero(1e-12)) {
                tau += -1.0 - I * M_1_PI * expint_Ei(k * k / (4.0 * eta * eta));
            }

            cpxd spectral_sum = 0.0;
            for (int m1 = -options.reciprocal_cutoff; m1 <= options.reciprocal_cutoff; ++m1) {
                for (int m2 = -options.reciprocal_cutoff; m2 <= options.reciprocal_cutoff; ++m2) {
                    Vector2d beta_m = beta + m1 * b1 + m2 * b2;
                    const double beta2 = beta_m.squaredNorm();
                    const cpxd denominator = k * k - beta2;
                    if (std::abs(denominator) <= options.wood_tolerance) {
                        throw std::domain_error("q_function is singular at this Wood anomaly");
                    }

                    const cpxd angular = angular_power_over_k(beta_m, k, n);
                    const cpxd damping = std::exp((k * k - beta2) / (4.0 * eta * eta));
                    spectral_sum += damping * angular / denominator * exp(-I * beta_m.dot(d));
                }
            }
            tau += 4.0 * I * i_power(n) * spectral_sum / area;

            cpxd spatial_sum = 0.0;
            for (int m1 = -options.spatial_cutoff; m1 <= options.spatial_cutoff; ++m1) {
                for (int m2 = -options.spatial_cutoff; m2 <= options.spatial_cutoff; ++m2) {
                    if (m1 == 0 && m2 == 0 && d.isZero(1e-12)) {
                        continue;
                    }

                    Vector2d lattice_point = m1 * a1 + m2 * a2;
                    const double radius = (lattice_point + d).norm();
                    const cpxd phase = std::exp(I * beta.dot(lattice_point));
                    const cpxd angular = angular_power_over_k(lattice_point + d, k, n);
                    const cpxd integral = spatial_integral(p, k, radius, eta, options);
                    spatial_sum += phase * angular * integral;
                }
            }

            const double sign = (n < 0) ? static_cast<double>(parity(p)) : 1.0;
            tau += sign * (-std::pow(2.0, p + 1) * I * M_1_PI) * spatial_sum;

            return tau;
        }

    } // namespace detail

    /**
     * @brief Evaluate the square-lattice cylindrical wave sum Q_n(k, alpha).
     *
     * Computes the quasi-periodic lattice sum
     *
     *     Q_n = sum_{m in Z^2, m != 0} H_n^{(1)}(k |m|)
     *           exp(i n arg(m)) exp(i m . alpha),
     *
     * for the unit square lattice away from Wood anomalies
     * k^2 = |alpha + 2 pi p|^2. The direct series is only conditionally
     * convergent for real k and exponentially convergent when Im(k) > 0, so
     * this uses Linton's Ewald decomposition for two-dimensional Helmholtz
     * lattice sums. Reciprocal terms, direct-lattice Ewald terms, and the local
     * singular correction are combined through tau_n = (-1)^n Q_{-n}; the
     * complex logarithm and Ei correction use their principal branches.
     *
     * @param n Cylindrical harmonic order.
     * @param k Nonzero complex wavenumber.
     * @param alpha Bloch vector; components are periodic modulo 2 pi.
     * @param options Ewald splitting parameter, cutoffs, and tolerances.
     * @return The outgoing Hankel lattice sum Q_n(k, alpha).
     */
    inline cpxd q_function(int n, cpxd k, const Vector2d& alpha, const Options& options = {}) {
        return static_cast<double>(detail::parity(n)) *
               detail::tau_lattice_sum(-n, k, alpha, Vector2d(1, 0), Vector2d(0, 1), Vector2d(0, 0), options);
    }

    inline cpxd q_function(int n, cpxd k, const Vector2d& alpha, Vector2d a1, Vector2d a2, Vector2d d, const Options& options = {}) {
        return static_cast<double>(detail::parity(n)) *
               detail::tau_lattice_sum(-n, k, alpha, a1, a2, d, options);
    }

} // namespace QFunction

inline cpxd q_function(int n, cpxd k, const Vector2d& alpha,
                       const QFunction::Options& options = {}) {
    return QFunction::q_function(n, k, alpha, options);
}

inline cpxd q_function(int n, double k, const Vector2d& alpha,
                       const QFunction::Options& options = {}) {
    return QFunction::q_function(n, k, alpha, options);
}

#endif // SUBWAVELENGTHRESONATORS_Q_FUNCTION_H
