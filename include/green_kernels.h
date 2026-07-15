//
// Boundary-integral Green function policies.
//

#ifndef SUBWAVELENGTHRESONATORS_GREEN_KERNELS_H
#define SUBWAVELENGTHRESONATORS_GREEN_KERNELS_H

#include <complex>
#include <concepts>
#include <cmath>

#include "Eigen/Dense"
#include "basis.h"
#include "kernels.h"
#include "math_constants.h"

using cpxd = std::complex<double>;

namespace GreenKernels {

inline cpxd normal_dot(const Eigen::Vector2cd& value, const Eigen::Vector2d& normal) {
    return value.cwiseProduct(normal).sum();
}

template <typename Kernel>
concept BoundaryIntegralKernel = requires(const Kernel kernel,
                                          const Eigen::Vector2d& x,
                                          const Eigen::Vector2d& y,
                                          const Eigen::Vector2d& normal,
                                          const Vertex& vertex,
                                          double r) {
    { kernel.single_layer_value(x, y) } -> std::convertible_to<cpxd>;
    { kernel.single_layer_log_coefficient(x, y, r) } -> std::convertible_to<cpxd>;
    { kernel.single_layer_self_log_coefficient(vertex) } -> std::convertible_to<cpxd>;
    { kernel.single_layer_self_regular(vertex) } -> std::convertible_to<cpxd>;
    { kernel.target_normal_value(x, y, normal) } -> std::convertible_to<cpxd>;
    { kernel.target_normal_log_coefficient(x, y, normal, r) } -> std::convertible_to<cpxd>;
    { kernel.target_normal_self(vertex) } -> std::convertible_to<cpxd>;
    { kernel.source_normal_value(x, y, normal) } -> std::convertible_to<cpxd>;
    { kernel.source_normal_log_coefficient(x, y, normal, r) } -> std::convertible_to<cpxd>;
    { kernel.source_normal_self(vertex) } -> std::convertible_to<cpxd>;
};

struct FreeSpaceKernel {
    cpxd k;

    [[nodiscard]] bool is_static() const {
        return std::abs(k) < 1e-6;
    }

    [[nodiscard]] cpxd single_layer_value(const Eigen::Vector2d& x,
                                          const Eigen::Vector2d& y) const {
        const double r = (x - y).norm();
        if (is_static()) {
            return 0.5 * M_1_PI * std::log(r);
        }
        return Kernels::helmholtz_2D(k, x, y);
    }

    [[nodiscard]] cpxd single_layer_log_coefficient(const Eigen::Vector2d&,
                                                    const Eigen::Vector2d&,
                                                    double r) const {
        if (is_static()) {
            return 0.25 * M_1_PI;
        }
        return 0.25 * M_1_PI * bessel::cyl_j(0, k * r);
    }

    [[nodiscard]] cpxd single_layer_self_log_coefficient(const Vertex&) const {
        return 0.25 * M_1_PI;
    }

    [[nodiscard]] cpxd single_layer_self_regular(const Vertex& vertex) const {
        if (is_static()) {
            return 0.5 * M_1_PI * std::log(vertex.tnorm) * vertex.sigma;
        }
        return vertex.sigma * (cpxd(0.0, -0.25) +
                               0.5 * M_1_PI *
                               (EULER_GAMMA + std::log(0.5 * k * vertex.tnorm)));
    }

    [[nodiscard]] cpxd target_normal_value(const Eigen::Vector2d& x,
                                           const Eigen::Vector2d& y,
                                           const Eigen::Vector2d& normal) const {
        const Eigen::Vector2d r_vec = x - y;
        if (is_static()) {
            return 0.5 * M_1_PI * r_vec.dot(normal) / r_vec.squaredNorm();
        }
        return normal_dot(Kernels::grad_helmholtz_2D(k, x, y), normal);
    }

    [[nodiscard]] cpxd target_normal_log_coefficient(const Eigen::Vector2d& x,
                                                     const Eigen::Vector2d& y,
                                                     const Eigen::Vector2d& normal,
                                                     double r) const {
        if (is_static()) {
            return 0.0;
        }
        return -k * 0.25 * M_1_PI * bessel::cyl_j(1, k * r)
               * (x - y).dot(normal) / r;
    }

    [[nodiscard]] cpxd target_normal_self(const Vertex& vertex) const {
        return 0.25 * M_1_PI * vertex.curvature * vertex.sigma;
    }

    [[nodiscard]] cpxd source_normal_value(const Eigen::Vector2d& x,
                                           const Eigen::Vector2d& y,
                                           const Eigen::Vector2d& normal) const {
        const Eigen::Vector2d r_vec = x - y;
        if (is_static()) {
            return -0.5 * M_1_PI * r_vec.dot(normal) / r_vec.squaredNorm();
        }
        return -normal_dot(Kernels::grad_helmholtz_2D(k, x, y), normal);
    }

    [[nodiscard]] cpxd source_normal_log_coefficient(const Eigen::Vector2d& x,
                                                     const Eigen::Vector2d& y,
                                                     const Eigen::Vector2d& normal,
                                                     double r) const {
        if (is_static()) {
            return 0.0;
        }
        return k * 0.25 * M_1_PI * bessel::cyl_j(1, k * r)
               * (x - y).dot(normal) / r;
    }

    [[nodiscard]] cpxd source_normal_self(const Vertex& vertex) const {
        return 0.25 * M_1_PI * vertex.curvature * vertex.sigma;
    }
};

struct PeriodicKernel {
    cpxd k;
    double kbar;
    double period;
    cpxd ewald_constant;

    PeriodicKernel(cpxd k_value, double kbar_value, double period_value)
        : k(k_value), kbar(kbar_value), period(period_value), ewald_constant(0.0) {
        if (!is_static()) {
            const double epsilon = std::sqrt(M_PI) / period;
            const cpxd ratio = k / (2.0 * epsilon);
            for (int q = 1; q <= 20; q++) {
                ewald_constant += std::pow(ratio, 2 * q) / (std::tgamma(q + 1) * q);
            }
        }
    }

    [[nodiscard]] bool is_static() const {
        return std::abs(k) < 1e-6 && std::abs(kbar) < 1e-6;
    }

    [[nodiscard]] cpxd single_layer_value(const Eigen::Vector2d& x,
                                          const Eigen::Vector2d& y) const {
        if (is_static()) {
            return Kernels::laplace_2D_periodic(x, y, period);
        }
        return Kernels::helmholtz_2D_periodic(k, kbar, x, y, period);
    }

    [[nodiscard]] cpxd single_layer_log_coefficient(const Eigen::Vector2d&,
                                                    const Eigen::Vector2d&,
                                                    double r) const {
        if (is_static()) {
            return 0.25 * M_1_PI;
        }
        return 0.25 * M_1_PI * bessel::cyl_j(0, k * r);
    }

    [[nodiscard]] cpxd single_layer_self_log_coefficient(const Vertex&) const {
        return 0.25 * M_1_PI;
    }

    [[nodiscard]] cpxd single_layer_self_regular(const Vertex& vertex) const {
        if (is_static()) {
            return 0.5 * M_1_PI * std::log(M_PI / period * vertex.tnorm) * vertex.sigma;
        }
        return vertex.sigma *
               (Kernels::helmholtz_2D_periodic(k, kbar, vertex.point, vertex.point, period) +
                0.25 * M_1_PI *
                (-ewald_constant + EULER_GAMMA +
                 std::log(M_PI / period / period * vertex.tnorm * vertex.tnorm)));
    }

    [[nodiscard]] cpxd target_normal_value(const Eigen::Vector2d& x,
                                           const Eigen::Vector2d& y,
                                           const Eigen::Vector2d& normal) const {
        if (is_static()) {
            return normal_dot(Kernels::grad_laplace_2D_periodic(x, y, period), normal);
        }
        return normal_dot(Kernels::grad_helmholtz_2D_periodic(k, kbar, x, y, period), normal);
    }

    [[nodiscard]] cpxd target_normal_log_coefficient(const Eigen::Vector2d& x,
                                                     const Eigen::Vector2d& y,
                                                     const Eigen::Vector2d& normal,
                                                     double r) const {
        if (is_static()) {
            return 0.0;
        }
        return -k * 0.25 * M_1_PI * bessel::cyl_j(1, k * r)
               * (x - y).dot(normal) / r;
    }

    [[nodiscard]] cpxd target_normal_self(const Vertex& vertex) const {
        cpxd value = 0.25 * M_1_PI * vertex.curvature * vertex.sigma;
        if (!is_static()) {
            value += vertex.sigma *
                     normal_dot(Kernels::grad_helmholtz_2D_periodic(k, kbar, vertex.point,
                                                                     vertex.point, period),
                                vertex.normal);
        }
        return value;
    }

    [[nodiscard]] cpxd source_normal_value(const Eigen::Vector2d& x,
                                           const Eigen::Vector2d& y,
                                           const Eigen::Vector2d& normal) const {
        if (is_static()) {
            return -normal_dot(Kernels::grad_laplace_2D_periodic(x, y, period), normal);
        }
        return -normal_dot(Kernels::grad_helmholtz_2D_periodic(k, kbar, x, y, period), normal);
    }

    [[nodiscard]] cpxd source_normal_log_coefficient(const Eigen::Vector2d& x,
                                                     const Eigen::Vector2d& y,
                                                     const Eigen::Vector2d& normal,
                                                     double r) const {
        if (is_static()) {
            return 0.0;
        }
        return k * 0.25 * M_1_PI * bessel::cyl_j(1, k * r)
               * (x - y).dot(normal) / r;
    }

    [[nodiscard]] cpxd source_normal_self(const Vertex& vertex) const {
        cpxd value = 0.25 * M_1_PI * vertex.curvature * vertex.sigma;
        if (!is_static()) {
            value -= vertex.sigma *
                     normal_dot(Kernels::grad_helmholtz_2D_periodic(k, kbar, vertex.point,
                                                                     vertex.point, period),
                                vertex.normal);
        }
        return value;
    }
};

struct DirichletPeriodicKernel {
    cpxd k;
    double kbar;
    double period;
    cpxd ewald_constant;

    DirichletPeriodicKernel(cpxd k_value, double kbar_value, double period_value)
        : k(k_value), kbar(kbar_value), period(period_value), ewald_constant(0.0) {
        if (!is_static()) {
            const double epsilon = std::sqrt(M_PI) / period;
            const cpxd ratio = k / (2.0 * epsilon);
            for (int q = 1; q <= 20; q++) {
                ewald_constant += std::pow(ratio, 2 * q) / (std::tgamma(q + 1) * q);
            }
        }
    }

    [[nodiscard]] bool is_static() const {
        return std::abs(k) < 1e-6 && std::abs(kbar) < 1e-6;
    }

    [[nodiscard]] cpxd single_layer_value(const Eigen::Vector2d& x,
                                          const Eigen::Vector2d& y) const {
        if (is_static()) {
            return Kernels::dir_laplace_2D_periodic(x, y, period);
        }
        return Kernels::dir_helmholtz_2D_periodic(k, kbar, x, y, period);
    }

    [[nodiscard]] cpxd single_layer_log_coefficient(const Eigen::Vector2d&,
                                                    const Eigen::Vector2d&,
                                                    double r) const {
        if (is_static()) {
            return 0.25 * M_1_PI;
        }
        return 0.25 * M_1_PI * bessel::cyl_j(0, k * r);
    }

    [[nodiscard]] cpxd single_layer_self_log_coefficient(const Vertex&) const {
        return 0.25 * M_1_PI;
    }

    [[nodiscard]] cpxd single_layer_self_regular(const Vertex& vertex) const {
        if (is_static()) {
            const Eigen::Vector2d mirror(vertex.point.x(), -vertex.point.y());
            return vertex.sigma *
                   (0.5 * M_1_PI * std::log(M_PI / period * vertex.tnorm) -
                    Kernels::laplace_2D_periodic(vertex.point, mirror, period));
        }
        return vertex.sigma *
               (Kernels::dir_helmholtz_2D_periodic(k, kbar, vertex.point,
                                                    vertex.point, period) +
                0.25 * M_1_PI *
                (-ewald_constant + EULER_GAMMA +
                 std::log(M_PI / period / period * vertex.tnorm * vertex.tnorm)));
    }

    [[nodiscard]] cpxd target_normal_value(const Eigen::Vector2d& x,
                                           const Eigen::Vector2d& y,
                                           const Eigen::Vector2d& normal) const {
        if (is_static()) {
            return normal_dot(Kernels::dir_grad_laplace_2D_periodic(x, y, period), normal);
        }
        return normal_dot(Kernels::dir_grad_helmholtz_2D_periodic(k, kbar, x, y, period), normal);
    }

    [[nodiscard]] cpxd target_normal_log_coefficient(const Eigen::Vector2d& x,
                                                     const Eigen::Vector2d& y,
                                                     const Eigen::Vector2d& normal,
                                                     double r) const {
        if (is_static()) {
            return 0.0;
        }
        return -k * 0.25 * M_1_PI * bessel::cyl_j(1, k * r)
               * (x - y).dot(normal) / r;
    }

    [[nodiscard]] cpxd target_normal_self(const Vertex& vertex) const {
        cpxd value = 0.25 * M_1_PI * vertex.curvature * vertex.sigma;
        if (is_static()) {
            const Eigen::Vector2d mirror(vertex.point.x(), -vertex.point.y());
            value -= vertex.sigma *
                     normal_dot(Kernels::grad_laplace_2D_periodic(vertex.point, mirror, period),
                                vertex.normal);
        } else {
            value += vertex.sigma *
                     normal_dot(Kernels::dir_grad_helmholtz_2D_periodic(k, kbar,
                                                                         vertex.point,
                                                                         vertex.point,
                                                                         period),
                                vertex.normal);
        }
        return value;
    }

    [[nodiscard]] cpxd source_normal_value(const Eigen::Vector2d& x,
                                           const Eigen::Vector2d& y,
                                           const Eigen::Vector2d& normal) const {
        if (is_static()) {
            return -normal_dot(Kernels::dir_grad_laplace_2D_periodic(x, y, period), normal);
        }
        return -normal_dot(Kernels::dir_grad_helmholtz_2D_periodic(k, kbar, x, y, period), normal);
    }

    [[nodiscard]] cpxd source_normal_log_coefficient(const Eigen::Vector2d& x,
                                                     const Eigen::Vector2d& y,
                                                     const Eigen::Vector2d& normal,
                                                     double r) const {
        if (is_static()) {
            return 0.0;
        }
        return k * 0.25 * M_1_PI * bessel::cyl_j(1, k * r)
               * (x - y).dot(normal) / r;
    }

    [[nodiscard]] cpxd source_normal_self(const Vertex& vertex) const {
        cpxd value = 0.25 * M_1_PI * vertex.curvature * vertex.sigma;
        if (!is_static()) {
            value -= vertex.sigma *
                     normal_dot(Kernels::dir_grad_helmholtz_2D_periodic(k, kbar,
                                                                         vertex.point,
                                                                         vertex.point,
                                                                         period),
                                vertex.normal);
        }
        return value;
    }
};

struct BiperiodicKernel {
    cpxd k;
    Eigen::Vector2d alpha;
    Eigen::Vector2d a1;
    Eigen::Vector2d a2;
    double epsilon;
    cpxd ewald_constant;

    BiperiodicKernel(cpxd k_value,
                     Eigen::Vector2d alpha_value,
                     Eigen::Vector2d a1_value,
                     Eigen::Vector2d a2_value)
        : k(k_value),
          alpha(alpha_value),
          a1(a1_value),
          a2(a2_value),
          epsilon(M_PI / std::abs(a1.x() * a2.y() - a1.y() * a2.x())),
          ewald_constant(0.0) {
        const cpxd ratio = k * k / (4.0 * epsilon);
        for (int q = 1; q <= 20; q++) {
            ewald_constant += std::pow(ratio, q) / (std::tgamma(q + 1) * q);
        }
    }

    [[nodiscard]] cpxd single_layer_value(const Eigen::Vector2d& x,
                                          const Eigen::Vector2d& y) const {
        return Kernels::helmholtz_2D_biperiodic(k, alpha, x, y, a1, a2);
    }

    [[nodiscard]] cpxd single_layer_log_coefficient(const Eigen::Vector2d&,
                                                    const Eigen::Vector2d&,
                                                    double r) const {
        return 0.25 * M_1_PI * bessel::cyl_j(0, k * r);
    }

    [[nodiscard]] cpxd single_layer_self_log_coefficient(const Vertex&) const {
        return 0.25 * M_1_PI;
    }

    [[nodiscard]] cpxd single_layer_self_regular(const Vertex& vertex) const {
        return vertex.sigma *
               (Kernels::helmholtz_2D_biperiodic(k, alpha, vertex.point, vertex.point, a1, a2) +
                0.25 * M_1_PI *
                (-ewald_constant + EULER_GAMMA +
                 std::log(epsilon * vertex.tnorm * vertex.tnorm)));
    }

    [[nodiscard]] cpxd target_normal_value(const Eigen::Vector2d& x,
                                           const Eigen::Vector2d& y,
                                           const Eigen::Vector2d& normal) const {
        return normal_dot(Kernels::grad_helmholtz_2D_biperiodic(k, alpha, x, y, a1, a2), normal);
    }

    [[nodiscard]] cpxd target_normal_log_coefficient(const Eigen::Vector2d& x,
                                                     const Eigen::Vector2d& y,
                                                     const Eigen::Vector2d& normal,
                                                     double r) const {
        return -k * 0.25 * M_1_PI * bessel::cyl_j(1, k * r)
               * (x - y).dot(normal) / r;
    }

    [[nodiscard]] cpxd target_normal_self(const Vertex& vertex) const {
        return vertex.sigma *
               (0.25 * M_1_PI * vertex.curvature +
                normal_dot(Kernels::grad_helmholtz_2D_biperiodic(k, alpha, vertex.point,
                                                                  vertex.point, a1, a2),
                           vertex.normal));
    }

    [[nodiscard]] cpxd source_normal_value(const Eigen::Vector2d& x,
                                           const Eigen::Vector2d& y,
                                           const Eigen::Vector2d& normal) const {
        return -normal_dot(Kernels::grad_helmholtz_2D_biperiodic(k, alpha, x, y, a1, a2), normal);
    }

    [[nodiscard]] cpxd source_normal_log_coefficient(const Eigen::Vector2d& x,
                                                     const Eigen::Vector2d& y,
                                                     const Eigen::Vector2d& normal,
                                                     double r) const {
        return k * 0.25 * M_1_PI * bessel::cyl_j(1, k * r)
               * (x - y).dot(normal) / r;
    }

    [[nodiscard]] cpxd source_normal_self(const Vertex& vertex) const {
        return vertex.sigma *
               (0.25 * M_1_PI * vertex.curvature -
                normal_dot(Kernels::grad_helmholtz_2D_biperiodic(k, alpha, vertex.point,
                                                                  vertex.point, a1, a2),
                           vertex.normal));
    }
};

static_assert(BoundaryIntegralKernel<FreeSpaceKernel>);
static_assert(BoundaryIntegralKernel<PeriodicKernel>);
static_assert(BoundaryIntegralKernel<DirichletPeriodicKernel>);
static_assert(BoundaryIntegralKernel<BiperiodicKernel>);

} // namespace GreenKernels

#endif // SUBWAVELENGTHRESONATORS_GREEN_KERNELS_H
