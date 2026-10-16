#include "workflows/colbrook_convergence.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace workflows {

ColbrookMinimum find_colbrook_minimum(const Eigen::MatrixXcd& C, int row_offset,
                                     double target, double half_width,
                                     int intervals, int passes) {
    if (C.cols() == 0 || C.rows() < C.cols() || row_offset < 0 ||
        row_offset > C.rows() - C.cols() || !C.allFinite() ||
        !std::isfinite(target) || !std::isfinite(half_width) || half_width <= 0 ||
        intervals < 4 || passes < 1)
        throw std::invalid_argument("Invalid Colbrook matrix, embedding, or search settings");
    const double search_lo = target - half_width, search_hi = target + half_width;
    if (!std::isfinite(search_lo) || !std::isfinite(search_hi) || search_lo >= search_hi)
        throw std::invalid_argument("Target interval must be finite and increasing");

    Eigen::MatrixXcd E = Eigen::MatrixXcd::Zero(C.rows(), C.cols());
    E.block(row_offset, 0, C.cols(), C.cols()).setIdentity();
    ColbrookMinimum best{target, std::numeric_limits<double>::infinity(), 0.0, false};
    double lo = search_lo, hi = search_hi;
    for (int pass = 0; pass < passes; ++pass) {
        const double step = (hi - lo) / intervals;
        if (step <= 0 || lo + step == lo)
            throw std::runtime_error("Search grid reached floating-point resolution; reduce passes");
        // Keep the best previous sample as well: nested grids need not contain its exact z.
        for (int i = 0; i <= intervals; ++i) {
            const double z = std::lerp(lo, hi, double(i) / intervals);
            const Eigen::MatrixXcd shifted = C - z * E;
            Eigen::BDCSVD<Eigen::MatrixXcd> svd(shifted);
            if (svd.info() != Eigen::Success || !svd.singularValues().allFinite())
                throw std::runtime_error("Colbrook SVD failed at z=" + std::to_string(z));
            const double sigma = svd.singularValues().minCoeff();
            if (sigma < best.sigma_min) {
                best.z = z;
                best.sigma_min = sigma;
            }
        }
        best.z_resolution = step;
        lo = std::max(search_lo, best.z - step);
        hi = std::min(search_hi, best.z + step);
    }
    best.at_boundary = best.z - search_lo <= best.z_resolution ||
                       search_hi - best.z <= best.z_resolution;
    return best;
}

} // namespace workflows
