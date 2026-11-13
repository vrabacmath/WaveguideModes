#pragma once

#include <algorithm>
#include <complex>
#include <numbers>

namespace Ewald {
// Bound |k^2/(4 epsilon)| by one to avoid high-frequency Ewald cancellation.
// Reciprocal-space cutoffs must still be checked separately.
inline double biperiodic_epsilon(std::complex<double> k, double cell_area) {
    return std::max(std::numbers::pi / cell_area, std::norm(k) / 4.0);
}
}
