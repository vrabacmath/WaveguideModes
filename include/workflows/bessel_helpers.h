#ifndef SUBWAVELENGTHRESONATORS_WORKFLOWS_BESSEL_HELPERS_H
#define SUBWAVELENGTHRESONATORS_WORKFLOWS_BESSEL_HELPERS_H

#include <complex>

#include "bessel-library.hpp"

namespace workflows {

// First positive zero j'_{m,1} of J_m' for the interior Neumann eigenvalue.
inline double first_neumann_zero(int m) {
    switch (m) {
        case 0: return 3.8317059702;  // breathing (= j_{1,1})
        case 1: return 1.8411837813;  // dipole (doubly degenerate)
        case 2: return 3.0542369282;  // quadrupole
        case 3: return 4.2011889412;  // octupole
        default: return 1.8411837813;
    }
}

inline std::complex<double> cyl_j_prime(int n, std::complex<double> z) {
    return (n == 0) ? -bessel::cyl_j(1, z)
                    : 0.5 * (bessel::cyl_j(n - 1, z) - bessel::cyl_j(n + 1, z));
}

}  // namespace workflows

#endif  // SUBWAVELENGTHRESONATORS_WORKFLOWS_BESSEL_HELPERS_H
