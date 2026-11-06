#ifndef SUBWAVELENGTHRESONATORS_WORKFLOWS_BESSEL_HELPERS_H
#define SUBWAVELENGTHRESONATORS_WORKFLOWS_BESSEL_HELPERS_H

#include <complex>
#include <stdexcept>

#include "bessel-library.hpp"

namespace workflows {

struct NeumannMode {
    int n_rad;  // one-based POSITIVE radial root (excludes the zero-frequency constant mode)
    int m_ang;  // angular order, never the encoded mode identifier
};

// Decimal identifier nm = 10*n + m: 10 monopole, 11 dipole, 21 second radial dipole.
// The last digit is m (0..9); all preceding digits are n (>=1).
inline NeumannMode decode_nm(int nm) {
    if (nm < 10)
        throw std::invalid_argument("Mode must use nm = 10*n + m (n >= 1, 0 <= m <= 9): "
                                    "10 monopole, 11 dipole, 12 quadrupole, 21 second radial dipole");
    return {nm / 10, nm % 10};
}

// Positive Neumann roots j'_{m,n}, keyed directly by nm (obtained from
// scipy.special.jnp_zeros). No zero-frequency constant mode or silent fallback.
inline double neumann_zero(int nm) {
    decode_nm(nm);
    switch (nm) {
        case 10: return 3.8317059702075125;
        case 11: return 1.8411837813406595;
        case 12: return 3.0542369282271404;
        case 13: return 4.2011889412105283;
        case 14: return 5.3175531260839941;
        case 15: return 6.4156163757002398;
        case 20: return 7.0155866698156188;
        case 21: return 5.3314427735250325;
        case 22: return 6.7061331941584594;
        case 23: return 8.0152365983759513;
        case 24: return 9.2823962852416120;
        case 25: return 10.5198608737723074;
        case 30: return 10.1734681350627216;
        case 31: return 8.5363163663462860;
        case 32: return 9.9694678230875962;
        case 33: return 11.3459243107430066;
        case 34: return 12.6819084426388908;
        case 35: return 13.9871886301403006;
        default:
            throw std::invalid_argument("Neumann root is not tabulated: supported nm codes are 10..15, 20..25, and 30..35");
    }
}

inline std::complex<double> cyl_j_prime(int n, std::complex<double> z) {
    return (n == 0) ? -bessel::cyl_j(1, z)
                    : 0.5 * (bessel::cyl_j(n - 1, z) - bessel::cyl_j(n + 1, z));
}

}  // namespace workflows

#endif  // SUBWAVELENGTHRESONATORS_WORKFLOWS_BESSEL_HELPERS_H
