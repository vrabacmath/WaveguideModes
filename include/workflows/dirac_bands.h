#ifndef DIRAC_BANDS_H
#define DIRAC_BANDS_H

#include "hex_crystal.h"
#include "bessel_helpers.h"

namespace workflows {

    void capacitance_dirac_bands(Crystal& crystal, int r_idx, int max_m, int Npath);

} // namespace dirac_bands

#endif // DIRAC_BANDS_H