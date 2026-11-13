#ifndef DIRAC_BANDS_H
#define DIRAC_BANDS_H

#include "hex_crystal.h"
#include "bessel_helpers.h"

namespace workflows {

    struct DiracModeFamily {
        int nm, angular_modes;
        double omega0;
        std::vector<int> disk_indices; // Compact column blocks, in disk/angular order.
        Eigen::MatrixXcd G;            // Unit interior mode traces, without speed scaling.
        Eigen::VectorXd source_speeds; // One speed per column of G.
    };

    // Select disks sharing v_b/r with the reference disk, up to floating-point roundoff.
    // Disks with other resonance frequencies remain in the exterior solve.
    DiracModeFamily build_dirac_mode_family(const Crystal& crystal, int reference_disk, int nm);
    Eigen::MatrixXcd dirac_capacitance_matrix(const Crystal& crystal,
                                             const DiracModeFamily& family, Eigen::Vector2d alpha);
    void capacitance_dirac_bands(const Crystal& crystal, int reference_disk, int max_nm, int Npath);

} // namespace dirac_bands

#endif // DIRAC_BANDS_H
