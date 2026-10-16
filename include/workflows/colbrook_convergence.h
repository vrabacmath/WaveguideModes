#ifndef WAVEGUIDE_COLBROOK_CONVERGENCE_H
#define WAVEGUIDE_COLBROOK_CONVERGENCE_H

#include "Eigen/Dense"

namespace workflows {

struct ColbrookMinimum {
    double z;
    double sigma_min;
    double z_resolution; // spacing of the last search grid, not a spectral error bound
    bool at_boundary;
};

// Scan a fixed target interval, then repeatedly scan between the winning sample's
// neighbours. E embeds the trial coefficients into their matching output rows.
ColbrookMinimum find_colbrook_minimum(const Eigen::MatrixXcd& C, int row_offset,
                                     double target, double half_width,
                                     int intervals, int passes);

} // namespace workflows
#endif
