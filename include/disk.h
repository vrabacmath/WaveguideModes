#ifndef SUBWAVELENGTHRESONATORS_DISK_H
#define SUBWAVELENGTHRESONATORS_DISK_H

#include "Eigen/Dense"

struct Disk {
    double r;
    Eigen::Vector2d c;
    double v_b = -100.;
};

#endif // SUBWAVELENGTHRESONATORS_DISK_H
