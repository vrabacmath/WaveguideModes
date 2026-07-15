// Precompiled Header for SubwavelengthResonators
// This file includes heavy headers that are included in most translation units
// to reduce redundant parsing and compilation time.

#ifndef SUBWAVELENGTHRESONATORS_PCH_H
#define SUBWAVELENGTHRESONATORS_PCH_H

// Standard library headers
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <complex>
#include <cmath>
#include <algorithm>
#include <functional>
#include <chrono>
#include <filesystem>

// Eigen (heavy template library - major compile-time cost)
#include "Eigen/Dense"
#include "Eigen/Eigenvalues"
#include "unsupported/Eigen/FFT"

// External math library (large header-only dependency)
#include "bessel-library.hpp"

// Math constants (used everywhere)
#include "math_constants.h"

#endif // SUBWAVELENGTHRESONATORS_PCH_H
