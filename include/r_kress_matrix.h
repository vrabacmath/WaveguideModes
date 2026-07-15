//
// Created by lara on 12/5/25.
//

#ifndef SUBWAVELENGTHRESONATORS_R_KRESS_MATRIX_H
#define SUBWAVELENGTHRESONATORS_R_KRESS_MATRIX_H

#include "Eigen/Dense"
#include <complex>

using namespace Eigen;

class RKressMatrix {
public:
    /**
     * @brief Return pointer to precomputed Kress correction coefficients for size N.
     */
    const double* returnR(int N);

    /**
     * @brief Compute Kress correction vector in physical space for N nodes.
     */
    static VectorXd computeRVec(int N);

    /**
     * @brief Fill matrix whose rows contain Kress correction vectors up to Nmax.
     */
    static void computeRMat(MatrixXd& R_mat, int Nmax);
};

#endif //SUBWAVELENGTHRESONATORS_R_KRESS_MATRIX_H
