//
// Created by lara on 10/9/25.
//

#ifndef SUBWAVELENGTHRESONATORS_OPERATORS_H
#define SUBWAVELENGTHRESONATORS_OPERATORS_H


#include <complex>
#include "Eigen/Dense"
#include "basis.h"
#include "boundary_mesh.h"
#include <math.h>

using namespace std;
using namespace Eigen;

class Operators {
public:
    /**
     * @brief Assemble single-layer boundary integral operator.
     * @param S Output matrix.
     * @param k Wavenumber.
     * @param mesh Boundary mesh.
     * @param eta Coupling/regularization parameter used by implementation.
     */
    static void S(MatrixXcd& S, double k, BoundaryMesh mesh, double eta);

    /**
     * @brief Assemble adjoint double-layer boundary integral operator.
     */
    static void Kstar(MatrixXcd& Kstar, double k, BoundaryMesh mesh, double eta);

    /**
     * @brief Assemble double-layer boundary integral operator.
     */
    static void K(MatrixXcd& K, double k, BoundaryMesh mesh, double eta);
};


#endif //SUBWAVELENGTHRESONATORS_OPERATORS_H

/*
 * //
// Created by lara on 10/26/25.
//

#ifndef SUBWAVELENGTHRESONATORS_SPECTRAL_OPERATORS_H
#define SUBWAVELENGTHRESONATORS_SPECTRAL_OPERATORS_H

#include <complex>
#include "Eigen/Dense"
#include "basis.h"
#include "boundary_mesh.h"
#include <math.h>
#include "omp.h"
using cpxd = complex<double>;

using namespace std;
using namespace Eigen;

class SpectralOperators {
    public:
        static void S(MatrixXcd& S, cpxd k, const BoundaryMesh& mesh);
        static void Kstar(MatrixXcd& Kstar, cpxd k, const BoundaryMesh& mesh);
        static void dS(MatrixXcd& dS, cpxd k, const BoundaryMesh& mesh);
        static void K(MatrixXcd& K, cpxd k, const BoundaryMesh& mesh);
        static void DirPeriodicS(MatrixXcd& S, cpxd k, double kbar, double a, const BoundaryMesh& mesh);
        static void DirPeriodicKstar_diagonal(MatrixXcd &Kstar_diagonal, cpxd k, double kbar, double a, const BoundaryMesh& mesh);

//        static void DirPeriodicS_diagonal(MatrixXcd& S, double k, double a, double b, const BoundaryMesh& mesh);
        static void A(MatrixXcd &A_matrix, cpxd k, cpxd k_b, double delta, const BoundaryMesh& mesh);
        static void DirPeriodicA(MatrixXcd &A_matrix, cpxd k, cpxd k_b, double kbar, double delta, double a, const BoundaryMesh& mesh);

        static void makeDirPeriodicRHS(VectorXcd &rhs, cpxd k, double kbar, double delta, const BoundaryMesh& mesh);
        static MatrixXcd makeCapacitanceMatrix(MatrixXcd &C, MatrixXcd S, const BoundaryMesh& mesh);
};


#endif //SUBWAVELENGTHRESONATORS_SPECTRAL_OPERATORS_H

 */