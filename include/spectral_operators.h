//
// Created by lara on 10/26/25.
//

#ifndef SUBWAVELENGTHRESONATORS_SPECTRAL_OPERATORS_H
#define SUBWAVELENGTHRESONATORS_SPECTRAL_OPERATORS_H

#include <complex>
#include "Eigen/Dense"
#include "basis.h"
#include "boundary_mesh.h"
#include <math.h>
#include "green_kernels.h"
#include "kernels.h"
#if __has_include(<omp.h>)
#include <omp.h>
#endif
#include "unsupported/Eigen/FFT"
using cpxd = complex<double>;

using namespace std;
using namespace Eigen;

class SpectralOperators {
    public:
        SpectralOperators(BoundaryMesh mesh);

        /**
         * Fourier synthesis matrix for equally spaced angular samples on each component.
         * Shape: (point_count * component_count) x ((2 * order_cutoff + 1) * component_count).
         * Rows: component * point_count + point. Columns: (mode + order_cutoff) * component_count + component.
         * Modes run from -order_cutoff to +order_cutoff, matching Multipole's ordering.
         * Samples are at theta = 2*pi*point/point_count, with entries exp(i*mode*theta).
         * For B = fourier_basis(...), nodal values = B * coefficients, and the unweighted
         * Fourier projection of a nodal operator A is B.adjoint() * A * B / point_count.
         * With point_count >= 2*order_cutoff+1, B.adjoint()*B / point_count is the identity.
         */
        static MatrixXcd fourier_basis(int point_count, int order_cutoff, int component_count = 1);

        void S_diagonal(MatrixXcd& S, cpxd k) const;
        void S(MatrixXcd& S, cpxd k) const;
        void Kstar_diagonal(MatrixXcd& Kstar, cpxd k) const;
        void Kstar(MatrixXcd& Kstar, cpxd k) const;
        void dS(MatrixXcd& dS, cpxd k);
        void K_diagonal(MatrixXcd& K, cpxd k);
        void K(MatrixXcd& K, cpxd k);
        void DirPeriodicS_diagonal(MatrixXcd& S, cpxd k, double kbar, double a);
        void PeriodicS_diagonal(MatrixXcd& S, cpxd k, double kbar, double a);
        void PeriodicS(MatrixXcd& S, cpxd k, double kbar, double a);
        void PeriodicKstar_diagonal(MatrixXcd& Kstar, cpxd k, double kbar, double a);
        void PeriodicKstar(MatrixXcd& Kstar, cpxd k, double kbar, double a);
        void QuasiPeriodicStaticS(MatrixXcd& S, double alpha, double a = 1.0, int M = 200);
        void DirPeriodicS(MatrixXcd& S, cpxd k, double kbar, double a);
        void DirPeriodicKstar_diagonal(MatrixXcd &Kstar, cpxd k, double kbar, double a);
        void DirPeriodicKstar(MatrixXcd &Kstar, cpxd k, double kbar, double a);

//        static void DirPeriodicS_diagonal(MatrixXcd& S, double k, double a, double b, const BoundaryMesh& mesh);
        void A(MatrixXcd &A_matrix, cpxd k, cpxd k_b, double delta) const;
        void DirPeriodicA(MatrixXcd &A_matrix, cpxd k, cpxd k_b, double kbar, double delta, double a);
        VectorXcd PeriodicDensityStabilized(cpxd k, cpxd k_b, double kbar, double delta, double a, cpxd eta = 0.);

        void makeDirPeriodicRHS(VectorXcd &rhs, cpxd k, double kbar, double delta);
        MatrixXcd makeCapacitanceMatrix(MatrixXcd &C, const MatrixXcd& S);

        void CrystalS_diagonal(MatrixXcd &S, cpxd k, Vector2d alpha, Vector2d a1, Vector2d a2);
        void CrystalS(MatrixXcd &S, cpxd k, Vector2d alpha, Vector2d a1, Vector2d a2);
        void CrystalKstar_diagonal(MatrixXcd &Kstar, cpxd k, Vector2d alpha, Vector2d a1, Vector2d a2);
        void CrystalKstar(MatrixXcd &Kstar, cpxd k, Vector2d alpha, Vector2d a1, Vector2d a2);
        void CrystalK_diagonal(MatrixXcd &K, cpxd k, Vector2d alpha, Vector2d a1, Vector2d a2);
        void CrystalK(MatrixXcd &K, cpxd k, Vector2d alpha, Vector2d a1, Vector2d a2);

        void CrystalA(MatrixXcd &A_matrix, cpxd k, cpxd k_b, Vector2d alpha, Vector2d a1, Vector2d a2, double delta);

        void DefectCrystalA(MatrixXcd &A_matrix, BoundaryMesh &phantom_mesh, cpxd k, cpxd k_b, Vector2d alpha, Vector2d a1, Vector2d a2, double delta);

    private:
        VectorXcd R_j_Kress;
        BoundaryMesh mesh;
};


#endif //SUBWAVELENGTHRESONATORS_SPECTRAL_OPERATORS_H
