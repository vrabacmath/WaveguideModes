//
// Created by lara on 10/11/25.
// the Gauss Legendre quadrature matrix from a previous project
//

#include "legendre_matrix.h"

const double* LegendreMatrix::getWeights(int order) {
    return weightsMatrix[order - 1];
}

const double* LegendreMatrix::getPositions(int order) {
    return positionMatrix[order - 1];
}