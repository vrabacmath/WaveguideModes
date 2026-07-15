//
// Created by lara on 10/9/25.
//

#ifndef SUBWAVELENGTHRESONATORS_BASIS_H
#define SUBWAVELENGTHRESONATORS_BASIS_H

#include "Eigen/Dense"
using namespace Eigen;
using namespace std;

struct Segment {
    Vector2d start;
    Vector2d end;
    Vector2d tangent;
    Vector2d normal;
    double length;
    double curvature;
    Vector2d center;
};

struct Vertex {
    Vector2d point;
    Vector2d tangent;
    Vector2d normal;
    double curvature;
    double sigma;
    double tnorm;
};

#endif //SUBWAVELENGTHRESONATORS_BASIS_H
