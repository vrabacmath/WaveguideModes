//
// Created by lara on 10/9/25.
//

#include <iostream>
#include <cassert>
#include "boundary_mesh.h"

void BoundaryMesh::generate_circle(double radius, Vector2d shift) {
    double dtheta = 2.0 * M_PI / N;
    segments.clear();
    vertices.clear();
    for (int i = 0; i < N; i++) {
        double theta1 = i * dtheta;
        double theta2 = (i + 1) * dtheta;

        Vector2d start(radius * cos(theta1), radius * sin(theta1));
        Vector2d end(radius * cos(theta2), radius * sin(theta2));

        Vector2d tangentS = (end - start).normalized();
        Vector2d normalS(tangentS.y(), -tangentS.x()); // outward normal

        Vector2d tangentV(-sin(theta1), cos(theta1));
        Vector2d normalV(cos(theta1), sin(theta1));

        double length = (end - start).norm();
        double curvature = 1.0 / radius;
        double sign = -(tangentV.x() * normalV.y() - tangentV.y() * normalV.x()) > 0 ? 1.0 : -1.0;
        curvature *= sign;
        Vector2d center = 0.5 * (start + end);

        segments.push_back({start + shift, end + shift, tangentS, normalS, length, curvature, center + shift});
        vertices.push_back({start + shift, tangentV, normalV, curvature, radius * dtheta, radius});
    }
//    // last vertex
//    Segment s = segments[N-1];
//    vertices.push_back({s.end, s.tangent, s.normal, s.curvature});
    std::pair<int, int> indices;
    indices.first = 0;
    indices.second = N - 1;
    mesh_indices.push_back(indices);
    volume.push_back(M_PI * radius * radius);
}

void BoundaryMesh::generate_line(Vector2d start, Vector2d end) {
    segments.clear();
    vertices.clear();

    Vector2d tangentS = (end - start).normalized();
    Vector2d normalS(tangentS.y(), -tangentS.x()); // outward normal

    Vector2d tangentV = tangentS;
    Vector2d normalV = normalS;

    double curvature = 0.0;

    Vector2d dl = (end - start) / N;
    for (int i = 0; i < N - 1; i++) {
        Vector2d seg_start = start + i * dl;
        Vector2d seg_end = start + (i + 1) * dl;
        Vector2d center = 0.5 * (seg_start + seg_end);

        segments.push_back({seg_start, seg_end, tangentS, normalS, dl.norm(), curvature, center});
        vertices.push_back({seg_start, tangentV, normalV, curvature, dl.norm()});
    }
    // last segment
    segments.push_back({start + (N - 1) * dl, end, tangentS, normalS, dl.norm(), curvature, 0.5 * (start + (N - 1) * dl + end)});
    vertices.push_back({start + (N - 1) * dl, tangentV, normalV, curvature, dl.norm(), 1.});

    std::pair<int, int> indices;
    indices.first = 0;
    indices.second = 0;
    mesh_indices.push_back(indices);

    this->N_meshes = 1;
}

BoundaryMesh::BoundaryMesh(int N) {
    this->N = N;
    segments.reserve(N);
    vertices.reserve(N);
    this->N_meshes = 1;
}

int BoundaryMesh::get_num_segments() const {
    return N;
}

const vector<Segment> &BoundaryMesh::get_segments() const {
    return segments;
}

const Segment &BoundaryMesh::get_segment(int index) const {
    return segments[index];
}

void BoundaryMesh::generate_ellipse(double a, double b, Vector2d shift) {
    double d_theta = 2.0 * M_PI / N;
    segments.clear();
    vertices.clear();
    for (int i = 0; i < N; i++) {
        double theta1 = i * d_theta;
        double theta2 = (i + 1) * d_theta;

        Vector2d start(a * cos(theta1), b * sin(theta1));
        Vector2d end(a * cos(theta2), b * sin(theta2));

        Vector2d tangentS = (end - start).normalized();
        Vector2d normalS(tangentS.y(), -tangentS.x());

        Vector2d tangentV(-a * sin(theta1), b * cos(theta1));
        Vector2d normalV(b * cos(theta1), a * sin(theta1)); // outward normal

        double length = (end - start).norm();
        double curvature = (a * b) / std::pow(b*b*std::cos(theta1)*std::cos(theta1)
        + a*a*std::sin(theta1)*std::sin(theta1), 1.5);
        double sign = -(tangentV.x() * normalV.y() - tangentV.y() * normalV.x()) > 0 ? 1.0 : -1.0;
        curvature *= sign;
//        if (i == 44) std::cout << curvature;
        Vector2d center = 0.5 * (start + end);

        segments.push_back({start + shift, end + shift, tangentS, normalS, length, curvature, center + shift});
        vertices.push_back({start + shift, tangentV.normalized(), normalV.normalized(),
                            curvature, tangentV.norm() * d_theta, tangentV.norm()});
    }
//    // last vertex
//    Segment s = segments[N-1];
//    vertices.push_back({s.end, s.tangent, s.normal, s.curvature});
    std::pair<int, int> indices;
    indices.first = 0;
    indices.second = N - 1;
    mesh_indices.push_back(indices);
    volume.push_back(M_PI * a * b);
}

void BoundaryMesh::generate_squircle(double radius, double m, Vector2d shift) {
    double dtheta = 2.0 * M_PI / N;
    segments.clear();
    vertices.clear();
    for (int i = 0; i < N; i++) {
        double theta1 = (i + 0.5) * dtheta; // never touch the axes because the normal is undefined there
        double theta2 = (i + 1.5) * dtheta;
        double cos_t1 = cos(theta1);
        double sin_t1 = sin(theta1);
        double cos_t2 = cos(theta2);
        double sin_t2 = sin(theta2);

        double pow_c1 = pow(abs(cos_t1), m - 1.);
        double pow_s1 = pow(abs(sin_t1), m - 1.);
        double pow_c2 = pow(abs(cos_t2), m - 1.);
        double pow_s2 = pow(abs(sin_t2), m - 1.);

        Vector2d start(radius * pow_c1 * cos_t1,
                       radius * pow_s1 * sin_t1);
        Vector2d end(radius * pow_c2 * cos_t2,
                     radius * pow_s2 * sin_t2);

        Vector2d tangentS = (end - start).normalized();
        Vector2d normalS(tangentS.y(), -tangentS.x()); // outward normal

        Vector2d tangentV(m * radius * pow_c1 * -sin_t1,
                          m * radius * pow_s1 * cos_t1);
        // outward normal
        Vector2d normalV(-m * radius * ((m - 1.) * sin_t1 * sin_t1 * pow_c1 / cos_t1 - cos_t1 * cos_t1 * pow_c1),
                         -m * radius * ((m - 1.) * cos_t1 * cos_t1 * pow_s1 / sin_t1 - sin_t1 * sin_t1 * pow_s1));

        double length = (end - start).norm();

        double curvature = -(tangentV.x() * normalV.y() - tangentV.y() * normalV.x()) /
                            pow(tangentV.norm(), 3);
        Vector2d center = 0.5 * (start + end);
        segments.push_back({start + shift, end + shift, tangentS, normalS, length, curvature, center + shift});
        vertices.push_back({start + shift, tangentV.normalized(), normalV.normalized(),
                            curvature, tangentV.norm() * dtheta, tangentV.norm()});
        cout << "curvature at vertex " << i << ": " << curvature << endl;
    }
    std::pair<int, int> indices;
    indices.first = 0;
    indices.second = N - 1;
    mesh_indices.push_back(indices);
    // approximate area for squircle
    double area = 4.0 * radius * radius * tgamma(1.0 + 0.5 * m) * tgamma(1.0 + 0.5 * m) / tgamma(1.0 + m);
    volume.push_back(area);
}


const Vertex &BoundaryMesh::get_vertex(int index) const {
    return vertices[index];
}

BoundaryMesh::BoundaryMesh() {
    this->N = 0;
    this->N_meshes = 0;
    segments.reserve(0);
    vertices.reserve(0);
}

void BoundaryMesh::add_mesh(const BoundaryMesh &mesh) {
    int N_mesh = mesh.get_num_segments();

    segments.reserve(segments.size() + N_mesh);
    vertices.reserve(vertices.size() + N_mesh);
    volume.reserve(N_meshes + 1);

    for (int i = 0; i < N_mesh; i++) {
        segments.push_back(mesh.get_segment(i));
        vertices.push_back(mesh.get_vertex(i));
    }
    volume.push_back(mesh.volume[0]);
    N += N_mesh;
    N_meshes += 1;

    mesh_indices.reserve(N_meshes);

    std::pair<int, int> indices;
    if (N_meshes == 1) {
        indices.first = 0;
        indices.second = N_mesh - 1;
    } else {
        indices.first = mesh_indices[N_meshes - 2].second + 1;
        indices.second = indices.first + N_mesh - 1;
    }
    mesh_indices.push_back(indices);
}


void BoundaryMesh::add_polar_mesh(const BoundaryMesh &mesh) {
    int N_mesh = mesh.get_num_segments();

    segments.reserve(segments.size() + N_mesh);
    vertices.reserve(vertices.size() + N_mesh);
    volume.reserve(N_meshes + 1);
    polar_coeffs.reserve(N_meshes + 1);
    polar_shifts.reserve(N_meshes + 1);

    for (int i = 0; i < N_mesh; i++) {
        segments.push_back(mesh.get_segment(i));
        vertices.push_back(mesh.get_vertex(i));
    }
    volume.push_back(mesh.volume[0]);
    polar_coeffs.push_back(mesh.polar_coeffs[0]);
    polar_shifts.push_back(mesh.polar_shifts[0]);
    N += N_mesh;
    N_meshes += 1;

    mesh_indices.reserve(N_meshes);

    std::pair<int, int> indices;
    if (N_meshes == 1) {
        indices.first = 0;
        indices.second = N_mesh - 1;
    } else {
        indices.first = mesh_indices[N_meshes - 2].second + 1;
        indices.second = indices.first + N_mesh - 1;
    }
    mesh_indices.push_back(indices);
}

int BoundaryMesh::get_num_meshes() const {
    return N_meshes;
}

int BoundaryMesh::get_start_index(int mesh_index) const {
    return mesh_indices[mesh_index].first;
}

int BoundaryMesh::get_end_index(int mesh_index) const {
    return mesh_indices[mesh_index].second;
}

VectorXd BoundaryMesh::get_volume_vector() const {
    VectorXd V = VectorXd::Zero(N_meshes);
    for (int i = 0; i < N_meshes; i++) {
        V(i) = volume[i];
    }
    return V;
}

void BoundaryMesh::set_new_vertices(const vector<Vertex> &new_vertices) {
    assert(new_vertices.size() == vertices.size());
    this->vertices = new_vertices;

    // shoelace formula to compute enclosed volume
    for (int mesh_i = 0; mesh_i < N_meshes; mesh_i++) {
        int start_idx = get_start_index(mesh_i);
        int end_idx = get_end_index(mesh_i);
        double vol = 0.0;//, vol_s = 0.0;
        for (int i = start_idx; i <= end_idx; i++) {
//            Vector2d p1 = new_vertices[i].point;
//            Vector2d p2;
//            if (i < end_idx) {
//                p2 = new_vertices[i + 1].point;
//            } else {
//                p2 = new_vertices[start_idx].point;
//            }
//            vol_s += p1.x() * p2.y() - p2.x() * p1.y();

            // Green's theorem version
            Vector2d p = new_vertices[i].point;
            Vector2d t = new_vertices[i].tangent;
            vol += (p.x() * t.y() - p.y() * t.x()) * new_vertices[i].sigma;
        }
        vol *= 0.5;
//        vol_s *= 0.5;
//        cout << "Volume from shoelace formula for mesh " << mesh_i << ": " << vol_s << endl;
        volume[mesh_i] = vol;
//        cout << "Updated volume of mesh " << mesh_i << ": " << vol << endl;
    }

    // set segments based on new vertices
    for (int i = 0; i < N; i++) {
        segments[i].start = new_vertices[i].point;
        if (i < N - 1) {
            segments[i].end = new_vertices[i + 1].point;
        } else {
            segments[i].end = new_vertices[0].point;
        }
        segments[i].tangent = (segments[i].end - segments[i].start).normalized();
        segments[i].normal = Vector2d(segments[i].tangent.y(), -segments[i].tangent.x());
        segments[i].length = (segments[i].end - segments[i].start).norm();
        segments[i].curvature = new_vertices[i].curvature;
        segments[i].center = 0.5 * (segments[i].start + segments[i].end);
    }
}

/**
 * a0 (1 + sum of a_j cos(j*theta) + b_j sin(j*theta)) is the polar representation of the boundary, where a0 is
 * the first element of coeffs, and a_j, b_j are the subsequent elements.
 * shift is the translation applied to the entire mesh.
 */
void BoundaryMesh::generate_polar_mesh(VectorXd coeffs, Vector2d shift) {
    double dtheta = 2.0 * M_PI / N;
    segments.clear();
    vertices.clear();
    polar_coeffs.clear();
    polar_shifts.clear();

    double vol = 0.;
    for (int i = 0; i < N; i++) {
        double theta1 = i * dtheta;
        double theta2 = (i + 1) * dtheta;

        double r1 = 1., r2 = 1.;
        double dr1 = 0., dr2 = 0.;
        double ddr1 = 0., ddr2 = 0.;

        for (int j = 2; j < coeffs.size(); j += 2) {
            int k = j / 2.;
            r1 += coeffs(j) * cos(k * theta1);
            r2 += coeffs(j) * cos(k * theta2);
            dr1 += -k * coeffs(j) * sin(k * theta1);
            dr2 += -k * coeffs(j) * sin(k * theta2);
            ddr1 += -k * k * coeffs(j) * cos(k * theta1);
            ddr2 += -k * k * coeffs(j) * cos(k * theta2);
        }

        for (int j = 1; j < coeffs.size(); j += 2) {
            int k = (j + 1.) / 2.;
            r1 += coeffs(j) * sin(k * theta1);
            r2 += coeffs(j) * sin(k * theta2);
            dr1 += k * coeffs(j) * cos(k * theta1);
            dr2 += k * coeffs(j) * cos(k * theta2);
            ddr1 += -k * k * coeffs(j) * sin(k * theta1);
            ddr2 += -k * k * coeffs(j) * sin(k * theta2);
        }

        r1 *= coeffs(0);
        r2 *= coeffs(0);
        dr1 *= coeffs(0);
        dr2 *= coeffs(0);
        ddr1 *= coeffs(0);
        ddr2 *= coeffs(0);

        Vector2d start(r1 * cos(theta1), r1 * sin(theta1));
        Vector2d end(r2 * cos(theta2), r2 * sin(theta2));

        Vector2d tangentS = (end - start).normalized();
        Vector2d normalS(tangentS.y(), -tangentS.x()); // outward normal

        Vector2d tangentV(-sin(theta1) * r1 + dr1 * cos(theta1), cos(theta1) * r1 + dr1 * sin(theta1));
        Vector2d normalV(cos(theta1) * r1 + dr1 * sin(theta1) - ddr1 * cos(theta1) + dr1 * sin(theta1),
                         sin(theta1) * r1 - dr1 * cos(theta1) - ddr1 * sin(theta1) - dr1 * cos(theta1));

        double length = (end - start).norm();
        double curvature = -(tangentV.x() * normalV.y() - tangentV.y() * normalV.x()) /
                           pow(tangentV.squaredNorm(), 1.5);

        normalV = Vector2d(tangentV.y(), -tangentV.x());
        normalV.normalize();
//        if (normalV.dot(start) < 0.0) {
//            normalV *= -1.0; // flip inward -> outward
//        }

//        cout << normalV << endl;
//        cout << "(" << cos(theta1) << ", " << sin(theta1) << ")" << endl << endl;
        Vector2d center = 0.5 * (start + end);

        segments.push_back({start + shift, end + shift, tangentS, normalS, length, curvature, center + shift});
        vertices.push_back({start + shift, tangentV, normalV, curvature, tangentV.norm() * dtheta, tangentV.norm()});

        vol += (start.x() * tangentV.y() - start.y() * tangentV.x()) * dtheta;
    }
    vol *= 0.5;
//    cout << vol << endl;
    std::pair<int, int> indices;
    indices.first = 0;
    indices.second = N - 1;
    mesh_indices.push_back(indices);
    volume.push_back(vol);
    polar_coeffs.push_back(coeffs);
    polar_shifts.push_back(shift);
}

VectorXd BoundaryMesh::get_polar_coeffs(int index) const {
    return polar_coeffs[index];
}

VectorXd BoundaryMesh::get_polar_shifts(int index) const {
    return polar_shifts[index];
}
