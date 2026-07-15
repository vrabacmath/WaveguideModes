//
// Created by lara on 10/9/25.
//

#ifndef SUBWAVELENGTHRESONATORS_BOUNDARY_MESH_H
#define SUBWAVELENGTHRESONATORS_BOUNDARY_MESH_H


#include "basis.h"
#include "Eigen/Dense"
#include "math_constants.h"
using namespace Eigen;

class BoundaryMesh {
private:
    int N; // Number of segments
    int N_meshes; // Number of disjoint meshes
    vector<Segment> segments; // Vector of segments
    vector<Vertex> vertices; // Vector of vertices
    vector<std::pair<int, int>> mesh_indices; // start and end indices of each mesh
    vector<double> volume; // Enclosed volume
    vector<VectorXd> polar_coeffs;
    vector<Vector2d> polar_shifts;

public:
    /**
     * @brief Construct an empty boundary mesh with a prescribed number of segments.
     * @param N Number of boundary segments/vertices used for discretization.
     */
    explicit BoundaryMesh(int N);

    /**
     * @brief Construct an empty boundary mesh with no preset discretization size.
     */
    explicit BoundaryMesh();

    /**
     * @brief Generate a circular boundary and discretize it into the current mesh.
     * @param radius Circle radius.
     * @param shift Optional center shift.
     */
    void generate_circle(double radius, Vector2d shift = Vector2d(0.0, 0.0));

    /**
     * @brief Generate an ellipse boundary and discretize it into the current mesh.
     * @param a Semi-axis length in x direction.
     * @param b Semi-axis length in y direction.
     * @param shift Optional center shift.
     */
    void generate_ellipse(double a, double b, Vector2d shift = Vector2d(0.0, 0.0));

    /**
     * @brief Generate a superellipse/squircle boundary and discretize it.
     * @param radius Radius-like scaling parameter.
     * @param m Shape exponent controlling corner sharpness.
     * @param shift Optional center shift.
     */
    void generate_squircle(double radius, double m, Vector2d shift = Vector2d(0.0, 0.0));

    /**
     * @brief Generate a regular polygon boundary.
     * @param sides Number of polygon sides.
     * @param radius Circumradius of the polygon.
     */
    void generate_polygon(int sides, double radius);

    /**
     * @brief Generate a straight-line boundary segment between two points.
     * @param start Start point.
     * @param end End point.
     */
    void generate_line(Vector2d start, Vector2d end);

    /**
     * @brief Generate a boundary from polar/Fourier coefficients.
     * @param coeffs Coefficient vector describing radial profile versus angle.
     * @param shift Optional center shift.
     */
    void generate_polar_mesh(VectorXd coeffs, Vector2d shift = Vector2d(0.0, 0.0));

    /**
     * @brief Replace mesh vertices and recompute derived geometric quantities.
     * @param new_vertices New vertex list in boundary order.
     */
    void set_new_vertices(const vector<Vertex>& new_vertices);

    /**
     * @brief Append a disconnected mesh component to the current mesh union.
     * @param mesh Component mesh to append.
     */
    void add_mesh(const BoundaryMesh& mesh);

    /**
     * @brief Append a disconnected mesh component and preserve polar metadata.
     * @param mesh Component mesh to append.
     */
    void add_polar_mesh(const BoundaryMesh& mesh);

    /**
     * @brief Get the number of disconnected mesh components.
     */
    int get_num_meshes() const;

    /**
     * @brief Get the global start vertex index of a component.
     * @param mesh_index Component index.
     */
    [[nodiscard]] int get_start_index(int mesh_index) const;

    /**
     * @brief Get the global end vertex index of a component.
     * @param mesh_index Component index.
     */
    [[nodiscard]] int get_end_index(int mesh_index) const;

    /**
     * @brief Get the total number of segments/vertices across all components.
     */
    int get_num_segments() const;

    /**
     * @brief Access all segments in global indexing order.
     */
    [[nodiscard]] const vector<Segment>& get_segments() const;

    /**
     * @brief Access one segment by global index.
     * @param index Segment index.
     */
    [[nodiscard]] const Segment& get_segment(int index) const;

    /**
     * @brief Access one vertex by global index.
     * @param index Vertex index.
     */
    [[nodiscard]] const Vertex& get_vertex(int index) const;

    /**
     * @brief Get per-component enclosed volumes/areas as a vector.
     */
    [[nodiscard]] VectorXd get_volume_vector() const;

    /**
     * @brief Get stored polar coefficients for one component.
     * @param index Component index.
     */
    [[nodiscard]] VectorXd get_polar_coeffs(int index) const;

    /**
     * @brief Get stored polar center shift for one component.
     * @param index Component index.
     */
    [[nodiscard]] VectorXd get_polar_shifts(int index) const;
};

//class DisjointBoundaryMesh : public BoundaryMesh {
//private:
//    int N_meshes; // Number of disjoint meshes
//    int N_total; // Total number of segments
//    vector<Segment> segments; // Vector of segments
//    vector<Vertex> vertices; // Vector of vertices
//
//public:
//    explicit DisjointBoundaryMesh();
//    void add_mesh(const BoundaryMesh& mesh);
//
//    int get_num_segments() const;
//    [[nodiscard]] const vector<Segment>& get_segments() const;
//    [[nodiscard]] const Segment& get_segment(int index) const;
//    [[nodiscard]] const Vertex& get_vertex(int index) const;
//};

#endif //SUBWAVELENGTHRESONATORS_BOUNDARY_MESH_H
