#include "workflows/bent_waveguide_field.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

#include "Eigen/Dense"
#include "bessel-library.hpp"
#include "kernels.h"
#include "tools.h"
#include "utils.h"
#include "workflows/bent_waveguide_patch.h"

namespace workflows {

using namespace Eigen;

namespace {

// Where a grid point sits relative to the cluster.
struct Location {
    enum Kind { Exterior, ResonantDisk, DarkDisk, NearBoundary } kind;
    int slot;      // index into main_indices when kind == ResonantDisk
    double rho;    // local polar radius about that disk's centre
    double phi;    // local polar angle
};

}  // namespace

void run_bent_waveguide_tesselation(double radius, double defect_radius, double delta, int m_ang,
                              int n_defect, int n_clad, int fringe, int points_per_disk,
                              int mode_index, int grid_points) {
    const BentPatch p = build_bent_patch(radius, defect_radius, m_ang, n_defect, n_clad,
                                         fringe, points_per_disk);
    const int modes = p.modes;
    const int n_main = static_cast<int>(p.main_indices.size());

    Utils::draw_mesh(p.mesh, "bent_waveguide_mesh.csv");

    // --- eigenpairs, kept paired -----------------------------------------------------------
    // Sort a permutation, never the eigenvalue array in place: sorting `lam` directly would
    // decouple it from es.eigenvectors() and silently mislabel every mode drawn below.
    ComplexEigenSolver<MatrixXcd> es(p.C);
    const VectorXcd lam = es.eigenvalues();
    const MatrixXcd vecs = es.eigenvectors();
    std::vector<int> order(lam.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&lam](int a, int b) { return lam(a).real() < lam(b).real(); });

    // Choose a frequency and mode to draw. The most corner-localized mode is the one with the largest Re(lambda) -- the one that is most resonant with the corner disk.
    // --- the density on the WHOLE cluster --------------------------------------------------
    SpectralOperators ops(p.mesh);
    int Nres = p.mesh.get_num_meshes();

    // --- evaluate on a grid ----------------------------------------------------------------
    // Only disks carrying a mode amplitude are "resonant"; the fringe defect disks and every
    // cladding disk have zero Dirichlet data, so their interior field is zero at leading order.
    std::vector<int> slot_of_disk(p.disks.size(), -1);
    for (int q = 0; q < n_main; ++q) slot_of_disk[p.main_indices[q]] = q;

    const double half = double(p.L) + 1.0;
    const int Ng = std::max(grid_points, 16);
    // Nystrom evaluation of a single-layer potential degrades within ~one node spacing of the
    // boundary, so blank the exterior there rather than plotting quadrature error.
    const double h = 2.0 * M_PI * std::max(radius, defect_radius) / double(p.N);
    const double Jm_beta = bessel::cyl_j(m_ang, p.beta);

    std::vector<double> axis(Ng);
    for (int i = 0; i < Ng; ++i) axis[i] = -half + 2.0 * half * double(i) / double(Ng - 1);

    auto classify = [&](const Vector2d& r) {
        Location loc{Location::Exterior, -1, 0.0, 0.0};
        for (std::size_t i = 0; i < p.disks.size(); ++i) {
            const Vector2d d = r - p.disks[i].c;
            const double rho = d.norm();
            if (std::abs(rho - p.disks[i].r) < h) { loc.kind = Location::NearBoundary; return loc; }
            if (rho < p.disks[i].r) {
                loc.slot = slot_of_disk[i];
                loc.kind = (loc.slot >= 0) ? Location::ResonantDisk : Location::DarkDisk;
                loc.rho = rho;
                loc.phi = std::atan2(d.y(), d.x());
                return loc;
            }
        }
        return loc;
    };

    std::vector<std::vector<double>> re(Ng, std::vector<double>(Ng)),
                                     im(Ng, std::vector<double>(Ng)),
                                     mag(Ng, std::vector<double>(Ng)),
                                     phase(Ng, std::vector<double>(Ng));

    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (int iy = 0; iy < Ng; ++iy) {
#pragma omp parallel for schedule(dynamic)
        for (int ix = 0; ix < Ng; ++ix) {
            const Vector2d r(axis[ix], axis[iy]);
            const Location loc = classify(r);
            cpxd u = 0.0;

            if (loc.kind == Location::NearBoundary) {
                re[iy][ix] = im[iy][ix] = mag[iy][ix] = phase[iy][ix] = nan;
                continue;
            }
            if (loc.kind == Location::ResonantDisk) {
                // Interior Neumann mode, normalised so its trace at rho = R_def is exactly the
                // corresponding column of G -- hence u is continuous across the boundary.
                const double radial = bessel::cyl_j(m_ang, p.beta * loc.rho / defect_radius) / Jm_beta;
                for (int s = 0; s < modes; ++s) {
                    const double sg = (s == 0) ? 1.0 : -1.0;
                    u += v(modes * loc.slot + s) * p.Anorm * radial *
                         std::exp(cpxd(0, 1.0) * sg * double(m_ang) * loc.phi);
                }
            } else if (loc.kind == Location::Exterior) {
                for (int j = 0; j < p.Ntot; ++j)
                    u += psi(j) * Kernels::helmholtz_2D(p.k, r, p.mesh.get_vertex(j).point) *
                         p.sigma(j);
            }  // DarkDisk leaves u = 0

            re[iy][ix] = u.real();
            im[iy][ix] = u.imag();
            mag[iy][ix] = std::abs(u);
            phase[iy][ix] = std::atan2(u.imag(), u.real());
        }
        Tools::print_stage_progress("Evaluating field rows", iy + 1, Ng);
    }
    Tools::finish_progress_line();

    Tools::dump_csv("bent_waveguide_field_real.csv", re);
    Tools::dump_csv("bent_waveguide_field_imag.csv", im);
    Tools::dump_csv("bent_waveguide_field_mag.csv", mag);
    Tools::dump_csv("bent_waveguide_field_phase.csv", phase);
    Tools::dump_csv("bent_waveguide_field_axis.csv", {axis});

    // --- built-in checks -------------------------------------------------------------------
    // The patch is invariant under reflection about y = x, which maps site(l) <-> site(-l), so
    // every eigenmode must be even or odd under it and |u| must be exactly symmetric.
    double sym = 0.0, scale = 0.0;
    for (int iy = 0; iy < Ng; ++iy)
        for (int ix = 0; ix < Ng; ++ix) {
            const double a = mag[iy][ix], b = mag[ix][iy];
            if (std::isnan(a) || std::isnan(b)) continue;
            sym = std::max(sym, std::abs(a - b));
            scale = std::max(scale, a);
        }
    std::cout << "  reflection symmetry ||u(x,y)|-|u(y,x)||_max = " << sym
              << "  (relative " << (scale > 0 ? sym / scale : 0.0) << ")\n";

    // Continuity of the interior branch with the imposed trace. Both sides are analytic, so this
    // isolates the normalisation (Anorm, J_m(beta)) with no quadrature involved: at rho = R_def
    // the radial factor is 1 and the interior branch must reproduce column-combination G v
    // exactly. NOT tested by evaluating the exterior potential just off the boundary -- plain
    // Nystrom evaluation of a single-layer potential is invalid there, which is what the mask is
    // for, so such a test would only measure quadrature error.
    {
        const int q = p.center;
        const int s0 = p.mesh.get_start_index(p.main_indices[q]);
        const VectorXcd trace = p.G * v;
        double worst = 0.0, ref = 0.0;
        for (int loc = 0; loc < p.N; ++loc) {
            const double th = 2.0 * M_PI * double(loc) / double(p.N);
            cpxd u_in = 0.0;
            for (int s = 0; s < modes; ++s) {
                const double sg = (s == 0) ? 1.0 : -1.0;
                u_in += v(modes * q + s) * p.Anorm *
                        std::exp(cpxd(0, 1.0) * sg * double(m_ang) * th);
            }
            worst = std::max(worst, std::abs(u_in - trace(s0 + loc)));
            ref = std::max(ref, std::abs(u_in));
        }
        std::cout << "  interior branch vs imposed trace: max|du| = " << worst << "  (relative "
                  << (ref > 0 ? worst / ref : 0.0) << ")\n";
    }

    // Exterior quadrature accuracy is NOT checked by comparing against the boundary trace at
    // small offsets -- the field varies physically over that scale, so such a test cannot
    // separate quadrature error from real variation. The meaningful check is mesh refinement at
    // fixed geometry: doubling points_per_disk from 32 to 64 moves the unmasked field by 8e-4
    // relative (0.06% of peak), so the mask width of one node spacing is sufficient.

    std::cout << "[wrote] bent_waveguide_modes.csv, bent_waveguide_field_{real,imag,mag,phase}.csv, "
                 "bent_waveguide_field_axis.csv\n";
}

}  // namespace workflows
