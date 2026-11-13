#include "workflows/interface_field.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "Eigen/Dense"
#include "bessel-library.hpp"
#include "kernels.h"
#include "tools.h"
#include "utils.h"
#include "workflows/interface_patch.h"

namespace workflows {

using namespace Eigen;

namespace {

// Where a grid point sits relative to the cluster.
struct Location {
    enum Kind { Exterior, ResonantDisk, DarkDisk, NearBoundary } kind;
    int slot;      // index into active_indices when kind == ResonantDisk
    double rho;    // local polar radius about that disk's centre
    double phi;    // local polar angle
};

}  // namespace

void run_interface_field(const InterfacePatch& p, int grid_points,
                                    const std::string& prefix) {
    if (grid_points < 16)
        throw std::invalid_argument("Field grid requires at least sixteen points per axis");
    if (!(p.N > 2 * p.Nmultipole + 1 && p.Nmultipole >= p.m_ang)) {
        throw std::invalid_argument("The following must hold: N > 2*Nmultipole+1 and Nmultipole >= m_ang");
    }
    const int Ng = std::max(grid_points, 16);
    const int template_count = static_cast<int>(p.meshes.size());
    double max_r = 0.0;
    for (auto disk : p.disks_up) {
        max_r = std::max(max_r, disk.r);
    }
    for (auto disk : p.disks_down) {
        max_r = std::max(max_r, disk.r);
    }
    const double h = 2.0 * M_PI * max_r / double(p.N);
    const double Jm_beta = bessel::cyl_j(p.m_ang, p.beta);
    const double nan = std::numeric_limits<double>::quiet_NaN();

    auto open_csv = [](const std::string& filename) {
        std::ofstream out;
        out.exceptions(std::ios::failbit | std::ios::badbit);
        out.open(filename);
        out.precision(std::numeric_limits<double>::max_digits10);
        return out;
    };
    using Grid = std::vector<std::vector<double>>;
    auto save_grid = [&](const std::string& filename, const Grid& grid) {
        auto out = open_csv(filename);
        for (const auto& row : grid) {
            for (std::size_t i = 0; i < row.size(); ++i) {
                if (i) out << ',';
                out << row[i];
            }
            out << '\n';
        }
    };

    auto parameters = open_csv(prefix + "_parameters.csv");
    parameters << "window_radius,patch_radius,colbrook,nm,n_rad,m_ang,modes,points_per_disk,grid_points,"
                  "max_radius,omega0,k_real,k_imag,beta,Anorm,fringe,Nmultipole,template_count,"
                  "format_version,a1_x,a1_y,up_a2_x,up_a2_y,down_a2_x,down_a2_y\n"
               << p.window_radius << ',' << p.patch_radius << ',' << p.colbrook << ','
               << p.nm << ',' << p.nm / 10 << ',' << p.m_ang << ',' << p.modes << ',' << p.N << ',' << Ng << ','
               << max_r << ',' << p.omega0 << ',' << p.k.real() << ','
               << p.k.imag() << ',' << p.beta << ',' << p.Anorm << ',' << p.fringe << ','
               << p.Nmultipole << ',' << template_count << ",2,"
               << p.lattice_up.a1.x() << ',' << p.lattice_up.a1.y() << ','
               << p.lattice_up.a2.x() << ',' << p.lattice_up.a2.y() << ','
               << p.lattice_down.a2.x() << ',' << p.lattice_down.a2.y() << '\n';


    auto windows = open_csv(prefix + "_windows.csv");
    windows << "window_index,source_y,source_count,basis_count\n";

    // Save each central angular basis separately. No global eigenvector is applied here.
    const int mesh_radius = p.window_radius + p.fringe;
    for (int k = 0; k < template_count; ++k) {
        const auto& mesh = p.meshes[k];
        const auto& disks = p.disks[k];
        const int source_y = mesh_radius - k;
        auto& disk_indices = p.disk_indices[k];
        auto& row_offsets = p.row_offsets[k];

        // find the active disk indices
        std::vector<int> active_indices;
        for (auto iter = disk_indices.begin(); iter != disk_indices.end(); ++iter) {
            if (iter -> first == std::make_pair(0, 0)) {
                active_indices.push_back(iter -> second);
            }
        }
        const int basis_count = p.modes * static_cast<int>(active_indices.size());
        if (basis_count == 0 || p.patch_X[k].cols() != basis_count)
            throw std::invalid_argument("Interface field density columns do not match central disk/mode layout");
        windows << k << ',' << source_y << ',' << active_indices.size() << ',' << basis_count << '\n';

        const int Ntot = mesh.get_num_segments();
        const std::string patch_prefix = prefix + "_patch_" + std::to_string(k);
        auto sources = open_csv(patch_prefix + "_sources.csv");
        sources << "disk,mesh_index,basis_start,radius,center_x,center_y,source_speed\n";
        for (std::size_t a = 0; a < active_indices.size(); ++a) {
            const auto& disk = disks[active_indices[a]];
            sources << a << ',' << active_indices[a] << ',' << a * p.modes << ','
                    << disk.r << ',' << disk.c.x() << ',' << disk.c.y() << ',' << disk.v_b << '\n';
        }
        auto rows = open_csv(patch_prefix + "_rows.csv");
        rows << "local_x,local_y,disk_index\n";
        for (std::size_t q = 0; q < disk_indices.size(); ++q) {
            rows << row_offsets[q].local_x << ',' << row_offsets[q].local_y << ',' << row_offsets[q].disk_number << '\n';
        }
        auto geometry = open_csv(patch_prefix + "_disks.csv");
        geometry << "radius,center_x,center_y,v_b\n";
        for (std::size_t i = 0; i < disks.size(); ++i) {
            geometry << disks[i].r << ',' << disks[i].c.x() << ',' << disks[i].c.y()
                     << ',' << disks[i].v_b << '\n';
        }

        // Physical coordinates: skew lattices need more than [-R,R], and R=0
        // must still give a nondegenerate grid containing every central disk.
        double extent = 0.5 * p.lattice_up.a1.norm();
        for (const auto& disk : disks)
            extent = std::max(extent, disk.c.cwiseAbs().maxCoeff() + disk.r + h);
        std::vector<double> axis(Ng);
        for (int i = 0; i < Ng; ++i) axis[i] = -extent + 2.0 * extent * double(i) / double(Ng - 1);
        save_grid(patch_prefix + "_axis.csv", {axis});

        MatrixXcd B = SpectralOperators::fourier_basis(p.N, p.Nmultipole, p.disks[k].size());
        MatrixXcd weighted_X = B * p.patch_X[k];
        for (int j = 0; j < Ntot; ++j) weighted_X.row(j) *= mesh.get_vertex(j).sigma;

        auto classify = [&](const Vector2d& r) {
            Location loc{Location::Exterior, -1, 0.0, 0.0};
            for (std::size_t i = 0; i < disks.size(); ++i) {
                const Vector2d d = r - disks[i].c;
                const double rho = d.norm();
                if (rho < disks[i].r) {
                    bool is_resonant = false;
                    for (std::size_t j = 0; j < active_indices.size(); ++j) {
                        if (static_cast<int>(i) == active_indices[j]) {
                            is_resonant = true;
                            loc.slot = j;
                            break;
                        }
                    }
                    loc.kind = is_resonant ? Location::ResonantDisk : Location::DarkDisk;
                    loc.rho = rho;
                    loc.phi = std::atan2(d.y(), d.x());
                    return loc;
                }
                // Only exterior evaluation needs the near-boundary quadrature mask.
                if (rho - disks[i].r < h) loc.kind = Location::NearBoundary;
            }
            return loc;
        };

        std::vector<Grid> re(basis_count, Grid(Ng, std::vector<double>(Ng))),
                          im = re, mag = re, phase = re;
        for (int iy = 0; iy < Ng; ++iy) {
#pragma omp parallel for schedule(dynamic)
            for (int ix = 0; ix < Ng; ++ix) {
                const Vector2d r(axis[ix], axis[iy]);
                const Location loc = classify(r);
                std::vector<cpxd> u(basis_count);
                if (loc.kind == Location::ResonantDisk) {
                    // Evaluate the resonant disk field.
                    const auto& disk = disks[active_indices[loc.slot]];
                    const double norm = 1.0 / (std::sqrt(M_PI) * disk.r *
                        std::sqrt(std::max(1.0 - double(p.m_ang * p.m_ang) / (p.beta * p.beta), 1e-12)));
                    const double radial = bessel::cyl_j(p.m_ang, p.beta * loc.rho / disk.r)
                                          / Jm_beta;
                    for (int s = 0; s < p.modes; ++s) {
                        const double sign = (s == 0) ? 1.0 : -1.0;
                        u[p.modes * loc.slot + s] = norm * radial * std::exp(cpxd(0, sign * p.m_ang * loc.phi));
                    }
                } else if (loc.kind == Location::Exterior) {
                    for (int j = 0; j < Ntot; ++j) {
                        const cpxd kernel = Kernels::helmholtz_2D(p.k, r, mesh.get_vertex(j).point);
                        for (int s = 0; s < basis_count; ++s) u[s] += weighted_X(j, s) * kernel;
                    }
                }  // All other disk interiors have zero imposed data for this source.
                for (int s = 0; s < basis_count; ++s) {
                    if (loc.kind == Location::NearBoundary) {
                        re[s][iy][ix] = im[s][iy][ix] = mag[s][iy][ix] = phase[s][iy][ix] = nan;
                    } else {
                        re[s][iy][ix] = u[s].real();
                        im[s][iy][ix] = u[s].imag();
                        mag[s][iy][ix] = std::abs(u[s]);
                        phase[s][iy][ix] = std::arg(u[s]);
                    }
                }
            }
            Tools::print_stage_progress("Evaluating window " + std::to_string(k), iy + 1, Ng);
        }
        Tools::finish_progress_line();
        for (int s = 0; s < basis_count; ++s) {
            const std::string basis_prefix = patch_prefix + "_basis_" + std::to_string(s);
            save_grid(basis_prefix + "_real.csv", re[s]);
            save_grid(basis_prefix + "_imag.csv", im[s]);
            save_grid(basis_prefix + "_mag.csv", mag[s]);
            save_grid(basis_prefix + "_phase.csv", phase[s]);
        }
        std::cout << "[wrote] " << patch_prefix << "_{axis,disks}.csv and "
                  << basis_count << " central-source basis fields\n";
    }
}

}  // namespace workflows
