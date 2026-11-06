#include "workflows/interface_patch.h"
#include "multipole.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <iostream>
#include <stdexcept>


namespace workflows {

using namespace Eigen;

Vector2d interface_cell_origin(const InterfacePatch& ip, int x, int y) {
    if (y >= 0)
        return x * ip.lattice_up.a1 + y * ip.lattice_up.a2;
    return x * ip.lattice_down.a1 + (-y - 1) * ip.lattice_down.a2;
}

InterfaceLayout make_interface_layout(const InterfacePatch& ip) {
    if (ip.patch_radius < 0 || ip.colbrook < 0 || ip.modes < 1 ||
        ip.disks_up.empty() || ip.disks_down.empty())
        throw std::invalid_argument("Interface layout requires nonnegative radii and nonempty cells");
    const auto width = [&](int y) -> Eigen::Index {
        return ip.modes * static_cast<Eigen::Index>(
            y < 0 ? ip.disks_down.size() : ip.disks_up.size());
    };
    InterfaceLayout layout;
    const auto fill = [&](int radius, auto& starts, Eigen::Index& next) {
        for (int x = -radius; x <= radius; ++x) {
            for (int y = -radius; y <= radius; ++y) {
                starts.emplace(InterfaceLayout::Cell{x, y}, next);
                next += width(y);
            }
        }
    };
    fill(ip.patch_radius, layout.column_start, layout.cols);
    fill(ip.patch_radius + ip.colbrook, layout.row_start, layout.rows);
    layout.source_rows.resize(layout.cols);
    for (const auto& [cell, c0] : layout.column_start) {
        const Eigen::Index r0 = layout.row_start.at(cell);
        for (Eigen::Index j = 0; j < width(cell.second); ++j)
            layout.source_rows[c0 + j] = r0 + j;
    }
    return layout;
}

MatrixXcd assemble_interface_matrix(const InterfacePatch& ip) {
    const auto layout = make_interface_layout(ip);
    const int R = ip.window_radius + ip.fringe;
    if (ip.window_radius < 0 || ip.fringe < 0 ||
        ip.patch_colC.size() != static_cast<std::size_t>(2 * R + 2) ||
        ip.row_offsets.size() != ip.patch_colC.size())
        throw std::invalid_argument("Interface templates do not match the mesh radius");
    MatrixXcd bigC = MatrixXcd::Zero(layout.rows, layout.cols);
    for (const auto& [cell, col_start] : layout.column_start) {
        const auto [cell_x, cell_y] = cell;
        const int t = std::clamp(R - cell_y, 0, 2 * R + 1);
        const auto& localC = ip.patch_colC[t];
        const auto& targets = ip.row_offsets[t];
        const Eigen::Index width = ip.modes * static_cast<Eigen::Index>(
            cell_y < 0 ? ip.disks_down.size() : ip.disks_up.size());
        if (localC.rows() != ip.modes * static_cast<Eigen::Index>(targets.size()) ||
            localC.cols() != width || !localC.allFinite())
            throw std::invalid_argument("Invalid interface block column for template " + std::to_string(t));
        for (std::size_t q = 0; q < targets.size(); ++q) {
            const auto& target = targets[q];
            if (std::abs(target.local_x) > ip.window_radius ||
                std::abs(target.local_y) > ip.window_radius)
                continue;
            const int target_y = cell_y + target.local_y;
            const auto it = layout.row_start.find({cell_x + target.local_x, target_y});
            if (it == layout.row_start.end()) continue;
            const int count = static_cast<int>(
                target_y < 0 ? ip.disks_down.size() : ip.disks_up.size());
            if (target.disk_number < 0 || target.disk_number >= count)
                throw std::invalid_argument("Interface target disk does not belong to its global cell");
            const Eigen::Index row_start = it->second + ip.modes * target.disk_number;
            bigC.block(row_start, col_start, ip.modes, width) =
                localC.block(static_cast<Eigen::Index>(q) * ip.modes, 0, ip.modes, width);
        }
    }
    return bigC;
}

InterfacePatch build_interface_patch(int window_radius, int patch_radius, int colbrook, 
                                        LatticeVectors up, LatticeVectors down, int nm,
                                        vector<Disk> disks_up, vector<Disk> disks_down,
                                        int fringe, int points_per_disk, double omega0, 
                                        double v, int Nmultipole, bool verbose) {
    const int m_ang = decode_nm(nm).m_ang;
    if (window_radius < 0 || patch_radius < window_radius || colbrook < 0 || fringe < 0)
        throw std::invalid_argument("Require 0 <= window_radius <= patch_radius and nonnegative buffers");
    if (!std::isfinite(v) || v <= 0 || !std::isfinite(omega0) || omega0 <= 0 ||
        disks_up.empty() || disks_down.empty())
        throw std::invalid_argument("Require positive finite frequency/speed, a valid nm mode and nonempty cells");
    const auto cross = [](const Vector2d& a, const Vector2d& b) {
        return a.x() * b.y() - a.y() * b.x();
    };
    if (!up.a1.allFinite() || !up.a2.allFinite() || !down.a1.allFinite() || !down.a2.allFinite() ||
        !up.a1.isApprox(down.a1, 1e-12) || cross(up.a1, up.a2) <= 0 || cross(down.a1, down.a2) >= 0)
        throw std::invalid_argument("Lattices must share a1, with up.a2 and down.a2 pointing into opposite half-planes");
    for (const auto* cell : {&disks_up, &disks_down}) {
        for (const auto& d : *cell) {
            if (!d.c.allFinite() || !std::isfinite(d.r) || d.r <= 0 || !std::isfinite(d.v_b) || d.v_b <= 0)
                throw std::invalid_argument("Disk centers must be finite and disk radii/speeds positive and finite");
        }
    }
 
    InterfacePatch ip;
    ip.kV = v;
    ip.window_radius = window_radius;
    ip.patch_radius = patch_radius;
    ip.colbrook = colbrook;
    ip.fringe = fringe;
    ip.N = std::max(points_per_disk, 8);
    ip.nm = nm;
    ip.m_ang = m_ang;
    ip.modes = (m_ang == 0) ? 1 : 2;
    ip.disks_up = disks_up;
    ip.disks_down = disks_down;
    ip.lattice_up = up;
    ip.lattice_down = down;
    ip.beta = neumann_zero(nm);
    ip.omega0 = omega0;
    ip.k = cpxd(ip.omega0, 0) / ip.kV;

    ip.Nmultipole = Nmultipole;
    if (!(ip.N > 2 * ip.Nmultipole + 1 && ip.Nmultipole >= ip.m_ang)) {
        throw std::invalid_argument("The following must hold: N > 2*Nmultipole+1 and Nmultipole >= m_ang");
    }

    const int mesh_radius = window_radius + fringe;
    const int template_count = 2 * mesh_radius + 2;
    ip.meshes.resize(template_count);
    ip.disks.resize(template_count);
    ip.disk_indices.resize(template_count);
    ip.row_offsets.resize(template_count);
    ip.patch_colC.resize(template_count);
    ip.patch_X.resize(template_count);
    ip.solve_diagnostics.resize(template_count);

    // moving window of size (2*mesh_radius+1)^2 around each source site, with the source at the center
    // we are moving from up to down, crossing an interface line.
    // The source is always at (0,0) in local coordinates, and the mesh is translated to each source site.
    for (int t = 0; t < template_count; ++t) {
        const auto started = std::chrono::steady_clock::now();
        const int source_y = mesh_radius - t;
        const Vector2d source_origin = interface_cell_origin(ip, 0, source_y);
        const int mesh = t;
        auto& patch_mesh = ip.meshes[mesh];
        auto& disks = ip.disks[mesh];
        auto& disk_indices = ip.disk_indices[mesh];
        auto& row_offsets = ip.row_offsets[mesh];

        // Translate a square window to each source site; store coordinates relative to it.
        // Keep all disks in the square (including fringe) in the exterior solve.
        for (int x = -mesh_radius; x <= mesh_radius; ++x) {
            for (int y = -mesh_radius; y <= mesh_radius; ++y) {
                // The source is at the center of the window
                const int global_y = source_y + y;
                const Vector2d shift = interface_cell_origin(ip, x, global_y) - source_origin;
                const auto& cell = global_y < 0 ? ip.disks_down : ip.disks_up;
                for (std::size_t a = 0; a < cell.size(); ++a) {
                    const auto& d = cell[a];
                    disks.push_back(Disk{d.r, d.c + shift, d.v_b});
                    BoundaryMesh disk(ip.N);
                    disk.generate_circle(disks.back().r, disks.back().c);
                    patch_mesh.add_mesh(disk);
                    disk_indices.emplace(std::make_pair(x, y), static_cast<int>(disks.size()) - 1);
                    row_offsets.push_back({x, y, static_cast<int>(a)});
                }
            }
        }
        // in this construction x = 0 and y = 0 is always the source

        const int window_size = static_cast<int>(disks.size());
        for (int i = 0; i < window_size; ++i) {
            for (int j = 0; j < i; ++j) {
                if ((disks[i].c - disks[j].c).norm() <= disks[i].r + disks[j].r)
                    throw std::invalid_argument("Interface template " + std::to_string(t) +
                        " has touching/overlapping disks " + std::to_string(j) + " and " + std::to_string(i));
            }
        }

        // compute the interior Neumann traces for this template, one column-block per defect resonator and mode
        const int Ntot = patch_mesh.get_num_segments();
        VectorXd sigma(Ntot);
        for (int i = 0; i < Ntot; ++i) sigma(i) = patch_mesh.get_vertex(i).sigma;
        MatrixXcd G = MatrixXcd::Zero(Ntot, ip.modes * window_size);
        for (int q = 0; q < window_size; ++q) {
            const int s0 = patch_mesh.get_start_index(q);

            ip.Anorm = 1.0 / (std::sqrt(M_PI) * disks[q].r *
                     std::sqrt(std::max(1.0 - double(m_ang * m_ang) / (ip.beta * ip.beta), 1e-12)));

            for (int loc = 0; loc < ip.N; ++loc) {
                const double theta = 2.0 * M_PI * double(loc) / double(ip.N);
                for (int sgn = 0; sgn < ip.modes; ++sgn) {
                    const double sign = (sgn == 0) ? 1.0 : -1.0;
                    G(s0 + loc, ip.modes * q + sgn) =
                        ip.Anorm * std::exp(cpxd(0, sign * double(m_ang) * theta));
                }
            }
        }
        // Excite each disk/mode in the central cell independently.
        const MatrixXcd B = SpectralOperators::fourier_basis(
                                        ip.N, ip.Nmultipole, disks.size());

        vector<MatrixXcd> G0vecs;
        vector<double> v0list;
        for (auto iter = disk_indices.begin(); iter != disk_indices.end(); ++iter) {
            if (iter -> first == std::make_pair(0, 0)) {
                G0vecs.push_back(G.middleCols(ip.modes * iter -> second, ip.modes));
                v0list.push_back(disks[iter -> second].v_b);
            }
        }

        MatrixXcd G0 = MatrixXcd::Zero(Ntot, ip.modes * G0vecs.size());
        MatrixXd V0 = MatrixXd::Zero(ip.modes * G0vecs.size(), ip.modes * G0vecs.size());
        for (std::size_t i = 0; i < G0vecs.size(); ++i) {
            G0.middleCols(ip.modes * i, ip.modes) = G0vecs[i];
            V0.block(i * ip.modes, i * ip.modes, ip.modes, ip.modes) = 
                v0list[i] * MatrixXd::Identity(ip.modes, ip.modes);
        }

        const MatrixXcd g0 = B.adjoint() * G0 / ip.N;
        MatrixXcd S, Kstar;
        Multipole::S(S, ip.Nmultipole, ip.disks[mesh], ip.k);
        Multipole::Kstar(Kstar, ip.Nmultipole, ip.disks[mesh], ip.k);
        if (!S.allFinite() || !Kstar.allFinite())
            throw std::runtime_error("Non-finite multipole operator in interface template " + std::to_string(t));
        auto temp = S.partialPivLu();
        const MatrixXcd X = temp.solve(g0);
        const double residual_norm = (S * X - g0).norm();
        const double relative_residual = residual_norm / g0.norm();
        const double rcond = temp.rcond();
        if (!X.allFinite() || !std::isfinite(relative_residual) || relative_residual > 1e-8)
            throw std::runtime_error("Interface template " + std::to_string(t) +
                " solve failed: relative residual=" + std::to_string(relative_residual));
        ip.solve_diagnostics[mesh] = {rcond, relative_residual,
            residual_norm / (S.norm() * X.norm() + g0.norm()), X.norm()};
        ip.patch_X[mesh] = X;
        const MatrixXcd LamG = B * (0.5 * X + Kstar * X);

        // speed matrix
        MatrixXcd Vb = MatrixXcd::Zero(disks.size() * ip.modes, disks.size() * ip.modes);
        for (std::size_t i = 0; i < disks.size(); ++i) {
            Vb.block(i * ip.modes, i * ip.modes, ip.modes, ip.modes) = 
                disks[i].v_b * MatrixXcd::Identity(ip.modes, ip.modes);
        }

        // Speed-scaled target/source coordinates for the projected capacitance.
        ip.patch_colC[mesh] = -Vb / (2.0 * ip.omega0) *
                              (G.adjoint() * (sigma.asDiagonal() * LamG)) * V0;
        if (!ip.patch_colC[mesh].allFinite())
            throw std::runtime_error("Non-finite capacitance column in interface template " + std::to_string(t));
        if (verbose)
            std::cout << "Interface template " << mesh << ": source y=" << source_y
                      << ", " << patch_mesh.get_num_meshes() << " disks, " << Ntot << " boundary points, "
                      << S.rows() << " Fourier DOFs, " << g0.cols() << " source modes, residual="
                      << relative_residual << ", "
                      << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()
                      << " s" << std::endl;
    }

    ip.bigC = assemble_interface_matrix(ip);
    return ip;
}

}  // namespace workflows
