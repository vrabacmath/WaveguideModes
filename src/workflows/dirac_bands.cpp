#include "workflows/dirac_bands.h"

namespace workflows {

    DiracModeFamily build_dirac_mode_family(const Crystal& crystal, int reference_disk, int nm) {
        const auto mode = decode_nm(nm);
        const double beta = neumann_zero(nm);
        const auto& disks = crystal.get_disks();
        if (reference_disk < 0 || reference_disk >= static_cast<int>(disks.size()))
            throw std::invalid_argument("Capacitance reference disk is outside the crystal cell");
        const auto& reference = disks[reference_disk];
        const double ratio = reference.v_b / reference.r;
        DiracModeFamily family;
        family.nm = nm;
        family.angular_modes = mode.m_ang == 0 ? 1 : 2;
        family.omega0 = ratio * beta;
        for (std::size_t i = 0; i < disks.size(); ++i) {
            const double candidate = disks[i].v_b / disks[i].r;
            const double tolerance = 64 * std::numeric_limits<double>::epsilon() * std::max(ratio, candidate);
            if (std::abs(candidate - ratio) <= tolerance) family.disk_indices.push_back(i);
        }
        const auto mesh = crystal.get_mesh();
        const int columns = family.angular_modes * family.disk_indices.size();
        family.G = MatrixXcd::Zero(mesh.get_num_segments(), columns);
        family.source_speeds.resize(columns);
        for (std::size_t a = 0; a < family.disk_indices.size(); ++a) {
            const int index = family.disk_indices[a];
            const auto& disk = disks[index];
            const int start = mesh.get_start_index(index);
            const int N = mesh.get_end_index(index) - start + 1;
            if (N <= 2 * mode.m_ang)
                throw std::invalid_argument("Too few boundary points to resolve this angular mode");
            const double norm = 1.0 / (std::sqrt(M_PI) * disk.r *
                std::sqrt(1.0 - double(mode.m_ang * mode.m_ang) / (beta * beta)));
            for (int s = 0; s < family.angular_modes; ++s) {
                const int column = a * family.angular_modes + s;
                family.source_speeds(column) = disk.v_b;
                for (int j = 0; j < N; ++j) {
                    const double theta = 2 * M_PI * j / N;
                    family.G(start + j, column) = norm * std::exp(cpxd(0, (s == 0 ? 1 : -1) * mode.m_ang * theta));
                }
            }
        }
        return family;
    }

    MatrixXcd dirac_capacitance_matrix(const Crystal& crystal, const DiracModeFamily& family, Vector2d alpha) {
        const auto mesh = crystal.get_mesh();
        SpectralOperators ops(mesh);
        const auto [a1, a2] = crystal.get_lattice_vectors();
        const cpxd k(family.omega0 / crystal.get_background_speed(), 0);
        MatrixXcd S, Kstar;
        ops.CrystalS(S, k, alpha, a1, a2);
        ops.CrystalKstar(Kstar, k, alpha, a1, a2);
        const MatrixXcd X = S.partialPivLu().solve(family.G);
        if (!X.allFinite() || (S * X - family.G).norm() > 1e-8 * family.G.norm())
            throw std::runtime_error("Capacitance single-layer solve failed");
        MatrixXcd weighted_flux = 0.5 * X + Kstar * X;
        for (int i = 0; i < mesh.get_num_segments(); ++i)
            weighted_flux.row(i) *= mesh.get_vertex(i).sigma;
        // C=-V H V/(2 omega0), the same speed-scaled coordinates as interface-modes.
        // It is similar to the row-weighted form -V^2 H/(2 omega0).
        return -1.0 / (2 * family.omega0) * family.source_speeds.asDiagonal() *
               (family.G.adjoint() * weighted_flux) * family.source_speeds.asDiagonal();
    }

    void capacitance_dirac_bands(const Crystal& crystal, int reference_disk, int max_nm, int Npath) {
        const auto limit = decode_nm(max_nm);
        std::vector<DiracModeFamily> families;
        for (int m_ang = 0; m_ang <= limit.m_ang; ++m_ang) {
            families.push_back(build_dirac_mode_family(crystal, reference_disk, 10 * limit.n_rad + m_ang));
            const auto& family = families.back();
            std::cout << "   nm=" << family.nm << ": omega_0=" << family.omega0
                      << ", resonant disks=" << family.disk_indices.size()
                      << ", multiplicity=" << family.G.cols() << '\n';
        }
        std::ofstream sources("dirac_capacitance_sources.csv");
        sources.precision(std::numeric_limits<double>::max_digits10);
        sources << "nm,column,disk,basis,radius,source_speed,omega0\n";
        for (const auto& family : families)
            for (std::size_t a = 0; a < family.disk_indices.size(); ++a) {
                const int index = family.disk_indices[a];
                const auto& disk = crystal.get_disks()[index];
                for (int s = 0; s < family.angular_modes; ++s)
                    sources << family.nm << ',' << a * family.angular_modes + s << ',' << index << ','
                            << s << ',' << disk.r << ',' << disk.v_b << ',' << family.omega0 << '\n';
            }

        std::ofstream out_lo("dirac_capacitance_bands_leading.csv");
        std::ofstream out_cap("dirac_capacitance_matrix_per_alpha.csv");
        out_cap.setf(std::ios::scientific);
        out_cap.precision(std::numeric_limits<double>::max_digits10);
        out_lo.setf(std::ios::scientific);
        out_lo.precision(std::numeric_limits<double>::max_digits10);

        std::vector<KPoint> path = crystal.make_m_gamma_k_m_path(Npath);
        for (std::size_t a = 0; a < path.size(); ++a) {
            const KPoint& kp = path[a];
            for (std::size_t fi = 0; fi < families.size(); ++fi) {
                const DiracModeFamily& fam = families[fi];
                const double omega0 = fam.omega0;
                const int modes = fam.G.cols();
                const MatrixXcd Creg = dirac_capacitance_matrix(crystal, fam, Vector2d(kp.alpha_x, kp.alpha_y));

                // Relative speed scaling preserves the Hermitian form for real Bloch problems.
                const double herm = (Creg - Creg.adjoint()).norm();

                ComplexEigenSolver<MatrixXcd> es(Creg);
                if (es.info() != Success || !es.eigenvalues().allFinite())
                    throw std::runtime_error("Capacitance eigenvalue solve failed");
                VectorXcd lam = es.eigenvalues();
                std::sort(lam.data(), lam.data() + lam.size(),
                        [](const cpxd& x, const cpxd& y) { return x.real() < y.real(); });

                for (int j = 0; j < modes; ++j) {
                    const cpxd omega = omega0 + crystal.get_delta() * lam(j);
                         //- crystal.get_delta() * crystal.get_delta() * lam(j) * lam(j) * 0.5 / omega0;  // leading order: Thm 4.3, eq (4.12)
                    out_lo << fam.nm << "," << kp.s << "," << omega.real() << "," << omega.imag()
                        << "," << lam(j).real() << "," << lam(j).imag() << "," << herm << "\n";
                    cout << "    nm=" << fam.nm << " leading-order seed: omega=" << omega
                            << " lambda=" << lam(j) << " herm=" << herm << "\n";
                }

                // header
                out_cap << "alpha,nm";
                for (int i = 0; i < modes; ++i) {
                    for (int j = 0; j < modes; ++j) {
                        out_cap << ",C_" << i << "_" << j << "_re";
                        out_cap << ",C_" << i << "_" << j << "_im";
                    }
                }
                out_cap << "\n";

                // inside alpha loop
                out_cap << kp.s << "," << fam.nm;
                for (int i = 0; i < modes; ++i) {
                    for (int j = 0; j < modes; ++j) {
                        auto z = Creg(i, j);
                        out_cap << "," << z.real() << "," << z.imag();
                    }
                }
                out_cap << "\n";
            }
        }
        out_lo.close();
        out_cap.close();
        std::cout << "[wrote] dirac_capacitance_bands_leading.csv (leading-order seeds), "
                    "dirac_capacitance_matrix_per_alpha.csv (C^reg per alpha)\n";
    }

} // namespace dirac_bands