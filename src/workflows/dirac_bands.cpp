#include "workflows/dirac_bands.h"

namespace workflows {

    void capacitance_dirac_bands(Crystal& crystal, int r_idx, int max_m, int Npath) {
        struct ModeFamily {
            int m;          // angular order
            double omega0;  // interior Neumann base frequency of the defect disk
            int modes;      // multiplicity (size of C^reg)
            MatrixXcd G;    // eigenmode traces on the supercell boundary
        };

        cout << "Computing Dirac capacitance bands using the Neumann resonance of disk " << r_idx << " (R=" << crystal.Rs.at(r_idx) << ")...\n";
        BoundaryMesh mesh = crystal.get_mesh();
        const int Ntot = mesh.get_num_segments();
        const int Nmesh = mesh.get_num_meshes();
        std::vector<ModeFamily> families;

        SpectralOperators ops(mesh);
        VectorXd sigma(Ntot);
        for (int i = 0; i < Ntot; ++i) sigma(i) = mesh.get_vertex(i).sigma;

        // cout << "num = " << mesh.get_num_meshes() << ", Ntot = " << Ntot << ", sigma = " << sigma.transpose() << "\n";

        for (int m_ang = 0; m_ang <= max_m; ++m_ang) {
            const double beta = first_neumann_zero(m_ang);
            ModeFamily fam{m_ang, crystal.v_b * beta / crystal.Rs.at(r_idx), Nmesh * ((m_ang == 0) ? 1 : 2),
                        MatrixXcd::Zero(Ntot, Nmesh * ((m_ang == 0) ? 1 : 2))};
            const double Anorm =
                1.0 / (std::sqrt(M_PI) * crystal.Rs.at(0)
                    * std::sqrt(std::max(1.0 - double(m_ang * m_ang) / (beta * beta), 1e-12)));

            for (int mesh_idx = 0; mesh_idx < Nmesh; ++mesh_idx) {
                const double R = crystal.Rs.at(mesh_idx);
                // std::cout << "Disk " << mesh_idx << ": R=" << R << "\n";
                cout << abs(R - crystal.Rs.at(r_idx)) << "\n";
                if (abs(R - crystal.Rs.at(r_idx)) > 1e-6) continue;

                int N = mesh.get_end_index(mesh_idx) - mesh.get_start_index(mesh_idx) + 1;
                const int s0 = mesh.get_start_index(mesh_idx);
                for (int loc = 0; loc < N; ++loc)
                    for (int sgn = 0; sgn < ((m_ang == 0) ? 1 : 2); ++sgn)
                        fam.G(s0 + loc, sgn + mesh_idx * ((m_ang == 0) ? 1 : 2)) = Anorm * std::exp(cpxd(0, 1.0) * ((sgn == 0) ? 1.0 : -1.0)
                                                                * double(m_ang) * 2. * M_PI / double(N) * double(loc));
            }
            families.push_back(std::move(fam));
            std::cout << "   m=" << m_ang << ": omega_0=" << families.back().omega0
                    << " (= v_bd * " << beta << "/R_def), multiplicity " << families.back().modes << "\n";
        }

        MatrixXcd S, Kstar;  // workspace shared by all assemblies

        std::ofstream out_lo("dirac_capacitance_bands_leading.csv");
        std::ofstream out_cap("dirac_capacitance_matrix_per_alpha.csv");
        out_cap.setf(std::ios::scientific);
        out_cap.precision(10);
        out_lo.setf(std::ios::scientific);
        out_lo.precision(10);

        std::vector<KPoint> path = crystal.make_m_gamma_k_m_path(Npath);
        for (int a = 0; a < 3*(Npath-1)+1; ++a) {
            const KPoint& kp = path[a];
            for (std::size_t fi = 0; fi < families.size(); ++fi) {
                const ModeFamily& fam = families[fi];
                const double omega0 = fam.omega0;
                const int modes = fam.modes;
                const cpxd k = cpxd(omega0, 1e-3) / crystal.v;  // small Im(k) regularises the periodic G^alpha

                // Frequency-dependent capacitance matrix (Fabry-Perot, arXiv:2605.27572 eq. 4.11):
                //   Lambda_ext = (1/2 I + K*) S^{-1},   k = omega_0 / v
                //   C^reg_{pq} = -(v_b^2 / 2 omega_0) <Lambda_ext[g_q], g_p>_{dD}
                // The boundary inner product <.,.>_{dD} using quadrature, weighted sum (diag(sigma)).
                ops.CrystalS(S, k, Vector2d(kp.alpha_x, kp.alpha_y), crystal.get_lattice_vectors().first, crystal.get_lattice_vectors().second);
                ops.CrystalKstar(Kstar, k, Vector2d(kp.alpha_x, kp.alpha_y), crystal.get_lattice_vectors().first, crystal.get_lattice_vectors().second);
                const MatrixXcd X = S.partialPivLu().solve(fam.G);
                const MatrixXcd LamG = 0.5 * X + Kstar * X;   // Lambda_ext G = (1/2 + K*) S^{-1} G
                const MatrixXcd Creg =
                    -(crystal.v_b * crystal.v_b) / (2.0 * omega0) * (fam.G.adjoint() * (sigma.asDiagonal() * LamG));

                // complex symmetry - should be symmetric
                const double herm = (Creg - Creg.adjoint()).norm();

                ComplexEigenSolver<MatrixXcd> es(Creg);
                VectorXcd lam = es.eigenvalues();
                std::sort(lam.data(), lam.data() + lam.size(),
                        [](const cpxd& x, const cpxd& y) { return x.real() < y.real(); });

                for (int j = 0; j < modes; ++j) {
                    const cpxd omega = omega0 + crystal.get_delta() * lam(j);  // leading order: Thm 4.3, eq (4.12)
                    out_lo << fam.m << "," << kp.s << "," << omega.real() << "," << omega.imag()
                        << "," << lam(j).real() << "," << lam(j).imag() << "," << herm << "\n";
                    cout << "    m=" << fam.m << " leading-order seed: omega=" << omega
                            << " lambda=" << lam(j) << " herm=" << herm << "\n";
                }

                // header
                out_cap << "alpha,fi";
                for (int i = 0; i < modes; ++i) {
                    for (int j = 0; j < modes; ++j) {
                        out_cap << ",C_" << i << "_" << j << "_re";
                        out_cap << ",C_" << i << "_" << j << "_im";
                    }
                }
                out_cap << "\n";

                // inside alpha loop
                out_cap << kp.s << "," << fi;
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