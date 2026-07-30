#include "workflows/bent_waveguide_exact.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <set>
#include <vector>

#include "Eigen/Dense"
#include "kernels.h"
#include "tools.h"
#include "utils.h"
#include "workflows/bent_waveguide_patch.h"

namespace workflows {

using namespace Eigen;

namespace {

// A(omega) on the whole cluster, same block layout as SpectralOperators::A. Assembled here
// rather than through ops.A so that each S/K* pair is built once per DISTINCT wavenumber
// (all three coincide in the single-material case) instead of once per block.
//
// With a defect material the interior wavenumber differs per disk: rows of the interior
// blocks belonging to defect disks (defect_row) use k_bd = omega/kVbd, cladding rows use
// k_b = omega/kVb. This is exact, not a splice of two problems: the interior single layer only
// has to represent the field inside its own disk (interior fields are local), and the trace /
// jump relations of S^k hold row-wise on each boundary, so evaluating the interior rows of a
// defect boundary at k_bd is precisely the transmission condition for that disk.
MatrixXcd assemble_A(SpectralOperators& ops, int Ntot, cpxd omega, cpxd kV, cpxd kVb, cpxd kVbd,
                     const std::vector<bool>& defect_row, double delta) {
    const cpxd k = omega / kV, kb = omega / kVb, kbd = omega / kVbd;
    MatrixXcd Sb, Kb;
    ops.S(Sb, kb);
    ops.Kstar(Kb, kb);
    MatrixXcd S = Sb, Kst = Kb;
    if (k != kb) {
        ops.S(S, k);
        ops.Kstar(Kst, k);
    }
    if (kbd != kb) {
        MatrixXcd Sbd, Kbd;
        if (kbd == k) {
            Sbd = S;
            Kbd = Kst;
        } else {
            ops.S(Sbd, kbd);
            ops.Kstar(Kbd, kbd);
        }
        for (int i = 0; i < Ntot; ++i)
            if (defect_row[i]) {
                Sb.row(i) = Sbd.row(i);
                Kb.row(i) = Kbd.row(i);
            }
    }
    MatrixXcd A = MatrixXcd::Zero(2 * Ntot, 2 * Ntot);
    const MatrixXcd I = MatrixXcd::Identity(Ntot, Ntot);
    A.block(0, 0, Ntot, Ntot) = Sb;
    A.block(0, Ntot, Ntot, Ntot) = -S;
    A.block(Ntot, 0, Ntot, Ntot) = -0.5 * I + Kb;
    A.block(Ntot, Ntot, Ntot, Ntot) = -delta * (0.5 * I + Kst);
    return A;
}

struct MullerResult {
    cpxd omega;
    int iters = 0;
    bool converged = false;
};

// Muller's method on mu(omega) = the eigenvalue of A(omega) nearest zero, estimated by inverse
// iteration plus a Rayleigh quotient on the same LU. mu is locally analytic and vanishes exactly
// at the resonances, so it can drive Muller -- unlike sigma_min, which is real-valued and
// non-analytic. It is also strictly better than the resolvent probe 1/(w^H A^{-1} r) used first:
// that scalar has POLES interlacing the resonances (wherever w^H A^{-1} r crosses zero), and a
// Muller iterate landing near one gets flung into a neighbouring basin -- observed as modes 6/9
// collapsing onto mode 5's resonance once delta shrank the basin spacing. mu has no such poles.
MullerResult muller_resonance(SpectralOperators& ops, int Ntot, cpxd kV, cpxd kVb, cpxd kVbd,
                              const std::vector<bool>& defect_row, double delta,
                              const VectorXcd& w, const VectorXcd& r, cpxd omega_guess) {
    auto g = [&](cpxd omega) {
        const MatrixXcd A = assemble_A(ops, Ntot, omega, kV, kVb, kVbd, defect_row, delta);
        const PartialPivLU<MatrixXcd> lu = A.partialPivLu();
        VectorXcd x = r;
        for (int it = 0; it < 3; ++it) x = lu.solve(x).normalized();
        return x.dot(A * x);  // eigenvalue of A nearest 0, to inverse-iteration accuracy
    };

    // The whole miniband spans delta*(lambda_max - lambda_min), so BOTH the seed spacing and the
    // step clamp must scale with delta: a fixed clamp that is fine at delta = 0.05 exceeds the
    // inter-resonance spacing at delta = 0.025 and lets Muller hop into a neighbouring basin
    // (observed: modes 6 and 9 both collapsing onto mode 4's resonance).
    const double d = std::min(5.0e-4, 1.0e-2 * delta);
    const double max_step = 0.2 * delta;
    const double tol = 1e-10;

    cpxd x0 = omega_guess - d, x1 = omega_guess + d, x2 = omega_guess;
    cpxd f0 = g(x0), f1 = g(x1), f2 = g(x2);

    MullerResult res;
    res.omega = x2;
    for (int it = 0; it < 30; ++it) {
        const cpxd h1 = x1 - x0, h2 = x2 - x1;
        const cpxd d1 = (f1 - f0) / h1, d2 = (f2 - f1) / h2;
        const cpxd a = (d2 - d1) / (h2 + h1);
        const cpxd b = a * h2 + d2;
        const cpxd disc = std::sqrt(b * b - 4.0 * a * f2);
        const cpxd den = (std::abs(b + disc) > std::abs(b - disc)) ? (b + disc) : (b - disc);
        cpxd dx = -2.0 * f2 / den;
        if (std::abs(dx) > max_step) dx *= max_step / std::abs(dx);

        x0 = x1; f0 = f1;
        x1 = x2; f1 = f2;
        x2 = x2 + dx;
        f2 = g(x2);
        res.iters = it + 1;
        res.omega = x2;
        if (std::abs(dx) < tol) {
            res.converged = true;
            break;
        }
    }
    if (std::abs(res.omega - omega_guess) > 0.6 * delta)
        std::cout << "    WARNING: converged " << std::abs(res.omega - omega_guess)
                  << " away from the seed -- possibly a neighbouring resonance\n";
    return res;
}

// Deterministic unit-norm probe vectors; <random> would make runs non-reproducible.
VectorXcd probe(int n, double p, double q) {
    VectorXcd v(n);
    for (int i = 0; i < n; ++i) v(i) = cpxd(std::cos(p * i + 0.3), std::sin(q * i + 0.7));
    return v.normalized();
}

}  // namespace

void run_bent_waveguide_exact(double radius, double defect_radius, double delta, int m_ang,
                              int n_defect, int n_clad, int fringe, int points_per_disk,
                              int mode_index, int grid_points, double seed_re, double seed_im,
                              double v, double v_b, double v_bd) {
    const BentPatch p = build_bent_patch(radius, defect_radius, m_ang, n_defect, n_clad, fringe,
                                         points_per_disk, v, v_b, v_bd, /*verbose=*/false);

    // Boundary rows belonging to defect disks: these carry the defect interior wavenumber in
    // assemble_A, and identify the defect interiors in the field evaluation below.
    std::vector<bool> defect_row(p.Ntot, false);
    std::vector<bool> is_defect_disk(p.disks.size(), false);
    for (const auto& kv : p.defect_index) {
        is_defect_disk[kv.second] = true;
        const int s0 = p.mesh.get_start_index(kv.second);
        for (int i = 0; i < p.N; ++i) defect_row[s0 + i] = true;
    }
    const int modes = p.modes;
    const int n_main = static_cast<int>(p.main_indices.size());
    const int n_eig = modes * n_main;

    std::cout << "Exact bent-waveguide resonances: " << p.disks.size() << " disks, " << p.Ntot
              << " boundary points, A is " << 2 * p.Ntot << "x" << 2 * p.Ntot
              << ", omega_0 = " << p.omega0 << ", delta = " << delta << "\n";

    // --- capacitance eigenpairs: the seeds ------------------------------------------------
    ComplexEigenSolver<MatrixXcd> es(p.C);
    const VectorXcd lam = es.eigenvalues();
    const MatrixXcd vecs = es.eigenvectors();
    std::vector<int> order(n_eig);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&lam](int a, int b) { return lam(a).real() < lam(b).real(); });

    std::vector<double> corner_weight(n_eig);
    for (int j = 0; j < n_eig; ++j) {
        const VectorXcd v = vecs.col(order[j]);
        double wc = 0.0;
        for (int s = 0; s < modes; ++s) wc += std::norm(v(modes * p.center + s));
        corner_weight[j] = wc / v.squaredNorm();
    }
    const int corner_mode = static_cast<int>(
        std::max_element(corner_weight.begin(), corner_weight.end()) - corner_weight.begin());

    // Which modes to refine. Spread {smallest, middle, largest} exposes how the O(delta) error
    // grows with |lambda|; the corner mode is the physically interesting one and gets drawn.
    std::set<int> refine;
    int draw;
    if (mode_index >= 0 && mode_index < n_eig) {
        refine.insert(mode_index);
        draw = mode_index;
    } else {
        refine.insert(0);
        refine.insert(n_eig / 2);
        refine.insert(n_eig - 1);
        refine.insert(corner_mode);
        draw = corner_mode;
    }

    // --- refine each seed to the exact resonance ------------------------------------------
    SpectralOperators ops(p.mesh);
    const VectorXcd wprobe = probe(2 * p.Ntot, 1.3, 2.1);
    const VectorXcd rprobe = probe(2 * p.Ntot, 0.9, 1.7);

    std::ofstream out(("bent_waveguide_exact_resonances.csv"));
    out.setf(std::ios::scientific);
    out.precision(std::numeric_limits<double>::max_digits10);

    cpxd omega_draw = 0.0;
    std::cout << "  j  Re(lambda)   omega_asym            omega_exact                |domega|   iters\n";
    for (int j : refine) {
        const cpxd l = lam(order[j]);
        const cpxd omega_asym = p.omega0 + delta * l;
        cpxd seed = omega_asym;
        if (mode_index >= 0 && seed_re > 0.0) {
            seed = cpxd(seed_re, seed_im);
            std::cout << "  (seed override: " << seed << ")\n";
        }
        const MullerResult mr = muller_resonance(ops, p.Ntot, p.kV, p.kVb, p.kVbd, defect_row,
                                                 delta, wprobe, rprobe, seed);
        if (!mr.converged)
            std::cout << "    WARNING: mode " << j << " did not converge in " << mr.iters
                      << " iterations\n";
        if (j == draw) omega_draw = mr.omega;

        // Residual of the null vector at the converged omega (computed again below for the drawn
        // mode; here just for the report).
        const MatrixXcd A = assemble_A(ops, p.Ntot, mr.omega, p.kV, p.kVb, p.kVbd, defect_row, delta);
        VectorXcd x = rprobe;
        const PartialPivLU<MatrixXcd> lu = A.partialPivLu();
        for (int it = 0; it < 2; ++it) x = lu.solve(x).normalized();
        const double resid = (A * x).norm();

        const double err = std::abs(mr.omega - omega_asym);
        std::cout << "  " << j << "  " << l.real() << "   (" << omega_asym.real() << ","
                  << omega_asym.imag() << ")  (" << mr.omega.real() << "," << mr.omega.imag()
                  << ")  " << err << "   " << mr.iters << "\n";
        out << j << "," << l.real() << "," << l.imag() << "," << omega_asym.real() << ","
            << omega_asym.imag() << "," << mr.omega.real() << "," << mr.omega.imag() << "," << err
            << "," << mr.iters << "," << resid << "\n";
    }
    out.close();

    // --- exact mode: null vector of A(omega*) by inverse iteration -------------------------
    // Near the resonance A is almost singular, so one LU solve of a generic vector already lies
    // along the null direction to ~sigma_min/sigma_next; two iterations are plenty.
    std::cout << "  drawing exact mode j=" << draw << " at omega* = " << omega_draw << "\n";
    const MatrixXcd A = assemble_A(ops, p.Ntot, omega_draw, p.kV, p.kVb, p.kVbd, defect_row, delta);
    const PartialPivLU<MatrixXcd> lu = A.partialPivLu();
    VectorXcd psi_full = rprobe;
    for (int it = 0; it < 2; ++it) psi_full = lu.solve(psi_full).normalized();
    std::cout << "  null-vector residual ||A psi|| = " << (A * psi_full).norm() << "\n";

    const VectorXcd psi_int = psi_full.head(p.Ntot);  // interior densities (k_b layer)
    const VectorXcd psi_ext = psi_full.tail(p.Ntot);  // exterior densities (k layer)
    std::cout << "  ||psi_int|| = " << psi_int.norm() << ", ||psi_ext|| = " << psi_ext.norm()
              << "\n";

    // --- fingerprint: WHICH mode did we actually converge to? ------------------------------
    // The seed's label is not evidence: a strongly shifted neighbouring resonance can capture
    // the search (observed: the band-top pair sits ~0.23 below its asymptotic prediction and
    // swallowed mode 6's seed). Identify the converged state by its own structure instead:
    // trace u = S_b psi_int on each defect boundary, project onto e^{+-i m theta}, and compare
    // the per-site weight profile against every capacitance eigenvector.
    {
        // The fingerprint only reads the trace on DEFECT boundaries, whose interior layer lives
        // at the defect wavenumber, so assemble S there.
        MatrixXcd Sb;
        ops.S(Sb, omega_draw / p.kVbd);
        const VectorXcd trace = Sb * psi_int;

        std::vector<double> wsite;   // main sites, path order
        double wfringe = 0.0, wtot = 0.0;
        for (const auto& kv : p.defect_index) {
            const int s0 = p.mesh.get_start_index(kv.second);
            double wd = 0.0;
            for (int sgn = 0; sgn < p.modes; ++sgn) {
                cpxd c = 0.0;
                for (int loc = 0; loc < p.N; ++loc) {
                    const double th = 2.0 * M_PI * double(loc) / double(p.N);
                    const double s = (sgn == 0) ? 1.0 : -1.0;
                    c += trace(s0 + loc) * std::exp(cpxd(0, -1.0) * s * double(m_ang) * th);
                }
                wd += std::norm(c);
            }
            wtot += wd;
            const bool fringe_site = std::abs(kv.first.first) > p.L_main ||
                                     std::abs(kv.first.second) > p.L_main;
            if (fringe_site) wfringe += wd;
        }
        for (int q = 0; q < n_main; ++q) {
            const int s0 = p.mesh.get_start_index(p.main_indices[q]);
            double wd = 0.0;
            for (int sgn = 0; sgn < p.modes; ++sgn) {
                cpxd c = 0.0;
                for (int loc = 0; loc < p.N; ++loc) {
                    const double th = 2.0 * M_PI * double(loc) / double(p.N);
                    const double s = (sgn == 0) ? 1.0 : -1.0;
                    c += trace(s0 + loc) * std::exp(cpxd(0, -1.0) * s * double(m_ang) * th);
                }
                wd += std::norm(c);
            }
            wsite.push_back(wd);
        }
        double wmain = 0.0;
        for (double v : wsite) wmain += v;

        std::cout << "  converged-state fingerprint: fringe fraction = "
                  << (wtot > 0 ? wfringe / wtot : 0.0) << ", main-site weights =";
        for (double v : wsite) std::cout << " " << (wmain > 0 ? v / wmain : 0.0);
        std::cout << "\n  cos-similarity against capacitance modes:\n";
        int best = -1;
        double best_cs = -1.0;
        for (int j = 0; j < n_eig; ++j) {
            const VectorXcd v = vecs.col(order[j]);
            std::vector<double> w(n_main, 0.0);
            double n2 = 0.0;
            for (int q = 0; q < n_main; ++q) {
                for (int s = 0; s < modes; ++s) w[q] += std::norm(v(modes * q + s));
                n2 += w[q];
            }
            double dot = 0.0, na = 0.0, nb = 0.0;
            for (int q = 0; q < n_main; ++q) {
                dot += w[q] / n2 * wsite[q] / wmain;
                na += (w[q] / n2) * (w[q] / n2);
                nb += (wsite[q] / wmain) * (wsite[q] / wmain);
            }
            const double cs = dot / std::sqrt(na * nb);
            if (cs > best_cs) { best_cs = cs; best = j; }
            std::cout << "    j=" << j << "  Re(lambda)=" << lam(order[j]).real()
                      << "  cos=" << cs << "\n";
        }
        std::cout << "  => converged state most resembles capacitance mode j=" << best
                  << " (cos=" << best_cs << ")"
                  << (best != draw ? "  *** NOT the seeded mode ***" : "") << "\n";
    }

    // grid_points <= 0: resonance table + fingerprint only. Do NOT write tiny placeholder field
    // grids -- a validation sweep run this way used to overwrite a good 200x200 field with a
    // 16x16 stub, which then plotted as nothing.
    if (grid_points <= 0) {
        std::cout << "[wrote] bent_waveguide_exact_resonances.csv (field skipped, grid_points <= 0)\n";
        return;
    }

    // --- evaluate the field -----------------------------------------------------------------
    // Unlike the O(delta) reconstruction there are no dark disks: EVERY disk interior carries the
    // interior single layer of the full union boundary (at its own material's wavenumber), so the
    // cladding interior field -- which the leading-order model sets to zero -- comes out too.
    const cpxd k_ext = omega_draw / p.kV;
    const cpxd k_int_clad = omega_draw / p.kVb, k_int_def = omega_draw / p.kVbd;

    const double half = double(p.L) + 1.0;
    const int Ng = std::max(grid_points, 16);
    const double h = 2.0 * M_PI * std::max(radius, defect_radius) / double(p.N);

    std::vector<double> axis(Ng);
    for (int i = 0; i < Ng; ++i) axis[i] = -half + 2.0 * half * double(i) / double(Ng - 1);

    std::vector<std::vector<double>> re(Ng, std::vector<double>(Ng)),
                                     im(Ng, std::vector<double>(Ng)),
                                     mag(Ng, std::vector<double>(Ng)),
                                     phase(Ng, std::vector<double>(Ng));
    const double nan = std::numeric_limits<double>::quiet_NaN();
    double max_clad = 0.0, max_defect = 0.0;

    for (int iy = 0; iy < Ng; ++iy) {
#pragma omp parallel for schedule(dynamic) reduction(max : max_clad, max_defect)
        for (int ix = 0; ix < Ng; ++ix) {
            const Vector2d r(axis[ix], axis[iy]);

            int inside = -1;
            bool near = false;
            for (std::size_t i = 0; i < p.disks.size(); ++i) {
                const double rho = (r - p.disks[i].c).norm();
                if (std::abs(rho - p.disks[i].r) < h) { near = true; break; }
                if (rho < p.disks[i].r) { inside = static_cast<int>(i); break; }
            }
            if (near) {
                re[iy][ix] = im[iy][ix] = mag[iy][ix] = phase[iy][ix] = nan;
                continue;
            }

            const VectorXcd& dens = (inside >= 0) ? psi_int : psi_ext;
            const cpxd kk = (inside < 0) ? k_ext
                                         : (is_defect_disk[inside] ? k_int_def : k_int_clad);
            cpxd u = 0.0;
            for (int j = 0; j < p.Ntot; ++j)
                u += dens(j) * Kernels::helmholtz_2D(kk, r, p.mesh.get_vertex(j).point) *
                     p.sigma(j);

            if (inside >= 0) {
                if (is_defect_disk[inside]) max_defect = std::max(max_defect, std::abs(u));
                else max_clad = std::max(max_clad, std::abs(u));
            }
            re[iy][ix] = u.real();
            im[iy][ix] = u.imag();
            mag[iy][ix] = std::abs(u);
            phase[iy][ix] = std::atan2(u.imag(), u.real());
        }
        Tools::print_stage_progress("Evaluating exact field rows", iy + 1, Ng);
    }
    Tools::finish_progress_line();

    Utils::draw_mesh(p.mesh, "bent_waveguide_mesh.csv");
    Tools::dump_csv("bent_waveguide_exact_field_real.csv", re);
    Tools::dump_csv("bent_waveguide_exact_field_imag.csv", im);
    Tools::dump_csv("bent_waveguide_exact_field_mag.csv", mag);
    Tools::dump_csv("bent_waveguide_exact_field_phase.csv", phase);
    Tools::dump_csv("bent_waveguide_exact_field_axis.csv", {axis});

    // Cladding interior amplitude: nonzero here, identically zero in the O(delta) model. Its
    // size relative to the defect interiors measures that model error directly.
    std::cout << "  max |u| inside defect disks = " << max_defect
              << ", inside cladding disks = " << max_clad
              << "  (ratio " << (max_defect > 0 ? max_clad / max_defect : 0.0)
              << ", O(delta) model would give 0)\n";

    double sym = 0.0, scale = 0.0;
    for (int iy = 0; iy < Ng; ++iy)
        for (int ix = 0; ix < Ng; ++ix) {
            const double a = mag[iy][ix], b = mag[ix][iy];
            if (std::isnan(a) || std::isnan(b)) continue;
            sym = std::max(sym, std::abs(a - b));
            scale = std::max(scale, a);
        }
    std::cout << "  reflection symmetry ||u(x,y)|-|u(y,x)||_max = " << sym << "  (relative "
              << (scale > 0 ? sym / scale : 0.0) << ")\n";

    std::cout << "[wrote] bent_waveguide_exact_resonances.csv, "
                 "bent_waveguide_exact_field_{real,imag,mag,phase}.csv\n";
}

}  // namespace workflows
