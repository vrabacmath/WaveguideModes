#include "workflows/line_defect_bands.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

#include "Eigen/Dense"
#include "multipole.h"
#include "utils.h"

namespace workflows {

using namespace Eigen;


/* Return sigma_min(A), the smallest singular value of a complex matrix. The band searches use
sigma_min(A^alpha) and sigma_min(M^eps) because singular band/resonance points appear as zeros. */
double smallest_singular_value(const MatrixXcd& A) {
    return JacobiSVD<MatrixXcd>(A).singularValues().tail<1>()(0);
}

/* Golden section minimization of a scalar objective on [left, right]. This assumes the objective
is locally unimodal on the interval. */
template <typename Objective>
std::pair<double, double> golden_min_scalar(const Objective& f, double left, double right, int iters = 24) {
    const double phi = 0.5 * (std::sqrt(5.0) - 1.0);
    double x1 = right - phi * (right - left), x2 = left + phi * (right - left);
    double f1 = f(x1), f2 = f(x2);
    for (int i = 0; i < iters; ++i) {
        if (f1 > f2) { left = x1; x1 = x2; f1 = f2; x2 = left + phi * (right - left); f2 = f(x2); }
        else         { right = x2; x2 = x1; f2 = f1; x1 = right - phi * (right - left); f1 = f(x1); }
    }
    return (f1 < f2) ? std::make_pair(x1, f1) : std::make_pair(x2, f2);
}

/* From a coarse list of sampled objective values, choose the deepest local minimum that is not
located at an endpoint. Endpoint minima usually mean the requested search window is wrong. */
int vector_min_idx(const std::vector<double>& v) {
    int bidx = -1;
    double bval = std::numeric_limits<double>::infinity();
    for (int i = 1; i + 1 < static_cast<int>(v.size()); ++i)
        if (v[i] <= v[i - 1] && v[i] <= v[i + 1] && v[i] < bval) { bval = v[i]; bidx = i; }
    return bidx;
}

/* Same as scan_minimum, but without the margin check. Used after scan_minimum. */
template <typename Objective>
std::pair<double, double> fine_scan_golden(const Objective& f, double lo, double hi, int n = 41) {
    std::vector<double> xs(n), vs(n);
    for (int i = 0; i < n; ++i) { xs[i] = lo + (hi - lo) * i / (n - 1); vs[i] = f(xs[i]); }
    int b = 0;
    for (int i = 1; i < n; ++i)
        if (vs[i] < vs[b]) b = i;
    const int l = std::max(0, b - 1), r = std::min(n - 1, b + 1);
    return golden_min_scalar(f, xs[l], xs[r]);
}

// First positive zero of J_m' = interior Neumann eigenvalue of the disk (radial index 1). The
// defect resonances sit at omega_0 = v_b * j'_{m,1}/R_def; m>=1 is doubly degenerate.
double neumann_zero(int m) {
    switch (m) {
        case -1: return 0.0;          // static monopole
        case 0: return 3.8317059702;  // monopole
        case 1: return 1.8411837813;  // dipole
        case 2: return 3.0542369282;  // quadrupole
        case 3: return 4.2011889412;  // octupole
        default: return 1.8411837813;
    }
}

// Interior Dirichlet eigenvalues j_{p,q} (zeros of J_p). At omega = j_{p,q}/R the exterior
// single-layer / multipole operator is spuriously singular for every Bloch vector -> the flat
// bands. We exclude neighbourhoods of these from the defect-mode search.
const std::vector<double>& dirichlet_zeros() {
    static const std::vector<double> z = {
        2.4048255577, 3.8317059702, 5.1356223019, 5.5200781103, 6.3801618959,
        7.0155866699, 8.4172441404, 8.6537279129};
    return z;
}

// Is omega within `tol` of an interior Dirichlet eigenvalue j_{p,q}/R for any listed radius?
bool near_dirichlet(double omega, const std::vector<double>& radii, double tol) {
    for (double R : radii)
        for (double j : dirichlet_zeros())
            if (std::abs(omega - j / R) < tol) return true;
    return false;
}

void run_line_defect_bands_neumann(double radius, double defect_radius, double delta,
                                   int max_m, double window_factor, int num_alpha,
                                   int n_gauss, int n_multipole, double sub_lo, double sub_hi,
                                   double v, double v_b, double v_bd) {
    const double contour = 1e-5;       // small Im(omega): enough regularization without merging close dips
    const double dir_tol = 0.06;       // tolerance for an exterior Dirichlet eigenvalue
    const double gap_threshold = 0.25; // gap if min_ay(sigma_min A) > this * median_ay (see in_gap below)


    // select a multipole cutoff that is appropriate for the highest defect resonance to be searched
    double omega_hi = 0.0;
    for (int mm = 0; mm <= max_m; ++mm)
        omega_hi = std::max(omega_hi, v_bd * neumann_zero(mm) / defect_radius + window_factor * delta);
    if (n_multipole <= 0)
        n_multipole = std::max(max_m + 3, static_cast<int>(std::ceil(omega_hi * radius / v)) + 3);

    Multipole multipole_M(n_multipole, n_gauss, radius, 0.97 * radius, defect_radius, v, v_b, v_bd, delta);

    auto defect_sigma = [&](double omega, double alpha_x) {
        return smallest_singular_value(multipole_M.crystal_line_M_operator(cpxd(omega, contour), alpha_x));
    };

    auto crystal_sigma_ay = [&](double omega, double alpha_x, double alpha_y) {
        MatrixXcd A;
        const cpxd w(omega, contour);
        Utils::multipole_crystal_A(A, n_multipole, radius, w / v, w / v_b,
                                   Vector2d(alpha_x, alpha_y), delta);
        return smallest_singular_value(A);
    };

    // In a projected gap at (alpha_x, omega) sigma_min(A) does not dip as alpha_y
    // sweeps the transverse Brillouin zone. sigma_min(A) can be small even in the gap 
    // since high-order J_n(kR) are small, so an absolute threshold fails.
    // Compare the alpha_y-minimum to the alpha_y-median -- a real band always dips sharply
    // below the (flat) plateau.
    auto in_gap = [&](double omega, double alpha_x) {
        const int nay = 15;
        std::vector<double> vals(nay);
        for (int j = 0; j < nay; ++j) vals[j] = crystal_sigma_ay(omega, alpha_x, M_PI * j / (nay - 1));
        std::sort(vals.begin(), vals.end());
        return vals.front() > gap_threshold * vals[nay / 2];  // min > threshold * median => gap
    };
    // near_dirichlet works with the zeros of J_p(omega/v * R), the exterior representation is
    // spuriously singular at omega = v j_{p,q}/R and v j_{p,q}/R_def, the crystal interior at
    // omega = v_b j_{p,q}/R, and the defect interior at omega = v_bd j_{p,q}/R_def.
    const std::vector<double> dir_radii = {radius / v, defect_radius / v,
                                           radius / v_b, defect_radius / v_bd};

    // Subwavelength (static or 0th monopole) band uses its own, low-frequency-appropriate multipole cutoff
    // (at low omega the high-order J_n(kR) contamination is worse, so use fewer orders).
    const bool do_sub = sub_hi > sub_lo;
    const int n_sub = std::max(3, static_cast<int>(std::ceil(sub_hi * radius / v)) + 2);
    Multipole multipole_sub(n_sub, n_gauss, radius, 0.97 * radius, defect_radius, v, v_b, v_bd, delta);
    auto sub_sigma = [&](double omega, double alpha_x) {
        return smallest_singular_value(multipole_sub.crystal_line_M_operator(cpxd(omega, contour), alpha_x));
    };

    std::cout << "Line-defect bands near defect Neumann resonances (operator M): R=" << radius
              << ", R_def=" << defect_radius << ", delta=" << delta << ", N_multipole=" << n_multipole
              << ", search +/-" << window_factor << "*delta around m=0.." << max_m << "\n";
    for (int m = 0; m <= max_m; ++m)
        std::cout << "   m=" << m << ": omega_0=" << v_bd * neumann_zero(m) / defect_radius
                  << "  (multiplicity " << (m == 0 ? 1 : 2) << ")\n";

    std::ofstream out("defect_neumann_bands.csv");
    out.setf(std::ios::scientific);
    out.precision(10);

    for (int a = 0; a < num_alpha; ++a) {
        double alpha_x = M_PI * a / (num_alpha - 1);
        if (std::abs(alpha_x) < 1e-9) alpha_x = 1e-6;

        for (int m = 0; m <= max_m; ++m) {
            const double omega0 = v_bd * neumann_zero(m) / defect_radius;
            const double W = window_factor * delta * std::max(1, m);
            const double lo = omega0 - W, hi = omega0 + W;
            const int mult = (m == 0) ? 1 : 2;

            // coarse scan of sigma_min(M), blanking the exterior-Dirichlet neighbourhoods
            std::vector<double> omega_scan, singular_scan;
            std::vector<int> minima;
            auto scan_window = [&](double scan_lo, double scan_hi, int ns) {
                omega_scan.resize(ns);
                singular_scan.resize(ns);
                for (int i = 0; i < ns; ++i) {
                    omega_scan[i] = scan_lo + (scan_hi - scan_lo) * i / (ns - 1);
                    singular_scan[i] = near_dirichlet(omega_scan[i], dir_radii, dir_tol)
                                ? std::numeric_limits<double>::infinity()
                                : defect_sigma(omega_scan[i], alpha_x);
                }
                // Interior local minima, deepest first, up to the expected multiplicity. The
                // Neumann windows are intentionally wide enough that physical branches should not
                // be accepted from a monotone boundary value.
                minima.clear();
                for (int i = 1; i + 1 < ns; ++i)
                    if (std::isfinite(singular_scan[i]) && singular_scan[i] < singular_scan[i - 1] && singular_scan[i] <= singular_scan[i + 1])
                        minima.push_back(i);
                std::sort(minima.begin(), minima.end(),
                          [&](int x, int y) { return singular_scan[x] < singular_scan[y]; });
                if (static_cast<int>(minima.size()) > mult) minima.resize(mult);
            };
            scan_window(lo, hi, 61);
            // A branch dip can be narrower than the coarse spacing and sit barely below the
            // high-order contamination floor; when fewer than `mult` dips show up, rescan denser.
            if (static_cast<int>(minima.size()) < mult) scan_window(lo, hi, 301);
            // Some branches can nearly touch. When they merge into one coarse valley, zoom
            // around that valley instead of rescanning the whole resonance window.
            if (m >= 2 && static_cast<int>(minima.size()) < mult && !minima.empty()) {
                const double center = omega_scan[minima.front()];
                const double local_radius = std::min(0.003, 0.2 * W);
                scan_window(std::max(lo, center - local_radius), std::min(hi, center + local_radius), 401);
            }

            std::vector<double> accepted;
            for (int idx : minima) {
                const int left_idx = std::max(0, idx - 1);
                const int right_idx = std::min(static_cast<int>(omega_scan.size()) - 1, idx + 1);
                const auto refined = fine_scan_golden(
                    [&](double w) { return defect_sigma(w, alpha_x); }, omega_scan[left_idx], omega_scan[right_idx]);
                const double omega_refined = refined.first, sr = refined.second;
                // two coarse minima can refine into the same dip on a flat floor -- keep one
                const bool dup = std::any_of(accepted.begin(), accepted.end(),
                                             [&](double w) { return std::abs(w - omega_refined) < 5e-5; });
                if (dup) continue;
                accepted.push_back(omega_refined);
                const int gap = in_gap(omega_refined, alpha_x) ? 1 : 0;
                const int dir = near_dirichlet(omega_refined, dir_radii, dir_tol) ? 1 : 0;
                out << m << "," << alpha_x << "," << omega_refined << "," << sr << "," << gap << "," << dir << "\n";
            }
        }

        // subwavelength (static or 0th monopole, labelled m=-1) defect band: the deepest in-gap,
        // non-Dirichlet dip of sigma_min(M) over [sub_lo, sub_hi].
        if (do_sub) {
            const int nss = 120;
            std::vector<double> omega_scan(nss), singular_scan(nss);
            for (int i = 0; i < nss; ++i) {
                omega_scan[i] = sub_lo + (sub_hi - sub_lo) * i / (nss - 1);
                singular_scan[i] = near_dirichlet(omega_scan[i], dir_radii, dir_tol)
                            ? std::numeric_limits<double>::infinity()
                            : sub_sigma(omega_scan[i], alpha_x);
            }
            int best = -1;
            for (int i = 1; i + 1 < nss; ++i)
                if (std::isfinite(singular_scan[i]) && singular_scan[i] < singular_scan[i - 1] && singular_scan[i] <= singular_scan[i + 1] &&
                    (best < 0 || singular_scan[i] < singular_scan[best]))
                    best = i;
            if (best >= 0) {
                const auto refined = golden_min_scalar(
                    [&](double w) { return sub_sigma(w, alpha_x); }, omega_scan[best - 1], omega_scan[best + 1]);
                const int gap = in_gap(refined.first, alpha_x) ? 1 : 0;
                const int dir = near_dirichlet(refined.first, dir_radii, dir_tol) ? 1 : 0;
                out << -1 << "," << alpha_x << "," << refined.first << "," << refined.second << ","
                    << gap << "," << dir << "\n";
            }
        }
        std::cout << "  alpha_x=" << alpha_x << " done\n";
    }
    out.close();
    std::cout << "[wrote] defect_neumann_bands.csv (m, alpha_x, omega, sigma_min, in_gap, near_dirichlet)\n";
}

}  // namespace workflows
