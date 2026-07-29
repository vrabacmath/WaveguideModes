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

namespace {

// All scalar-search helpers return {argmin, minimum_value}. A failed search is represented by
// {NaN, infinity}; callers can then skip the point or fall back to a wider search.

// Return sigma_min(A), the smallest singular value of a complex matrix. The band searches use
// sigma_min(A^alpha) and sigma_min(M^eps) because singular band/resonance points appear as zeros.
double smallest_singular_value(const MatrixXcd& A) {
    return JacobiSVD<MatrixXcd>(A).singularValues().tail<1>()(0);
}

// Golden-section minimisation of a scalar objective on [left, right]. This assumes the objective
// is locally unimodal on the interval; the coarse scan below is what chooses such a bracket.
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

// From a coarse list of sampled objective values, choose the deepest local minimum that is not
// located at an endpoint. Endpoint minima usually mean the requested search window is wrong.
int best_interior_minimum(const std::vector<double>& v) {
    int best = -1;
    double bv = std::numeric_limits<double>::infinity();
    for (int i = 1; i + 1 < static_cast<int>(v.size()); ++i)
        if (v[i] <= v[i - 1] && v[i] <= v[i + 1] && v[i] < bv) { bv = v[i]; best = i; }
    return best;
}

// Coarse scan for the deepest interior local minimum, then golden-section refine around that
// three-point bracket. This is the robust first pass used when no previous alpha point is known.
template <typename Objective>
std::pair<double, double> scan_minimum(const Objective& f, double lo, double hi, int n) {
    std::vector<double> s(n);
    for (int i = 0; i < n; ++i) s[i] = f(lo + i * (hi - lo) / (n - 1));
    const int b = best_interior_minimum(s);
    const auto fail = std::make_pair(std::numeric_limits<double>::quiet_NaN(),
                                     std::numeric_limits<double>::infinity());
    if (b < 0) return fail;
    const double margin = 0.05 * (hi - lo);
    const double w = lo + b * (hi - lo) / (n - 1);
    if (w <= lo + margin || w >= hi - margin) return fail;
    return golden_min_scalar(f, lo + (b - 1) * (hi - lo) / (n - 1), lo + (b + 1) * (hi - lo) / (n - 1));
}

// Refine a bracketed minimum of a possibly noisy, floor-limited objective. A fine scan first
// locates the true dip -- golden-section alone assumes unimodality, which fails when sigma_min
// sits on a flat contamination floor (the high-order-multipole floor of M^eps) and then lands
// anywhere in the bracket -- and golden-section afterwards only polishes the best fine cell.
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

// Refine a minimum in a trusted local window, rejecting it if it lands too close to the window
// boundary. Used during continuation from one alpha_x sample to the next.
template <typename Objective>
std::pair<double, double> refine_local_minimum(const Objective& f, double lo, double hi) {
    const auto fail = std::make_pair(std::numeric_limits<double>::quiet_NaN(),
                                     std::numeric_limits<double>::infinity());
    if (!(lo < hi)) return fail;

    auto local = golden_min_scalar(f, lo, hi);
    const double margin = 0.05 * (hi - lo);
    if (local.first <= lo + margin || local.first >= hi - margin) return fail;
    return local;
}

// Track a band across alpha_x. If the previous accepted frequency is available, first refine in a
// small neighborhood of that frequency; otherwise fall back to a full coarse scan. The tolerance is
// a guard: local continuation is accepted only when the objective is small enough to be meaningful.
template <typename Objective>
std::pair<double, double> tracked_minimum(const Objective& f, double lo, double hi,
                                          int n, double prev, double radius, double tol) {
    if (std::isfinite(prev) && prev > lo && prev < hi) {
        const double l = std::max(lo, prev - radius), r = std::min(hi, prev + radius);
        auto local = refine_local_minimum(f, l, r);
        if (local.second < tol) return local;
    }
    return scan_minimum(f, lo, hi, n);
}

}  // namespace

// Compute the dilute line-defect band diagram used for the Figure 2-style plot.
//
// For each alpha_x in [0, 2pi], this writes:
//   1. the upper edge of the first crystal band, found from sigma_min(A^(alpha_x, pi));
//   2. the next visible crystal band edge, found from sigma_min(A^(alpha_x, 0));
//   3. the defect branch, found from sigma_min(M^eps(omega, alpha_x)).
//
// The defect search starts just above the first-band edge and accepts a point only if the
// line-defect operator is nearly singular. The parameters are explicit so this routine can be
// retuned, but the defaults in the header reproduce the R = 0.05, epsilon = -0.2R dilute case.
void run_line_defect_bands_M(double radius, double defect_radius, double delta,
                             double first_band_lo, double first_band_hi,
                             double second_band_lo, double second_band_hi,
                             double defect_band_hi, int num_alpha, int n_gauss) {
    const cpxd v = 1.0, v_b = 1.0;
    const int n_multipole = std::max(8, static_cast<int>(std::ceil(second_band_hi * radius)) + 7);
    // point_defect_radius is unused for the line defect; pass a benign value.
    Multipole multipole_M(n_multipole, n_gauss, radius, 0.97 * radius, defect_radius, v, v_b, delta);

    std::cout << "Subwavelength line-defect bands (operator M): R=" << radius
              << ", R_defect=" << defect_radius << " (epsilon=" << (defect_radius - radius) / radius
              << " R), delta=" << delta << ", N_multipole=" << n_multipole << "\n";

    // For the line-defect strip, the relevant edges of the bulk essential spectrum occur at
    // different transverse quasi-periodicities: the upper edge of the first band at alpha_y=pi,
    // and the lower edge of the next band at alpha_y=0.
    auto crystal_sigma = [&](double omega, double alpha_x, double alpha_y) {
        MatrixXcd A;
        Utils::multipole_crystal_A(A, n_multipole, radius, omega / v, omega / v_b,
                                   Vector2d(alpha_x, alpha_y), delta);
        return smallest_singular_value(A);
    };
    // sigma_min of the line-defect operator M^eps: its zeros are the defect band.
    auto defect_sigma = [&](double omega, double alpha_x) {
        return smallest_singular_value(multipole_M.crystal_line_M_operator(omega, alpha_x));
    };

    std::ofstream first_out("crystal_first_band.csv"), second_out("crystal_second_band.csv");
    std::ofstream legacy_crystal_out("crystal_bands.csv");
    std::ofstream defect_out("defect_bands.csv"), diag_out("defect_band_diagnostics.csv");

    double prev_first = std::numeric_limits<double>::quiet_NaN();
    double prev_second = std::numeric_limits<double>::quiet_NaN();
    double prev_defect = std::numeric_limits<double>::quiet_NaN();

    for (int a = 0; a <= num_alpha; ++a) {
        const double alpha_x = 2.0 * a * M_PI / num_alpha;

        // Crystal band edges. In the strip picture the first-band upper edge occurs at transverse
        // quasi-periodicity alpha_y = pi, while the next visible band edge occurs at alpha_y = 0.
        const auto first = tracked_minimum([&](double w) { return crystal_sigma(w, alpha_x, M_PI); },
                                           first_band_lo, first_band_hi, 100, prev_first, 0.01, 1e-4);
        const auto second = tracked_minimum([&](double w) { return crystal_sigma(w, alpha_x, 0.0); },
                                            second_band_lo, second_band_hi, 100, prev_second, 0.18,
                                            std::numeric_limits<double>::infinity());
        if (std::isfinite(first.first))  { prev_first = first.first;   first_out  << alpha_x << "," << first.first  << "\n"; }
        if (std::isfinite(second.first)) {
            prev_second = second.first;
            second_out << alpha_x << "," << second.first << "\n";
            legacy_crystal_out << alpha_x << "," << second.first << "\n";
        }

        // Line-defect band. Search in the gap above the first crystal edge and track the same
        // branch as alpha_x changes, rejecting candidates unless sigma_min(M^eps) is small.
        auto defect = std::make_pair(std::numeric_limits<double>::quiet_NaN(),
                                     std::numeric_limits<double>::infinity());
        const double search_lo = std::max(first_band_lo, first.first + 5e-4);
        if (std::isfinite(search_lo) && search_lo < defect_band_hi)
            defect = tracked_minimum([&](double w) { return defect_sigma(w, alpha_x); },
                                     search_lo, defect_band_hi, 40, prev_defect, 0.015, 1e-4);

        diag_out << alpha_x << "," << first.first << "," << first.second << "," << second.first << ","
                 << second.second << "," << defect.first << "," << defect.second << "\n";

        if (std::isfinite(defect.first) && defect.second < 1e-4) {
            prev_defect = defect.first;
            defect_out << alpha_x << "," << defect.first << "\n";
            std::cout << "  alpha_x=" << alpha_x << "  defect omega=" << defect.first
                      << "  sigma(M)=" << defect.second << "\n";
        } else {
            prev_defect = std::numeric_limits<double>::quiet_NaN();
            std::cout << "  alpha_x=" << alpha_x << "  no defect root (sigma(M)=" << defect.second << ")\n";
        }
    }
    std::cout << "[wrote] crystal_first_band.csv, crystal_second_band.csv, defect_bands.csv, "
                 "defect_band_diagnostics.csv\n";
}

namespace {

// First positive zero of J_m' = interior Neumann eigenvalue of the disk (radial index 1). The
// defect resonances sit at omega_0 = v_b * j'_{m,1}/R_def; m>=1 is doubly degenerate.
double neumann_zero(int m) {
    switch (m) {
        case 0: return 3.8317059702;  // breathing (= j_{1,1})
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

}  // namespace

// Line-defect bands found by searching the multipole defect operator M^eps AROUND the defect
// resonator's interior Neumann resonances -- the higher-frequency (dipole, quadrupole, ...)
// counterpart of run_line_defect_bands_M, meant for direct comparison with the capacitance bands
// (run_defect_capacitance_bands) at the SAME geometry.
//
// For each alpha_x in [0, pi] and each angular order m = 0..max_m:
//   * omega_0 = v_b * j'_{m,1}/R_def is the interior Neumann resonance (multiplicity 1 for m=0,
//     2 for m>=1);
//   * coarse-scan sigma_min(M^eps(omega, alpha_x)) over [omega_0 +/- window_factor*delta],
//     excluding neighbourhoods of the exterior Dirichlet eigenvalues j_{p,q}/R (the flat bands);
//   * take the up-to-`multiplicity` deepest interior minima and golden-refine each;
//   * tag each with in_gap (the bulk crystal A^{(alpha_x,alpha_y)} is nowhere singular over
//     alpha_y, i.e. omega is in a projected bulk gap) and near_dirichlet.
//
// Writes "defect_neumann_bands.csv": m, alpha_x, omega, sigma_min(M), in_gap, near_dirichlet.
// Keep the rows with in_gap=1, near_dirichlet=0 and small sigma_min for the physical band.
void run_line_defect_bands_neumann(double radius, double defect_radius, double delta,
                                   int max_m, double window_factor, int num_alpha,
                                   int n_gauss, int n_multipole, double sub_lo, double sub_hi) {
    const cpxd v = 1.0, v_b = 1.0;
    const double contour = 1e-5;       // small Im(omega): enough regularization without merging close dips
    const double dir_tol = 0.06;       // exclude within this of an exterior Dirichlet eigenvalue
    const double gap_threshold = 0.25; // gap if min_ay(sigma_min A) > this * median_ay (see in_gap)

    // frequency-appropriate multipole cutoff: too many orders -> vanishing J_n(kR) contamination.
    double omega_hi = 0.0;
    for (int mm = 0; mm <= max_m; ++mm)
        omega_hi = std::max(omega_hi, neumann_zero(mm) / defect_radius + window_factor * delta);
    if (n_multipole <= 0)
        n_multipole = std::max(max_m + 3, static_cast<int>(std::ceil(omega_hi * radius)) + 3);

    Multipole multipole_M(n_multipole, n_gauss, radius, 0.97 * radius, defect_radius, v, v_b, delta);

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
    // In a projected gap at (alpha_x, omega) the bulk crystal has no band, i.e. sigma_min(A) does
    // not dip as alpha_y sweeps the transverse Brillouin zone. sigma_min(A) has an alpha-independent
    // contamination floor (vanishing high-order J_n(kR)), so an ABSOLUTE threshold fails; instead
    // compare the alpha_y-minimum to the alpha_y-median -- a real band dips sharply below the
    // (flat) plateau, contamination does not.
    auto in_gap = [&](double omega, double alpha_x) {
        const int nay = 15;
        std::vector<double> vals(nay);
        for (int j = 0; j < nay; ++j) vals[j] = crystal_sigma_ay(omega, alpha_x, M_PI * j / (nay - 1));
        std::sort(vals.begin(), vals.end());
        return vals.front() > gap_threshold * vals[nay / 2];  // min > threshold * median => gap
    };
    const std::vector<double> dir_radii = {radius, defect_radius};

    // Subwavelength (static-monopole) band uses its own, low-frequency-appropriate multipole cutoff
    // (at low omega the high-order J_n(kR) contamination is worse, so use fewer orders).
    const bool do_sub = sub_hi > sub_lo;
    const int n_sub = std::max(3, static_cast<int>(std::ceil(sub_hi * radius)) + 2);
    Multipole multipole_sub(n_sub, n_gauss, radius, 0.97 * radius, defect_radius, v, v_b, delta);
    auto sub_sigma = [&](double omega, double alpha_x) {
        return smallest_singular_value(multipole_sub.crystal_line_M_operator(cpxd(omega, contour), alpha_x));
    };

    std::cout << "Line-defect bands near defect Neumann resonances (operator M): R=" << radius
              << ", R_def=" << defect_radius << ", delta=" << delta << ", N_multipole=" << n_multipole
              << ", search +/-" << window_factor << "*delta around m=0.." << max_m << "\n";
    for (int m = 0; m <= max_m; ++m)
        std::cout << "   m=" << m << ": omega_0=" << neumann_zero(m) / defect_radius
                  << "  (multiplicity " << (m == 0 ? 1 : 2) << ")\n";

    std::ofstream out("defect_neumann_bands.csv");
    out.setf(std::ios::scientific);
    out.precision(10);

    for (int a = 0; a < num_alpha; ++a) {
        double alpha_x = M_PI * a / (num_alpha - 1);
        if (std::abs(alpha_x) < 1e-9) alpha_x = 1e-6;

        for (int m = 0; m <= max_m; ++m) {
            const double omega0 = neumann_zero(m) / defect_radius;  // v_b = 1
            const double W = window_factor * delta * std::max(1, m);
            const double lo = omega0 - W, hi = omega0 + W;
            const int mult = (m == 0) ? 1 : 2;

            // coarse scan of sigma_min(M), blanking the exterior-Dirichlet neighbourhoods
            std::vector<double> ws, ss;
            std::vector<int> minima;
            auto scan_window = [&](double scan_lo, double scan_hi, int ns) {
                ws.resize(ns);
                ss.resize(ns);
                for (int i = 0; i < ns; ++i) {
                    ws[i] = scan_lo + (scan_hi - scan_lo) * i / (ns - 1);
                    ss[i] = near_dirichlet(ws[i], dir_radii, dir_tol)
                                ? std::numeric_limits<double>::infinity()
                                : defect_sigma(ws[i], alpha_x);
                }
                // Interior local minima, deepest first, up to the expected multiplicity. The
                // Neumann windows are intentionally wide enough that physical branches should not
                // be accepted from a monotone boundary value.
                minima.clear();
                for (int i = 1; i + 1 < ns; ++i)
                    if (std::isfinite(ss[i]) && ss[i] < ss[i - 1] && ss[i] <= ss[i + 1])
                        minima.push_back(i);
                std::sort(minima.begin(), minima.end(),
                          [&](int x, int y) { return ss[x] < ss[y]; });
                if (static_cast<int>(minima.size()) > mult) minima.resize(mult);
            };
            scan_window(lo, hi, 61);
            // A branch dip can be narrower than the coarse spacing and sit barely below the
            // high-order contamination floor; when fewer than `mult` dips show up, rescan denser.
            if (static_cast<int>(minima.size()) < mult) scan_window(lo, hi, 301);
            // Quadrupole branches can nearly touch. When they merge into one coarse valley, zoom
            // around that valley instead of rescanning the whole resonance window.
            if (m >= 2 && static_cast<int>(minima.size()) < mult && !minima.empty()) {
                const double center = ws[minima.front()];
                const double local_radius = std::min(0.003, 0.2 * W);
                scan_window(std::max(lo, center - local_radius), std::min(hi, center + local_radius), 401);
            }

            std::vector<double> accepted;
            for (int idx : minima) {
                const int left_idx = std::max(0, idx - 1);
                const int right_idx = std::min(static_cast<int>(ws.size()) - 1, idx + 1);
                const auto refined = fine_scan_golden(
                    [&](double w) { return defect_sigma(w, alpha_x); }, ws[left_idx], ws[right_idx]);
                const double wr = refined.first, sr = refined.second;
                // two coarse minima can refine into the same dip on a flat floor -- keep one
                const bool dup = std::any_of(accepted.begin(), accepted.end(),
                                             [&](double w) { return std::abs(w - wr) < 5e-5; });
                if (dup) continue;
                accepted.push_back(wr);
                const int gap = in_gap(wr, alpha_x) ? 1 : 0;
                const int dir = near_dirichlet(wr, dir_radii, dir_tol) ? 1 : 0;
                out << m << "," << alpha_x << "," << wr << "," << sr << "," << gap << "," << dir << "\n";
            }
        }

        // subwavelength (static-monopole, labelled m=-1) defect band: the deepest in-gap,
        // non-Dirichlet dip of sigma_min(M) over [sub_lo, sub_hi].
        if (do_sub) {
            const int nss = 120;
            std::vector<double> ws(nss), ss(nss);
            for (int i = 0; i < nss; ++i) {
                ws[i] = sub_lo + (sub_hi - sub_lo) * i / (nss - 1);
                ss[i] = near_dirichlet(ws[i], dir_radii, dir_tol)
                            ? std::numeric_limits<double>::infinity()
                            : sub_sigma(ws[i], alpha_x);
            }
            int best = -1;
            for (int i = 1; i + 1 < nss; ++i)
                if (std::isfinite(ss[i]) && ss[i] < ss[i - 1] && ss[i] <= ss[i + 1] &&
                    (best < 0 || ss[i] < ss[best]))
                    best = i;
            if (best >= 0) {
                const auto refined = golden_min_scalar(
                    [&](double w) { return sub_sigma(w, alpha_x); }, ws[best - 1], ws[best + 1]);
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
