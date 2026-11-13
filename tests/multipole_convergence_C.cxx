// Standalone convergence experiment; intentionally not part of the automatic unit-test suite.
#include "workflows/interface_patch.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace workflows;

namespace {
std::ofstream csv(const std::string& filename) {
    std::ofstream out;
    out.exceptions(std::ios::failbit | std::ios::badbit);
    out.open(filename);
    out.precision(std::numeric_limits<double>::max_digits10);
    return out;
}
void usage() {
    std::cout << "MultipoleConvergenceC [nm=12] [sweep=nmult] [max_order=20] [window=1] "
                 "[scan_intervals=0] [global_radius=window] [points=auto] "
                 "[omega0=mode resonance] [fringe=0] [colbrook=window]\n"
                 "sweep: nmult, window, fringe, patch, colbrook (may be omitted for old commands).\n"
                 "The selected parameter is swept up to its supplied value; others stay fixed.\n"
                 "window: 1..window (or just 0); fringe/colbrook: 0..value; patch: window..global_radius.\n"
                 "Spatial sweeps use order=max_order and require scan_intervals > 0.\n"
                 "Cutoff starts at max(m,ceil(k*r)+5), increases by 2, and includes max_order.\n"
                 "Local template convergence is always saved. A positive scan_intervals also\n"
                 "scans z in [0,10]; global_radius is used only for that optional scan.\n"
                 "points=0 chooses max(50,4*max_order+4), FIXED throughout the cutoff sweep.\n"
                 "Run in a separate output directory; CSVs with matching names are overwritten.\n";
}
}

int main(int argc, char** argv) try {
    if (argc > 1 && std::string(argv[1]) == "--help") { usage(); return 0; }
    // Optional word after nm; numeric second arguments retain the old CLI.
    const bool named_sweep = argc > 2 && std::string(argv[2]).find_first_not_of("+-0123456789") != std::string::npos;
    const std::string sweep = named_sweep ? argv[2] : "nmult";
    if (sweep != "nmult" && sweep != "window" && sweep != "fringe" &&
        sweep != "patch" && sweep != "colbrook")
        throw std::invalid_argument("Unknown sweep: use nmult, window, fringe, patch, or colbrook");
    const int shift = named_sweep ? 1 : 0;
    if (argc > 10 + shift) { usage(); return 1; }
    const auto integer_arg = [&](int position, int fallback) {
        return argc > position + shift ? std::stoi(argv[position + shift]) : fallback;
    };
    const int nm = argc > 1 ? std::stoi(argv[1]) : 12;
    const double beta = neumann_zero(nm);
    const int max_order = integer_arg(2, 20);
    const int window_radius = integer_arg(3, 1);
    const int Nz = integer_arg(4, 0);
    const int M = integer_arg(5, window_radius);
    const int requested_points = integer_arg(6, 0);
    const int fringe = integer_arg(8, 0);
    const int colbrook = integer_arg(9, window_radius);
    const double v = 1.0;
    const double z_min = 0.8, z_max = 1.0; // for this frequency!!!
    const LatticeVectors up{Vector2d(1, 0), Vector2d(0.5, std::sqrt(3.0) / 2)};
    const LatticeVectors down{Vector2d(1, 0), Vector2d(0.5, -std::sqrt(3.0) / 2)};
    const std::vector<Disk> disks_up{
        {0.1001, Vector2d(0.5, std::sqrt(3.0) / 6), 0.1},
        {0.09999, Vector2d(1.0, std::sqrt(3.0) / 3), 0.1}
    };
    auto disks_down = disks_up;
    for (auto& d : disks_down) d.c.y() = -d.c.y();
    const double omega0 = argc > 7 + shift ? std::stod(argv[7 + shift]) : disks_up[0].v_b * beta / disks_up[0].r;
    if (max_order < 1 || max_order > 1000 || window_radius < 0 || M < window_radius ||
        Nz < 0 || fringe < 0 || colbrook < 0 || requested_points < 0 || !std::isfinite(omega0) || omega0 <= 0)
        throw std::invalid_argument("Require 1 <= max_order <= 1000, M >= window >= 0, "
                                    "nonnegative counts/buffers and positive finite omega0");
    const double kr_max = omega0 * std::max_element(disks_up.begin(), disks_up.end(),
        [](const Disk& a, const Disk& b) { return a.r < b.r; })->r / v;
    if (sweep != "nmult" && Nz == 0)
        throw std::invalid_argument("Spatial sweeps require scan_intervals > 0 to compare global spectra");
    if (sweep == "nmult" && std::ceil(kr_max) + 5 > max_order)
        throw std::invalid_argument("max_order is below the starting cutoff ceil(k*r)+5");
    const int first_order = std::max(decode_nm(nm).m_ang, int(std::ceil(kr_max)) + 5);
    if (max_order < decode_nm(nm).m_ang)
        throw std::invalid_argument("max_order is below the angular order");
    const int npd = requested_points == 0 ? std::max(50, 4 * max_order + 4) : requested_points;
    if (npd <= 2 * max_order + 1)
        throw std::invalid_argument("points must exceed 2*max_order+1 for the ENTIRE sweep");

    std::vector<int> values;
    if (sweep == "nmult") {
        for (int L = first_order; L <= max_order; L += 2) values.push_back(L);
        if (values.back() != max_order) values.push_back(max_order);
    } else {
        const int last = sweep == "window" ? window_radius : sweep == "fringe" ? fringe :
                         sweep == "patch" ? M : colbrook;
        const int first = sweep == "window" ? std::min(1, window_radius) : sweep == "patch" ? window_radius : 0;
        for (int value = first; value <= last; ++value) values.push_back(value);
    }
    auto diagnostics = csv("multipole_convergence_C.csv");
    diagnostics << "nm,omega0,window,fringe,colbrook,order,points,template_id,local_rcond,"
                   "relative_solve_residual,backward_error,density_norm,C_norm,"
                   "absolute_C_change,relative_C_change,sweep,value,global_radius\n";
    std::ofstream summary;
    if (Nz > 0) {
        summary = csv("convergence_scan_summary.csv");
        summary << "sweep,value,nm,omega0,order,points,window,fringe,colbrook,global_radius,"
                   "z_min,sigma_min,z_spacing,at_boundary\n";
    }
    std::vector<MatrixXcd> previous;
    std::optional<InterfacePatch> cached;
    std::cout << "nm=" << nm << ", omega0=" << omega0 << ", k*r=" << kr_max
              << ", fixed points=" << npd << "; sweep=" << sweep << ", " << values.size() << " values\n";
    for (int value : values) {
        const int nmult = sweep == "nmult" ? value : max_order;
        const int W = sweep == "window" ? value : window_radius;
        const int F = sweep == "fringe" ? value : fringe;
        const int P = sweep == "patch" ? value : M;
        const int B = sweep == "colbrook" ? value : colbrook;
        // Only patch/colbrook sweeps can reuse the local solves.
        if (!cached || (sweep != "patch" && sweep != "colbrook"))
            cached = build_interface_patch(W, Nz > 0 ? P : W, B, up, down, nm,
                                           disks_up, disks_down, F, npd, omega0, v, nmult, false);
        else {
            cached->patch_radius = P;
            cached->colbrook = B;
            cached->bigC = assemble_interface_matrix(*cached);
        }
        const auto& p = *cached;
        double max_change = 0;
        for (std::size_t t = 0; t < p.patch_colC.size(); ++t) {
            const auto& C = p.patch_colC[t];
            const auto& d = p.solve_diagnostics[t];
            const double change = previous.empty() ? std::numeric_limits<double>::quiet_NaN()
                                                   : (C - previous[t]).norm();
            const double relative = C.norm() > 0 ? change / C.norm() : change;
            if (!previous.empty()) max_change = std::max(max_change, relative);
            diagnostics << nm << ',' << omega0 << ',' << W << ',' << F << ','
                        << B << ',' << nmult << ',' << npd << ',' << t << ',' << d.rcond << ','
                        << d.relative_residual << ',' << d.backward_error << ',' << d.density_norm
                        << ',' << C.norm() << ',' << change << ',' << relative << ','
                        << sweep << ',' << value << ',' << p.patch_radius << '\n';
        }
        diagnostics.flush();
        std::cout << sweep << '=' << value << ", order=" << nmult;
        if (sweep == "nmult")
            std::cout << ", max relative template change="
                      << (previous.empty() ? std::numeric_limits<double>::quiet_NaN() : max_change);
        std::cout << std::endl;
        // Spatial sweeps may change template shapes/identities: do not subtract them.
        if (sweep == "nmult") previous = p.patch_colC;
        if (Nz == 0) continue;

        const auto layout = make_interface_layout(p);
        const std::string tag = (sweep == "nmult" ? "mult" : sweep) + std::to_string(value);
        auto out = csv("interface_singular_values_" + tag + ".csv");
        auto out2 = csv("rcond_" + tag + ".csv");
        out << "# nm=" << nm << ", omega0=" << omega0 << ", order=" << nmult
            << ", window=" << W << ", fringe=" << F << ", patch=" << P << ", colbrook=" << B << '\n';
        out2 << "# z,sigma_min/sigma_max of C-zE (NOT local S rcond)\n";
        double best_sigma = std::numeric_limits<double>::infinity(), best_z = z_min;
        for (int zi = 0; zi <= Nz; ++zi) {
            const double z = std::lerp(z_min, z_max, double(zi) / Nz);
            MatrixXcd shifted = p.bigC;
            for (Eigen::Index j = 0; j < shifted.cols(); ++j) shifted(layout.source_rows[j], j) -= z;
            Eigen::BDCSVD<MatrixXcd> svd(shifted);
            if (svd.info() != Eigen::Success || !svd.singularValues().allFinite())
                throw std::runtime_error("Interface SVD failed at z=" + std::to_string(z));
            const double sigma = svd.singularValues().minCoeff();
            const double largest = svd.singularValues().maxCoeff();
            out << z << ',' << sigma << '\n';
            out2 << z << ',' << (largest > 0 ? sigma / largest : 0.0) << '\n';
            if (sigma < best_sigma) { best_sigma = sigma; best_z = z; }
        }
        summary << sweep << ',' << value << ',' << nm << ',' << omega0 << ',' << nmult << ','
                << npd << ',' << W << ',' << F << ',' << B << ',' << P << ','
                << best_z << ',' << best_sigma << ',' << (z_max-z_min)/Nz << ','
                << (best_z == z_min || best_z == z_max) << '\n';
        summary.flush();
        std::cout << "  global radius=" << P << ", minimum sampled sigma=" << best_sigma
                  << " at z=" << best_z << std::endl;
    }
    std::cout << "[wrote] multipole_convergence_C.csv\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "MultipoleConvergenceC: " << error.what() << '\n';
    return 1;
}
