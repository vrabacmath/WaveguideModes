// Entry point: a small dispatcher over the two line-defect band workflows.
//
//   line-defect-M     Subwavelength line-defect band via the multipole operator M^eps
//                     (arXiv:2512.05370). Reliable for the subwavelength / monopole band.
//                     -> workflows::run_line_defect_bands_M (src/workflows/line_defect_bands.cpp)
//
//   defect-capacitance  Higher-frequency (dipole) line-defect band via the frequency-dependent
//                     capacitance matrix / exterior DtN map (arXiv:2605.27572), on a crystal
//                     supercell. -> workflows::run_defect_capacitance_bands
//                     (src/workflows/defect_capacitance_bands.cpp)
//
// Both write CSVs to the repository root; render them with `python src/plot.py`.
//
// Other experiments (SSH Fig 6.6 reproduction, Zak phase, full-wave cluster, Fabry-Perot finite
// chain, channel-projected M, Gamma-X-M bulk bands, ...) were removed from this dispatcher to keep
// it focused; they live in the git history -- see docs/archived_experiments.md for the map.

#include <iostream>
#include <string>

#include "workflows/line_defect_bands.h"
#include "workflows/defect_capacitance_bands.h"
#include "workflows/crystal_matrix_bands.h"
#include "workflows/patch_capacitance.h"
#include "workflows/bent_waveguide.h"
#include "workflows/bent_waveguide_field.h"
#include "workflows/bent_waveguide_exact.h"
#include "workflows/bent_waveguide_patch.h"
#include "workflows/colbrook_convergence.h"
#include "workflows/dirac_bands.h"
#include "hex_crystal.h"


#include <algorithm>
#include <cmath>
#include <complex>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

#include "Eigen/Dense"

using namespace Eigen;

namespace {

std::vector<int> convergence_radii(const std::string& text) {
    std::vector<int> values;
    std::istringstream stream(text);
    std::string token;
    while (std::getline(stream, token, ',')) {
        std::size_t used = 0;
        const int value = std::stoi(token, &used);
        if (used != token.size() || value < 1)
            throw std::invalid_argument("Radii must be positive comma-separated integers");
        values.push_back(value);
    }
    if (values.empty() || text.back() == ',')
        throw std::invalid_argument("Radius lists must not be empty or end with a comma");
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    return values;
}

int run_colbrook_convergence(int argc, char** argv) {
    // Add/change cases here. Material contrast keeps equal radii; size contrast keeps equal speeds.
    struct Case { const char* name; int m_ang; double defect_radius, v_bd, target; };
    const std::vector<Case> cases{
        {"material_dipole",     1, 0.35, 0.35 / 0.455, 1.75},
        {"material_quadrupole", 2, 0.35, 0.35 / 0.455, 5.47},
        {"size_dipole",         1, 0.455, 1.0, 3.60},
        {"size_quadrupole",     2, 0.455, 1.0, 5.64},
    };
    const std::string selection = argc > 2 ? argv[2] : "all";
    const auto Ns = convergence_radii(argc > 3 ? argv[3] : "8,12,16,20,30,40,50,60,70,80,90,100");
    const auto Ws = convergence_radii(argc > 4 ? argv[4] : "1,2,3,4,5,6");
    const int npd = argc > 5 ? std::stoi(argv[5]) : 32;
    const int intervals = argc > 6 ? std::stoi(argv[6]) : 100;
    const int passes = argc > 7 ? std::stoi(argv[7]) : 3;
    const double half_width = argc > 8 ? std::stod(argv[8]) : 0.2;
    const int multipole_order = argc > 9 ? std::stoi(argv[9]) : 10;
    if (argc > 10 || Ns.front() < Ws.back() || npd < 8 || multipole_order < 2 ||
        multipole_order > (npd - 2) / 2 || intervals < 4 || passes < 1 ||
        !std::isfinite(half_width) || half_width <= 0)
        throw std::invalid_argument("Require min(N) >= max(W), points_per_disk >= 8 and > 2*order+1, "
                                    "multipole_order >= 2, intervals >= 4, passes >= 1, half_width > 0");
    if (selection != "all" && std::none_of(cases.begin(), cases.end(),
                                          [&](const Case& c) { return selection == c.name; }))
        throw std::invalid_argument("Unknown case: use all, material_dipole, material_quadrupole, "
                                    "size_dipole, or size_quadrupole");

    const int fixed_N = Ns.back(), fixed_W = Ws.back();
    for (const auto& c : cases) {
        if (selection != "all" && selection != c.name) continue;
        const std::string filename = "colbrook_convergence_" + std::string(c.name) + ".csv";
        std::ofstream out;
        out.exceptions(std::ios::failbit | std::ios::badbit);
        out.open(filename);
        out.precision(std::numeric_limits<double>::max_digits10);
        out << "case,sweep,N,W,N_colbrook,fringe,m_ang,radius,defect_radius,v,v_b,v_bd,"
               "points_per_disk,multipole_order,target_z,search_lo,search_hi,intervals,passes,"
               "z_min,sigma_min,z_resolution,at_boundary\n";
        auto save = [&](const char* sweep, int N, int W, const workflows::ColbrookMinimum& result) {
            out << c.name << ',' << sweep << ',' << N << ',' << W << ',' << W << ",1,"
                << c.m_ang << ",0.35," << c.defect_radius << ",1,1," << c.v_bd << ','
                << npd << ',' << multipole_order << ',' << c.target << ','
                << c.target - half_width << ',' << c.target + half_width << ','
                << intervals << ',' << passes << ',' << result.z << ',' << result.sigma_min
                << ',' << result.z_resolution << ',' << result.at_boundary << '\n';
            out.flush(); // Preserve completed points during long window sweeps.
            std::cout << c.name << " sweep=" << sweep << " N=" << N << " W=" << W
                      << " z=" << result.z << " sigma_min=" << result.sigma_min
                      << " dz=" << result.z_resolution
                      << (result.at_boundary ? " [search boundary: widen interval]" : "") << std::endl;
        };

        // One set of local solves per W and case. The largest W also supplies the N sweep.
        for (int W : Ws) {
            std::cout << "Building " << c.name << " templates: W=" << W << std::endl;
            const auto patch = workflows::build_tessellated_patch(W, fixed_N, W, true,
                0.35, c.defect_radius, c.m_ang, 1, npd, 1.0, 1.0, c.v_bd, multipole_order, false);
            const auto result = workflows::find_colbrook_minimum(patch.bigC, patch.modes * W,
                c.target, half_width, intervals, passes);
            save("W", fixed_N, W, result);
            if (W == fixed_W) {
                for (int N : Ns) {
                    if (N == fixed_N) {
                        save("N", N, W, result); // Same point belongs to both sweeps.
                    } else {
                        const auto C = workflows::assemble_tessellated_columns(
                            patch.patch_colC, patch.row_offsets, N, W);
                        save("N", N, W, workflows::find_colbrook_minimum(C, patch.modes * W,
                            c.target, half_width, intervals, passes));
                    }
                }
            }
        }
        out.close();
        std::cout << "[wrote] " << filename << std::endl;
    }
    return 0;
}

void save_singular_vector(const std::string& filename, const VectorXcd& vector,
                          int radius, int modes, double z, double sigma) {
    std::ofstream out;
    out.exceptions(std::ios::failbit | std::ios::badbit);
    out.open(filename);
    out.precision(std::numeric_limits<double>::max_digits10);
    out << "# z=" << z << ", sigma_min=" << sigma << '\n';
    out << "ell,basis,real,imag\n";
    for (int ell = -radius; ell <= radius; ++ell)
        for (int s = 0; s < modes; ++s) {
            const int row = (ell + radius) * modes + s;
            out << ell << ',' << s << ',' << vector(row).real() << ',' << vector(row).imag() << '\n';
        }
    out.close();
    std::cout << "[wrote] " << filename << " at z=" << z << ", sigma_min=" << sigma << '\n';
}

void print_usage(const char* prog) {
    std::cout
        << "Usage: " << prog << " <mode> [options]\n\n"
        << "Wave speeds (trailing optional args, default 1.0 except hexagonal-dirac v_b=0.1):\n"
        << "  v = background medium, v_b = crystal resonator interior, v_bd = defect interior.\n\n"
        << "Modes:\n"
        << "  line-defect-M        Subwavelength line-defect band via the multipole operator M^eps.\n"
        << "                       Optional: <num_alpha> [n_gauss] [v] [v_b] [v_bd]\n"
        << "  defect-capacitance   Higher-frequency line-defect bands (m = 0..max_m) via the\n"
        << "                       frequency-dependent capacitance matrix + exact transmission\n"
        << "                       solve. Optional: <max_m> <n_rows> <points_per_disk>\n"
        << "                       <num_alpha> [alpha_lo_frac] [alpha_hi_frac] [v] [v_b] [v_bd]\n"
        << "  projected-bulk       Projected bulk spectrum over alpha_y for the line-defect\n"
        << "                       diagrams. Optional: <num_alpha_x> <n_omega> <n_alpha_y>\n"
        << "                       [omega_lo] [omega_hi] [delta] [omega_imag] [v] [v_b]\n"
        << "  line-defect-neumann  Higher-freq line-defect bands from M^eps, searched around the\n"
        << "                       defect Neumann resonances (compare to defect-capacitance).\n"
        << "                       Optional: <max_m> <num_alpha> <n_multipole> <window_factor>\n"
        << "                       [v] [v_b] [v_bd]\n"
        << "  crystal-matrix-bands Bulk M-Gamma-X-M bands from CrystalA and multipoleA.\n"
        << "                       Optional: <points_per_segment> <n_omega> <points_per_disk> <n_multipole>\n"
        << "                       [v] [v_b]\n"
        << "  hexagonal-dirac     Direct local scan near K; defaults target omega=2.546138.\n"
        << "                       Optional: [n_omega=121] [n_alpha=21] [omega_lo=2.546132]\n"
        << "                       [omega_hi=2.546144] [omega_imag=1e-8] [radius=0.15]\n"
        << "                       [alpha_x=4pi/3] [alpha_y=0] [r1=0.12] [r2=0.12]\n"
        << "                       [delta=0.001] [v=1] [v_b=0.1]\n"
        << "  patch-capacitance    Real-space capacitance matrix of a finite, CENTERED patch of the\n"
        << "                       line defect (no Floquet). Optional: <n_defect> <n_clad>\n"
        << "                       <points_per_disk> <m_ang> <fringe> [v] [v_b] [v_bd]\n"
        << "  bent-waveguide       Real-space capacitance couplings of an L-shaped (90 deg bent)\n"
        << "                       defect chain in a finite patch. Optional: <n_defect> <n_clad>\n"
        << "                       <fringe> <points_per_disk> <m_ang> [v] [v_b] [v_bd]\n"
        << "  bent-waveguide-field Field u(r) of one eigenmode of that patch, reconstructed from\n"
        << "                       the capacitance eigenvector. Optional: <n_defect> <n_clad>\n"
        << "                       <fringe> <points_per_disk> <m_ang> <mode_index> <grid_points>\n"
        << "                       [v] [v_b] [v_bd]\n"
        << "                       mode_index < 0 picks the most corner-localized mode.\n"
        << "  tessellated-waveguide-field Save local central-source basis fields for each window.\n"
        << "                       Optional: [window_radius=2] [patch_radius=8] [colbrook=2]\n"
        << "                       [fringe=1] [points_per_disk=32] [m_ang=1] [grid_points=101]\n"
        << "                       [bent=1] [v=1] [v_b=1] [v_bd=1]\n"
        << "  bent-waveguide-exact Exact resonances omega* of that patch: Muller's method on the\n"
        << "                       full transmission operator A(omega), seeded by the capacitance\n"
        << "                       predictions; draws the null-vector field. Optional: <n_defect>\n"
        << "                       <n_clad> <fringe> <points_per_disk> <m_ang> <mode_index>\n"
        << "                       <grid_points> <delta> <seed_re> <seed_im> [v] [v_b] [v_bd]\n"
        << "  colbrook-tests      Tessellated bent-guide singular-value scans; radius increases by 2.\n"
        << "  colbrook-field      Scan, save the minimum right singular vector at each radius,\n"
        << "                       and export cached field bases for the last radius (alias: interface-modes).\n"
        << "                       Both accept: [window_radius=4] [colbrook=4] [min_radius=12]\n"
        << "                       [max_radius=20] [num_intervals=15000] [points_per_disk=32]\n"
        << "                       [m_ang=2] [z_min=0] [z_max=auto] [radius=0.35] [defect_radius=0.35]\n"
        << "                       [v=1] [v_b=1] [v_bd=0.35/0.455] [grid_points=101] [fringe=1]\n"
        << "  colbrook-convergence Track a local minimum for all four size/material mode cases.\n"
        << "                       Optional: [case=all] [N_list=8,12,16,20,30,40] [W_list=1,2,3,4,5,6]\n"
        << "                       [points_per_disk=32] [intervals=100] [passes=3] [half_width=0.2]\n"
        << "                       [multipole_order=10]. N_colbrook=W, fringe=1.\n"
        << "                       Cases: material_dipole, material_quadrupole, size_dipole, size_quadrupole.\n"
        << "  help, -h, --help     Show this message.\n";
}

}  // namespace

int main(int argc, char** argv) {
    const std::string mode = (argc > 1) ? argv[1] : "help";

    if (mode == "colbrook-convergence") {
        try {
            return run_colbrook_convergence(argc, argv);
        } catch (const std::exception& error) {
            std::cerr << "colbrook-convergence: " << error.what() << '\n';
            return 1;
        }
    }

    // reference method
    if (mode == "line-defect-neumann") {
        // Optional argv: max_m [num_alpha] [n_multipole] [window_factor] [v] [v_b] [v_bd].
        // Same geometry as defect-capacitance.
        const int max_m     = (argc > 2) ? std::stoi(argv[2]) : 2;
        const int num_alpha = (argc > 3) ? std::stoi(argv[3]) : 31;
        const int n_mult    = (argc > 4) ? std::stoi(argv[4]) : 0;   // 0 = auto
        const double window = (argc > 5) ? std::stod(argv[5]) : 20.0;
        const double v    = (argc > 6) ? std::stod(argv[6]) : 1.0;
        const double v_b  = (argc > 7) ? std::stod(argv[7]) : 1.0;
        const double v_bd = (argc > 8) ? std::stod(argv[8]) : 1.0;//0.35 / 0.455; //1.0;
        workflows::run_line_defect_bands_neumann(0.35, 0.455, 0.001, max_m, window, num_alpha, 20,
                                                 n_mult, 0.5, 3.6, v, v_b, v_bd);
        // workflows::run_line_defect_bands_neumann(0.35, 0.35, 0.001, max_m, window, num_alpha, 20,
        //                                          n_mult, 0.5, 3.6, v, v_b, v_bd);
        return 0;
    }

    // capacitance matrix with the supercell method for a line defect
    if (mode == "defect-capacitance") {
        // Optional argv: max_m [n_rows] [points_per_disk] [num_alpha] [alpha_lo] [alpha_hi]
        //                [v] [v_b] [v_bd].
        const int max_m  = (argc > 2) ? std::stoi(argv[2]) : 2;
        const int n_rows = (argc > 3) ? std::stoi(argv[3]) : 4;
        const int npd    = (argc > 4) ? std::stoi(argv[4]) : 64;
        const int nal    = (argc > 5) ? std::stoi(argv[5]) : 61;
        const double alo = (argc > 6) ? std::stod(argv[6]) : 0.0;
        const double ahi = (argc > 7) ? std::stod(argv[7]) : 1.0;
        const double v    = (argc > 8) ? std::stod(argv[8]) : 1.0;
        const double v_b  = (argc > 9) ? std::stod(argv[9]) : 1.0;
        const double v_bd = (argc > 10) ? std::stod(argv[10]) : 1.0;//0.35 / 0.455; //1.0;
        workflows::run_defect_capacitance_bands(0.35, 0.455, 0.001, max_m, n_rows, npd, nal,
                                                alo, ahi, v, v_b, v_bd);
        // workflows::run_defect_capacitance_bands(0.35, 0.35, 0.001, max_m, n_rows, npd, nal,
        //                                         alo, ahi, v, v_b, v_bd);
        return 0;
    }

    // projected bulk bands for the unperturbed crystal
    if (mode == "projected-bulk") {
        // Optional argv: num_alpha_x [n_omega] [n_alpha_y] [omega_lo] [omega_hi] [delta]
        //                [omega_imag] [v] [v_b].
        const int nax      = (argc > 2) ? std::stoi(argv[2]) : 31;
        const int n_omega  = (argc > 3) ? std::stoi(argv[3]) : 500;
        const int nay      = (argc > 4) ? std::stoi(argv[4]) : 13;
        const double wlo   = (argc > 5) ? std::stod(argv[5]) : 0.0;
        const double whi   = (argc > 6) ? std::stod(argv[6]) : 8.0;
        const double delta = (argc > 7) ? std::stod(argv[7]) : 0.001;
        const double oimag = (argc > 8) ? std::stod(argv[8]) : 1e-4;
        const double v     = (argc > 9) ? std::stod(argv[9]) : 1.0;
        const double v_b   = (argc > 10) ? std::stod(argv[10]) : 1.0;
        workflows::run_projected_bulk_bands(0.35, delta, 7, nax, nay, wlo, whi, n_omega, oimag,
                                            v, v_b);
        return 0;
    }

    // comparison of the multipole method and the standard singular value search for an unperturbed crystal
    // the results are expected to be very similar, but much faster for the multipole method
    if (mode == "crystal-matrix-bands" || mode == "m-gamma-x-m") {
        // Optional argv: points_per_segment [n_omega] [points_per_disk] [n_multipole] [v] [v_b].
        const int points_per_segment = (argc > 2) ? std::stoi(argv[2]) : 13;
        const int n_omega            = (argc > 3) ? std::stoi(argv[3]) : 140;
        const int points_per_disk    = (argc > 4) ? std::stoi(argv[4]) : 12;
        const int n_multipole        = (argc > 5) ? std::stoi(argv[5]) : 8;
        const double v               = (argc > 6) ? std::stod(argv[6]) : 1.0;
        const double v_b             = (argc > 7) ? std::stod(argv[7]) : 1.0;
        workflows::run_crystal_matrix_bands(0.35, 0.001, points_per_disk, n_multipole,
                                            points_per_segment, 0.001, 7.5, n_omega, 1.0e-3,
                                            v, v_b);
        return 0;
    }

    // finite patch capacitance method for a line defect
    if (mode == "patch-capacitance") {
        // Optional argv: n_defect [n_clad] [points_per_disk] [m_ang] [fringe] [v] [v_b] [v_bd].
        const int n_defect = (argc > 2) ? std::stoi(argv[2]) : 5;
        const int n_clad   = (argc > 3) ? std::stoi(argv[3]) : 4;
        const int npd      = (argc > 4) ? std::stoi(argv[4]) : 64;
        const int m_ang    = (argc > 5) ? std::stoi(argv[5]) : 1;
        const int fringe   = (argc > 6) ? std::stoi(argv[6]) : 1;
        const double v    = (argc > 7) ? std::stod(argv[7]) : 1.0;
        const double v_b  = (argc > 8) ? std::stod(argv[8]) : 1.0;
        const double v_bd = (argc > 9) ? std::stod(argv[9]) : 1.0;//0.35 / 0.455; //1.0;
        workflows::run_patch_capacitance(0.35, 0.455, 0.001, m_ang, n_defect, n_clad, fringe, npd,
                                         v, v_b, v_bd);
        // workflows::run_patch_capacitance(0.35, 0.35, 0.001, m_ang, n_defect, n_clad, fringe, npd,
        //                                  v, v_b, v_bd);
        return 0;
    }

    // finite patch capacitance method for a bent waveguide (L-shaped) defect
    if (mode == "bent-waveguide") {
        // Optional argv: n_defect [n_clad] [fringe] [points_per_disk] [m_ang] [v] [v_b] [v_bd].
        const int n_defect = (argc > 2) ? std::stoi(argv[2]) : 3;
        const int n_clad   = (argc > 3) ? std::stoi(argv[3]) : 3;
        const int fringe   = (argc > 4) ? std::stoi(argv[4]) : 1;
        const int npd      = (argc > 5) ? std::stoi(argv[5]) : 64;
        const int m_ang    = (argc > 6) ? std::stoi(argv[6]) : 1;
        const double v    = (argc > 7) ? std::stod(argv[7]) : 1.0;
        const double v_b  = (argc > 8) ? std::stod(argv[8]) : 1.0;
        const double v_bd = (argc > 9) ? std::stod(argv[9]) : 1.0;//0.35 / 0.455; //1.0;
        workflows::run_bent_waveguide(0.35, 0.455, 0.001, m_ang, n_defect, n_clad, fringe, npd,
                                      v, v_b, v_bd);
        // workflows::run_bent_waveguide(0.35, 0.35, 0.001, m_ang, n_defect, n_clad, fringe, npd,
        //                               v, v_b, v_bd);
        return 0;
    }

    // same as bent-waveguide (reuses the code), but reconstructs a chosen approximate localized eigenmode
    if (mode == "bent-waveguide-field") {
        // Optional argv: n_defect [n_clad] [fringe] [points_per_disk] [m_ang] [mode_index]
        //                [grid_points] [v] [v_b] [v_bd].
        const int n_defect = (argc > 2) ? std::stoi(argv[2]) : 3;
        const int n_clad   = (argc > 3) ? std::stoi(argv[3]) : 3;
        const int fringe   = (argc > 4) ? std::stoi(argv[4]) : 1;
        const int npd      = (argc > 5) ? std::stoi(argv[5]) : 64;
        const int m_ang    = (argc > 6) ? std::stoi(argv[6]) : 1;
        // pick the most corner-localized mode (index < 0) or a specific mode (index >= 0)
        const int mode_idx = (argc > 7) ? std::stoi(argv[7]) : -1;
        const int ngrid    = (argc > 8) ? std::stoi(argv[8]) : 300;
        const double v    = (argc > 9) ? std::stod(argv[9]) : 1.0;
        const double v_b  = (argc > 10) ? std::stod(argv[10]) : 1.0;
        const double v_bd = (argc > 11) ? std::stod(argv[11]) : 1.0;//0.35 / 0.455; //1.0;
        workflows::run_bent_waveguide_field(0.35, 0.455, 0.001, m_ang, n_defect, n_clad, fringe,
                                            npd, mode_idx, ngrid, v, v_b, v_bd);
        // workflows::run_bent_waveguide_field(0.35, 0.35, 0.001, m_ang, n_defect, n_clad, fringe,
        //                                     npd, mode_idx, ngrid, v, v_b, v_bd);
        return 0;
    }


    if (mode == "tessellated-waveguide-field") {
        const int window_radius = (argc > 2) ? std::stoi(argv[2]) : 2;
        const int patch_radius = (argc > 3) ? std::stoi(argv[3]) : 8;
        const int colbrook = (argc > 4) ? std::stoi(argv[4]) : 2;
        const int fringe = (argc > 5) ? std::stoi(argv[5]) : 1;
        const int npd = (argc > 6) ? std::stoi(argv[6]) : 32;
        const int m_ang = (argc > 7) ? std::stoi(argv[7]) : 1;
        const int grid_points = (argc > 8) ? std::stoi(argv[8]) : 101;
        const bool bent = (argc > 9) ? std::stoi(argv[9]) != 0 : true;
        const double v = (argc > 10) ? std::stod(argv[10]) : 1.0;
        const double v_b = (argc > 11) ? std::stod(argv[11]) : 1.0;
        const double v_bd = (argc > 12) ? std::stod(argv[12]) : 0.35 / 0.455;
        const auto patch = workflows::build_tessellated_patch(window_radius, patch_radius,
            colbrook, bent, 0.35, 0.35, m_ang, fringe, npd, v, v_b, v_bd);
        workflows::run_tessellated_waveguide_field(patch, grid_points);
        return 0;
    }

    if (mode == "bent-waveguide-exact") {
        // Optional argv: n_defect [n_clad] [fringe] [points_per_disk] [m_ang] [mode_index]
        //                [grid_points] [delta] [seed_re] [seed_im] [v] [v_b] [v_bd].
        const int n_defect  = (argc > 2) ? std::stoi(argv[2]) : 3;
        const int n_clad    = (argc > 3) ? std::stoi(argv[3]) : 3;
        const int fringe    = (argc > 4) ? std::stoi(argv[4]) : 1;
        const int npd       = (argc > 5) ? std::stoi(argv[5]) : 64;
        const int m_ang     = (argc > 6) ? std::stoi(argv[6]) : 1;
        const int mode_idx  = (argc > 7) ? std::stoi(argv[7]) : -1;
        const int ngrid     = (argc > 8) ? std::stoi(argv[8]) : 200;
        const double delta  = (argc > 9) ? std::stod(argv[9]) : 0.001;
        const double sre    = (argc > 10) ? std::stod(argv[10]) : -1.0;
        const double sim    = (argc > 11) ? std::stod(argv[11]) : 0.0;
        const double v      = (argc > 12) ? std::stod(argv[12]) : 1.0;
        const double v_b    = (argc > 13) ? std::stod(argv[13]) : 1.0;
        const double v_bd   = (argc > 14) ? std::stod(argv[14]) : 0.35 / 0.455; //1.0;
        // workflows::run_bent_waveguide_exact(0.35, 0.455, delta, m_ang, n_defect, n_clad, fringe,
        //                                     npd, mode_idx, ngrid, sre, sim, v, v_b, v_bd);
        workflows::run_bent_waveguide_exact(0.35, 0.35, delta, m_ang, n_defect, n_clad, fringe,
                                            npd, mode_idx, ngrid, sre, sim, v, v_b, v_bd);
        return 0;
    }

    // Direct local scan: avoid the heuristic candidate search when the center is known.
    if (mode == "hexagonal-dirac") {
        const int n_omega = (argc > 2) ? std::stoi(argv[2]) : 121;
        const int n_alpha = (argc > 3) ? std::stoi(argv[3]) : 21;
        const double omega_lo = (argc > 4) ? std::stod(argv[4]) : 2.546132;
        const double omega_hi = (argc > 5) ? std::stod(argv[5]) : 2.546144;
        const double omega_imag = (argc > 6) ? std::stod(argv[6]) : 1e-8;
        const double radius = (argc > 7) ? std::stod(argv[7]) : 0.15;
        const double alpha_x = (argc > 8) ? std::stod(argv[8]) : 4 * M_PI / 3;
        const double alpha_y = (argc > 9) ? std::stod(argv[9]) : 0.0;
        const double r1 = (argc > 10) ? std::stod(argv[10]) : 0.12;
        const double r2 = (argc > 11) ? std::stod(argv[11]) : 0.12;
        const double delta = (argc > 12) ? std::stod(argv[12]) : 0.001;
        const double v = (argc > 13) ? std::stod(argv[13]) : 1.0;
        const double v_b = (argc > 14) ? std::stod(argv[14]) : 0.1;
        BoundaryMesh mesh(36), mesh2(36);
        mesh.generate_circle(r1, Vector2d(0.5, sqrt(3) / 6));
        mesh2.generate_circle(r2, Vector2d(1., sqrt(3) / 3));
        mesh.add_mesh(mesh2);
        Crystal crystal(mesh, delta, Vector2d(1, 0), Vector2d(0.5, sqrt(3) / 2));
        crystal.Rs = {r1, r2};
        crystal.shifts = {Vector2d(0.5, sqrt(3) / 6), Vector2d(1., sqrt(3) / 3)};
        crystal.circles_mesh = true;
        crystal.v = v;
        crystal.v_b = v_b;
        crystal.compute_local_bands(omega_lo, omega_hi, n_omega, omega_imag,
                                    "multipoleA_dirac_local.csv", Vector2d(alpha_x, alpha_y),
                                    radius, n_alpha);
        return 0;
    }

    // brand new experiment: hexagonal lattice of circular disks, compute the high-symmetry bands, the python file is now plotting this
    if (mode == "hexagonal-lattice") {
        const int n_omega = (argc > 2) ? std::stoi(argv[2]) : 20;
        const int n_path = (argc > 3) ? std::stoi(argv[3]) : 20;
        const int n_alpha = (argc > 4) ? std::stoi(argv[4]) : 20;
        const double r1 = (argc > 5) ? std::stod(argv[5]) : 0.12;
        const double r2 = (argc > 6) ? std::stod(argv[6]) : 0.12;
        const double omega_lo = (argc > 7) ? std::stod(argv[7]) : 0.0;
        const double omega_hi = (argc > 8) ? std::stod(argv[8]) : 5.0;
        const double delta = (argc > 9) ? std::stod(argv[9]) : 0.001;
        const double v = (argc > 10) ? std::stod(argv[10]) : 1.0;
        const double v_b = (argc > 11) ? std::stod(argv[11]) : 1.0;
        std::cout << "Computing high-symmetry bands for a hexagonal lattice of circular disks with parameters:\n"
                  << "  omega in [" << omega_lo << "," << omega_hi << "], " << n_omega << " omega samples, delta = " << delta
                  << ", v = " << v << ", v_b = " << v_b << "\n";
        int N = 36;
        BoundaryMesh mesh(N), mesh2(N);
        mesh.generate_circle(r1, Vector2d(0.5, sqrt(3) / 6));
        mesh2.generate_circle(r2, Vector2d(1., sqrt(3) / 3));
        mesh.add_mesh(mesh2);
        Vector2d a1(1.0, 0.0), a2(0.5, sqrt(3) / 2);

        Crystal crystal(mesh, delta, a1, a2);
        crystal.Rs = {r1, r2};
        crystal.shifts = {Vector2d(0.5, sqrt(3) / 6), Vector2d(1., sqrt(3) / 3)};
        crystal.circles_mesh = true;
        crystal.v = v;
        crystal.v_b = v_b;

        crystal.compute_high_symmetry_bands(omega_lo, omega_hi, n_omega, 1e-6, "crystal_hex_bands.csv", n_path, false);
        // workflows::capacitance_dirac_bands(crystal, 0, 2, n_alpha);

        // ./bin/WaveguideModes hexagonal-lattice 200 20 20 0.12 0.12 3.1925 3.1936 0.001 1.0 0.1
        // ./bin/WaveguideModes hexagonal-lattice 300 20 20 0.12 0.12 1.5345 1.5354 0.001 1.0 0.1
        // ./bin/WaveguideModes hexagonal-lattice 300 20 20 0.12 0.12 2.5460 2.5463 0.001 1.0 0.1

        return 0;
    }

    if (mode == "colbrook-tests" || mode == "colbrook-field" || mode == "interface-modes") {
        const bool export_field = mode != "colbrook-tests";
        const int window_radius = (argc > 2) ? std::stoi(argv[2]) : 4;
        const int colbrook = (argc > 3) ? std::stoi(argv[3]) : 4;
        const int min_radius = (argc > 4) ? std::stoi(argv[4]) : 12;
        const int max_radius = (argc > 5) ? std::stoi(argv[5]) : 20;
        const int Nz = (argc > 6) ? std::stoi(argv[6]) : 15000;
        const int npd = (argc > 7) ? std::stoi(argv[7]) : 32;
        const int m_ang = (argc > 8) ? std::stoi(argv[8]) : 2;
        const double z_min = (argc > 9) ? std::stod(argv[9]) : 0.0;
        double z_max = (argc > 10) ? std::stod(argv[10]) : 10.0;
        const double r = (argc > 11) ? std::stod(argv[11]) : 0.35;
        const double r_def = (argc > 12) ? std::stod(argv[12]) : 0.35; //0.455;
        const double v = (argc > 13) ? std::stod(argv[13]) : 1.0;
        const double v_b = (argc > 14) ? std::stod(argv[14]) : 1.0;
        const double v_bd = (argc > 15) ? std::stod(argv[15]) : 0.35 / 0.455;
        const int grid_points = (argc > 16) ? std::stoi(argv[16]) : 801;
        const int fringe = (argc > 17) ? std::stoi(argv[17]) : 1;
        if (window_radius < 0 || min_radius < window_radius || max_radius < min_radius ||
            colbrook < 0 || fringe < 0 || m_ang < 0 || Nz < 1 || npd < 8 || grid_points < 16 ||
            !std::isfinite(z_min) || (argc > 10 && (!std::isfinite(z_max) || z_max <= z_min))) {
            std::cerr << "Require 0 <= window_radius <= min_radius <= max_radius, nonnegative buffers "
                         "and angular order, num_intervals >= 1, points_per_disk >= 8, "
                         "grid_points >= 16, and a finite increasing z range.\n";
            return 1;
        }

        // The expensive local solves are independent of the assembled patch radius.
        auto p = workflows::build_tessellated_patch(window_radius, min_radius, colbrook,
            true, r, r_def, m_ang, fringe, npd, v, v_b, v_bd);
        if (!p.bigC.allFinite())
            throw std::runtime_error("Tessellated operator contains non-finite entries");
        const int row_offset = p.modes * colbrook;
        if (argc <= 10) {
            // Only use eigenvalues of the SQUARE section as a heuristic scan bound.
            // Keep all rows of bigC for the actual rectangular singular-value calculation.
            const MatrixXcd squareC = p.bigC.middleRows(row_offset, p.bigC.cols());
            ComplexEigenSolver<MatrixXcd> es(squareC, /*computeEigenvectors=*/false);
            if (es.info() != Eigen::Success || !es.eigenvalues().allFinite())
                throw std::runtime_error("Failed to estimate scan bound; supply z_min and z_max explicitly");
            z_max = 1.5 * es.eigenvalues().real().maxCoeff();
        }
        if (!std::isfinite(z_max) || z_max <= z_min)
            throw std::runtime_error("Invalid scan bound; supply z_min and z_max explicitly");
        std::cout << "Colbrook scan: z in [" << z_min << ", " << z_max << "], "
                  << Nz + 1 << " samples per radius\n";

        for (int M = min_radius; M <= max_radius; M+=2) {
            if (M != min_radius)
                p.bigC = workflows::assemble_tessellated_columns(p.patch_colC, p.row_offsets, M, colbrook);
            p.patch_radius = M;  // Keep the exported sites and coefficient indices consistent.
            const MatrixXcd& C = p.bigC;
            // Hermiticity compares matching input/output sites; retain the extra rows for the scan.
            const MatrixXcd squareC = C.middleRows(row_offset, C.cols());
            const double hermiticity_defect = (squareC - squareC.adjoint()).norm();
            cout << "Size: " << C.rows() << " x " << C.cols()
                 << ", square-section relative Hermiticity defect: "
                 << (squareC.norm() > 0.0 ? hermiticity_defect / squareC.norm() : 0.0)
                 << ", absolute defect: " << hermiticity_defect << "\n";
            MatrixXcd E = MatrixXcd::Zero(C.rows(), C.cols());
            E.block(row_offset, 0, C.cols(), C.cols()).setIdentity();
            const std::string filename = "bent_waveguide_singular_values" + std::to_string(M) + ".csv";
            std::ofstream out_z;
            out_z.exceptions(std::ios::failbit | std::ios::badbit);
            out_z.open(filename);
            out_z.precision(std::numeric_limits<double>::max_digits10);
            double best_sigma = std::numeric_limits<double>::infinity();
            double best_z = z_min;
            VectorXcd best_vector;
            std::cout << "Scanning radius " << M << ": " << C.rows() << " x " << C.cols() << "\n";
            for (int zi = 0; zi <= Nz; ++zi) {
                // An indexed grid gives every radius the same samples and includes both endpoints.
                const double z = std::lerp(z_min, z_max, double(zi) / double(Nz));
                const MatrixXcd shifted = C - z * E;
                Eigen::BDCSVD<MatrixXcd> svd(shifted, Eigen::ComputeThinV); 
                if (svd.info() != Eigen::Success || !svd.singularValues().allFinite())
                    throw std::runtime_error("Singular-value scan failed at z=" + std::to_string(z));
                out_z << z << "," << svd.singularValues().minCoeff() << "\n";
                if (export_field && svd.singularValues().minCoeff() < best_sigma) {
                    best_sigma = svd.singularValues().minCoeff();
                    best_z = z;
                    best_vector = svd.matrixV().col(svd.singularValues().size() - 1);
                }

                // 5.621 quad r
                // 5.471 quad m
                // 1.733 dip m

                if (!export_field && abs(z - 1.733) < 1e-3) {
                    // Get the last singular vector
                    VectorXcd v_z = svd.matrixV().col(svd.singularValues().size() - 1);
                    const std::string filename_v = "bent_waveguide_singular_vector_" + std::to_string(z) + ".csv";
                    save_singular_vector(filename_v, v_z, M, p.modes, z, svd.singularValues().minCoeff());
                    return 0;
                }
            }
            out_z.close();
            std::cout << "[wrote] " << filename << "\n";
            if (export_field)
                save_singular_vector("bent_waveguide_singular_vector_radius_" + std::to_string(M) + ".csv",
                                     best_vector, M, p.modes, best_z, best_sigma);
            // if (M > max_radius / 2) break;
        }
        if (export_field) {
            workflows::run_tessellated_waveguide_field(p, grid_points);
            std::cout << "Plot with: python src/plot_tessellated_waveguide_field.py "
                      << "bent_waveguide_singular_vector_radius_" << p.patch_radius << ".csv --no-show\n";
        }
        return 0;

        int n_min = 1, n_max = 6;
        for (int n = n_min; n <= n_max; ++n) {
            const std::string filename = "bent_waveguide_singular_values_window_" + std::to_string(n) + ".csv";
            std::ofstream out_z;
            out_z.exceptions(std::ios::failbit | std::ios::badbit);
            out_z.open(filename);
            
            const auto p = workflows::build_tessellated_patch(n, 20, 0,
                                    true, 0.35, 0.455, m_ang, 1, npd, 1.0, 1.0, 1.0);

            cout << "Hermiticity: " << (p.bigC - p.bigC.adjoint()).norm() / p.bigC.norm() << "\n";

            const MatrixXcd C = workflows::assemble_tessellated_columns(p.patch_colC, p.row_offsets, n, colbrook);
            MatrixXcd E = MatrixXcd::Zero(C.rows(), C.cols());
            E.block(row_offset, 0, C.cols(), C.cols()).setIdentity();

            for (int zi = 0; zi <= Nz; ++zi) {
                const double z = std::lerp(z_min, z_max, double(zi) / double(Nz));
                const MatrixXcd shifted = C - z * E;
                Eigen::BDCSVD<MatrixXcd> svd(shifted); 
                if (svd.info() != Eigen::Success || !svd.singularValues().allFinite())
                    throw std::runtime_error("Singular-value scan failed at z=" + std::to_string(z));
                out_z << z << "," << svd.singularValues().minCoeff() << "\n";
            }
            out_z.close();
            std::cout << "[wrote] " << filename << "\n";
        }
        // int n_min = 0, n_max = 6;
        // std::cout.precision(std::numeric_limits<double>::max_digits10);
        // const auto p2 = workflows::build_tessellated_patch(4, 7, colbrook,
        //                             true, 0.35, 0.455, m_ang, 1, npd, 1.0, 1.0, 1.0);
        // for (int n = n_min; n <= n_max; ++n) {
        //     const std::string filename = "bent_waveguide_singular_values_colbr_" + std::to_string(n) + ".csv";
        //     std::ofstream out_z;
        //     out_z.exceptions(std::ios::failbit | std::ios::badbit);
        //     out_z.open(filename);
        //     out_z.precision(std::numeric_limits<double>::max_digits10);

        //     int row_offset = p2.modes * n;
        //     const MatrixXcd C = workflows::assemble_tessellated_columns(p2.patch_colC, p2.row_offsets, 4, n);
        //     MatrixXcd E = MatrixXcd::Zero(C.rows(), C.cols());
        //     E.block(row_offset, 0, C.cols(), C.cols()).setIdentity();
        //     for (int j = 0; j < C.rows(); j += p2.modes) {
        //         const int ell = j / p2.modes - (4 + n);
        //         cout << "Row ell=" << ell << ": " << C.block(j, 0, p2.modes, C.cols()).norm() << "\n";
        //     }
        //     // cout << "Scanning Colbrook padding " << n << ": " << C.rows() << " x " << C.cols() << "\n";

        //     for (int zi = 0; zi <= Nz; ++zi) {
        //         const double z = std::lerp(z_min, z_max, double(zi) / double(Nz));
        //         const MatrixXcd shifted = C - z * E;
        //         Eigen::BDCSVD<MatrixXcd> svd(shifted, Eigen::ComputeThinV); 
        //         if (svd.info() != Eigen::Success || !svd.singularValues().allFinite())
        //             throw std::runtime_error("Singular-value scan failed at z=" + std::to_string(z));
        //         out_z << z << "," << svd.singularValues().minCoeff() << "\n";

        //         if (abs(z - 1.258) < 1e-3 || abs(z - 3.354) < 1e-3 || abs(z - 9.052) < 1e-3) {
        //             cout << "(C - zE) * v_z and v_z at z = " << z << ": ";

        //             VectorXcd v_z = svd.matrixV().col(svd.singularValues().size() - 1);

        //             cout << "v_z = " << v_z.transpose() << "\n";
        //             const VectorXcd res = shifted * v_z;
        //             cout << "sigma_min = " << svd.singularValues().minCoeff()
        //                  << ", residual norm = " << res.norm() << "\n";
        //             double tail_squared = 0.0;
        //             for (int j = 0; j < res.rows(); j += p2.modes) {
        //                 const int ell = j / p2.modes - (4 + n);
        //                 const double block_norm = res.segment(j, p2.modes).norm();
        //                 cout << "Residual ell=" << ell << ": " << block_norm << "\n";
        //                 // n=1 already contains ell=-5..5. Isolate the rows added beyond it.
        //                 if (std::abs(ell) > 5) tail_squared += block_norm * block_norm;
        //             }
        //             cout << "Squared residual outside n=1 rows = " << tail_squared << "\n";
                    
        //             cout << "\n";
        //         }
        //     }
        //     out_z.close();
        //     std::cout << "[wrote] " << filename << "\n";
        // }
        return 0;
    }

    print_usage(argv[0]);
    return (mode == "help" || mode == "-h" || mode == "--help") ? 0 : 1;
}
