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

namespace {

void print_usage(const char* prog) {
    std::cout
        << "Usage: " << prog << " <mode> [options]\n\n"
        << "Wave speeds (each mode takes them as trailing optional args, all default 1.0):\n"
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
        << "  bent-waveguide-exact Exact resonances omega* of that patch: Muller's method on the\n"
        << "                       full transmission operator A(omega), seeded by the capacitance\n"
        << "                       predictions; draws the null-vector field. Optional: <n_defect>\n"
        << "                       <n_clad> <fringe> <points_per_disk> <m_ang> <mode_index>\n"
        << "                       <grid_points> <delta> <seed_re> <seed_im> [v] [v_b] [v_bd]\n"
        << "  help, -h, --help     Show this message.\n";
}

}  // namespace

int main(int argc, char** argv) {
    const std::string mode = (argc > 1) ? argv[1] : "help";

    if (mode == "line-defect-M") {
        // Optional argv: num_alpha [n_gauss] [v] [v_b] [v_bd].
        const int num_alpha = (argc > 2) ? std::stoi(argv[2]) : 30;
        const int n_gauss = (argc > 3) ? std::stoi(argv[3]) : 21;
        const double v    = (argc > 4) ? std::stod(argv[4]) : 1.0;
        const double v_b  = (argc > 5) ? std::stod(argv[5]) : 1.0;
        const double v_bd = (argc > 6) ? std::stod(argv[6]) : 0.35 / 0.455; //1.0;
        // workflows::run_line_defect_bands_M(0.35, 0.455, 0.001, 0.20, 0.27, 1.5, 3.5, 0.31,
        //                                    num_alpha, n_gauss, v, v_b, v_bd);
        workflows::run_line_defect_bands_M(0.35, 0.35, 0.001, 0.20, 0.27, 1.5, 3.5, 0.31,
                                           num_alpha, n_gauss, v, v_b, v_bd);
        return 0;
    }

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
        const double v_bd = (argc > 10) ? std::stod(argv[10]) : 0.35 / 0.455; //1.0;
        // workflows::run_defect_capacitance_bands(0.35, 0.455, 0.001, max_m, n_rows, npd, nal,
        //                                         alo, ahi, v, v_b, v_bd);
        workflows::run_defect_capacitance_bands(0.35, 0.35, 0.001, max_m, n_rows, npd, nal,
                                                alo, ahi, v, v_b, v_bd);
        return 0;
    }

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

    if (mode == "line-defect-neumann") {
        // Optional argv: max_m [num_alpha] [n_multipole] [window_factor] [v] [v_b] [v_bd].
        // Same geometry as defect-capacitance.
        const int max_m     = (argc > 2) ? std::stoi(argv[2]) : 2;
        const int num_alpha = (argc > 3) ? std::stoi(argv[3]) : 31;
        const int n_mult    = (argc > 4) ? std::stoi(argv[4]) : 0;   // 0 = auto
        const double window = (argc > 5) ? std::stod(argv[5]) : 20.0;
        const double v    = (argc > 6) ? std::stod(argv[6]) : 1.0;
        const double v_b  = (argc > 7) ? std::stod(argv[7]) : 1.0;
        const double v_bd = (argc > 8) ? std::stod(argv[8]) : 0.35 / 0.455; //1.0;
        // workflows::run_line_defect_bands_neumann(0.35, 0.455, 0.001, max_m, window, num_alpha, 20,
        //                                          n_mult, 0.5, 3.6, v, v_b, v_bd);
        workflows::run_line_defect_bands_neumann(0.35, 0.35, 0.001, max_m, window, num_alpha, 20,
                                                 n_mult, 0.5, 3.6, v, v_b, v_bd);
        return 0;
    }

    if (mode == "crystal-matrix-bands" || mode == "m-gamma-x-m") {
        // Optional argv: points_per_segment [n_omega] [points_per_disk] [n_multipole] [v] [v_b].
        const int points_per_segment = (argc > 2) ? std::stoi(argv[2]) : 13;
        const int n_omega            = (argc > 3) ? std::stoi(argv[3]) : 140;
        const int points_per_disk    = (argc > 4) ? std::stoi(argv[4]) : 12;
        const int n_multipole        = (argc > 5) ? std::stoi(argv[5]) : 8;
        const double v               = (argc > 6) ? std::stod(argv[6]) : 1.0;
        const double v_b             = (argc > 7) ? std::stod(argv[7]) : 1.0;
        workflows::run_crystal_matrix_bands(0.35, 5.0e-2, points_per_disk, n_multipole,
                                            points_per_segment, 0.001, 7.5, n_omega, 1.0e-3,
                                            v, v_b);
        return 0;
    }

    if (mode == "patch-capacitance") {
        // Optional argv: n_defect [n_clad] [points_per_disk] [m_ang] [fringe] [v] [v_b] [v_bd].
        const int n_defect = (argc > 2) ? std::stoi(argv[2]) : 5;
        const int n_clad   = (argc > 3) ? std::stoi(argv[3]) : 4;
        const int npd      = (argc > 4) ? std::stoi(argv[4]) : 64;
        const int m_ang    = (argc > 5) ? std::stoi(argv[5]) : 1;
        const int fringe   = (argc > 6) ? std::stoi(argv[6]) : 1;
        const double v    = (argc > 7) ? std::stod(argv[7]) : 1.0;
        const double v_b  = (argc > 8) ? std::stod(argv[8]) : 1.0;
        const double v_bd = (argc > 9) ? std::stod(argv[9]) : 0.35 / 0.455; //1.0;
        // workflows::run_patch_capacitance(0.35, 0.455, 0.001, m_ang, n_defect, n_clad, fringe, npd,
        //                                  v, v_b, v_bd);
        workflows::run_patch_capacitance(0.35, 0.35, 0.001, m_ang, n_defect, n_clad, fringe, npd,
                                         v, v_b, v_bd);
        return 0;
    }

    if (mode == "bent-waveguide") {
        // Optional argv: n_defect [n_clad] [fringe] [points_per_disk] [m_ang] [v] [v_b] [v_bd].
        const int n_defect = (argc > 2) ? std::stoi(argv[2]) : 3;
        const int n_clad   = (argc > 3) ? std::stoi(argv[3]) : 3;
        const int fringe   = (argc > 4) ? std::stoi(argv[4]) : 1;
        const int npd      = (argc > 5) ? std::stoi(argv[5]) : 64;
        const int m_ang    = (argc > 6) ? std::stoi(argv[6]) : 1;
        const double v    = (argc > 7) ? std::stod(argv[7]) : 1.0;
        const double v_b  = (argc > 8) ? std::stod(argv[8]) : 1.0;
        const double v_bd = (argc > 9) ? std::stod(argv[9]) : 0.35 / 0.455; //1.0;
        // workflows::run_bent_waveguide(0.35, 0.455, 0.001, m_ang, n_defect, n_clad, fringe, npd,
        //                               v, v_b, v_bd);
        workflows::run_bent_waveguide(0.35, 0.35, 0.001, m_ang, n_defect, n_clad, fringe, npd,
                                      v, v_b, v_bd);
        return 0;
    }

    if (mode == "bent-waveguide-field") {
        // Optional argv: n_defect [n_clad] [fringe] [points_per_disk] [m_ang] [mode_index]
        //                [grid_points] [v] [v_b] [v_bd].
        const int n_defect = (argc > 2) ? std::stoi(argv[2]) : 3;
        const int n_clad   = (argc > 3) ? std::stoi(argv[3]) : 3;
        const int fringe   = (argc > 4) ? std::stoi(argv[4]) : 1;
        const int npd      = (argc > 5) ? std::stoi(argv[5]) : 64;
        const int m_ang    = (argc > 6) ? std::stoi(argv[6]) : 1;
        const int mode_idx = (argc > 7) ? std::stoi(argv[7]) : -1;
        const int ngrid    = (argc > 8) ? std::stoi(argv[8]) : 300;
        const double v    = (argc > 9) ? std::stod(argv[9]) : 1.0;
        const double v_b  = (argc > 10) ? std::stod(argv[10]) : 1.0;
        const double v_bd = (argc > 11) ? std::stod(argv[11]) : 0.35 / 0.455; //1.0;
        // workflows::run_bent_waveguide_field(0.35, 0.455, 0.001, m_ang, n_defect, n_clad, fringe,
        //                                     npd, mode_idx, ngrid, v, v_b, v_bd);
        workflows::run_bent_waveguide_field(0.35, 0.35, 0.001, m_ang, n_defect, n_clad, fringe,
                                            npd, mode_idx, ngrid, v, v_b, v_bd);
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

    print_usage(argv[0]);
    return (mode == "help" || mode == "-h" || mode == "--help") ? 0 : 1;
}
