"""Overlay the real-space capacitance couplings C_l computed two independent ways:

  * FINITE PATCH (no Floquet): build/bent_waveguide_couplings.csv, the center resonator's row of
    blocks C_{center, center+l} from `SubwavelengthResonators patch-capacitance`;
  * INVERSE FLOQUET: the quasi-periodic C^alpha (capacitance_matrix_per_alpha.csv from
    `SubwavelengthResonators defect-capacitance`) inverse-Fourier-transformed over the Bloch
    phase, C_l = (1/M) sum_k C^{alpha_k} e^{-i alpha_k l}.

If the two agree in the bulk (small |l|), truncating the infinite defect line to a finite patch
is justified -- the couplings decay fast enough that the far ones don't matter. Run both modes at
the SAME geometry first (defaults match: R=0.35, R_def=0.455, delta=0.05, dipole, 4 cladding rows):
    ./build/bin/SubwavelengthResonators defect-capacitance 1 4 16 41
    ./build/bin/SubwavelengthResonators patch-capacitance 10 4 16
    python src/plot_patch_vs_floquet.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path.cwd()
TARGET_FI = 1  # family index in capacitance_matrix_per_alpha.csv: 0 = breathing, 1 = dipole, ...

plt.rcParams.update({
        "font.size": 18,
        "axes.labelsize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "mathtext.fontset": "cm",
        "axes.linewidth": 1.2,
    })

def resolve(name):
    p = DATA_ROOT / name
    return p if p.exists() else SCRIPT_ROOT / name

def main():
    filenames = [resolve("build/bent_waveguide_couplings10_10_1.csv"),
                 resolve("build/bent_waveguide_couplings9_9_1.csv"),
                 resolve("build/bent_waveguide_couplings8_8_1.csv"),
                 resolve("build/bent_waveguide_couplings7_7_1.csv"),
                 resolve("build/bent_waveguide_couplings6_6_1.csv"),
                 resolve("build/bent_waveguide_couplings5_5_1.csv"),
                 resolve("build/bent_waveguide_couplings4_4_1.csv"),
                 resolve("build/bent_waveguide_couplings3_3_1.csv"),
                 resolve("build/bent_waveguide_couplings2_2_1.csv")]
    
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    first = []
    for filename in filenames[::-1]:
        
        patch = np.loadtxt(filename, delimiter=",")
        if patch.ndim == 1:
            patch = patch.reshape(1, -1)
        patch_ell, patch_abs = patch[:, 0].astype(int), patch[:, 1]

        patch_len = len(patch_ell)
        idx = patch_len // 2 - 1
        first.append(patch_abs[idx])

        ax.semilogy(patch_ell, patch_abs, "o", ms=8, mfc="none",
                    mew=1.6, label="$N = $" + filename.stem.split("_")[-2], zorder=10-patch_len)
        ax.set_xlim(-9, 9)

    # fpath = resolve("capacitance_matrix_per_alpha.csv")
    # if fpath.exists() and fpath.stat().st_size > 0:
    #     alphas, C_alpha = read_floquet_blocks(fpath, TARGET_FI)
    #     if len(alphas):
    #         ell, C_ell = inverse_floquet(alphas, C_alpha)
    #         fab = np.linalg.norm(C_ell, axis=(1, 2)) if C_ell.ndim == 3 else np.abs(C_ell)
    #         ax.semilogy(ell, np.maximum(fab, 1e-18), "x", ms=7, color="tab:red",
    #                     label="inverse Floquet (ifft of $C^\\alpha$)")
    #         print_comparison(patch_ell, patch_abs, ell, fab)
    # else:
    #     print("(capacitance_matrix_per_alpha.csv not found -- run `defect-capacitance` first "
    #           "for the numeric comparison; plotting the patch couplings only.)")

    ax.set_xlabel(r"$\text{real-space separation}~\ell$")
    ax.set_ylabel(r"$\|C_\ell\|_F$")
    ax.set_title(r"$\text{Bent waveguide: finite patches of different sizes}$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(resolve("decay_bent.pdf"), dpi=250)
    plt.show()

    plt.figure(figsize=(7.5, 5.0))
    error = np.array(first) / first[-1] - 1.0
    print("Relative error in |C_0| vs largest patch:", error)
    plt.semilogy(range(len(error) - 1), np.abs(error[:-1]), "o", ms=8, mfc="none",
                 mew=1.6, label="relative error in |C_0| vs largest patch")
    plt.xlabel(r"$\text{patch size (number of resonators along each side)}$")
    plt.ylabel(r"$\text{relative error in}~|C_0|$")
    plt.title(r"$\text{Convergence of finite patch to infinite Floquet}$")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(resolve("patch_convergence.pdf"), dpi=250)
    plt.show()


if __name__ == "__main__":
    main()
