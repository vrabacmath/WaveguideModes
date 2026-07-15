"""Overlay the real-space capacitance couplings C_l computed two independent ways:

  * FINITE PATCH (no Floquet): patch_capacitance_couplings.csv, the center resonator's row of
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


def resolve(name):
    p = DATA_ROOT / name
    return p if p.exists() else SCRIPT_ROOT / name


def read_floquet_blocks(path, target_fi):
    """Read C^alpha (square complex blocks) for one mode family from capacitance_matrix_per_alpha.csv.
    Returns (alphas sorted, C_alpha of shape (M, modes, modes))."""
    alphas, mats, header = [], [], None
    for line in open(path):
        row = line.strip().split(",")
        if row[0] == "alpha":
            header = row
            continue
        if header is None or int(round(float(row[1]))) != target_fi:
            continue
        vals = row[2:]
        modes = int(round(np.sqrt(len(vals) // 2)))
        block = np.array([float(vals[2 * i]) + 1j * float(vals[2 * i + 1])
                          for i in range(modes * modes)]).reshape(modes, modes)
        alphas.append(float(row[0]))
        mats.append(block)
    alphas = np.array(alphas)
    order = np.argsort(alphas)
    return alphas[order], np.array(mats)[order]


def inverse_floquet(alphas, C_alpha):
    """C_l = (1/M) sum_k C^{alpha_k} e^{-i alpha_k l}, with alpha_k = 2*pi*k/M uniform on [0, 2pi).
    The CSV includes alpha = 2*pi (a duplicate of 0), so drop the last sample first."""
    if np.isclose(alphas[-1], 2 * np.pi) or np.isclose(alphas[-1] - alphas[0], 2 * np.pi):
        alphas, C_alpha = alphas[:-1], C_alpha[:-1]
    M = len(alphas)
    C_ell = np.fft.ifft(C_alpha, axis=0)          # C_ell[m] = (1/M) sum_k C_alpha[k] e^{+2pi i k m/M}
    ell = np.fft.fftfreq(M, d=1.0 / M).astype(int)
    order = np.argsort(ell)
    return ell[order], C_ell[order]


def print_comparison(patch_ell, patch_abs, floquet_ell, floquet_abs):
    """Print the per-l relative difference between the finite-patch and inverse-Floquet couplings
    for l >= 0 -- the numeric form of the overlay. Where both are near their numerical floors the
    coupling is physically zero, so the relative difference there is meaningless (flagged)."""
    p = dict(zip(patch_ell.tolist(), patch_abs.tolist()))
    f = dict(zip(floquet_ell.tolist(), floquet_abs.tolist()))
    print(f"{'l':>3} {'patch |C_l|':>15} {'floquet |C_l|':>15} {'rel.diff':>11}")
    for ell in sorted(k for k in p if k >= 0 and k in f):
        pv, fv = p[ell], f[ell]
        rel = abs(pv - fv) / max(abs(fv), 1e-30)
        floor = " (both ~ noise floor)" if max(pv, fv) < 1e-9 else ""
        print(f"{ell:>3} {pv:>15.4e} {fv:>15.4e} {rel:>11.2e}{floor}")


def main():
    patch = np.loadtxt(resolve("patch_capacitance_couplings.csv"), delimiter=",")
    if patch.ndim == 1:
        patch = patch.reshape(1, -1)
    patch_ell, patch_abs = patch[:, 0].astype(int), patch[:, 1]

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.semilogy(patch_ell, np.maximum(patch_abs, 1e-18), "o", ms=8, mfc="none", mec="tab:blue",
                mew=1.6, label="finite patch (no Floquet)")

    fpath = resolve("capacitance_matrix_per_alpha.csv")
    if fpath.exists() and fpath.stat().st_size > 0:
        alphas, C_alpha = read_floquet_blocks(fpath, TARGET_FI)
        if len(alphas):
            ell, C_ell = inverse_floquet(alphas, C_alpha)
            fab = np.linalg.norm(C_ell, axis=(1, 2)) if C_ell.ndim == 3 else np.abs(C_ell)
            ax.semilogy(ell, np.maximum(fab, 1e-18), "x", ms=7, color="tab:red",
                        label="inverse Floquet (ifft of $C^\\alpha$)")
            print_comparison(patch_ell, patch_abs, ell, fab)
    else:
        print("(capacitance_matrix_per_alpha.csv not found -- run `defect-capacitance` first "
              "for the numeric comparison; plotting the patch couplings only.)")

    ax.set_xlabel(r"real-space separation $\ell$")
    ax.set_ylabel(r"$\|C_\ell\|_F$")
    ax.set_title("Real-space capacitance coupling: finite patch vs inverse Floquet")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(resolve("patch_vs_floquet.pdf"), dpi=250)
    plt.show()


if __name__ == "__main__":
    main()
