"""Plot the unperturbed CrystalA / multipoleA bulk band comparison.

Run from the repository root after:
    ./build/bin/SubwavelengthResonators crystal-matrix-bands
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path.cwd()
METRIC = "sigma_min"  # "sigma_min" or "log_abs_det"

# --- band-line extraction settings ----------------------------------------------------------
CRYSTAL_RADIUS = 0.35    # MUST match the radius passed to run_crystal_matrix_bands in main.cpp
DEFECT_RADIUS = 0.455    # line-defect resonator radius (set None to hide the defect overlay)
SIGMA_THRESHOLD = 0.02   # keep sigma_min minima below this as band points (lower = crisper)
NEUMANN_ZEROS = {0: 3.8317059702, 1: 1.8411837813, 2: 3.0542369282, 3: 4.2011889412}   # j'_{m,1}
DIRICHLET_ZEROS = [("j_{0,1}", 2.4048255577), ("j_{1,1}", 3.8317059702),
                   ("j_{2,1}", 5.1356223019), ("j_{0,2}", 5.5200781103),
                   ("j_{1,2}", 7.0155866699)]  # interior Dirichlet eigenvalues -> spurious BIE lines


def resolve_data_path(relative_path):
    cwd_path = DATA_ROOT / relative_path
    if cwd_path.exists() or cwd_path.parent.exists():
        return cwd_path
    return SCRIPT_ROOT / relative_path


def load_csv(relative_path, min_columns):
    path = resolve_data_path(relative_path)
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"{relative_path} is missing or empty")

    data = np.loadtxt(path, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < min_columns:
        raise ValueError(f"Expected at least {min_columns} columns in {path}, found {data.shape}")
    return data


def mark_eigenvalue_lines(ax, omega_lo, omega_hi):
    """Mark the relevant interior disk eigenvalues, the three families drawn separately:
      * green dashed -- CRYSTAL Neumann eigenvalues j'_{m,1}/R_crystal: the real high-contrast bulk
        bands sit here;
      * red dotted   -- CRYSTAL Dirichlet eigenvalues j_{m,n}/R_crystal: these make the single-layer
        BIE / multipole operator singular for every Bloch vector, i.e. the *spurious* perfectly-flat
        lines (e.g. j_{0,1}/R = 2.4048/0.35 = 6.857). Ignore those when reading the band structure;
      * gold dash-dot -- DEFECT Neumann eigenvalues j'_{m,1}/R_defect: the line-defect resonances we
        want to land in the bulk gaps (e.g. dipole 1.841/0.455 = 4.05, quadrupole 3.054/0.455 = 6.71).
    """
    def draw(value, **kw):
        if omega_lo <= value <= omega_hi:
            ax.axhline(value, **kw)

    # Thick and high zorder so the lines stay visible IN FRONT of the band markers.
    for z in NEUMANN_ZEROS.values():
        draw(z / CRYSTAL_RADIUS, color="tab:green", ls="--", lw=1.4, alpha=0.9, zorder=4)
    for _, z in DIRICHLET_ZEROS:
        draw(z / CRYSTAL_RADIUS, color="tab:red", ls=":", lw=4.6, alpha=0.9, zorder=4)
    if DEFECT_RADIUS:
        for z in NEUMANN_ZEROS.values():
            draw(z / DEFECT_RADIUS, color="goldenrod", ls="-.", lw=2.4, alpha=1.0, zorder=6)


def eigenvalue_legend_handles():
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color="tab:green", ls="--"),
               Line2D([], [], color="tab:red", ls=":")]
    labels = [r"crystal Neumann $j'_{m,1}/R$ (real bands)",
              r"crystal Dirichlet $j_{m,n}/R$ (spurious flat lines)"]
    if DEFECT_RADIUS:
        handles.append(Line2D([], [], color="goldenrod", ls="-."))
        labels.append(r"defect Neumann $j'_{m,1}/R_{\rm def}$ (target, in gaps)")
    return handles, labels


def plot_grid(ax, filename, title):
    data = load_csv(filename, 6)
    s, omega, sigma, log_abs_det = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

    if METRIC == "log_abs_det":
        values = log_abs_det
        label = r"$\log|\det A^\alpha|$"
    else:
        values = np.log10(np.maximum(sigma, 1e-12))
        label = r"$\log_{10}\sigma_{\min}(A^\alpha)$"

    sc = ax.scatter(s, omega, c=values, s=25, cmap="viridis_r", rasterized=True)
    # mark_eigenvalue_lines(ax, omega.min(), omega.max())
    ax.set_title(title)
    ax.set_xlim(0, 3)
    ax.set_xlabel("Bloch path")
    ax.grid(True, alpha=0.22)
    return sc, label


def plot_heatmaps():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    sc, label = plot_grid(axes[0], "build/crystal_hex_bands.csv", "CrystalA")
    plot_grid(axes[1], "build/multipoleA_m_gamma_x_m_hex.csv", "multipoleA")

    axes[0].set_ylabel(r"Frequency $\omega$")
    for ax in axes:
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels([r"$M$", r"$\Gamma$", r"$X$", r"$M$"])
    handles, labels = eigenvalue_legend_handles()
    axes[1].legend(handles, labels, fontsize=7, loc="upper right")
    fig.colorbar(sc, ax=axes, label=label)
    fig.suptitle("Square-lattice bulk bands along M-Gamma-X-M")
    fig.savefig(resolve_data_path("crystal_matrix_bands.pdf"), dpi=250, bbox_inches="tight")


def extract_bands(filename, sigma_threshold=SIGMA_THRESHOLD):
    """Turn a (k-point, omega) sigma_min grid into band points: at each k-point, the local minima
    of sigma_min(omega) below `sigma_threshold` are band crossings. Returns (s, omega, sigma) of
    those minima and the (omega_lo, omega_hi) range. No band-sorting is attempted -- the dense set
    of minima already reads as lines, and that avoids fragile branch-tracking across k."""
    data = load_csv(filename, 6)
    s, omega, sigma = data[:, 0], data[:, 1], data[:, 2]
    s_round = np.round(s, 9)
    bs, bw, bsig = [], [], []
    for sv in np.unique(s_round):
        m = s_round == sv
        ww, ss = omega[m], sigma[m]
        order = np.argsort(ww)
        ww, ss = ww[order], ss[order]
        for i in range(1, len(ss) - 1):
            if ss[i] <= ss[i - 1] and ss[i] <= ss[i + 1] and ss[i] < sigma_threshold:
                bs.append(sv); bw.append(ww[i]); bsig.append(ss[i])
    return np.array(bs), np.array(bw), np.array(bsig), (omega.min(), omega.max())


def plot_band_lines():
    """Plot the bulk bands as the (color-coded) minima of sigma_min along M-Gamma-X-M, instead of a
    heatmap. Green dashed lines mark the disk's interior Neumann eigenvalues j'_{m,1}/R (where the
    real high-contrast bands sit); red dotted lines mark the interior Dirichlet eigenvalues
    j_{m,n}/R, which produce *spurious* perfectly-flat lines in the single-layer BIE / multipole
    formulation (e.g. the flat feature at j_{0,1}/R = 2.4048/0.35 = 6.857) -- ignore those."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    sc = None
    for ax, filename, title in [(axes[0], "build/crystal_hex_bands.csv", "CrystalA"),
                                (axes[1], "build/multipoleA_m_gamma_x_m_hex.csv", "multipoleA")]:
        bs, bw, bsig, (omega_lo, omega_hi) = extract_bands(filename)
        if bs.size:
            sc = ax.scatter(bs, bw, c=np.log10(np.maximum(bsig, 1e-12)), s=12, cmap="viridis",
                            vmax=np.log10(SIGMA_THRESHOLD), zorder=1)
        # mark_eigenvalue_lines(ax, omega_lo, omega_hi)  # green/red/gold, thick and in front
        ax.set_title(title)
        ax.set_xlim(0, 3)
        ax.set_ylim(omega_lo, omega_hi)
        ax.set_xlabel("Bloch path")
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels([r"$M$", r"$\Gamma$", r"$X$", r"$M$"])
        ax.grid(True, alpha=0.22)

    axes[0].set_ylabel(r"Frequency $\omega$")
    handles, labels = eigenvalue_legend_handles()
    fig.legend(handles, labels, fontsize=7, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04), frameon=True)
    if sc is not None:
        fig.colorbar(sc, ax=axes, label=r"$\log_{10}\sigma_{\min}$")
    fig.suptitle(f"Square-lattice bulk bands along M-Gamma-X-M  (R={CRYSTAL_RADIUS}, band minima)")
    fig.savefig(resolve_data_path("crystal_matrix_band_lines.pdf"), dpi=250, bbox_inches="tight")


if __name__ == "__main__":
    plot_band_lines()
    plt.show()
