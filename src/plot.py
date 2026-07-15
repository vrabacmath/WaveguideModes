"""Render the CSV outputs of the two line-defect band workflows.

Toggle-driven: set the booleans below and run `python src/plot.py` from the directory holding the
CSVs (usually the repository root). The two workflows are:
  * line-defect-M       -> crystal_first_band.csv, crystal_second_band.csv, defect_bands.csv,
                           defect_band_diagnostics.csv   (plot_line_defect_bands / _diagnostics)
  * defect-capacitance  -> defect_capacitance_bands.csv  (plot_defect_capacitance_bands)
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path.cwd()


def resolve_data_path(relative_path):
    cwd_path = DATA_ROOT / relative_path
    if cwd_path.exists() or cwd_path.parent.exists():
        return cwd_path
    return SCRIPT_ROOT / relative_path


# --- toggles --------------------------------------------------------------------------------
line_defect_bands = True        # subwavelength line-defect band (operator M): crystal + defect bands
line_defect_diagnostics = False  # sigma_min(A) / sigma_min(M) heat maps over the scan grid
defect_capacitance_bands = True  # higher-frequency (dipole) band from the capacitance matrix
neumann_vs_capacitance = True    # overlay the M^eps Neumann-resonance bands vs the capacitance bands
nvc_ylim = None                  # set to (lo, hi) to zoom the frequency axis, e.g. (3.95, 4.6) dipole


def load_xy_csv(path):
    path = resolve_data_path(path)
    if not path.exists() or path.stat().st_size == 0:
        return np.empty((0, 2))

    data = np.loadtxt(path, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != 2:
        raise ValueError(f"Expected two columns in {path}, found shape {data.shape}")
    return data


def plot_line_defect_bands():
    """Dilute Fig. 2-style line-defect band: unperturbed crystal bands in blue, the
    asymptotic/defect branch in red, and a zoom around the first subwavelength band."""
    first_band_data = load_xy_csv("crystal_first_band.csv")
    second_band_data = load_xy_csv("crystal_second_band.csv")
    defect_data = load_xy_csv("defect_bands.csv")
    asymptotic_data = load_xy_csv("defect_asymptotic.csv")

    if not (first_band_data.size and second_band_data.size):
        raise FileNotFoundError("crystal band CSVs are missing or empty. Run 'line-defect-M' first.")

    first_band_data = first_band_data[np.argsort(first_band_data[:, 0])]
    second_band_data = second_band_data[np.argsort(second_band_data[:, 0])]
    if defect_data.size:
        defect_data = defect_data[np.argsort(defect_data[:, 0])]
    if asymptotic_data.size:
        asymptotic_data = asymptotic_data[np.argsort(asymptotic_data[:, 0])]
    elif defect_data.shape[0] >= 2:
        # If the asymptotic formula has not been exported separately, draw a smooth visual guide
        # through the discretized operator points so the Fig. 2 layout remains readable.
        alpha_dense = np.linspace(defect_data[:, 0].min(), defect_data[:, 0].max(), 300)
        asymptotic_data = np.column_stack((alpha_dense, np.interp(alpha_dense, defect_data[:, 0], defect_data[:, 1])))

    fig, axes = plt.subplots(2, 1, figsize=(7.4, 7.0), sharex=True)
    for ax in axes:
        ax.plot(first_band_data[:, 0], first_band_data[:, 1], "-", color="tab:blue",
                linewidth=1.2, label="Unperturbed")
        if asymptotic_data.size:
            ax.plot(asymptotic_data[:, 0], asymptotic_data[:, 1], "--", color="tab:red",
                    linewidth=1.2, label="Asymptotic formula")
        if defect_data.size:
            ax.plot(defect_data[:, 0], defect_data[:, 1], "o", color="tab:red",
                    markerfacecolor="none", markersize=4, linewidth=0, label="Discretized operator")
    axes[0].plot(second_band_data[:, 0], second_band_data[:, 1], "-", color="tab:blue", linewidth=1.2)
    axes[0].set_ylim(0.0, 3.5)
    axes[1].set_ylim(0.24, 0.31)
    axes[1].set_xlabel(r"Quasi-periodicity $\alpha_1$")
    for ax in axes:
        ax.set_ylabel(r"Frequency $\omega$")
        ax.set_xlim(0, 2 * np.pi)
        ax.set_xticks([0, np.pi, 2 * np.pi])
        ax.set_xticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")
    fig.tight_layout()
    plt.savefig(resolve_data_path("bands_defect_line.pdf"), dpi=300)
    plt.show()


def plot_line_defect_diagnostics():
    """Heat maps of sigma_min(A) (crystal) and sigma_min(M) (defect) over the (alpha_x, omega) scan,
    from defect_band_diagnostics.csv -- useful for retuning the band-search windows."""
    path = resolve_data_path("defect_band_diagnostics.csv")
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError("defect_band_diagnostics.csv is missing or empty. Run 'line-defect-M' first.")

    data = np.loadtxt(path, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != 7:
        raise ValueError(f"Expected seven columns in {path}, found shape {data.shape}")

    alpha_x = data[:, 0]
    first_omega, first_sigma = data[:, 1], data[:, 2]
    second_omega, second_sigma = data[:, 3], data[:, 4]
    defect_omega, defect_sigma = data[:, 5], data[:, 6]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True)
    p0 = axes[0].scatter(alpha_x, first_omega, c=np.log10(np.maximum(first_sigma, 1e-16)), s=22, cmap="viridis")
    axes[0].set_title(r"first band, $\log_{10}\sigma_{\min}(A)$")
    fig.colorbar(p0, ax=axes[0])
    p1 = axes[1].scatter(alpha_x, second_omega, c=np.log10(np.maximum(second_sigma, 1e-16)), s=22, cmap="viridis")
    axes[1].set_title(r"second band, $\log_{10}\sigma_{\min}(A)$")
    fig.colorbar(p1, ax=axes[1])
    mask = np.isfinite(defect_omega)
    p2 = axes[2].scatter(alpha_x[mask], defect_omega[mask], c=np.log10(np.maximum(defect_sigma[mask], 1e-16)),
                         s=22, cmap="magma_r")
    axes[2].set_title(r"defect candidates, $\log_{10}\sigma_{\min}(M)$")
    if mask.any():
        fig.colorbar(p2, ax=axes[2])

    for ax in axes:
        ax.set_xlabel(r"$\alpha_x$")
        ax.set_xlim(0, 2 * np.pi)
        ax.set_xticks([0, np.pi, 2 * np.pi])
        ax.set_xticklabels([r"$\Gamma$", r"$X$", r"$\Gamma$"])
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel(r"$\omega$")
    fig.tight_layout()
    plt.savefig(resolve_data_path("defect_band_diagnostics.pdf"), dpi=300)
    plt.show()


def plot_defect_capacitance_bands():
    """Line-defect band from the capacitance-matrix workflow (Fabry-Perot, arXiv:2605.27572):
    the leading-order eigenvalues of C^reg = -(v_b^2/2 omega_0) <Lambda_ext g_q, g_p> seed an
    exact supercell transmission solve [delta (1/2 + K*) - B_int(omega) S] phi = 0, and the CSV
    holds the exact roots (docs/defect_capacitance_bands.md section 4b). Works for ANY multipole
    order m: the base frequency and the y-range are inferred from the data, so changing m (e.g.
    m=0 monopole at omega_0 = j'_{0,1}/R_def, m=1 dipole at j'_{1,1}/R_def, ...) re-centres it.
    CSV columns: alpha_x, Re(omega), Im(omega), Re(lambda_eff), Im(lambda_eff), secant_step with
    lambda_eff = (omega - omega_0)/delta. The bound branch (near-real omega) is the defect band;
    near-zero |Im| means a true bound mode in a gap."""
    cap_path = resolve_data_path("defect_capacitance_bands.csv")
    if not cap_path.exists() or cap_path.stat().st_size == 0:
        raise FileNotFoundError("defect_capacitance_bands.csv missing. Run 'defect-capacitance' first.")
    c = np.loadtxt(cap_path, delimiter=",")
    if c.ndim == 1:
        c = c.reshape(1, -1)
    if c.shape[1] >= 7:   # multi-m schema: m, alpha, Re w, Im w, Re lam_eff, Im lam_eff, step
        cm, cax, cre, cim, lre = c[:, 0].astype(int), c[:, 1], c[:, 2], c[:, 3], c[:, 4]
    else:                 # single-m schema: alpha, Re w, Im w, Re lam, Im lam, resid
        cm, cax, cre, cim, lre = np.zeros(len(c), int), c[:, 0], c[:, 1], c[:, 2], c[:, 3]
    bound = np.abs(cim) < 1e-2    # near-real root -> genuine bound mode
    leaky = ~bound                # radiating branch

    # Recover each family's Neumann base frequency from the data: Re(omega) = omega_0 +
    # delta * Re(lambda_eff), so omega_0 is the intercept of a line fit per m family.
    omega0s = []
    for m in sorted(set(cm)):
        sel = cm == m
        if np.ptp(lre[sel]) > 1e-9:
            _, w0 = np.polyfit(lre[sel], cre[sel], 1)
        else:
            w0 = float(np.mean(cre[sel]))
        omega0s.append(w0)

    ymin, ymax = float(np.min(cre)), float(np.max(cre))
    pad = max(0.05 * (ymax - ymin), 0.05)
    ylo, yhi = min(ymin, min(omega0s)) - pad, max(ymax, max(omega0s)) + pad

    fig, axp = plt.subplots(figsize=(7.5, 5.5))
    # axp.hlines([2.4048255577/0.35], color="tab:red", ls="--", lw=1.4, alpha=0.9, label="Exterior Dirichlet Eigenvalue", xmin=0, xmax=np.pi)
    gap_lo, gap_hi = 1.5, 4.6  # R=0.35 crystal bulk gap, shaded only when in view
    # gap_lo, gap_hi = 6.32, 7.5  # R=0.35 crystal bulk gap, shaded only when in view
    if gap_lo < yhi and gap_hi > ylo:
        axp.axhspan(max(gap_lo, ylo), min(gap_hi, yhi), color="0.85", alpha=0.55, label="bulk gap")

    for j, w0 in enumerate(omega0s):
        axp.axhline(w0, color="0.4", ls=":", lw=1,
                    label=(r"$\omega_0$ (Neumann base)" if j == 0 else None))
    if leaky.any():
        axp.plot(cax[leaky], cre[leaky], "x", color="0.6", ms=5, label="radiating branch")
    if bound.any():
        axp.plot(cax[bound], cre[bound], ".", color="red", ms=9,
                 label=r"bound band $\omega=\omega_0+\delta\lambda$")

    axp.set_xlabel(r"$\alpha_x$")
    axp.set_ylabel(r"$\omega$")
    axp.set_xlim(0, np.pi)
    axp.set_ylim(ylo, yhi)
    axp.set_xticks([0, np.pi])
    axp.set_xticklabels([r"$\Gamma$", r"$X$"])
    axp.set_title("Line-defect band from the capacitance matrix (BIE / DtN)")
    axp.legend(loc="best", fontsize=8)
    axp.grid(True, alpha=0.25)
    fig.tight_layout()
    plt.savefig(resolve_data_path("defect_capacitance_bands.pdf"), dpi=300)
    plt.show()


def load_capacitance_bands(path):
    """Load defect_capacitance_bands*.csv in either schema. Old: alpha, Re w, Im w, ...
    New (multi-m): m, alpha, Re w, Im w, ... Returns (m, alpha, Re w, Im w) arrays with m = -1
    for the old schema."""
    c = np.loadtxt(path, delimiter=",")
    if c.ndim == 1:
        c = c.reshape(1, -1)
    if c.shape[1] >= 7:
        return c[:, 0].astype(int), c[:, 1], c[:, 2], c[:, 3]
    return np.full(len(c), -1), c[:, 0], c[:, 1], c[:, 2]


def overlay_projected_bulk(ax):
    """Shade the alpha_y-projected bulk spectrum (mode projected-bulk) behind a line-defect band
    diagram: a bulk band passes through (alpha_x, omega) iff min over alpha_y of sigma_min dips
    well below its median (the same relative criterion as the in_gap test). Returns a legend
    proxy patch, or None when projected_bulk_bands.csv is absent."""
    path = resolve_data_path("projected_bulk_bands.csv")
    if not path.exists() or path.stat().st_size == 0:
        return None
    p = np.loadtxt(path, delimiter=",")
    a, w = np.unique(p[:, 0]), np.unique(p[:, 1])
    if len(a) * len(w) != len(p):
        return None
    band = (p[:, 2] < 0.25 * p[:, 3]).reshape(len(a), len(w))
    from matplotlib.colors import ListedColormap
    ax.pcolormesh(a, w, np.where(band, 1.0, np.nan).T, cmap=ListedColormap(["0.85"]),
                  vmin=0, vmax=1, shading="nearest", zorder=0, rasterized=True)
    from matplotlib.patches import Patch
    return Patch(facecolor="0.85", label="bulk bands (projected over $\\alpha_y$)")


def plot_neumann_vs_capacitance(ylim=None):
    """Compare the higher-frequency line-defect bands from the multipole operator M^eps (mode
    line-defect-neumann, searched around the defect Neumann resonances) with the capacitance-matrix
    bands (mode defect-capacitance) at the same geometry. Only the physical M rows are shown
    (in_gap=1, near_dirichlet=0). Pass ylim=(lo, hi) to zoom into one band, e.g. (3.8, 4.8) for the
    dipole, (6.2, 7.2) for the quadrupole. CSV: m, alpha_x, omega, sigma_min, in_gap, near_dirichlet
    with m = -1 subwavelength, 0 breathing, 1 dipole, 2 quadrupole."""
    npath = resolve_data_path("defect_neumann_bands.csv")
    if not npath.exists() or npath.stat().st_size == 0:
        raise FileNotFoundError("defect_neumann_bands.csv missing. Run 'line-defect-neumann' first.")
    d = np.loadtxt(npath, delimiter=",")
    if d.ndim == 1:
        d = d.reshape(1, -1)
    d = d[(d[:, 4] == 1) & (d[:, 5] == 0)]   # in a gap, not on a spurious Dirichlet line

    labels = {-1: r"$M$: subwavelength ($m=-1$)", 0: r"$M$: breathing ($m=0$)",
              1: r"$M$: dipole ($m=1$)", 2: r"$M$: quadrupole ($m=2$)", 3: r"$M$: octupole ($m=3$)"}
    colors = {-1: "tab:green", 0: "tab:brown", 1: "tab:red", 2: "tab:purple", 3: "tab:orange"}

    fig, ax = plt.subplots(figsize=(8, 6.5))
    bulk_patch = overlay_projected_bulk(ax)
    for m in sorted(set(d[:, 0].astype(int))):
        sel = d[:, 0].astype(int) == m
        ax.plot(d[sel, 1], d[sel, 2], "o", mfc="none", mec=colors.get(m, "tab:gray"),
                ms=6, mew=1.3, label=labels.get(m, f"$M$: $m={m}$"))

    # cpath = resolve_data_path("defect_capacitance_bands.csv")
    # if cpath.exists() and cpath.stat().st_size > 0:
    #     _, calpha, cre, cim = load_capacitance_bands(cpath)
    #     bound = np.abs(cim) < 1e-2   # exact solve: bound roots have |Im omega| ~ 1e-9 in a gap
    #     ax.plot(calpha[bound], cre[bound], ".", color="k", ms=8, label="capacitance (exact roots)")
    #     if (~bound).any():
    #         ax.plot(calpha[~bound], cre[~bound], "x", color="0.45", ms=5, mew=1.1,
    #                 label="capacitance (leaky)")

    ax.set_xlabel(r"$\alpha_x$")
    ax.set_ylabel(r"$\omega$")
    ax.set_xlim(0, np.pi)
    ax.set_xticks([0, np.pi])
    ax.set_xticklabels([r"$\Gamma$", r"$X$"])
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_title(r"Line-defect bands: multipole $M^\epsilon$ (Neumann search) vs capacitance")
    handles, labels_ = ax.get_legend_handles_labels()
    if bulk_patch is not None:
        handles.append(bulk_patch)
        labels_.append(bulk_patch.get_label())

    
    lpath = resolve_data_path("defect_capacitance_bands_leading.csv")
    if lpath.exists() and lpath.stat().st_size > 0:
        _, lalpha, lre, _ = load_capacitance_bands(lpath)
        ax.plot(lalpha, lre, ".", color="black", ms=5, mew=1.0, label=r"capacitance (leading order $\omega_0+\delta\lambda$)")

    ax.legend(handles, labels_, loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    plt.savefig(resolve_data_path("neumann_vs_capacitance.pdf"), dpi=300)
    plt.show()


if line_defect_bands:
    plot_line_defect_bands()

if line_defect_diagnostics:
    plot_line_defect_diagnostics()

if defect_capacitance_bands:
    plot_defect_capacitance_bands()

if neumann_vs_capacitance:
    plot_neumann_vs_capacitance(nvc_ylim)
