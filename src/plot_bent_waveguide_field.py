"""Plot the reconstructed field u(r) of one eigenmode of the bent-waveguide patch.

Reads what `WaveguideModes bent-waveguide-field` writes:
  bent_waveguide_field_{mag,real,phase}.csv   Ng x Ng field maps (NaN = masked near-boundary band)
  bent_waveguide_field_axis.csv               the shared x/y coordinate axis
  bent_waveguide_mesh.csv                     row 0 = boundary x, row 1 = boundary y
  bent_waveguide_modes.csv                    j, Re/Im(lambda), Re/Im(omega), corner weight,
                                              participation, then the per-site weights

Blank rings around each disk are the masked band where Nystrom evaluation of the single-layer
potential is not valid -- not missing physics.

    ./build/bin/WaveguideModes bent-waveguide-field 3 3 1 32 1 -1 300
    ./venv/bin/python3 src/plot_bent_waveguide_field.py

An optional prefix argument selects which field files to render, e.g. the exact solve:

    ./venv/bin/python3 src/plot_bent_waveguide_field.py bent_waveguide_exact_field
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path.cwd()

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
    for base in (DATA_ROOT, DATA_ROOT / "build", SCRIPT_ROOT, SCRIPT_ROOT / "build"):
        if (base / name).exists():
            return base / name
    raise FileNotFoundError(f"{name} not found - run `bent-waveguide-field` first")


def mode_gallery():
    """Paper-style dot-lattice gallery (Ammari--Miao--Qiu, arXiv:2605.30951, Figs. 8-10): one
    panel per capacitance eigenmode, every disk drawn as a dot, defect sites colored by that
    mode's per-site weight |v|^2. This is the cheap plot -- it needs no field evaluation, only
    the eigenvectors -- and showing ALL modes side by side is what makes selecting one (e.g. the
    corner-heavy pair) a discovery rather than a choice: the gallery either contains a
    corner-localized panel or it does not.

    What it hides, by construction: the +-m structure inside each site, the relative phases
    between sites (even/odd character), and everything off the defect sites (radiation, cladding
    leakage, fringe hybridization). Those need the field maps or the exact solver's fingerprint.
    """
    try:
        modes = np.atleast_2d(np.loadtxt(resolve("bent_waveguide_modes.csv"), delimiter=","))
    except FileNotFoundError:
        print("(bent_waveguide_modes.csv not found -- run `bent-waveguide-field` first;"
              " skipping the mode gallery)")
        return
    w = modes[:, 7:]                      # per-site weights, path order ell = -Lm..+Lm
    n_modes, n_main = w.shape
    lm = (n_main - 1) // 2
    main_xy = np.array([(-l, 0) if l <= 0 else (0, l) for l in range(-lm, lm + 1)], float)

    # Disk centres sit on the integer lattice, so rounding any boundary point recovers its
    # centre (all radii < 0.5); no geometry parameters need to be passed in.
    mesh = np.loadtxt(resolve("bent_waveguide_mesh.csv"), delimiter=",")
    centers = np.unique(np.round(mesh[:2].T), axis=0)

    ncol = min(5, n_modes)
    nrow = (n_modes + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.8 * nrow),
                             constrained_layout=True)
    # One shared scale across panels: per-panel normalization would make every mode look
    # equally localized and defeat the comparison the gallery exists for.
    vmax = w.max()
    sc = None
    for j, ax in enumerate(np.atleast_1d(axes).ravel()):
        if j >= n_modes:
            ax.axis("off")
            continue
        ax.scatter(centers[:, 0], centers[:, 1], s=14, c="0.55", zorder=1)
        sc = ax.scatter(main_xy[:, 0], main_xy[:, 1], s=110, c=w[j], cmap="magma",
                        vmin=0.0, vmax=vmax, edgecolors="0.3", linewidths=0.4, zorder=2)
        ax.set_title(f"j={j}   " + rf"$\mathrm{{Re}}\,\lambda={modes[j, 1]:.3g}$"
                     + f"\ncorner={modes[j, 5]:.2f}, part.={modes[j, 6]:.2f}", fontsize=8)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(sc, ax=np.atleast_1d(axes).ravel().tolist(), shrink=0.8,
                 label=r"site weight $\sum_\pm |v_{\ell,\pm}|^2 / \|v\|^2$")
    fig.suptitle("Capacitance eigenmodes, per-site weights "
                 r"($O(\delta)$; near-degenerate pairs share their subspace)")
    out = DATA_ROOT / "bent_waveguide_mode_gallery.pdf"
    fig.savefig(out, dpi=200)
    print(f"[wrote] {out}")


def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else "bent_waveguide_field"
    mag = np.loadtxt(resolve(f"{prefix}_mag.csv"), delimiter=",")
    real = np.loadtxt(resolve(f"{prefix}_real.csv"), delimiter=",")
    axis = np.loadtxt(resolve(f"{prefix}_axis.csv"), delimiter=",")
    mesh = np.loadtxt(resolve("bent_waveguide_mesh.csv"), delimiter=",")
    bx, by = mesh[0], mesh[1]
    print("mag.shape", mag.shape, "real.shape", real.shape, "axis.shape", axis.shape, "mesh.shape", mesh.shape)

    # The mode diagnostics file belongs to the O(delta) workflow ONLY. For any other prefix
    # (e.g. the exact solve) it describes a different reconstruction -- the exact state can even
    # be a different physical mode than the seeded one (see the fingerprint in the exact run's
    # stdout) -- so never borrow it for the title there.
    subtitle = prefix
    if prefix == "bent_waveguide_field":
        try:
            modes = np.atleast_2d(np.loadtxt(resolve("bent_waveguide_modes.csv"), delimiter=","))
            j = int(np.argmax(modes[:, 5]))
            subtitle = (f"eigenmode j={j}:  " rf"$\mathrm{{Re}}\,\lambda={modes[j, 1]:.4g}$,  "
                        f"corner weight={modes[j, 5]:.3f},  participation={modes[j, 6]:.2f}")
        except FileNotFoundError:
            pass

    extent = [axis[0], axis[-1], axis[0], axis[-1]]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.8), constrained_layout=True)

    # |u| on a log scale: the mode spans many decades, so a linear map shows only the corner.
    finite = mag[np.isfinite(mag) & (mag > 0)]
    vmax = finite.max()
    floor = max(vmax * 1e-6, finite.min())
    im0 = axes[0].imshow(np.clip(mag, floor, None), origin="lower", extent=extent,
                         norm=plt.matplotlib.colors.LogNorm(vmin=floor, vmax=vmax),
                         cmap="magma", interpolation="nearest")
    axes[0].set_title(r"$\log|u|$")
    fig.colorbar(im0, ax=axes[0], shrink=0.85)

    lim = np.nanmax(np.abs(real))
    im1 = axes[1].imshow(real, origin="lower", extent=extent, cmap="RdBu_r",
                         vmin=-lim, vmax=lim, interpolation="nearest")
    axes[1].set_title(r"$\mathrm{Re}\,u$")
    fig.colorbar(im1, ax=axes[1], shrink=0.85)

    for ax in axes:
        ax.set_xlim(axis[0], axis[-1])
        ax.set_ylim(axis[0], axis[-1])
        ax.set_aspect("equal")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
    axes[1].plot(bx, by, ".", ms=0.6, color="0.7", zorder=3)

    fig.suptitle(f"Bent-waveguide {subtitle}")
    out = DATA_ROOT / f"{prefix.replace('_field', '')}_field.pdf"
    fig.savefig(out, dpi=200)
    print(f"[wrote] {out}")

    mode_gallery()
    plt.show()


if __name__ == "__main__":
    main()
