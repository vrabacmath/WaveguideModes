"""Batch-render the common figures from the current CSV outputs.

Run from the repository root:
    MPLBACKEND=Agg venv/bin/python src/plot_all.py

Or render one group:
    MPLBACKEND=Agg venv/bin/python src/plot_all.py bands patch

The numerical workflows can still be run separately; this script only consumes the CSVs that are
already present in the repository root or build/.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
HERE = Path.cwd()

BULK_THRESHOLD = 0.25
LOG_FLOOR = 1e-18
PATCH_MARKER_SIZE = 7
PATCH_MARKER_EDGE_WIDTH = 1.4
FLOQUET_MARKER_SIZE = 6.5
MULTISERIES_LEGEND_SIZE = 10

# Tweak these ranges/titles when the paper figures move.
BAND_VIEWS = [
    {
        "output": "bands.pdf",
        "title": r"$\text{Line defect bands: } M^\varepsilon \text{ vs. capacitance}$",
        "ylim": (0.0, 7.5),
        "families": None,
        "figsize": (6.5, 9.0),
        "legend_loc": "center right",
        "legend_anchor": (0.98, 0.16),
        "legend_fontsize": 14,
        "bulk": True,
    },
    {
        "output": "dipole.pdf",
        "title": "Dipole bands",
        "ylim": (4.04, 4.06),
        "families": [1],
        "figsize": (7.5, 5.0),
        "legend_loc": "lower right",
        "legend_anchor": None,
        "legend_fontsize": 16,
        "bulk": False,
    },
    {
        "output": "quadrupole.pdf",
        "title": "Quadrupole bands",
        "ylim": (6.67, 6.74),
        "families": [2],
        "figsize": (7.5, 5.0),
        "legend_loc": "lower left",
        "legend_anchor": None,
        "legend_fontsize": 16,
        "bulk": False,
    },
]

PATCH_VIEWS = [
    {"output": "patch_vs_floquet.pdf", "xlim": (-10, 10), "figsize": (8.4, 5.0)},
    {"output": "patch_vs_floquet_full.pdf", "xlim": None, "figsize": (8.4, 5.0)},
]

BENT_FIELD_PREFIXES = [
    "bent_waveguide_field",
    "bent_waveguide_exact_field",
]

plt.rcParams.update(
    {
        "font.size": 18,
        "axes.labelsize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 14,
        "mathtext.fontset": "cm",
        "axes.linewidth": 1.2,
    }
)


def data_path(name: str | Path) -> Path:
    path = Path(name)
    if path.is_absolute():
        return path

    for base in (HERE, HERE / "build", ROOT, ROOT / "build"):
        candidate = base / path
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    for base in (HERE, HERE / "build", ROOT, ROOT / "build"):
        candidate = base / path
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"{name} not found in current directory, repo root, or build/")


def output_path(out_dir: Path, name: str | Path) -> Path:
    path = Path(name)
    if not path.is_absolute():
        path = out_dir / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_numeric_csv(name: str | Path, min_columns: int) -> np.ndarray:
    path = data_path(name)
    rows = []
    for raw in path.read_text().splitlines():
        parts = raw.strip().split(",")
        if len(parts) < min_columns:
            continue
        try:
            rows.append([float(value) for value in parts[:min_columns]])
        except ValueError:
            continue
    if not rows:
        raise ValueError(f"{path} has no numeric rows with at least {min_columns} columns")
    return np.array(rows)


def save_figure(fig, out_dir: Path, name: str, dpi: int = 250) -> Path:
    out = output_path(out_dir, name)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[wrote] {out}")
    return out


def plot_projected_bulk(ax):
    try:
        data = load_numeric_csv("projected_bulk_bands.csv", 4)
    except FileNotFoundError:
        return None

    alpha = np.unique(data[:, 0])
    omega = np.unique(data[:, 1])
    if len(alpha) * len(omega) != len(data):
        return None

    in_bulk = (data[:, 2] < BULK_THRESHOLD * data[:, 3]).reshape(len(alpha), len(omega))
    alpha_grid, omega_grid = np.meshgrid(alpha, omega, indexing="ij")
    ax.scatter(
        alpha_grid[in_bulk],
        omega_grid[in_bulk],
        s=6,
        c="0.65",
        alpha=0.45,
        linewidths=0,
        zorder=0,
        rasterized=True,
    )
    return plt.Line2D(
        [],
        [],
        marker=".",
        linestyle="None",
        c="0.65",
        label=r"Bulk bands projected over $\alpha_y$",
    )


def plot_neumann_bands(ax, families=None):
    data = load_numeric_csv("defect_neumann_bands.csv", 6)
    data = data[(data[:, 4] == 1) & (data[:, 5] == 0)]
    if families is not None:
        data = data[np.isin(data[:, 0].astype(int), families)]

    labels = {
        -1: r"$M^\varepsilon$: subwavelength",
        0: r"$M^\varepsilon$: $m=0$",
        1: r"$M^\varepsilon$: $m=1$",
        2: r"$M^\varepsilon$: $m=2$",
        3: r"$M^\varepsilon$: $m=3$",
    }
    colors = {-1: "tab:green", 0: "tab:brown", 1: "tab:red", 2: "tab:purple", 3: "tab:orange"}

    for m in sorted(set(data[:, 0].astype(int))):
        rows = data[data[:, 0].astype(int) == m]
        ax.plot(
            rows[:, 1],
            rows[:, 2],
            "o",
            mfc="none",
            mec=colors.get(m, "tab:gray"),
            ms=6,
            mew=1.3,
            label=labels.get(m, rf"$M^\varepsilon$: $m={m}$"),
        )


def plot_capacitance_bands(ax, families=None):
    data = load_numeric_csv("defect_capacitance_bands_leading.csv", 7)
    if families is not None:
        data = data[np.isin(data[:, 0].astype(int), families)]

    family_values = sorted(set(data[:, 0].astype(int)))
    for index, m in enumerate(family_values):
        rows = data[data[:, 0].astype(int) == m]
        label = "Capacitance" if index == len(family_values) - 1 else None
        ax.plot(rows[:, 1], rows[:, 2], ".", c="black", ms=5, label=label)


def plot_band_view(out_dir: Path, view: dict) -> Path:
    fig, ax = plt.subplots(figsize=view["figsize"])
    bulk_handle = plot_projected_bulk(ax) if view["bulk"] else None
    plot_neumann_bands(ax, view["families"])
    plot_capacitance_bands(ax, view["families"])

    ax.set_xlabel(r"$\alpha_x$")
    ax.set_ylabel(r"$\omega$")
    ax.set_xlim(0, np.pi)
    ax.set_xticks([0, np.pi])
    ax.set_xticklabels([r"$\Gamma$", r"$X$"])
    ax.set_ylim(*view["ylim"])
    ax.set_title(view["title"])
    ax.grid(True, alpha=0.25)

    handles, labels = ax.get_legend_handles_labels()
    if bulk_handle is not None:
        handles.append(bulk_handle)
        labels.append(bulk_handle.get_label())
    legend_kwargs = {
        "loc": view["legend_loc"],
        "fontsize": view["legend_fontsize"],
        "framealpha": 0.9,
    }
    if view["legend_anchor"] is not None:
        legend_kwargs["bbox_to_anchor"] = view["legend_anchor"]
    ax.legend(handles, labels, **legend_kwargs)

    return save_figure(fig, out_dir, view["output"], dpi=300)


def plot_bands(out_dir: Path) -> list[Path]:
    return [plot_band_view(out_dir, view) for view in BAND_VIEWS]


def patch_file_key(path: Path) -> tuple[int, int, int, str]:
    match = re.match(r"patch_capacitance_couplings(\d+)_(\d+)_(\d+)$", path.stem)
    if not match:
        return (10**9, 10**9, 10**9, path.name)
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)), path.name)


def patch_label(path: Path) -> str:
    n_defect, n_clad, _fringe, _ = patch_file_key(path)
    if n_defect == 10**9:
        return path.stem
    return rf"finite patch $N={n_defect}$, $N_{{\rm clad}}={n_clad}$"


def patch_files() -> list[Path]:
    build = ROOT / "build"
    files = sorted(build.glob("patch_capacitance_couplings[0-9]*_*_*.csv"), key=patch_file_key)
    if files:
        return files
    return [data_path("patch_capacitance_couplings.csv")]


def read_floquet_blocks(path: Path, target_fi: int):
    alphas, mats = [], []
    for raw in path.read_text().splitlines():
        row = raw.strip().split(",")
        if not row or row[0] == "alpha":
            continue
        if int(round(float(row[1]))) != target_fi:
            continue
        values = row[2:]
        modes = int(round(np.sqrt(len(values) // 2)))
        block = np.array(
            [float(values[2 * i]) + 1j * float(values[2 * i + 1]) for i in range(modes * modes)]
        ).reshape(modes, modes)
        alphas.append(float(row[0]))
        mats.append(block)

    if not alphas:
        return np.array([]), np.array([])
    alphas = np.array(alphas)
    order = np.argsort(alphas)
    return alphas[order], np.array(mats)[order]


def inverse_floquet(alphas: np.ndarray, c_alpha: np.ndarray):
    if np.isclose(alphas[-1], 2 * np.pi) or np.isclose(alphas[-1] - alphas[0], 2 * np.pi):
        alphas, c_alpha = alphas[:-1], c_alpha[:-1]
    count = len(alphas)
    c_ell = np.fft.ifft(c_alpha, axis=0)
    ell = np.fft.fftfreq(count, d=1.0 / count).astype(int)
    order = np.argsort(ell)
    return ell[order], c_ell[order]


def plot_patch_view(out_dir: Path, view: dict, family: int) -> Path:
    fig, ax = plt.subplots(figsize=view["figsize"])
    for filename in patch_files():
        patch = np.loadtxt(filename, delimiter=",")
        if patch.ndim == 1:
            patch = patch.reshape(1, -1)
        patch_ell = patch[:, 0].astype(int)
        patch_abs = patch[:, 1]
        n_defect, _n_clad, _fringe, _name = patch_file_key(filename)
        zorder = 100 - n_defect if n_defect != 10**9 else 1
        ax.semilogy(
            patch_ell,
            np.maximum(patch_abs, LOG_FLOOR),
            "o",
            ms=PATCH_MARKER_SIZE,
            mfc="none",
            mew=PATCH_MARKER_EDGE_WIDTH,
            label=patch_label(filename),
            zorder=zorder,
        )

    fpath = data_path("capacitance_matrix_per_alpha.csv")
    alphas, c_alpha = read_floquet_blocks(fpath, family)
    if len(alphas):
        ell, c_ell = inverse_floquet(alphas, c_alpha)
        floquet_abs = np.linalg.norm(c_ell, axis=(1, 2)) if c_ell.ndim == 3 else np.abs(c_ell)
        ax.semilogy(
            ell,
            np.maximum(floquet_abs, LOG_FLOOR),
            "x",
            ms=FLOQUET_MARKER_SIZE,
            color="tab:red",
            label=r"inverse Floquet (IFFT of $C^\alpha$)",
            zorder=150,
        )
    else:
        print(f"[skip] no Floquet rows found for family {family}")

    ax.set_xlabel(r"real-space separation $\ell$")
    ax.set_ylabel(r"$\|C_\ell\|_F$")
    if view["xlim"] is not None:
        ax.set_xlim(*view["xlim"])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(
        loc="upper right",
        fontsize=MULTISERIES_LEGEND_SIZE,
        framealpha=0.9,
        borderpad=0.45,
        labelspacing=0.35,
        handletextpad=0.5,
    )
    return save_figure(fig, out_dir, view["output"])


def plot_patch_group(out_dir: Path, family: int, decay_output: str | None) -> list[Path]:
    written = [plot_patch_view(out_dir, view, family) for view in PATCH_VIEWS]
    if decay_output:
        decay_view = {"output": decay_output, "xlim": (-10, 10), "figsize": (8.4, 5.0)}
        written.append(plot_patch_view(out_dir, decay_view, family))
    return written


def bent_file_key(path: Path) -> tuple[int, int, int, str]:
    match = re.match(r"bent_waveguide_couplings(\d+)_(\d+)_(\d+)$", path.stem)
    if not match:
        return (10**9, 10**9, 10**9, path.name)
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)), path.name)


def bent_files() -> list[Path]:
    files = sorted((ROOT / "build").glob("bent_waveguide_couplings[0-7]_*_*.csv"), key=bent_file_key)
    if files:
        return files
    return [data_path("bent_waveguide_couplings.csv")]


def plot_bent_decay(out_dir: Path) -> list[Path]:
    files = bent_files()
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    center_values = []
    sizes = []

    for filename in files:
        data = np.loadtxt(filename, delimiter=",")
        if data.ndim == 1:
            data = data.reshape(1, -1)
        ell = data[:, 0].astype(int)
        norm = data[:, 1]
        n_defect, _n_clad, _fringe, _ = bent_file_key(filename)
        sizes.append(n_defect)
        center_values.append(norm[np.argmin(np.abs(ell))])
        zorder = 100 - n_defect if n_defect != 10**9 else 1
        ax.semilogy(
            ell,
            np.maximum(norm, LOG_FLOOR),
            "o",
            ms=PATCH_MARKER_SIZE,
            mfc="none",
            mew=PATCH_MARKER_EDGE_WIDTH,
            label=rf"$N={n_defect},~N_\text{{fringe}}={_fringe}$",
            zorder=zorder,
        )

    ax.set_xlim(-9, 9)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel(r"$\text{real-space separation}~\ell$")
    ax.set_ylabel(r"$\|C_\ell\|_F$")
    # ax.set_title(r"$\text{Bent waveguide: finite patches of different sizes}$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(
        loc="upper right",
        fontsize=MULTISERIES_LEGEND_SIZE,
        framealpha=0.9,
        borderpad=0.45,
        labelspacing=0.35,
        handletextpad=0.5,
    )
    written = [save_figure(fig, out_dir, "decay_bent.pdf")]

    if len(center_values) > 1:
        center_values = np.array(center_values)
        sizes = np.array(sizes)
        largest = center_values[np.argmax(sizes)]
        error = center_values / largest - 1.0
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        ax.semilogy(
            sizes[:-1],
            np.maximum(np.abs(error[:-1]), LOG_FLOOR),
            "o",
            ms=PATCH_MARKER_SIZE,
            mfc="none",
            mew=PATCH_MARKER_EDGE_WIDTH,
            label=r"$|C_0|$ relative error vs largest patch",
        )
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel(r"$\text{patch size }N$")
        ax.set_ylabel(r"$\text{relative error in}~|C_0|$")
        ax.set_title(r"$\text{Convergence of finite patch to infinite Floquet}$")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(loc="best", fontsize=12, framealpha=0.9)
        written.append(save_figure(fig, out_dir, "patch_convergence.pdf"))

    return written


def plot_mode_gallery(out_dir: Path) -> Path:
    modes = np.atleast_2d(np.loadtxt(data_path("bent_waveguide_modes.csv"), delimiter=","))
    weights = modes[:, 7:]
    n_modes, n_main = weights.shape
    lm = (n_main - 1) // 2
    main_xy = np.array([(-ell, 0) if ell <= 0 else (0, ell) for ell in range(-lm, lm + 1)], float)

    mesh = np.loadtxt(data_path("bent_waveguide_mesh.csv"), delimiter=",")
    centers = np.unique(np.round(mesh[:2].T), axis=0)

    ncol = min(5, n_modes)
    nrow = (n_modes + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.8 * nrow), constrained_layout=True)

    vmax = weights.max()
    sc = None
    for index, ax in enumerate(np.atleast_1d(axes).ravel()):
        if index >= n_modes:
            ax.axis("off")
            continue
        ax.scatter(centers[:, 0], centers[:, 1], s=14, c="0.55", zorder=1)
        sc = ax.scatter(
            main_xy[:, 0],
            main_xy[:, 1],
            s=110,
            c=weights[index],
            cmap="magma",
            vmin=0.0,
            vmax=vmax,
            edgecolors="0.3",
            linewidths=0.4,
            zorder=2,
        )
        ax.set_title(
            f"j={index}   " + rf"$\mathrm{{Re}}\,\lambda={modes[index, 1]:.3g}$"
            + f"\ncorner={modes[index, 5]:.2f}, part.={modes[index, 6]:.2f}",
            fontsize=8,
        )
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.colorbar(
        sc,
        ax=np.atleast_1d(axes).ravel().tolist(),
        shrink=0.8,
        label=r"site weight $\sum_\pm |v_{\ell,\pm}|^2 / \|v\|^2$",
    )
    fig.suptitle(
        "Capacitance eigenmodes, per-site weights "
        r"($O(\delta)$; near-degenerate pairs share their subspace)"
    )
    out = output_path(out_dir, "bent_waveguide_mode_gallery.pdf")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[wrote] {out}")
    return out


def plot_field_prefix(out_dir: Path, prefix: str) -> Path:
    mag = np.loadtxt(data_path(f"{prefix}_mag.csv"), delimiter=",")
    real = np.loadtxt(data_path(f"{prefix}_real.csv"), delimiter=",")
    axis = np.loadtxt(data_path(f"{prefix}_axis.csv"), delimiter=",")
    mesh = np.loadtxt(data_path("bent_waveguide_mesh.csv"), delimiter=",")
    bx, by = mesh[0], mesh[1]

    subtitle = prefix
    if prefix == "bent_waveguide_field":
        try:
            modes = np.atleast_2d(np.loadtxt(data_path("bent_waveguide_modes.csv"), delimiter=","))
            index = int(np.argmax(modes[:, 5]))
            # subtitle = (
            #     f"eigenmode j={index}:  "
            #     rf"$\mathrm{{Re}}\,\lambda={modes[index, 1]:.4g}$,  "
            #     f"corner weight={modes[index, 5]:.3f},  participation={modes[index, 6]:.2f}"
            # )
        except FileNotFoundError:
            pass

    extent = [axis[0], axis[-1], axis[0], axis[-1]]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.8), constrained_layout=True)

    finite = mag[np.isfinite(mag) & (mag > 0)]
    vmax = finite.max()
    floor = max(vmax * 1e-6, finite.min())
    im0 = axes[0].imshow(
        np.clip(mag, floor, None),
        origin="lower",
        extent=extent,
        norm=LogNorm(vmin=floor, vmax=vmax),
        cmap="magma",
        interpolation="nearest",
    )
    axes[0].set_title(r"$\log|u|$")
    fig.colorbar(im0, ax=axes[0], shrink=0.85)

    lim = np.nanmax(np.abs(real))
    im1 = axes[1].imshow(
        real,
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        vmin=-lim,
        vmax=lim,
        interpolation="nearest",
    )
    axes[1].set_title(r"$\mathrm{Re}\,u$")
    fig.colorbar(im1, ax=axes[1], shrink=0.85)

    for ax in axes:
        ax.set_xlim(axis[0], axis[-1])
        ax.set_ylim(axis[0], axis[-1])
        ax.set_aspect("equal")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
    axes[1].plot(bx, by, ".", ms=0.6, color="0.7", zorder=3)

    # fig.suptitle(f"Bent-waveguide {subtitle}")
    out_name = f"{prefix.replace('_field', '')}_field.pdf"
    out = output_path(out_dir, out_name)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[wrote] {out}")
    return out


def plot_fields(out_dir: Path) -> list[Path]:
    written = []
    for prefix in BENT_FIELD_PREFIXES:
        try:
            written.append(plot_field_prefix(out_dir, prefix))
        except FileNotFoundError as exc:
            print(f"[skip] {prefix}: {exc}")
    try:
        written.append(plot_mode_gallery(out_dir))
    except FileNotFoundError as exc:
        print(f"[skip] bent_waveguide_mode_gallery.pdf: {exc}")
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "groups",
        nargs="*",
        help="Groups to plot: bands, patch, bent, fields, or all. Defaults to all.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT,
        help="Directory for generated PDFs. Defaults to the repository root.",
    )
    parser.add_argument(
        "--patch-family",
        type=int,
        default=2,
        help="Floquet family for patch/Floquet comparisons: 1=dipole, 2=quadrupole. Defaults to 2.",
    )
    parser.add_argument(
        "--patch-decay-output",
        default="decay_quad.pdf",
        help="Extra patch/Floquet decay output name. Use decay_dip.pdf with --patch-family 1.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    requested = args.groups or ["all"]
    if "all" in requested:
        requested = ["bands", "patch", "bent", "fields"]

    plotters = {
        "bands": lambda: plot_bands(args.out_dir),
        "patch": lambda: plot_patch_group(args.out_dir, args.patch_family, args.patch_decay_output),
        "bent": lambda: plot_bent_decay(args.out_dir),
        "fields": lambda: plot_fields(args.out_dir),
    }

    written = []
    for group in requested:
        if group not in plotters:
            print(f"[skip] unknown plot group: {group}")
            continue
        try:
            written.extend(plotters[group]())
        except FileNotFoundError as exc:
            print(f"[skip] {group}: {exc}")

    print(f"[done] wrote {len(written)} PDF(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
