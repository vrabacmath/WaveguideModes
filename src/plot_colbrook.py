from pathlib import Path
import argparse
import csv

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter
import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
# PATCH_SIZES = [2, 3, 4, 7, 10, *range(12, 21, 2), 40]
PATCH_SIZES = [10,*range(12, 21, 2), 15, 40]
WINDOW_SIZES = [*range(0,7)]
NUM_MINIMA = 3
MIN_Z_SEPARATION = 1.0  # z units; suppress nearby secondary dips, not a physical criterion
CONVERGENCE_CASES = ("material_dipole", "material_quadrupole", "size_dipole", "size_quadrupole")
CONVERGENCE_COLORS = {
    "material_dipole": "tab:blue",
    "material_quadrupole": "tab:red",
    "size_dipole": "tab:green",
    "size_quadrupole": "tab:orange",
}

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 13,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 13,
        "mathtext.fontset": "cm",
        "axes.linewidth": 1.2,
    }
)


def resolve(name):
    for directory in (Path.cwd(), Path.cwd() / "build", SCRIPT_ROOT / "build",
                      SCRIPT_ROOT):
        p = directory / name
        if p.is_file():
            return p
    raise FileNotFoundError(f"Could not find {name} in the current directory or repository build directory")

def find_distinct_minima(z, sigma, count=NUM_MINIMA, min_separation=MIN_Z_SEPARATION):
    """Return sampled minima indices, ranked by depth then separated in z.

    Flat minima contribute one midpoint sample. Scan endpoints are excluded because
    they do not establish a dip. Returned indices are sorted by z, not by depth.
    """
    z, sigma = np.asarray(z), np.asarray(sigma)
    if z.ndim != 1 or sigma.shape != z.shape:
        raise ValueError("Expected matching one-dimensional z and sigma arrays")
    if not np.isfinite(z).all() or not np.isfinite(sigma).all() or np.any(np.diff(z) <= 0):
        raise ValueError("Expected finite samples with strictly increasing z")
    if count < 1 or min_separation < 0:
        raise ValueError("count must be positive and min_separation nonnegative")
    if len(z) < 3:
        return np.array([], dtype=int)

    starts = np.r_[0, np.flatnonzero(np.diff(sigma) != 0) + 1]
    ends = np.r_[starts[1:] - 1, len(sigma) - 1]
    candidates = [
        (left + right) // 2 for left, right in zip(starts, ends)
        if left > 0 and right < len(sigma) - 1
        and sigma[left] < sigma[left - 1] and sigma[right] < sigma[right + 1]
    ]
    selected = []
    for i in sorted(candidates, key=lambda i: sigma[i]):
        if all(abs(z[i] - z[j]) >= min_separation for j in selected):
            selected.append(i)
            if len(selected) == count:
                break
    return np.array(sorted(selected), dtype=int)


def _load_convergence(filenames=None):
    """Load and validate the case CSVs for either convergence plot layout."""
    if not filenames:
        filenames = []
        for case in CONVERGENCE_CASES:
            try:
                filenames.append(resolve(f"colbrook_convergence_{case}.csv"))
            except FileNotFoundError:
                pass
        if not filenames:
            raise FileNotFoundError("No convergence CSVs found; run WaveguideModes colbrook-convergence first")
    datasets = []
    seen = set()
    for filename in filenames:
        path = resolve(filename)
        with path.open(newline="") as file:
            rows = list(csv.DictReader(file))
        if not rows:
            raise ValueError(f"No completed convergence points in {path}")
        case = rows[0]["case"]
        if case not in CONVERGENCE_CASES or case in seen or any(r["case"] != case for r in rows):
            raise ValueError(f"Expected one distinct supported case per file: {path}")
        seen.add(case)
        for row in rows:
            if row["sweep"] not in ("N", "W") or int(row["N_colbrook"]) != int(row["W"]) or int(row["fringe"]) != 1:
                raise ValueError(f"Expected N/W sweeps with N_colbrook=W and fringe=1 in {path}")
            for key in ("z_min", "sigma_min", "z_resolution"):
                if not np.isfinite(float(row[key])):
                    raise ValueError(f"Non-finite {key} in {path}")
            if float(row["sigma_min"]) < 0 or float(row["z_resolution"]) <= 0 or row["at_boundary"] not in ("0", "1"):
                raise ValueError(f"Invalid residual, resolution, or boundary flag in {path}")
        for sweep, fixed in (("N", "W"), ("W", "N")):
            points = [r for r in rows if r["sweep"] == sweep]
            if points and (len({r[fixed] for r in points}) != 1 or
                           len({r[sweep] for r in points}) != len(points)):
                raise ValueError(f"Expected a fixed {fixed} and unique {sweep} values in {path}")
        datasets.append((case, rows))
    return datasets


def plot_convergence(filenames=None, output_dir="figures", show=True):
    """Plot each case separately; error bars indicate search-grid spacing only."""
    datasets = _load_convergence(filenames)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = []
    for case, rows in datasets:

        fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex="col", squeeze=False)
        for column, sweep in enumerate(("N", "W")):
            points = sorted((r for r in rows if r["sweep"] == sweep), key=lambda r: int(r[sweep]))
            if not points:
                axes[0, column].set_title(f"No {sweep} sweep points yet")
                continue
            fixed = "W" if sweep == "N" else "N"
            x = np.array([int(r[sweep]) for r in points])
            z = np.array([float(r["z_min"]) for r in points])
            sigma = np.array([float(r["sigma_min"]) for r in points])
            resolution = np.array([float(r["z_resolution"]) for r in points])
            boundary = np.array([r["at_boundary"] == "1" for r in points])
            axes[0, column].errorbar(x, np.where(boundary, np.nan, z), yerr=resolution,
                                    fmt="o-", capsize=3, label="tracked minimum (grid spacing)")
            axes[0, column].axhline(float(points[0]["target_z"]), color="0.5", ls="--",
                                   lw=1, label="search target")
            axes[1, column].plot(x, np.where(boundary, np.nan, sigma), "o-")
            axes[1, column].set_yscale("log" if np.all(sigma > 0) else "symlog",
                                       **({} if np.all(sigma > 0) else {"linthresh": 1e-12}))
            if boundary.any():
                for ax, y in zip(axes[:, column], (z, sigma)):
                    ax.scatter(x[boundary], y[boundary], color="tab:red", marker="x",
                               s=60, zorder=5, label="search boundary")
                print(f"{case}, {sweep} sweep: boundary minima at {x[boundary].tolist()}; widen half_width.")
            axes[0, column].set_title(f"Vary {sweep}; {fixed} = {points[0][fixed]}")
            axes[0, column].set_ylabel(r"Minimizing $z$")
            axes[1, column].set_ylabel(r"$\min_z\,\sigma_{\min}(C-zE)$")
            axes[1, column].set_xlabel(f"{'Patch' if sweep == 'N' else 'Window'} radius {sweep}")
            axes[0, column].legend(fontsize=8)
            for ax in axes[:, column]:
                ax.grid(True, alpha=0.25)
                ax.set_xticks(x)
        first = rows[0]
        # fig.suptitle(f"{case.replace('_', ' ').title()} — Colbrook convergence\n"
        #              f"R={float(first['radius']):g}, R_def={float(first['defect_radius']):g}, "
        #              f"v_bd={float(first['v_bd']):.6g}, N_colbrook=W, fringe=1")
        fig.tight_layout()
        for suffix in ("pdf", "png"):
            destination = output_dir / f"colbrook_convergence_{case}.{suffix}"
            fig.savefig(destination, dpi=180)
            print(f"[wrote] {destination}")
        figures.append(fig)
    if show:
        plt.show()
    return figures


def plot_convergence_combined(filenames=None, output_dir="figures", show=True):
    """Overlay all cases with stable colors in the same four N/W panels.

    Crosses mark boundary hits; dashed lines mark search targets. Error bars
    represent final search-grid spacing, not eigenvalue error estimates.
    """
    datasets = _load_convergence(filenames)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex="col", squeeze=False)
    for column, sweep in enumerate(("N", "W")):
        fixed = "W" if sweep == "N" else "N"
        ticks, fixed_values = set(), set()
        has_zero = False
        for case, rows in datasets:
            points = sorted((r for r in rows if r["sweep"] == sweep), key=lambda r: int(r[sweep]))
            if not points:
                continue
            fixed_value = int(points[0][fixed])
            fixed_values.add(fixed_value)
            x = np.array([int(r[sweep]) for r in points])
            ticks.update(x)
            z = np.array([float(r["z_min"]) for r in points])
            sigma = np.array([float(r["sigma_min"]) for r in points])
            resolution = np.array([float(r["z_resolution"]) for r in points])
            boundary = np.array([r["at_boundary"] == "1" for r in points])
            has_zero |= bool(np.any(sigma == 0))
            color = CONVERGENCE_COLORS[case]
            label = f"{case.replace('_', ' ').title()}"# ({fixed}={fixed_value})"
            axes[0, column].errorbar(x, np.where(boundary, np.nan, z), yerr=resolution,
                                    fmt="o-", color=color, capsize=3, label=label)
            axes[0, column].axhline(float(points[0]["target_z"]), color=color,
                                   ls="--", lw=1, alpha=0.45)
            axes[1, column].loglog(x, np.where(boundary, np.nan, sigma), "o-",
                                 color=color, label=label)
            if boundary.any():
                for ax, y in zip(axes[:, column], (z, sigma)):
                    ax.scatter(x[boundary], y[boundary], color=color, marker="x", s=60, zorder=5)
                print(f"{case}, {sweep} sweep: boundary minima at {x[boundary].tolist()}; widen half_width.")
        title = rf"$\text{{Varying }}{sweep}$"
        if len(fixed_values) == 1:
            title += rf"$; {fixed} = {next(iter(fixed_values))}$"
        elif fixed_values:
            title += rf"$; {fixed} differs by case (see legend)$"
        else:
            title += r"$ (no points yet)$"
        axes[0, column].set_title(title)
        axes[0, column].set_ylabel(r"$\text{Minimizing }z$")
        axes[1, column].set_ylabel(r"$\min_z\,\sigma_{\min}(C_\mathrm{rect}-zE)$")
        name = r"$\text{Patch}$" if sweep == "N" else r"$\text{Window}$"
        axes[1, column].set_xlabel(rf"{name}$\text{{ radius }}{sweep}$")
        axes[1, column].set_yscale("symlog" if has_zero else "log",
                                   **({"linthresh": 1e-12} if has_zero else {}))
        for ax in axes[:, column]:
            ax.grid(True, alpha=0.25)
            ax.set_xticks(sorted(ticks))
            # Keep logarithmic spacing, but label the sampled radii as plain integers.
            formatter = ScalarFormatter()
            formatter.set_scientific(False)
            formatter.set_useOffset(False)
            ax.xaxis.set_major_formatter(formatter)
            ax.xaxis.set_minor_formatter(NullFormatter())
            if ticks:
                ax.legend(loc="lower left",
                            bbox_to_anchor=(0.05, 0.1),
                            fontsize=8,
                            borderaxespad=0)
    # fig.suptitle("Colbrook convergence — all cases; N_colbrook=W, fringe=1\n"
    #              "Dashed: search targets · Crosses: search boundary · Error bars: grid spacing")
    fig.tight_layout()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        destination = output_dir / f"colbrook_convergence_all.{suffix}"
        fig.savefig(destination, dpi=180)
        print(f"[wrote] {destination}")
    if show:
        plt.show()
    return fig


def main():
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    print(f"Sampled local minima (minimum separation in z: {MIN_Z_SEPARATION:g})")
    for n_defect in sorted(PATCH_SIZES, reverse=True):
        filename = resolve(f"bent_waveguide_singular_values{n_defect}.csv")
        patch = np.loadtxt(filename, delimiter=",", ndmin=2)
        patch_z, patch_sigma_min = patch[:, 0], patch[:, 1]
        minima = find_distinct_minima(patch_z, patch_sigma_min)
        values = ", ".join(f"z={patch_z[i]:.6g} (sigma={patch_sigma_min[i]:.6g})" for i in minima)
        print(f"N={n_defect}: {values or 'no interior minima'}")
        if len(minima) < NUM_MINIMA:
            print(f"  Only {len(minima)} distinct minima found; not forcing {NUM_MINIMA}.")

        line, = ax.plot(patch_z, patch_sigma_min, label=rf"$N = {n_defect}$")
        # if n_defect == max(PATCH_SIZES):
        #     ax.scatter(patch_z[minima], patch_sigma_min[minima], marker="x", s=70,
        #                color=line.get_color(), zorder=5)
        #     for i in minima:
        #         ax.annotate(f"z = {patch_z[i]:.6g}", (patch_z[i], patch_sigma_min[i]),
        #                     xytext=(0, 12), textcoords="offset points", ha="center",
        #                     fontsize=9, color=line.get_color())

    ax.set_xlabel("$z$")
    ax.set_ylabel("min singular value of $C_{cb} - z E$")
    ax.legend()
    fig.tight_layout()
    plt.show()

    # quadrupole material F(z) = 0.0108329
    # quadrupole radius F(z) = 0.0554881
    # dipole material F(z) = 0.038105
    # dipole radius F(z) = 0.0566392

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    maxsdiff = []
    n = 0
    for n_window in WINDOW_SIZES:
        filename = resolve(f"bent_waveguide_singular_values_colbr_{n_window}.csv")
        patch = np.loadtxt(filename, delimiter=",", ndmin=2)
        patch_z, patch_sigma_min = patch[:, 0], patch[:, 1]
        minima = find_distinct_minima(patch_z, patch_sigma_min)
        values = ", ".join(f"z={patch_z[i]:.6g} (sigma={patch_sigma_min[i]:.6g})" for i in minima)
        print(f"N={n_window}: {values or 'no interior minima'}")
        if len(minima) < NUM_MINIMA:
            print(f"  Only {len(minima)} distinct minima found; not forcing {NUM_MINIMA}.")
        line, = ax.plot(patch_z, patch_sigma_min, label=rf"$N_\mathrm{{Colbr}} = {n_window}$")
        # # if n_window == max(WINDOW_SIZES):
        # ax.scatter(patch_z[minima], patch_sigma_min[minima], marker="x", s=70,
        #             color=line.get_color(), zorder=5)
        # for i in minima:
        #     ax.annotate(f"z = {patch_z[i]:.6g}", (patch_z[i], patch_sigma_min[i]),
        #                 xytext=(0, 12), textcoords="offset points", ha="center",
        #                 fontsize=9, color=line.get_color())
        # print(f"Window size {n_window}: {minima} minima found.")
        if (n > 0):
            diff = np.abs(patch_sigma_min - last_patchsigma_min)
            maxsdiff.append(diff.max())
            print(f"Window size {n_window}: max difference from previous window: {diff.max()} at z={patch_z[diff.argmax()]:.6g}")
        n+= 1
        last_patchsigma_min = patch_sigma_min
        last_patchz = patch_z


    ax.set_xlabel("$z$")
    ax.set_title("Min singular value of $C_{cb} - z E$ for increasing window/subpatch sizes")
    ax.set_ylabel("min singular value of $C_{cb} - z E$")
    ax.legend()
    fig.tight_layout()
    plt.show()

    # fig = plt.figure(figsize=(7.5, 5.0))
    # plt.semilogy([1, 2, 3, 4, 5], maxsdiff, marker="o")
    # plt.xlabel("Window size index")
    # plt.ylabel("Max difference in min singular value from previous window")
    # plt.title("Convergence of min singular value with increasing window size")
    # plt.grid()
    # plt.tight_layout()
    # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Colbrook scans or N/W convergence summaries.")
    layout = parser.add_mutually_exclusive_group()
    layout.add_argument("--convergence", nargs="*", metavar="CSV",
                        help="plot convergence CSVs; with no filenames, find available cases")
    layout.add_argument("--convergence-combined", nargs="*", metavar="CSV",
                        help="overlay convergence cases in one four-panel figure")
    parser.add_argument("--output-dir", default="figures", help="directory for convergence PDF/PNG files")
    parser.add_argument("--no-show", action="store_true", help="save convergence plots without opening windows")
    args = parser.parse_args()
    if args.convergence_combined is not None:
        plot_convergence_combined(args.convergence_combined, args.output_dir, show=not args.no_show)
    elif args.convergence is not None:
        plot_convergence(args.convergence, args.output_dir, show=not args.no_show)
    else:
        main()
