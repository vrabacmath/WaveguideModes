"""Plot line-defect bands from the current CSV outputs.

Run from the repository root or from build:
    python ../src/plot.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = Path.cwd()

CAP_FILE = "defect_capacitance_bands_leading.csv"
NEUMANN_FILE = "defect_neumann_bands.csv"
PROJECTED_BULK_FILE = "projected_bulk_bands.csv"
OUTPUT_FILE = "neumann_vs_capacitance.pdf"

YLIM = (0, 7.5)  # Example: (3.95, 4.6)
BULK_THRESHOLD = 0.25

plt.rcParams.update({
        "font.size": 18,
        "axes.labelsize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 20,
        "mathtext.fontset": "cm",
        "axes.linewidth": 1.2,
    })


def data_path(filename):
    path = Path(filename)
    if path.is_absolute():
        return path

    candidates = [
        HERE / path,
        HERE / "build" / path,
        ROOT / path,
        ROOT / "build" / path,
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"{filename} not found in current directory, repo root, or build/")


def load_csv(filename, columns):
    path = data_path(filename)
    if path.stat().st_size == 0:
        raise FileNotFoundError(f"{path} is empty")
    rows = []
    for line in path.read_text().splitlines():
        parts = line.split(",")
        if len(parts) >= columns:
            rows.append([float(value) for value in parts[:columns]])
    if not rows:
        raise ValueError(f"{path} has no complete {columns}-column rows")
    return np.array(rows)


def output_path(filename):
    build_dir = HERE if HERE.name == "build" else ROOT / "build"
    build_dir.mkdir(exist_ok=True)
    return build_dir / filename


def plot_projected_bulk(ax):
    try:
        data = load_csv(PROJECTED_BULK_FILE, 4)
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
    return plt.Line2D([], [], marker=".", linestyle="None", c="0.65",
                      label=r"$\text{Bulk bands projected over}~\alpha_y$")


def plot_neumann_bands(ax):
    data = load_csv(NEUMANN_FILE, 6)
    data = data[(data[:, 4] == 1) & (data[:, 5] == 0)]

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


def plot_capacitance_bands(ax):
    data = load_csv(CAP_FILE, 7)
    for m in sorted(set(data[:, 0].astype(int)))[:-1]:
        rows = data[data[:, 0].astype(int) == m]
        ax.plot(rows[:, 1], rows[:, 2], ".", c="black", ms=5)

    rows = data[data[:, 0].astype(int) == sorted(set(data[:, 0].astype(int)))[-1]]
    ax.plot(rows[:, 1], rows[:, 2], ".", c="black", ms=5,
            label=r"$\text{Capacitance}$")


def main():
    fig, ax = plt.subplots(figsize=(6.5, 9))
    bulk_handle = plot_projected_bulk(ax)
    plot_neumann_bands(ax)
    plot_capacitance_bands(ax)

    ax.set_xlabel(r"$\alpha_x$")
    ax.set_ylabel(r"$\omega$")
    ax.set_xlim(0, np.pi)
    ax.set_xticks([0, np.pi])
    ax.set_xticklabels([r"$\Gamma$", r"$X$"])
    if YLIM is not None:
        ax.set_ylim(*YLIM)
    ax.set_title(r"$\text{Line defect bands: } M^\varepsilon \text{ vs. capacitance}$")
    # ax.set_title(r"$\text{Quadrupole bands}$")
    ax.grid(True, alpha=0.25)

    handles, labels = ax.get_legend_handles_labels()
    if bulk_handle is not None:
        handles.append(bulk_handle)
        labels.append(bulk_handle.get_label())
    ax.legend(
        handles,
        labels,
        loc="center right",
        bbox_to_anchor=(0.98, 0.16),
        fontsize=14,
        framealpha=0.9,
    )

    fig.tight_layout()
    out = output_path(OUTPUT_FILE)
    fig.savefig(out, dpi=300)
    plt.show()
    print(f"[wrote] {out}")


if __name__ == "__main__":
    main()
