"""Plot local crystal bands in (alpha_x, alpha_y, omega).

    python src/plot_dirac_cone.py build/multipoleA_dirac_candidate_omega_*.csv
    python src/plot_dirac_cone.py scan.csv --mode cloud --sigma-threshold 0.002

Minima mode follows plot_crystal_matrix_bands.py: only frequency minima below
the threshold are drawn. Cloud mode draws every sub-threshold sample, fading
to fully transparent at the threshold. Neither mode connects separate bands.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


def load_samples(path):
    """Read the named columns so their order in the CSV does not matter."""
    with path.open() as stream:
        names = stream.readline().strip().split(",")
    required = ("alpha_x", "alpha_y", "omega", "sigma_min")
    if not set(required).issubset(names):
        raise ValueError(f"{path}: expected columns {', '.join(required)}; "
                         "regenerate older candidate files with the updated C++ sweep")
    table = np.genfromtxt(path, delimiter=",", names=True, ndmin=1)
    samples = np.column_stack([table[name] for name in required])
    if not len(samples):
        raise ValueError(f"{path}: no samples")
    if not np.isfinite(samples).all() or np.any(samples[:, 3] < 0):
        raise ValueError(f"{path}: samples must be finite with nonnegative sigma_min")
    return samples


def frequency_minima(samples):
    """Find interior minima separately at each Bloch vector, as in the 2D plot."""
    order = np.lexsort((samples[:, 2], samples[:, 1], samples[:, 0]))
    rows = samples[order]
    if np.any(np.all(rows[1:, :3] == rows[:-1, :3], axis=1)):
        raise ValueError("Duplicate (alpha_x, alpha_y, omega) samples")
    same_left = np.all(rows[1:-1, :2] == rows[:-2, :2], axis=1)
    same_right = np.all(rows[1:-1, :2] == rows[2:, :2], axis=1)
    sigma = rows[:, 3]
    minima = (same_left & same_right & (sigma[1:-1] <= sigma[:-2])
              & (sigma[1:-1] <= sigma[2:])
              & ((sigma[1:-1] < sigma[:-2]) | (sigma[1:-1] < sigma[2:])))
    return rows[1:-1][minima]


def plot_samples(samples, args, title):
    frequencies = np.unique(samples[:, 2])
    if len(frequencies) > 1:
        print(f"[grid] {len(frequencies)} frequencies, largest step "
              f"{np.diff(frequencies).max():.6g}; sigma_min range "
              f"{samples[:, 3].min():.6g} to {samples[:, 3].max():.6g}")
    rows = frequency_minima(samples) if args.mode == "minima" else samples
    rows = rows[rows[:, 3] < args.sigma_threshold]
    if not len(rows):
        raise ValueError("No visible band samples: raise --sigma-threshold, use "
                         "--mode cloud, or resample a wider/finer frequency window")

    if args.mode == "minima" and len(np.unique(rows[:, 2])) <= 3:
        print("[resolution] Minima occupy at most 3 frequency bins. Flat sheets can "
              "mean unresolved dispersion; refine the frequency grid to distinguish it "
              "from genuinely flat bands.")
    if args.mode == "cloud" and len(rows) == len(samples):
        print("[opacity] The threshold retains every sample. Lower --sigma-threshold "
              "below the reported sigma_min maximum to hide gap samples.")

    log_sigma = np.log10(np.maximum(rows[:, 3], 1e-15))
    vmax = np.log10(args.sigma_threshold) if args.mode == "cloud" else float(log_sigma.max())
    vmin = min(float(log_sigma.min()), vmax - 1e-6)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("viridis_r")
    colors = cmap(norm(log_sigma))
    if args.mode == "cloud":
        # Large sigma values (gap samples) have zero opacity and are omitted.
        # The power controls how quickly weaker evidence fades from view.
        colors[:, 3] = args.opacity * (1 - rows[:, 3] / args.sigma_threshold) ** args.fade_power
    else:
        colors[:, 3] = args.opacity

    fig = plt.figure(figsize=(9, 7), layout="constrained")
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(rows[:, 0], rows[:, 1], rows[:, 2] - args.omega_offset, c=colors,
               s=args.point_size, linewidths=0, depthshade=False, rasterized=True)
    ax.set(xlabel=r"$\alpha_x$", ylabel=r"$\alpha_y$", zlabel=r"Frequency $\omega$",
           title=title)
    if args.omega_offset:
        ax.set_zlabel(rf"$\omega - {args.omega_offset:.10g}$")
    # Preserve the scan extent even when almost every sample is transparent.
    for axis, setter in enumerate((ax.set_xlim, ax.set_ylim, ax.set_zlim)):
        lo, hi = samples[:, axis].min(), samples[:, axis].max()
        if axis == 2:
            lo, hi = lo - args.omega_offset, hi - args.omega_offset
        if lo < hi:
            setter(lo, hi)
    ax.set_box_aspect((1, 1, 1.1))
    ax.view_init(elev=args.elev, azim=args.azim)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                 shrink=0.65, pad=0.08, label=r"$\log_{10}\sigma_{\min}$")
    print(f"[visible] {len(rows):,} / {len(samples):,} samples ({args.mode})")
    return fig


def positive_float(value):
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", type=Path,
                        help="candidate CSVs; default: discover in cwd or repository build/")
    parser.add_argument("--mode", choices=("minima", "cloud"), default="minima")
    parser.add_argument("--sigma-threshold", type=positive_float, default=0.02,
                        help="samples at or above this value are invisible (default: 0.02)")
    parser.add_argument("--fade-power", type=positive_float, default=2,
                        help="cloud opacity exponent; larger hides more samples (default: 2)")
    parser.add_argument("--opacity", type=positive_float, default=0.9)
    parser.add_argument("--point-size", type=positive_float, default=8)
    parser.add_argument("--omega-range", nargs=2, type=float, metavar=("LOW", "HIGH"))
    parser.add_argument("--omega-offset", type=float, default=0,
                        help="subtract a reference frequency for readable near-degenerate bands")
    parser.add_argument("--elev", type=float, default=22)
    parser.add_argument("--azim", type=float, default=-55)
    parser.add_argument("--output-dir", type=Path, help="default: alongside each CSV")
    parser.add_argument("--no-show", action="store_true", help="save without opening a window")
    args = parser.parse_args()
    if args.opacity > 1:
        parser.error("--opacity must be at most 1")
    if not np.isfinite(args.omega_offset):
        parser.error("--omega-offset must be finite")
    if args.omega_range and (not np.isfinite(args.omega_range).all()
                             or args.omega_range[0] >= args.omega_range[1]):
        parser.error("--omega-range needs finite LOW < HIGH")
    files = args.files
    if not files:
        pattern = "multipoleA_dirac_*.csv"
        files = sorted(Path.cwd().glob(pattern))
        if not files:
            files = sorted((Path(__file__).resolve().parents[1] / "build").glob(pattern))
    if not files:
        parser.error("No candidate CSVs found; run the hexagonal-lattice sweep first")
    succeeded = 0
    for path in files:
        try:
            samples = load_samples(path)
            if args.omega_range:
                lo, hi = args.omega_range
                samples = samples[(samples[:, 2] >= lo) & (samples[:, 2] <= hi)]
                if not len(samples):
                    raise ValueError("No samples within --omega-range")
            fig = plot_samples(samples, args, f"Local crystal bands ({args.mode})")
            output_dir = args.output_dir or path.parent
            output_dir.mkdir(parents=True, exist_ok=True)
            output = output_dir / f"{path.stem}_{args.mode}.png"
            fig.savefig(output, dpi=250)
            print(f"[wrote] {output}")
            succeeded += 1
            if args.no_show:
                plt.close(fig)
        except (OSError, ValueError) as exc:
            print(f"[skipped] {path}: {exc}")
    if not succeeded:
        parser.exit(1, "No plots generated.\n")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
