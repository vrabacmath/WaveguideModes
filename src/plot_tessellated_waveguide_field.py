"""Reconstruct a global mode from the saved central-source window fields.

    python src/plot_tessellated_waveguide_field.py mode.csv --no-show
    python src/plot_tessellated_waveguide_field.py mode.csv --prefix data/tessellated_waveguide
    python src/plot_tessellated_waveguide_field.py --patches --prefix build/tessellated_waveguide

--patches saves each unweighted window as a PNG and all windows in one PDF.
It uses the original local grids, shows both angular bases, and needs no eigenvector.

The eigenvector CSV has a header: ell,basis,real,imag. Include every ell=-M..M
and basis=0..modes-1, including zero coefficients. Basis 0 is +m; basis 1 is -m.
For m=0 there is only basis 0. Rows may be shuffled; # comment lines are ignored.

Complex fields are translated, multiplied by the supplied coefficients, and SUMMED.
Amplitudes and global phase are preserved. Bilinear interpolation propagates any
NaN with nonzero interpolation weight. Each contribution is zero outside its
saved grid; locations outside all grids are blank. This finite-window approximation
does not reconstruct the missing far field or perform a global boundary solve.
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Window:
    axis: np.ndarray
    basis: np.ndarray  # [basis, y, x], complex


def read_table(path, columns):
    with Path(path).open() as stream:
        reader = csv.DictReader(line for line in stream
                                if line.strip() and not line.lstrip().startswith("#"))
        if not set(columns).issubset(reader.fieldnames or ()):
            raise ValueError(f"{path}: expected columns {','.join(columns)}")
        try:
            rows = [{name: float(row[name]) for name in columns} for row in reader]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}: every row must contain finite numeric values for {','.join(columns)}") from exc
    if not rows or not all(np.isfinite(list(row.values())).all() for row in rows):
        raise ValueError(f"{path}: expected nonempty, finite numeric data")
    return rows


def integer(value, name):
    if not np.isfinite(value) or value != int(value):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def load_exports(prefix):
    try:
        params = read_table(f"{prefix}_parameters.csv",
                            ("window_radius", "patch_radius", "m_ang", "modes", "bent", "defect_radius",
                             "fringe", "template_radius", "template_count"))
    except ValueError as exc:
        raise ValueError(f"{exc}. Regenerate the C++ field exports with the straight endpoint templates") from exc
    if len(params) != 1:
        raise ValueError("Expected one parameter row for the exported window set")
    params = params[0]
    for key in ("window_radius", "patch_radius", "m_ang", "modes", "bent", "fringe",
                "template_radius", "template_count"):
        params[key] = integer(params[key], key)
    w, M, modes = params["window_radius"], params["patch_radius"], params["modes"]
    T = params["template_radius"]
    if not (0 <= w <= M and params["m_ang"] >= 0 and params["bent"] in (0, 1)
            and modes == (1 if params["m_ang"] == 0 else 2) and params["defect_radius"] > 0
            and params["fringe"] >= 0 and T == w + params["fringe"] + 1
            and params["template_count"] == 2 * T + 1):
        raise ValueError("Invalid or inconsistent window parameters")
    sites = read_table(f"{prefix}_sites.csv", ("ell", "x", "y", "window_index", "coefficient_start"))
    sites.sort(key=lambda site: site["ell"])
    if [site["ell"] for site in sites] != list(range(-M, M + 1)):
        raise ValueError("Sites must contain each global label ell=-patch_radius..patch_radius once")
    for q, site in enumerate(sites):
        ell = int(site["ell"])
        site["ell"] = ell
        site["window_index"] = integer(site["window_index"], "window_index")
        xy = ((-ell, 0) if ell <= 0 else (0, ell)) if params["bent"] else (ell, 0)
        if ((site["x"], site["y"]) != xy or site["coefficient_start"] != modes * q
                or site["window_index"] != np.clip(ell, -T, T) + T):
            raise ValueError(f"Inconsistent global mapping at ell={ell}")
    windows = {}
    for k in range(params["template_count"]):
        base = f"{prefix}_patch_{k}"
        axis = np.loadtxt(f"{base}_axis.csv", delimiter=",", ndmin=1)
        if (axis.ndim != 1 or axis.size < 2 or not np.isfinite(axis).all()
                or np.any(np.diff(axis) <= 0)):
            raise ValueError(f"{base}: expected a finite, increasing coordinate axis")
        fields = []
        for s in range(modes):
            real = np.loadtxt(f"{base}_basis_{s}_real.csv", delimiter=",", ndmin=2)
            imag = np.loadtxt(f"{base}_basis_{s}_imag.csv", delimiter=",", ndmin=2)
            if real.shape != (axis.size, axis.size) or imag.shape != real.shape:
                raise ValueError(f"{base}, basis {s}: grid shape does not match axis")
            if np.isinf(real).any() or np.isinf(imag).any():
                raise ValueError(f"{base}, basis {s}: field contains infinities")
            fields.append(real + 1j * imag)
        windows[k] = Window(axis, np.array(fields))
    return params, sites, windows


def load_eigenvector(path, sites, modes):
    rows = read_table(path, ("ell", "basis", "real", "imag"))
    site_index = {site["ell"]: q for q, site in enumerate(sites)}
    coefficients = np.zeros((len(sites), modes), dtype=complex)
    seen = set()
    for row in rows:
        ell, s = integer(row["ell"], "ell"), integer(row["basis"], "basis")
        if ell not in site_index or not 0 <= s < modes:
            raise ValueError(f"{path}: unexpected coefficient ell={ell}, basis={s}")
        if (ell, s) in seen:
            raise ValueError(f"{path}: duplicate coefficient ell={ell}, basis={s}")
        seen.add((ell, s))
        coefficients[site_index[ell], s] = complex(row["real"], row["imag"])
    if len(seen) != coefficients.size:
        raise ValueError(f"{path}: expected {coefficients.size} coefficients, got {len(seen)}; "
                         "include zero coefficients too")
    if not np.any(coefficients):
        raise ValueError(f"{path}: the entire mode vector is zero")
    return coefficients


def interpolate(axis, field, x, y):
    """Sample a Cartesian grid inside the local domain without filling masked values."""
    def brackets(coords):
        lo = np.clip(np.searchsorted(axis, coords, side="right") - 1, 0, len(axis) - 2)
        fraction = np.clip((coords - axis[lo]) / (axis[lo + 1] - axis[lo]), 0, 1)
        # Exact nodes should not acquire NaNs from a zero-weight adjacent sample.
        fraction[np.abs(fraction) < 1e-12] = 0
        fraction[np.abs(fraction - 1) < 1e-12] = 1
        return lo, fraction

    ix, fx = brackets(x)
    iy, fy = brackets(y)
    result = np.zeros((len(y), len(x)), dtype=complex)
    invalid = np.zeros(result.shape, dtype=bool)
    for dy, wy in ((0, 1 - fy), (1, fy)):
        for dx, wx in ((0, 1 - fx), (1, fx)):
            weight = wy[:, None] * wx[None, :]
            values = field[np.ix_(iy + dy, ix + dx)]
            finite = np.isfinite(values)
            invalid |= ~finite & (weight > 0)
            result += np.where(finite, values, 0) * weight
    result[invalid] = complex(np.nan, np.nan)
    return result


def assemble_field(sites, windows, coefficients, spacing=None, extent=None):
    if spacing is None:
        spacing = min(np.diff(window.axis).min() for window in windows.values())
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("Grid spacing must be finite and positive")
    if extent is None:
        extent = [min(site["x"] + windows[site["window_index"]].axis[0] for site in sites),
                  max(site["x"] + windows[site["window_index"]].axis[-1] for site in sites),
                  min(site["y"] + windows[site["window_index"]].axis[0] for site in sites),
                  max(site["y"] + windows[site["window_index"]].axis[-1] for site in sites)]
    xmin, xmax, ymin, ymax = extent
    if not np.isfinite(extent).all() or xmin >= xmax or ymin >= ymax:
        raise ValueError("Extent requires finite xmin < xmax and ymin < ymax")
    nx = int(np.ceil((xmax - xmin) / spacing - 1e-10)) + 1
    ny = int(np.ceil((ymax - ymin) / spacing - 1e-10)) + 1
    if nx * ny > 16_000_000:
        raise ValueError("Grid exceeds 16 million points; increase --spacing or restrict --extent")
    x = xmin + spacing * np.arange(nx)
    y = ymin + spacing * np.arange(ny)
    field = np.zeros((ny, nx), dtype=complex)
    coverage = np.zeros((ny, nx), dtype=np.int32)
    invalid = np.zeros((ny, nx), dtype=bool)
    for site, amplitudes in zip(sites, coefficients):
        window = windows[site["window_index"]]
        axis = window.axis
        ix = np.flatnonzero((x >= site["x"] + axis[0] - 1e-12)
                           & (x <= site["x"] + axis[-1] + 1e-12))
        iy = np.flatnonzero((y >= site["y"] + axis[0] - 1e-12)
                           & (y <= site["y"] + axis[-1] + 1e-12))
        if not ix.size or not iy.size:
            continue
        block = np.ix_(iy, ix)
        coverage[block] += 1
        active = np.flatnonzero(amplitudes != 0)
        if not active.size:
            continue  # In particular, 0 * NaN must not mask another source's field.
        local = sum(amplitudes[s] * window.basis[s] for s in active)
        values = interpolate(axis, local, x[ix] - site["x"], y[iy] - site["y"])
        finite = np.isfinite(values)
        field[block] += np.where(finite, values, 0)
        invalid[block] |= ~finite
    # field[invalid | (coverage == 0)] = complex(np.nan, np.nan)
    field[invalid] = complex(np.nan, np.nan)
    return x, y, field, coverage


def plot_field(x, y, field, sites, radius, title, coefficients):
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Circle

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), layout="constrained")
    magnitude = np.abs(field)
    positive = magnitude[np.isfinite(magnitude) & (magnitude > 0)]
    if positive.size:
        vmax = positive.max()
        floor = max(vmax * 1e-6, np.finfo(float).tiny)
        image = axes[0].pcolormesh(x, y, np.maximum(magnitude, floor), shading="nearest",
                                   cmap="magma", norm=LogNorm(vmin=floor, vmax=max(vmax, 10 * floor)),
                                   rasterized=True)
    else:
        image = axes[0].pcolormesh(x, y, magnitude, shading="nearest", cmap="magma",
                                   vmin=0, vmax=1, rasterized=True)
    fig.colorbar(image, ax=axes[0], label=r"$|u|$")
    finite = field.real[np.isfinite(field)]
    limit = max(float(np.max(np.abs(finite))), 1e-15)
    image = axes[1].pcolormesh(x, y, field.real, shading="nearest", cmap="RdBu_r",
                               vmin=-limit, vmax=limit, rasterized=True)
    fig.colorbar(image, ax=axes[1], label=r"$\mathrm{Re}\,u$")
    block_norms = np.linalg.norm(coefficients, axis=1)
    disks = PatchCollection([Circle((site["x"], site["y"]), radius) for site in sites],
                            cmap="magma", edgecolor="0.6", linewidth=0.35)
    disks.set_array(block_norms)
    disks.set_clim(0, max(float(block_norms.max()), 1e-15))
    axes[2].add_collection(disks)
    fig.colorbar(disks, ax=axes[2], label=r"$\|a_\ell\|_2$")
    for ax, label in zip(axes, ("Magnitude", "Real part", "Eigenvector block norms")):
        # Outline the global source resonators; local cladding meshes can overlap differently.
        if ax is not axes[2]:
            circles = [Circle((site["x"], site["y"]), radius) for site in sites]
            ax.add_collection(PatchCollection(circles, facecolor="none", edgecolor="0.6", linewidth=0.35))
        ax.set(xlabel="$x$", ylabel="$y$", title=label, aspect="equal",
               xlim=(x[0], x[-1]), ylim=(y[0], y[-1]))
    fig.suptitle(f"{title}\nSuperposition of finite window fields")
    return fig


def window_disk_roles(params, disks):
    """Retain all defect centers in the local square; only the outer fringe is excluded."""
    roles = []
    for disk in disks:
        xy = (disk["x"], disk["y"])
        if disk["is_source"]:
            role = "source"
        elif not disk["is_defect"]:
            role = "cladding"
        elif max(abs(xy[0]), abs(xy[1])) > params["window_radius"]:
            role = "fringe_defect"
        else:
            role = "retained"
        roles.append(role)
    return roles


def window_color_limits(windows):
    """One scale for each component across every window and angular basis."""
    limits = np.zeros(3)
    for window in windows.values():
        values = window.basis[np.isfinite(window.basis)]
        if values.size:
            limits = np.maximum(limits, [np.abs(values).max(), np.abs(values.real).max(),
                                         np.abs(values.imag).max()])
    return limits


def plot_window(k, window, params, metadata, disks, limits):
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.colors import LogNorm, Normalize
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, Patch, Rectangle

    styles = {"source": ("#e69f00", 1.6, "Source at (0,0)"),
              "retained": ("#00a6c8", 1.0, "Retained window defects"),
              "fringe_defect": ("#cc79a7", 1.1, "Fringe defects"),
              "cladding": ("#777777", 0.5, "Cladding")}
    roles = window_disk_roles(params, disks)
    axis = window.axis
    modes = window.basis.shape[0]
    fig, axes = plt.subplots(modes, 3, figsize=(15, 3.7 * modes + 1.5), squeeze=False,
                             layout="constrained")
    vmax = max(float(limits[0]), 1e-15)
    norms = [LogNorm(vmin=vmax * 1e-6, vmax=vmax),
             Normalize(vmin=-max(limits[1], 1e-15), vmax=max(limits[1], 1e-15)),
             Normalize(vmin=-max(limits[2], 1e-15), vmax=max(limits[2], 1e-15))]
    cmaps = [plt.get_cmap(name).with_extremes(bad="#dedede")
             for name in ("magma", "RdBu_r", "RdBu_r")]
    images = []
    core = params["window_radius"] + 0.5  # boundary between lattice cells, not disk boundaries
    outer = params["window_radius"] + params["fringe"] + 0.5
    for s in range(modes):
        field = window.basis[s]  # No coefficients, phase rotation, interpolation, or summation.
        components = [np.maximum(np.abs(field), norms[0].vmin), field.real, field.imag]
        angular_order = params["m_ang"] if s == 0 else -params["m_ang"]
        for column, (values, label) in enumerate(zip(components, (r"$|u|$", r"$\mathrm{Re}\,u$", r"$\mathrm{Im}\,u$"))):
            ax = axes[s, column]
            image = ax.pcolormesh(axis, axis, values, shading="nearest", cmap=cmaps[column],
                                  norm=norms[column], rasterized=True)
            if s == 0:
                images.append(image)
            if params["fringe"]:
                for left, bottom, width, height in [(-outer, -outer, outer-core, 2*outer),
                                                    (core, -outer, outer-core, 2*outer),
                                                    (-core, -outer, 2*core, outer-core),
                                                    (-core, core, 2*core, outer-core)]:
                    ax.add_patch(Rectangle((left, bottom), width, height, facecolor="#e69f00",
                                           alpha=0.10, edgecolor="none"))
                ax.add_patch(Rectangle((-core, -core), 2*core, 2*core, fill=False,
                                       edgecolor="#e69f00", linewidth=1.3, linestyle="--"))
            # Enclose every lattice cell in the solved mesh, including its fringe.
            # The exported field grid extends farther, into the exterior medium.
            ax.add_patch(Rectangle((-outer, -outer), 2*outer, 2*outer, fill=False,
                                   edgecolor="#0072b2", linewidth=1.5, linestyle="-"))
            for role, (color, width, _) in styles.items():
                circles = [Circle((disk["x"], disk["y"]), disk["radius"])
                           for disk, assigned in zip(disks, roles) if assigned == role]
                ax.add_collection(PatchCollection(circles, facecolor="none", edgecolor=color,
                                                  linewidth=width))
            ax.set(xlabel="Local x", ylabel="Local y", aspect="equal",
                   xlim=(axis[0], axis[-1]), ylim=(axis[0], axis[-1]),
                   title=f"Basis {s}: angular order {angular_order:+d} — {label}")
    for column, label in enumerate((r"$|u|$ (log scale)", r"$\mathrm{Re}\,u$", r"$\mathrm{Im}\,u$")):
        fig.colorbar(images[column], ax=axes[:, column].tolist(), shrink=0.75, label=label)
    t = integer(metadata["t"], "t")
    if metadata["is_straight"]:
        kind = "Straight horizontal arm" if not params["bent"] or t < 0 else "Straight vertical arm"
    else:
        kind = "Bend window"
    fig.suptitle(f"Window {k} / {params['template_count'] - 1} · {kind}\n"
                 f"Source ell={t:+d}, global center=({metadata['source_x']:g}, {metadata['source_y']:g}), "
                 f"mesh disk {int(metadata['center_disk'])} · unit-amplitude bases\n"
                 f"Window: |x|, |y| ≤ {params['window_radius']} (disk centers); full mesh: "
                 f"|x|, |y| ≤ {params['window_radius'] + params['fringe']}\n"
                 "Common color scales across all windows; gray samples are masked")
    handles = [Line2D([], [], marker="o", linestyle="none", markerfacecolor="none",
                      markeredgecolor=color, label=label) for color, _, label in styles.values()]
    if params["fringe"]:
        handles.append(Line2D([], [], color="#e69f00", linestyle="--", label="Window square (cell edges)"))
        handles.append(Patch(facecolor="#e69f00", alpha=0.3, label="Geometric fringe"))
    handles.append(Line2D([], [], color="#0072b2", label="Full mesh window (including fringe)"))
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=9)
    return fig


def save_window_gallery(prefix, params, windows, output, dpi=200):
    """Write one PNG per window and a PDF with one page per window; close pages as we go."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    rows = read_table(f"{prefix}_windows.csv",
                      ("window_index", "t", "source_x", "source_y", "center_disk", "is_straight"))
    metadata = {integer(row["window_index"], "window_index"): row for row in rows}
    if len(metadata) != len(rows) or set(metadata) != set(windows):
        raise ValueError("The window metadata must contain every template exactly once")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    limits = window_color_limits(windows)
    pdf_path = Path(f"{output}.pdf")
    with PdfPages(pdf_path) as pdf:
        for k, window in sorted(windows.items()):
            disks = read_table(f"{prefix}_patch_{k}_disks.csv",
                               ("disk", "x", "y", "radius", "is_source", "is_defect"))
            fig = plot_window(k, window, params, metadata[k], disks, limits)
            png_path = Path(f"{output}_patch_{k:02d}.png")
            try:
                fig.savefig(png_path, dpi=dpi)
                pdf.savefig(fig, dpi=dpi)
            finally:
                plt.close(fig)
            print(f"[wrote] {png_path}", flush=True)
    print(f"[wrote] {pdf_path} ({len(windows)} pages)")
    return pdf_path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("eigenvector", type=Path, nargs="?", help="CSV with ell,basis,real,imag columns")
    parser.add_argument("--patches", action="store_true",
                        help="save every unweighted window as PNG and a combined PDF; no eigenvector needed")
    parser.add_argument("--prefix", type=Path, default=Path("tessellated_waveguide"),
                        help="C++ export prefix, including its directory")
    parser.add_argument("--output", type=Path,
                        help="output prefix; default: <eigenvector stem>_field or <basis prefix>_windows")
    parser.add_argument("--spacing", type=float, help="global grid step; default: finest local grid step")
    parser.add_argument("--extent", type=float, nargs=4, metavar=("XMIN", "XMAX", "YMIN", "YMAX"))
    parser.add_argument("--dpi", type=int, default=200, help="saved image resolution (default: 200)")
    parser.add_argument("--no-show", action="store_true", help="save plots without opening a window")
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    if args.patches and (args.eigenvector or args.spacing is not None or args.extent is not None):
        parser.error("--patches uses the full native window grids; omit eigenvector, --spacing, and --extent")
    if not args.patches and args.eigenvector is None:
        parser.error("provide an eigenvector CSV, or use --patches to inspect the unweighted bases")
    if args.no_show or args.patches:
        import matplotlib
        matplotlib.use("Agg")
    try:
        params, sites, windows = load_exports(args.prefix)
        if args.patches:
            output = args.output or Path(f"{args.prefix}_windows")
            save_window_gallery(args.prefix, params, windows, output, args.dpi)
            return
        coefficients = load_eigenvector(args.eigenvector, sites, params["modes"])
        x, y, field, coverage = assemble_field(sites, windows, coefficients, args.spacing, args.extent)
        if not np.isfinite(field).any():
            raise ValueError("No valid field samples in this extent; inspect the window masks or grid spacing")
        fig = plot_field(x, y, field, sites, params["defect_radius"], args.eigenvector.stem,
                         coefficients)
        output = args.output or args.eigenvector.with_name(args.eigenvector.stem + "_field")
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(f"{output}.npz", x=x, y=y, u=field, coverage=coverage,
                            ell=np.array([site["ell"] for site in sites]), coefficients=coefficients,
                            basis_prefix=str(args.prefix.resolve()), outside_window="zero")
        for suffix in ("png", "pdf"):
            fig.savefig(f"{output}.{suffix}", dpi=args.dpi)
        print(f"[wrote] {output}.png, {output}.pdf, {output}.npz")
        print(f"[grid] {len(x)} x {len(y)}; {np.isfinite(field).sum()} valid samples")
        print("Contributions outside each saved window are zero; overlapping complex fields are summed.")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    import matplotlib.pyplot as plt
    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
