"""Reconstruct an interface field using interface_field.cxx exports and a mode CSV.

From build/:
    python ../src/plot_tessellated_interface_field.py interface_singular_vector_radius_10.csv --no-show
    python ../src/plot_tessellated_interface_field.py mode.csv --repeats 21
    python ../src/plot_tessellated_interface_field.py --patches

Templates are reused at every (cell_x, cell_y), with that cell's actual complex
coefficients multiplied by source_speed. Basis column = disk*modes + basis.
No amplitudes are copied periodically. --repeats N sets the displayed horizontal
width to N lattice periods; it neither changes the coefficients nor creates new
ones outside the patch. By default the entire source patch is shown in a square
physical viewport (including the shear of nonrectangular lattices).

The exported grids are finite local approximations: contributions outside each
local grid are zero; uncovered points and near-boundary masks remain blank.
Use matching exports and eigenvectors from the same geometry/frequency run.
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from neumann_modes import decode_nm


@dataclass
class Window:
    axis: np.ndarray
    basis: np.ndarray  # [disk*modes + angular_basis, y, x], complex
    sources: list
    disks: list


def read_table(path, columns):
    with Path(path).open() as stream:
        reader = csv.DictReader(line for line in stream
                               if line.strip() and not line.lstrip().startswith("#"))
        if not set(columns).issubset(reader.fieldnames or ()):
            raise ValueError(f"{path}: expected columns {','.join(columns)}")
        try:
            rows = [{name: float(row[name]) for name in columns} for row in reader]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}: malformed numeric row") from exc
    if not rows or not all(np.isfinite(list(row.values())).all() for row in rows):
        raise ValueError(f"{path}: expected nonempty, finite numeric data")
    return rows


def integer(value, name):
    if not np.isfinite(value) or value != int(value):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def load_exports(prefix):
    names = ("format_version", "window_radius", "patch_radius", "fringe", "template_count",
             "nm", "n_rad", "m_ang", "modes", "omega0", "beta", "a1_x", "a1_y",
             "up_a2_x", "up_a2_y", "down_a2_x", "down_a2_y")
    try:
        rows = read_table(f"{prefix}_parameters.csv", names)
    except ValueError as exc:
        raise ValueError(f"{exc}. Regenerate with the updated interface_field exporter; "
                         "old exports omit source-disk columns.") from exc
    if len(rows) != 1:
        raise ValueError("Expected exactly one parameter row")
    params = rows[0]
    for key in names[:9]:
        params[key] = integer(params[key], key)
    n, m = decode_nm(params["nm"])
    R = params["window_radius"] + params["fringe"]
    if (params["format_version"] != 2 or params["window_radius"] < 0 or params["fringe"] < 0
            or params["template_count"] != 2 * R + 2 or params["n_rad"] != n
            or params["m_ang"] != m or params["modes"] != (1 if m == 0 else 2)):
        raise ValueError("Inconsistent interface template parameters")
    if params["a1_x"] <= 0 or abs(params["a1_y"]) > 1e-12:
        raise ValueError("Expected a positive horizontal interface period")
    metadata = read_table(f"{prefix}_windows.csv",
                          ("window_index", "source_y", "source_count", "basis_count"))
    if sorted(row["window_index"] for row in metadata) != list(range(2 * R + 2)):
        raise ValueError("Window metadata must contain each template once")
    windows = {}
    for row in metadata:
        k = integer(row["window_index"], "window_index")
        count = integer(row["basis_count"], "basis_count")
        if row["source_y"] != R - k or count != row["source_count"] * params["modes"]:
            raise ValueError(f"Window {k}: inconsistent source layout")
        base = f"{prefix}_patch_{k}"
        axis = np.loadtxt(f"{base}_axis.csv", delimiter=",", ndmin=1)
        if (axis.ndim != 1 or axis.size < 2 or not np.isfinite(axis).all()
                or np.any(np.diff(axis) <= 0)):
            raise ValueError(f"{base}: expected a finite, increasing coordinate axis")
        sources = read_table(f"{base}_sources.csv",
                             ("disk", "mesh_index", "basis_start", "radius", "center_x", "center_y", "source_speed"))
        sources.sort(key=lambda source: source["disk"])
        disks = read_table(f"{base}_disks.csv", ("radius", "center_x", "center_y", "v_b"))
        if len(sources) != row["source_count"]:
            raise ValueError(f"{base}: source count mismatch")
        for a, source in enumerate(sources):
            mesh_index = integer(source["mesh_index"], "mesh_index")
            if (source["disk"] != a or source["basis_start"] != a * params["modes"]
                    or not 0 <= mesh_index < len(disks) or source["radius"] <= 0
                    or source["source_speed"] <= 0):
                raise ValueError(f"{base}: invalid source-disk mapping")
            disk = disks[mesh_index]
            if not np.allclose([source[t] for t in ("radius", "center_x", "center_y", "source_speed")],
                               [disk[t] for t in ("radius", "center_x", "center_y", "v_b")],
                               rtol=1e-10, atol=1e-12):
                raise ValueError(f"{base}: source geometry mismatch")
        fields = []
        for s in range(count):
            real = np.loadtxt(f"{base}_basis_{s}_real.csv", delimiter=",", ndmin=2)
            imag = np.loadtxt(f"{base}_basis_{s}_imag.csv", delimiter=",", ndmin=2)
            if real.shape != (axis.size, axis.size) or imag.shape != real.shape:
                raise ValueError(f"{base}, basis {s}: grid shape does not match axis")
            if np.isinf(real).any() or np.isinf(imag).any():
                raise ValueError(f"{base}, basis {s}: field contains infinities")
            fields.append(real + 1j * imag)
        windows[k] = Window(axis, np.array(fields), sources, disks)
    return params, windows


def load_eigenvector(path, params, windows):
    # Check optional run metadata when present, before combining unrelated runs.
    with Path(path).open() as stream:
        for line in stream:
            if line.lstrip().startswith("#"):
                for item in line.lstrip()[1:].split(","):
                    key, separator, value = item.partition("=")
                    key = key.strip()
                    if separator and key in ("nm", "n_rad", "m_ang", "omega0", "beta"):
                        if not np.isclose(float(value), params[key], rtol=1e-10, atol=1e-12):
                            raise ValueError(f"Eigenvector {key} does not match the field templates")
    columns = ("dof", "cell_x", "cell_y", "template_id", "origin_x", "origin_y",
               "disk", "basis", "source_speed", "real", "imag")
    rows = read_table(path, columns)
    sites, seen, dofs = {}, set(), set()
    R, modes = params["window_radius"] + params["fringe"], params["modes"]
    for row in rows:
        for key in ("dof", "cell_x", "cell_y", "template_id", "disk", "basis"):
            row[key] = integer(row[key], key)
        x, y = row["cell_x"], row["cell_y"]
        k = int(np.clip(R - y, 0, 2 * R + 1))
        if row["template_id"] != k:
            raise ValueError(f"Wrong template for cell {(x, y)}")
        a, s = row["disk"], row["basis"]
        window = windows[k]
        if not 0 <= a < len(window.sources) or not 0 <= s < modes:
            raise ValueError(f"Invalid disk/basis at cell {(x, y)}")
        origin = np.array([x * params["a1_x"], x * params["a1_y"]])
        side, height = ("up", y) if y >= 0 else ("down", -y - 1)
        origin += height * np.array([params[f"{side}_a2_x"], params[f"{side}_a2_y"]])
        if not np.allclose(origin, [row["origin_x"], row["origin_y"]], rtol=1e-10, atol=1e-12):
            raise ValueError(f"Inconsistent cell origin at {(x, y)}")
        if not np.isclose(row["source_speed"], window.sources[a]["source_speed"], rtol=1e-10, atol=1e-12):
            raise ValueError(f"Source speed mismatch at cell {(x, y)}, disk {a}")
        key = (x, y, a, s)
        if key in seen or row["dof"] in dofs or row["dof"] < 0:
            raise ValueError("Duplicate coefficient or invalid DOF index")
        seen.add(key)
        dofs.add(row["dof"])
        if (x, y) not in sites:
            sites[x, y] = dict(cell_x=x, cell_y=y, x=origin[0], y=origin[1], window_index=k,
                               coefficients=np.zeros(len(window.basis), dtype=complex))
        sites[x, y]["coefficients"][a * modes + s] = row["source_speed"] * complex(row["real"], row["imag"])
    radius = max(max(abs(x), abs(y)) for x, y in sites)
    expected = {(x, y) for x in range(-radius, radius + 1) for y in range(-radius, radius + 1)}
    if set(sites) != expected or len(seen) != sum(len(site["coefficients"]) for site in sites.values()):
        raise ValueError("Expected the complete square patch eigenvector, including zero coefficients")
    if dofs != set(range(len(rows))):
        raise ValueError("DOF indices must run from 0 through the vector length minus one")
    sites = [sites[key] for key in sorted(sites)]
    if not any(np.any(site["coefficients"]) for site in sites):
        raise ValueError("The entire eigenvector is zero")
    return sites, radius


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


def global_disks(sites, windows):
    return [(site["x"] + disk["center_x"], site["y"] + disk["center_y"], disk["radius"])
            for site in sites for disk in windows[site["window_index"]].sources]


def display_extent(sites, windows, period, repeats=None):
    disks = global_disks(sites, windows)
    xmin = min(x - r for x, y, r in disks) - period / 2
    xmax = max(x + r for x, y, r in disks) + period / 2
    ymin = min(y - r for x, y, r in disks) - period / 2
    ymax = max(y + r for x, y, r in disks) + period / 2
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    if repeats is None:
        half = max(xmax - xmin, ymax - ymin) / 2
        return (cx - half, cx + half, cy - half, cy + half)
    if integer(repeats, "repeats") <= 0:
        raise ValueError("--repeats must be a positive integer")
    return (cx - repeats * period / 2, cx + repeats * period / 2, ymin, ymax)


def assemble_field(sites, windows, extent, spacing=None):
    if spacing is None:
        spacing = min(np.diff(window.axis).min() for window in windows.values())
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("Grid spacing must be finite and positive")
    xmin, xmax, ymin, ymax = extent
    if not np.isfinite(extent).all() or xmin >= xmax or ymin >= ymax:
        raise ValueError("Extent requires finite xmin < xmax and ymin < ymax")
    nx = int(np.ceil((xmax - xmin) / spacing)) + 1
    ny = int(np.ceil((ymax - ymin) / spacing)) + 1
    if nx * ny > 16_000_000:
        raise ValueError("Grid exceeds 16 million points; increase --spacing or restrict --extent")
    x, y = np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)
    field = np.zeros((ny, nx), dtype=complex)
    coverage = np.zeros((ny, nx), dtype=np.int32)
    invalid = np.zeros((ny, nx), dtype=bool)
    for site in sites:
        window = windows[site["window_index"]]
        amplitudes, axis = site["coefficients"], window.axis
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
            continue  # 0 * NaN must not mask a different source's field.
        local = sum(amplitudes[s] * window.basis[s] for s in active)
        values = interpolate(axis, local, x[ix] - site["x"], y[iy] - site["y"])
        finite = np.isfinite(values)
        field[block] += np.where(finite, values, 0)
        invalid[block] |= ~finite
    field[invalid | (coverage == 0)] = complex(np.nan, np.nan)
    return x, y, field, coverage


def eigenvector_block_norms(sites, windows):
    """One raw eigenvector norm per disk, in global_disks order (undo speed scaling)."""
    norms = []
    for site in sites:
        sources = windows[site["window_index"]].sources
        blocks = site["coefficients"].reshape(len(sources), -1)
        norms.extend(np.linalg.norm(blocks, axis=1)
                     / np.array([disk["source_speed"] for disk in sources]))
    return np.asarray(norms)


def plot_field(x, y, field, disks, title, block_norms=None):
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Circle

    values = field[np.isfinite(field)]
    if not values.size:
        raise ValueError("No finite samples in the viewport; check masks and grid spacing")
    vmax = max(float(np.abs(values).max()), 1e-15)
    limit = max(float(np.abs(values.real).max()), 1e-15)
    panels = 3 if block_norms is not None else 2
    fig, axes = plt.subplots(1, panels, figsize=(6 * panels, 6), layout="constrained")
    axes[-1].set_facecolor("black")
    if block_norms is not None:
        colors = PatchCollection([Circle((dx, dy), r) for dx, dy, r in disks],
                                 cmap="magma", edgecolor="0.6", linewidth=0.35)
        colors.set_array(block_norms)
        colors.set_clim(0, max(float(block_norms.max()), 1e-15))
        axes[2].add_collection(colors)
        fig.colorbar(colors, ax=axes[2], shrink=0.8, label=r"$\|a_{\mathrm{cell,disk}}\|_2$")
        axes[2].set(xlabel="x", ylabel="y", title="Eigenvector block norms", aspect="equal",
                    xlim=(x[0], x[-1]), ylim=(y[0], y[-1]))
    for ax, component, label, cmap, scale in zip(
            axes, (np.maximum(np.abs(field), vmax * 1e-6), field.real),
            (r"$|u|$", r"$\mathrm{Re}\,u$"), ("magma", "RdBu_r"),
            (dict(norm=LogNorm(vmin=vmax * 1e-6, vmax=vmax)), dict(vmin=-limit, vmax=limit))):
        image = ax.pcolormesh(x, y, component, shading="nearest", cmap=cmap, rasterized=True, **scale)
        fig.colorbar(image, ax=ax, shrink=0.8)
        ax.add_collection(PatchCollection([Circle((dx, dy), r) for dx, dy, r in disks],
                                          facecolor="none", edgecolor="0.6", linewidth=0.35))
        ax.set(xlabel="x", ylabel="y", title=label, aspect="equal",
               xlim=(x[0], x[-1]), ylim=(y[0], y[-1]))
    fig.suptitle(title)
    return fig


def save_window_gallery(params, windows, output, dpi):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(f"{output}.pdf") as pdf:
        for k, window in sorted(windows.items()):
            disks = [(d["center_x"], d["center_y"], d["radius"]) for d in window.disks]
            for s, field in enumerate(window.basis):
                disk, angular_basis = divmod(s, params["modes"])
                angular = params["m_ang"] * (1 if angular_basis == 0 else -1)
                fig = plot_field(window.axis, window.axis, field, disks,
                                 f"Template {k}, source disk {disk}, angular order {angular:+d} (unit trace)")
                try:
                    fig.savefig(f"{output}_patch_{k}_basis_{s}.png", dpi=dpi)
                    pdf.savefig(fig, dpi=dpi)
                finally:
                    plt.close(fig)
    print(f"[wrote] {output}.pdf and individual basis PNGs")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("eigenvector", type=Path, nargs="?", help="interface-modes singular-vector CSV")
    parser.add_argument("--prefix", type=Path, default=Path("tessellated_interface"), help="interface_field export prefix")
    parser.add_argument("--output", type=Path, help="output prefix for PNG/PDF/NPZ")
    parser.add_argument("--repeats", "-N", type=int, help="display width in x lattice periods (actual coefficients are always used)")
    parser.add_argument("--spacing", type=float, help="maximum grid step; default: finest local step")
    parser.add_argument("--extent", nargs=4, type=float, metavar=("XMIN", "XMAX", "YMIN", "YMAX"))
    parser.add_argument("--patches", action="store_true", help="plot all unweighted template bases without a mode vector")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if args.dpi <= 0 or (args.repeats is not None and args.repeats <= 0):
        parser.error("--dpi and --repeats must be positive")
    if args.extent is not None and args.repeats is not None:
        parser.error("choose --extent or --repeats")
    if args.patches and (args.eigenvector or args.spacing is not None or args.extent is not None or args.repeats is not None):
        parser.error("--patches uses the native template grids; omit the eigenvector and viewport options")
    if not args.patches and args.eigenvector is None:
        parser.error("provide an eigenvector CSV, or use --patches")
    if args.no_show or args.patches:
        import matplotlib
        matplotlib.use("Agg")
    try:
        params, windows = load_exports(args.prefix)
        output = args.output or (Path(f"{args.prefix}_windows") if args.patches else
                                 args.eigenvector.with_name(args.eigenvector.stem + "_field"))
        output.parent.mkdir(parents=True, exist_ok=True)
        if args.patches:
            save_window_gallery(params, windows, output, args.dpi)
            return
        sites, radius = load_eigenvector(args.eigenvector, params, windows)
        extent = args.extent or display_extent(sites, windows, params["a1_x"], args.repeats)
        x, y, field, coverage = assemble_field(sites, windows, extent, args.spacing)
        fig = plot_field(x, y, field, global_disks(sites, windows),
                         f"nm={params['nm']}, {2 * radius + 1} × {2 * radius + 1} source cells",
                         eigenvector_block_norms(sites, windows))
        starts = np.cumsum([0] + [len(site["coefficients"]) for site in sites])
        np.savez_compressed(f"{output}.npz", x=x, y=y, u=field, coverage=coverage,
                            cells=np.array([(s["cell_x"], s["cell_y"]) for s in sites]),
                            coefficient_offsets=starts, coefficients=np.concatenate([s["coefficients"] for s in sites]),
                            coefficients_include_source_speed=True, basis_prefix=str(args.prefix.resolve()),
                            eigenvector=str(args.eigenvector.resolve()))
        for suffix in ("png", "pdf"):
            fig.savefig(f"{output}.{suffix}", dpi=args.dpi)
        print(f"[wrote] {output}.{{png,pdf,npz}}; grid {len(x)} × {len(y)}; actual coefficients at {len(sites)} cells")
        import matplotlib.pyplot as plt
        if not args.no_show:
            plt.show()
        plt.close(fig)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
