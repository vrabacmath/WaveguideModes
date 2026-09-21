from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
PATCH_SIZES = [2, 3, 4, 7, 10, *range(12, 21, 2)]
NUM_MINIMA = 3
MIN_Z_SEPARATION = 1.0  # z units; suppress nearby secondary dips, not a physical criterion


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


if __name__ == "__main__":
    main()
