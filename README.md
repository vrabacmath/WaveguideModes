# Waveguide Modes

[![C++ CI](https://github.com/vrabacmath/WaveguideModes/actions/workflows/ci.yml/badge.svg)](https://github.com/vrabacmath/SubwavelengthResonators/actions/workflows/ci.yml)


This repository implements the boundary integral operator and capacitance matrix ideas for waveguiding as done by Ammari et al. [2] beyond the subwavelength regime using the high-frequency capacitance ideas from Ammari et al. [1].
The core implementation of the repository is based on the previous SubwavelengthResonators repository https://github.com/vrabacmath/SubwavelengthResonators.

The code assumes circular resonators in a 2D crystal with square unit cells. And the workflow idea is the following:

### 1. Compute the projected bulk bands of the unperturbed crystal
This can be done by using the "projected-bulk" workflow. It will draw the crystal bands along the $x$-axis, and project on it along all $y$-coordinates. This way, we can directly see the entire band gaps and the plot is ready for introducing the defect.
A defect radius and material parameters can now be chosen depending on the band gap locations and a line defect can be created and its bands computed by running "line-defect-neumann-M" or "defect-capacitance." The results from these two methods should be very similar for identical parameters.

### 2. Choosing the defect radius
To ensure the defect resonances land in the bandgap and enable localized guided modes, we can change the radius and material parameters in such a way that $\frac{\tilde{v}_b / v}{R_d / R} j'_{m,n}$ lands in the band gap, away from the edges.

### 3. Defect bands, patches, and modes
After running the patch or line workflows, we can now plot the bands, exponential decay plots, or compute the approximate localized waveguide modes. 

## Quick Start (Build, Run, Plot)

### 1. Build

```bash
mkdir -p build
cd build
cmake ..
cmake --build .
ctest --output-on-failure
```

macOS compiler selection:

```bash
# Homebrew GCC (required on macOS — provides libstdc++ with the C++17 special functions)
brew install gcc
cmake -S . -B build -DCMAKE_C_COMPILER=gcc-15 -DCMAKE_CXX_COMPILER=g++-15
```

> **Do not** use Apple Clang (`cmake -S . -B build` with the default toolchain) or
> Homebrew LLVM Clang on macOS: both use libc++, which lacks `std::expint` /
> `std::cyl_bessel_j` / `std::cyl_neumann`, so compilation fails.

### 2. Run workflows

You can run from either `build/` or `build/bin/`.

E.g. from `build/`:

```bash
./bin/WaveguideModes
./bin/WaveguideModes line-defect-neumann
./bin/WaveguideModes bent-waveguide
./bin/WaveguideModes bent-waveguide 3 3 1 64 2
./bin/WaveguideModes defect-capacitance
./bin/WaveguideModes projected-bulk
./bin/WaveguideModes --help
```

### 3. Plot results

Set up a virtual environment once:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

From the repository root:

```bash
python src/plot_all.py
python
```

## Building

### Linux/macOS

```bash
# Create build directory
mkdir build && cd build

# Configure the project
cmake ..

# Build the project
cmake --build .

# Run tests
ctest --output-on-failure
# or
make run_tests

# Run the main executable (from build/)
./bin/WaveguideModes
```

### Windows (Visual Studio)

```powershell
# Create build directory
mkdir build
cd build

# Configure the project
cmake ..

# Build the project
cmake --build . --config Release

# Run tests
ctest -C Release --output-on-failure

# Run the main executable (from build/)
.\bin\Release\WaveguideModes.exe main
```

## Build Options

- `BUILD_TESTS` (default: ON) - Build unit tests
- `CMAKE_BUILD_TYPE` (default: Release) - Build configuration (Debug, Release, RelWithDebInfo, MinSizeRel)

Example:
```bash
cmake -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=OFF ..
```

## OpenMP / Parallel Computation
If OpenMP is available, you can set the thread count as follows:
### Linux/macOS
```
# run from build/
OMP_NUM_THREADS=8 OMP_DYNAMIC=FALSE ./bin/WaveguideModes demos
```
### Windows PowerShell
```
# run from build\
$env:OMP_NUM_THREADS = "8"
$env:OMP_DYNAMIC = "FALSE"
.\bin\Release\WaveguideModes.exe demos
```

## Workflows

The executable dispatches workflows from `src/main.cpp` by passing a mode as the first argument:

```text
./build/bin/WaveguideModes <mode> [options]
```

Currently available modes are:

```text
line-defect-neumann
    Higher-frequency line-defect bands from M^eps, searched around defect Neumann
    resonances. Optional: <max_m> <num_alpha> <n_multipole> <window_factor>
    [v] [v_b] [v_bd]

defect-capacitance
    Higher-frequency line-defect bands using the frequency-dependent capacitance
    matrix and exact transmission solve. Optional: <max_m> <n_rows>
    <points_per_disk> <num_alpha> [alpha_lo] [alpha_hi] [v] [v_b] [v_bd]

projected-bulk
    Projected bulk spectrum over transverse alpha_y values for line-defect band
    diagrams. Optional: <num_alpha_x> <n_omega> <n_alpha_y> [omega_lo]
    [omega_hi] [delta] [omega_imag] [v] [v_b]

crystal-matrix-bands
m-gamma-x-m
    Bulk M-Gamma-X-M bands from CrystalA and multipoleA. Optional:
    <points_per_segment> <n_omega> <points_per_disk> <n_multipole> [v] [v_b]

patch-capacitance
    Real-space capacitance matrix of a finite centered line-defect patch.
    Optional: <n_defect> <n_clad> <points_per_disk> <m_ang> <fringe>
    [v] [v_b] [v_bd]

bent-waveguide
    Real-space capacitance couplings for an L-shaped finite defect patch.
    Optional: <n_defect> <n_clad> <fringe> <points_per_disk> <m_ang>
    [v] [v_b] [v_bd]

bent-waveguide-field
    Reconstructs the field of one approximate localized eigenmode of the bent
    waveguide patch. Optional: <n_defect> <n_clad> <fringe> <points_per_disk>
    <m_ang> <mode_index> <grid_points> [v] [v_b] [v_bd]

bent-waveguide-exact
    Finds exact resonances of the bent patch using Muller's method on the full
    transmission operator and draws the null-vector field. Optional: <n_defect>
    <n_clad> <fringe> <points_per_disk> <m_ang> <mode_index> <grid_points>
    <delta> <seed_re> <seed_im> [v] [v_b] [v_bd]

hexagonal-lattice
    Computes high-symmetry bands for the current hexagonal-lattice crystal setup
    and writes crystal_hex_bands.csv.
```


## Project Structure

```text
WaveguideModes/
├── CMakeLists.txt                 # Build configuration
├── README.md
├── .github/workflows/             # CI configuration
├── include/                       # Public headers and core numerical types
│   ├── workflows/                 # Workflow declarations
│   ├── boundary_mesh.h            # Boundary mesh representation
│   ├── kernels.h                  # Green's functions and kernel routines
│   ├── spectral_operators.h       # Boundary integral and crystal operators
│   ├── multipole.h                # Multipole utilities
│   └── tools.h                    # Numerical helper routines
├── src/                           # Implementations and executable entry point
│   ├── workflows/                 # Workflow implementations
│   ├── main.cpp                   # CLI workflow selection
│   ├── *_*.py                     # Plotting and post-processing scripts
│   └── *.cxx / *.cpp              # Core C++ implementations
├── tests/                         # CTest / GoogleTest regression tests
├── ext/                           # Vendored third-party dependencies
│   ├── bessel-library/
│   ├── eigen/
│   └── spectra/

```


## Troubleshooting

### Compile errors about `std::expint` / `std::cyl_bessel_j` / `std::cyl_neumann`

These C++17 special functions are not implemented by **libc++**, so the build fails
with errors like `no member named 'expint' in namespace 'std'`. Build with a
**libstdc++** toolchain — on macOS install and use Homebrew GCC:

```bash
brew install gcc
cmake -S . -B build -DCMAKE_C_COMPILER=gcc-15 -DCMAKE_CXX_COMPILER=g++-15
```

### GoogleTest fails to build with Homebrew GCC on recent macOS SDKs

Building the tests with Homebrew GCC against a very recent macOS SDK can fail inside
system headers (for example `mach/message.h: expected constructor, destructor, or type
conversion`). This affects only the GoogleTest build. To build/run the library and
executable, disable tests:

```bash
cmake -S . -B build -DCMAKE_C_COMPILER=gcc-15 -DCMAKE_CXX_COMPILER=g++-15 -DBUILD_TESTS=OFF
```

Run the test suite on Linux (or an older SDK) where GCC + GoogleTest build cleanly.

### FetchContent / GoogleTest fails during configure (Linux)

If configure fails with errors around GoogleTest, such as:

- "Build step for googletest failed"
- "Generator: execution of make failed"

check whether your build cache has an invalid make program entry:

```bash
grep CMAKE_MAKE_PROGRAM build/CMakeCache.txt
```

`CMAKE_MAKE_PROGRAM` must be only the executable path (for example `/usr/bin/gmake`),
not a command with `-j` flags appended.

Fix it by clearing just that cache entry and reconfiguring:

```bash
cd build
cmake -U CMAKE_MAKE_PROGRAM ..
```

### Recommended parallel build invocation

Use CMake's portable parallel flag instead of manually editing make/ninja program variables:

```bash
cmake --build . --parallel
# or a fixed count
cmake --build . --parallel 16
```

### Conda/OpenMP runtime warning

If CMake warns that `libgomp.so.1` from Conda may hide the system OpenMP runtime,
prefer building/running from a clean shell (without Conda activated), or ensure your
runtime library paths prefer system GCC OpenMP libraries.

### Missing Submodules

If you see errors about missing Eigen or Spectra CMakeLists.txt, 
ensure you cloned the repository with `--recursive` or initialized submodules:

`git submodule update --init`


## References from the intro


[1] Ammari, Li, Shao, Uhlmann: Frequency-dependent capacitance matrix formulation for Fabry-Perot resonances in two and three dimensional systems, 10.48550/arXiv.2605.27572

[2] Ammari, Miao, Qiu: Resolvent Convergence and Patch Approximation for Subwavelength Guided Modes in Non-Periodic Systems of High-Contrast Resonators, 10.48550/arXiv.2605.30951 
