cmake_minimum_required(VERSION 3.14)
project(WaveguideModes VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# macOS 26 SDK Mach headers use the C spelling, which Apple Clang accepts as
# an extension but Homebrew GCC does not recognize when compiling C++.
if(APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_definitions(_Static_assert=static_assert)
endif()

# Set default build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler-specific flags
if(MSVC)
    add_compile_options(/W4 /WX-)
    # Enable multi-processor compilation for faster builds
    add_compile_options(/MP)
    # Use external includes to reduce header re-parsing
    add_compile_options(/external:W0)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(/O2)
        # Inline function expansion for Release
        add_compile_options(/Ox)
    endif()
else()
    add_compile_options(-Wall -Wextra -pedantic)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-O3)
        # Only add architecture-specific flags on native builds
        if(NOT CMAKE_CROSSCOMPILING)
            if(NOT APPLE OR CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
                add_compile_options(-march=native -mtune=native)
            endif()
        endif()
    endif()
endif()

# Find required packages
# Use local Eigen 5.0 from ext folder
set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_SOURCE_DIR}/ext/eigen)

## Build FFTW from ext folder
#set(ENABLE_FLOAT OFF CACHE BOOL "" FORCE)
#set(ENABLE_LONG_DOUBLE OFF CACHE BOOL "" FORCE)
#set(ENABLE_THREADS ON CACHE BOOL "" FORCE)
#set(ENABLE_OPENMP ON CACHE BOOL "" FORCE)
#set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
#add_subdirectory(${CMAKE_SOURCE_DIR}/ext/fftw-3.3.10)

# Spectra is header-only, so we just need to include it
# No need to add_subdirectory to avoid install/export target conflicts

# Optionally find Python3
find_package(Python3 COMPONENTS Development)

# AppleClang does not ship OpenMP headers or libraries with Xcode/Command Line
# Tools. Help CMake find a locally installed libomp from Homebrew or MacPorts.
if(APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    set(_OPENMP_APPLECLANG_CANDIDATES
        "/opt/homebrew/opt/libomp"
        "/usr/local/opt/libomp"
        "/opt/local"
    )

    foreach(_OPENMP_PREFIX IN LISTS _OPENMP_APPLECLANG_CANDIDATES)
        if(EXISTS "${_OPENMP_PREFIX}/include/omp.h")
            set(_OPENMP_INCLUDE_DIR "${_OPENMP_PREFIX}/include")
        elseif(EXISTS "${_OPENMP_PREFIX}/include/libomp/omp.h")
            set(_OPENMP_INCLUDE_DIR "${_OPENMP_PREFIX}/include/libomp")
        endif()

        if(EXISTS "${_OPENMP_PREFIX}/lib/libomp.dylib")
            set(_OPENMP_LIBRARY "${_OPENMP_PREFIX}/lib/libomp.dylib")
        elseif(EXISTS "${_OPENMP_PREFIX}/lib/libomp/libomp.dylib")
            set(_OPENMP_LIBRARY "${_OPENMP_PREFIX}/lib/libomp/libomp.dylib")
        endif()

        if(_OPENMP_INCLUDE_DIR AND _OPENMP_LIBRARY)
            if(NOT OpenMP_CXX_FLAGS OR OpenMP_CXX_FLAGS MATCHES "NOTFOUND")
                set(OpenMP_CXX_FLAGS "-Xclang -fopenmp" CACHE STRING "CXX compiler flags for OpenMP" FORCE)
            endif()
            if(NOT OpenMP_CXX_INCLUDE_DIR OR OpenMP_CXX_INCLUDE_DIR MATCHES "NOTFOUND")
                set(OpenMP_CXX_INCLUDE_DIR "${_OPENMP_INCLUDE_DIR}" CACHE PATH "CXX include directory for OpenMP" FORCE)
            endif()
            if(NOT OpenMP_CXX_LIB_NAMES OR OpenMP_CXX_LIB_NAMES MATCHES "NOTFOUND")
                set(OpenMP_CXX_LIB_NAMES "omp" CACHE STRING "CXX OpenMP library names" FORCE)
            endif()
            if(NOT OpenMP_omp_LIBRARY OR OpenMP_omp_LIBRARY MATCHES "NOTFOUND")
                set(OpenMP_omp_LIBRARY "${_OPENMP_LIBRARY}" CACHE FILEPATH "OpenMP library" FORCE)
            endif()
            break()
        endif()

        unset(_OPENMP_INCLUDE_DIR)
        unset(_OPENMP_LIBRARY)
    endforeach()
endif()

# Find OpenMP when available (optional for better cross-platform portability)
find_package(OpenMP)

# Compilation speed options
option(USE_PRECOMPILED_HEADERS "Use precompiled headers for faster compilation" ON)
option(USE_UNITY_BUILD "Combine source files for faster compilation" OFF)

# Option to build tests
option(BUILD_TESTS "Build the tests" ON)

# Set parallel compilation for non-MSVC (MSVC /MP already set above)
if(NOT MSVC)
    # Keep GCC/Clang diagnostics readable in terminal output.
    add_compile_options(-fdiagnostics-color)
endif()

# Create core library
if(USE_UNITY_BUILD)
    set(CMAKE_UNITY_BUILD ON)
    set(CMAKE_UNITY_BUILD_BATCH_SIZE 4)
endif()

add_library(waveguide_core STATIC
    src/boundary_mesh.cxx
    src/legendre_matrix.cxx
    src/spectral_operators.cxx
    src/tools.cxx
    src/multipole.cxx
    src/workflows/line_defect_bands.cpp
    src/workflows/defect_capacitance_bands.cpp
    src/workflows/crystal_matrix_bands.cpp
    src/workflows/patch_capacitance.cpp
    src/workflows/bent_waveguide.cpp
    src/workflows/bent_waveguide_patch.cpp
    src/workflows/bent_waveguide_field.cpp
    src/workflows/bent_waveguide_exact.cpp
    src/hex_crystal.cpp
)

# Apply precompiled headers if enabled
if(USE_PRECOMPILED_HEADERS)
    target_precompile_headers(waveguide_core PUBLIC include/pch.h)
endif()

# Set include directories for the library
target_include_directories(waveguide_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Link dependencies to the library
target_link_libraries(waveguide_core
    PUBLIC
        Eigen3::Eigen
#        Boost::boost
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(waveguide_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(waveguide_core PUBLIC HAS_OPENMP)
endif()

target_compile_definitions(waveguide_core PUBLIC PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

if(USE_PRECOMPILED_HEADERS)
    target_compile_definitions(waveguide_core PUBLIC USE_PRECOMPILED_HEADERS)
endif()

# Set include directories for external headers
target_include_directories(waveguide_core
    PUBLIC
        ${CMAKE_SOURCE_DIR}/ext/spectra/include
        ${CMAKE_SOURCE_DIR}/ext/bessel-library
#        ${CMAKE_SOURCE_DIR}/ext/fftw-3.3.10/api
)

# Optionally link Python if found
if(Python3_FOUND)
    target_link_libraries(waveguide_core PUBLIC Python3::Python)
    target_compile_definitions(waveguide_core PUBLIC HAS_PYTHON)
endif()

# Create main executable
add_executable(WaveguideModes src/main.cpp)

# Link libraries to executable
target_link_libraries(WaveguideModes
    PRIVATE
        waveguide_core
#        fftw3
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(WaveguideModes PRIVATE OpenMP::OpenMP_CXX)
endif()

# Apply PCH to executable too
if(USE_PRECOMPILED_HEADERS)
    target_precompile_headers(WaveguideModes REUSE_FROM waveguide_core)
endif()

# Set output directories
set_target_properties(WaveguideModes PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Enable testing and add tests subdirectory
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "Configuration Summary:")
message(STATUS "  Project: ${PROJECT_NAME} ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build tests: ${BUILD_TESTS}")
message(STATUS "  Python3 found: ${Python3_FOUND}")
message(STATUS "  OpenMP found: ${OpenMP_CXX_FOUND}")
message(STATUS "")
