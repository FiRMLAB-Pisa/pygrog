cmake_minimum_required(VERSION 3.15...3.27)

# Project definition
project(pygrog 
    VERSION 1.0.0
    DESCRIPTION "An interoperable GROG package to accelerate Non-Cartesian MR reconstructions"
    LANGUAGES CXX
)

# Add custom CMake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find Python and required packages
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Find NumPy
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Include our custom CMake functions
include(FindPybind11Extension)

# Detect SIMD support
detect_simd_support()

# Include directories
include_directories(${NUMPY_INCLUDE_DIR})
include_directories(csrc/cpu)

# Source files for the fast binning extension
set(FAST_BINNING_SOURCES
    csrc/cpu/binning_wrapper.cpp
)

# Create the Python extension using our custom function
add_python_extension(_fast_binning
    SOURCES ${FAST_BINNING_SOURCES}
    INCLUDE_DIRS ${NUMPY_INCLUDE_DIR} csrc/cpu
    COMPILE_DEFINITIONS 
        VERSION_INFO="${PROJECT_VERSION}"
        ${SIMD_COMPILE_DEFINITIONS}
)

# Add SIMD-specific compile flags
if(SIMD_COMPILE_FLAGS)
    target_compile_options(_fast_binning PRIVATE ${SIMD_COMPILE_FLAGS})
endif()

# Optional: Create a separate static library for the core functionality
add_library(fast_binning_core INTERFACE)
target_include_directories(fast_binning_core INTERFACE csrc/cpu)
target_sources(fast_binning_core INTERFACE csrc/cpu/fast_binning_cpu.h)

# Link the core library to the Python module
target_link_libraries(_fast_binning PRIVATE fast_binning_core)

# Install the module to the correct location for scikit-build-core
install(TARGETS _fast_binning 
    DESTINATION pygrog
    COMPONENT python_modules
)

# Testing support
option(BUILD_TESTS "Build C++ tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests/cpp)
endif()

# Benchmarking support
option(BUILD_BENCHMARKS "Build C++ benchmarks" OFF)
if(BUILD_BENCHMARKS)
    add_subdirectory(benchmarks/cpp)
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "PyGROG Fast Binning Configuration Summary:")
message(STATUS "=========================================")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Python version: ${Python_VERSION}")
message(STATUS "  Python executable: ${Python_EXECUTABLE}")
message(STATUS "  NumPy include: ${NUMPY_INCLUDE_DIR}")
message(STATUS "  pybind11 version: ${pybind11_VERSION}")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")

if(SIMD_COMPILE_FLAGS)
    message(STATUS "  SIMD flags: ${SIMD_COMPILE_FLAGS}")
    message(STATUS "  SIMD definitions: ${SIMD_COMPILE_DEFINITIONS}")
else()
    message(STATUS "  SIMD flags: None (scalar fallback)")
endif()
message(STATUS "")