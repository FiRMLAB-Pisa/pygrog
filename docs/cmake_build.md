# CMake Build System for PyGROG Fast Binning

This document describes the CMake-based build system for the PyGROG fast binning C++ extension.

## Overview

The build system uses:
- **CMake 3.15+** for cross-platform C++ compilation
- **scikit-build-core** for Python integration
- **pybind11** for Python-C++ bindings
- **Automatic SIMD detection** for optimal performance

## Quick Start

### Option 1: Automated Build Script (Recommended)
```bash
# Build in release mode (development install)
./build_cmake.sh

# Build in debug mode
./build_cmake.sh --debug

# Clean build
./build_cmake.sh --clean

# View all options
./build_cmake.sh --help
```

### Option 2: Direct pip Install
```bash
# Development install
pip install -e .

# Regular install
pip install .

# With specific CMake options
pip install -e . --config-settings=cmake.build-type=Debug
```

### Option 3: Manual CMake (Advanced)
```bash
# Configure
cmake -B build -S . --preset=release

# Build
cmake --build build --parallel

# Install Python module
cmake --install build
```

## Build Configuration

### Build Types
- **Release** (default): Optimized for performance
- **Debug**: No optimization, debug symbols
- **RelWithDebInfo**: Optimized with debug info

### SIMD Support
The build system automatically detects and enables:
- **AVX-512**: 8 complex numbers per instruction
- **AVX/AVX2**: 4 complex numbers per instruction  
- **SSE4.2**: 2 complex numbers per instruction
- **Scalar**: Fallback for unsupported platforms

### Platform Support
- **Linux**: GCC/Clang with full SIMD support
- **macOS**: Intel and Apple Silicon (M1/M2)
- **Windows**: MSVC with AVX2 support

## Project Structure

```
pygrog/
├── CMakeLists.txt              # Main CMake configuration
├── CMakePresets.json           # Modern CMake presets
├── build_cmake.sh              # Automated build script
├── pyproject.toml              # Python packaging (scikit-build-core)
├── setup.py                    # Legacy setup (now CMake-based)
├── cmake/                      # CMake modules
│   ├── FindPybind11Extension.cmake
│   └── OptimizationFlags.cmake
├── csrc/cpu/                   # C++ source code
│   ├── fast_binning_cpu.h      # SIMD implementation
│   └── binning_wrapper.cpp     # pybind11 wrapper
└── src/pygrog/operator/        # Python interface
    └── _fast_binning.py
```

## CMake Options

### Build Options
```bash
-DCMAKE_BUILD_TYPE=Release      # Build type
-DBUILD_TESTS=ON               # Build C++ tests
-DBUILD_BENCHMARKS=ON          # Build C++ benchmarks
```

### Optimization Options
```bash
-DENABLE_NATIVE_ARCH=ON        # Use -march=native
-DENABLE_FAST_MATH=ON          # Use -ffast-math
-DENABLE_LTO=ON                # Link-time optimization
-DENABLE_SANITIZERS=OFF        # Debug sanitizers
```

### Example Configurations
```bash
# Maximum optimization
cmake -B build -DCMAKE_BUILD_TYPE=Release \
                -DENABLE_NATIVE_ARCH=ON \
                -DENABLE_LTO=ON

# Portable build (no native arch)
cmake -B build -DCMAKE_BUILD_TYPE=Release \
                -DENABLE_NATIVE_ARCH=OFF

# Debug with sanitizers
cmake -B build -DCMAKE_BUILD_TYPE=Debug \
                -DENABLE_SANITIZERS=ON
```

## CMake Presets

Use modern CMake presets for common configurations:

```bash
# List available presets
cmake --list-presets

# Configure with preset
cmake --preset=release
cmake --preset=debug
cmake --preset=ninja           # Fast Ninja builds

# Build with preset
cmake --build --preset=release
```

## Integration with Python

### scikit-build-core Configuration
The `pyproject.toml` configures scikit-build-core:

```toml
[tool.scikit-build]
cmake.build-type = "Release"
cmake.verbose = true
wheel.expand-macos-universal-tags = true
```

### Runtime Configuration
The Python module automatically detects SIMD capabilities:

```python
from pygrog.operator import detect_simd_level
print(f"SIMD level: {detect_simd_level()}")  # AVX512, AVX, SSE, Scalar
```

## Development Workflow

### 1. Initial Setup
```bash
git clone <repository>
cd pygrog
pip install -e .  # Development install
```

### 2. Make Changes
Edit C++ code in `csrc/cpu/` or Python code in `src/pygrog/`

### 3. Rebuild
```bash
# Quick rebuild (incremental)
pip install -e . --force-reinstall

# Clean rebuild
./build_cmake.sh --clean
```

### 4. Test
```bash
python -c "from pygrog.operator import fast_binning_add_at"
python examples/fast_binning_example.py
python -m pytest tests/test_fast_binning.py
```

## Troubleshooting

### Common Issues

#### CMake Not Found
```bash
# Ubuntu/Debian
sudo apt install cmake

# Fedora/RHEL  
sudo dnf install cmake

# macOS
brew install cmake
```

#### Compiler Issues
```bash
# Install build tools
# Ubuntu/Debian
sudo apt install build-essential

# Fedora/RHEL
sudo dnf groupinstall "Development Tools"

# macOS
xcode-select --install
```

#### pybind11 Not Found
```bash
pip install pybind11
# Or for global CMake detection
pip install "pybind11[global]"
```

#### SIMD Not Working
- Check CPU support: `lscpu | grep -E "(avx|sse)"`
- Verify compiler flags in build output
- Try building with `--verbose` flag

### Debug Build Issues
```bash
# Clean everything
./build_cmake.sh --clean

# Try safe build (no optimizations)
cmake -B build -DCMAKE_BUILD_TYPE=Debug \
                -DENABLE_NATIVE_ARCH=OFF \
                -DENABLE_FAST_MATH=OFF

# Check build log
cat _skbuild/*/cmake-build/CMakeFiles/CMakeError.log
```

### Performance Verification
```bash
python -c "
from pygrog.operator import benchmark_binning
results = benchmark_binning(n_points=100000)
print(f'Speedup: {results.get(\"speedup\", \"N/A\")}x')
"
```

## Advanced Usage

### Custom Compiler
```bash
export CC=clang
export CXX=clang++
./build_cmake.sh --clean
```

### Cross-Compilation
```bash
# Example for ARM64
cmake -B build -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
                -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc \
                -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++
```

### Packaging
```bash
# Build wheel
python -m build

# Test wheel
pip install dist/*.whl
python -c "from pygrog.operator import fast_binning_add_at"
```

## Performance Notes

- **Release builds** are ~10-100x faster than debug builds
- **Native architecture** (`-march=native`) provides best performance
- **AVX-512** can provide 2-8x speedup over scalar code
- **Large arrays** (>10k elements) show the biggest improvements
- **Memory layout** matters: use C-contiguous arrays when possible