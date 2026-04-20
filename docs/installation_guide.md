# PyGROG Installation and Distribution Guide

This document explains how PyGROG handles installation, precompiled binaries, and CI/CD.

## Installation Methods

### 1. From PyPI (Recommended)
```bash
pip install pygrog
```

This will:
- **First try**: Download a precompiled wheel for your platform (OS + architecture)
- **Fallback**: Build from source if no compatible wheel is available

A C++17 compiler is required for source builds. Precompiled wheels are
provided for all major platforms.

### 2. From Source (Development)
```bash
# Clone repository
git clone https://github.com/FiRMLAB-Pisa/pygrog.git
cd pygrog

# Install build dependencies (Linux/macOS)
./scripts/install_build_deps.sh          # C++ toolchain only
./scripts/install_build_deps.sh --cuda   # C++ + CUDA

# Development install
pip install -e .
```

### 3. Skip CUDA (CPU-only Build)
```bash
PYGROG_NO_CUDA=1 pip install pygrog
```

## Precompiled Wheel Support

### Supported Platforms
| OS      | Architecture | Python Versions | SIMD Support |
|---------|-------------|----------------|--------------|
| Linux   | x86_64      | 3.10, 3.11, 3.12 | AVX-512, AVX2, SSE |
| Linux   | aarch64     | 3.10, 3.11, 3.12 | NEON |
| macOS   | x86_64      | 3.10, 3.11, 3.12 | AVX2, SSE |
| macOS   | arm64       | 3.10, 3.11, 3.12 | NEON |
| Windows | x86_64      | 3.10, 3.11, 3.12 | AVX2, SSE |

### Wheel Naming Convention
```
pygrog-1.0.0-cp311-cp311-linux_x86_64.whl
pygrog-1.0.0-cp311-cp311-macosx_10_15_x86_64.whl
pygrog-1.0.0-cp311-cp311-win_amd64.whl
```

## CI/CD Pipeline

### GitHub Actions Workflow

#### 1. Build Wheels (`.github/workflows/wheels.yml`)
Triggered on:
- Push to `main` branch
- Pull requests
- New releases
- Manual dispatch

**Process:**
```mermaid
graph TD
    A[Trigger Event] --> B[Build Wheels Matrix]
    B --> C[Linux x86_64]
    B --> D[Linux aarch64]
    B --> E[macOS Intel]
    B --> F[macOS ARM64]
    B --> G[Windows x86_64]
    
    C --> H[Test Wheels]
    D --> H
    E --> H
    F --> H
    G --> H
    
    H --> I{Release?}
    I -->|Yes| J[Publish to PyPI]
    I -->|No| K[Publish to TestPyPI]
```

#### 2. Test Local Build (`.github/workflows/test_build.yml`)
Tests local compilation on all platforms to ensure source distribution works.

### Build Matrix
```yaml
strategy:
  matrix:
    include:
      - os: ubuntu-latest
        arch: x86_64
        build: "cp310-* cp311-* cp312-*"
      - os: ubuntu-latest  
        arch: aarch64
        build: "cp310-* cp311-* cp312-*"
      - os: macos-13      # Intel
        arch: x86_64
        build: "cp310-* cp311-* cp312-*"
      - os: macos-14      # ARM64
        arch: arm64
        build: "cp310-* cp311-* cp312-*"
      - os: windows-latest
        arch: AMD64
        build: "cp310-* cp311-* cp312-*"
```

## Performance Optimization

### SIMD Instruction Sets
The build system automatically detects and enables:

**x86_64 platforms:**
- AVX-512F: 8 complex operations per instruction
- AVX2: 4 complex operations per instruction  
- SSE4.2: 2 complex operations per instruction
- Scalar: Fallback implementation

**ARM platforms:**
- NEON: 4 complex operations per instruction
- Scalar: Fallback implementation

### Build Optimization Flags

**Linux/macOS (GCC/Clang):**
```cmake
-O3 -march=native -mavx2 -mfma -fopenmp-simd -flto
```

**Windows (MSVC):**
```cmake
/O2 /arch:AVX2 /Oi /Ot /Oy /GL /LTCG
```

## Local Development

### Setup Development Environment
```bash
# Install build dependencies (cross-platform scripts provided)
# Linux/macOS:
./scripts/install_build_deps.sh
# Windows (PowerShell as Admin):
.\scripts\install_build_deps.ps1

# Clone and build
git clone https://github.com/FiRMLAB-Pisa/pygrog.git
cd pygrog
pip install -e .

# Test installation
python -c "import pygrog; print('OK')"
```

### Build Scripts
```bash
# Install build dependencies (Linux/macOS)
./scripts/install_build_deps.sh
./scripts/install_build_deps.sh --cuda   # with CUDA support

# Install build dependencies (Windows, PowerShell as Admin)
.\scripts\install_build_deps.ps1
.\scripts\install_build_deps.ps1 -Cuda   # with CUDA support
```

### Testing Different Scenarios
```bash
# Test precompiled wheel installation
pip install --find-links wheelhouse pygrog

# Test source installation
pip install --no-binary pygrog pygrog
```

## Distribution Workflow

### 1. Development Cycle
```bash
# Make changes
edit src/pygrog/operator/_fast_binning.py
edit csrc/cpu/fast_binning_cpu.h

# Test locally
pip install -e .
python examples/fast_binning_example.py
python -m pytest tests/

# Push changes
git commit -am "Improve fast binning performance"
git push origin feature-branch
```

### 2. Release Process
```bash
# Create release
git tag v1.0.0
git push origin v1.0.0

# GitHub Actions will:
# 1. Build wheels for all platforms
# 2. Test wheels
# 3. Publish to PyPI
```

### 3. User Installation
```bash
# Users install with
pip install pygrog

# pip will:
# 1. Check for compatible wheel on PyPI
# 2. Download and install wheel if available
# 3. Build from source if no wheel found (requires C++17 compiler)
```

## Troubleshooting

### Common Installation Issues

#### No Precompiled Wheel Available
```
Building wheel for pygrog (pyproject.toml) ... done
```
This means no compatible wheel was found and pip is building from source.

#### Build Failure
```
error: Failed building wheel for pygrog
```
Solutions:
1. Install build dependencies: `./scripts/install_build_deps.sh`
2. Skip CUDA: `PYGROG_NO_CUDA=1 pip install pygrog`
3. Use a precompiled wheel: `pip install pygrog` (recommended)

#### Missing SIMD Performance
```python
from pygrog.operator import detect_simd_level
print(detect_simd_level())  # Should show AVX512/AVX/SSE, not Scalar
```

If showing "Scalar" or "Unavailable":
1. Check if C++ extension built: `python -c "import pygrog.operator._fast_binning"`
2. Rebuild with optimizations: `pip install --force-reinstall --no-binary pygrog pygrog`

### Verification Commands
```bash
# Check extension is loaded
python -c "import pygrog._pygrog_torch; print('\u2713 C++ extension loaded')"

# Performance test
python -c "
from pygrog.operator import benchmark_binning
results = benchmark_binning(n_points=10000, num_runs=3)
print(f'Speedup: {results.get(\"speedup\", \"N/A\")}x')
"

# SIMD detection
python -c "
from pygrog.operator import detect_simd_level
print(f'SIMD level: {detect_simd_level()}')
"
```

## Platform-Specific Notes

### Linux
- Uses `manylinux2014` for maximum compatibility
- Static linking of C++ runtime
- Supports both glibc and musl (Alpine)

### macOS
- Universal wheels for Intel and Apple Silicon
- Deployment target: macOS 10.15+
- Code signing for distribution

### Windows
- MSVC runtime statically linked
- Support for Windows 10+
- Both x86_64 and potentially x86 (32-bit)

## Performance Expectations

| System | SIMD | Expected Speedup |
|--------|------|------------------|
| Modern Intel/AMD | AVX-512 | 4-8x |
| Intel/AMD | AVX2 | 2-4x |
| Older x86_64 | SSE4.2 | 1.5-2x |
| Apple M1/M2 | NEON | 2-4x |
| ARM64 Linux | NEON | 2-4x |

Actual performance depends on:
- Data size (larger = better speedup)
- Memory layout (C-contiguous = faster)
- System memory bandwidth
- Compiler optimizations