# Building PyGROG C++ Extensions

This document describes how PyGROG builds the `_pygrog_torch` C++/CUDA extension.

## Overview

The build system uses:
- **setuptools** as the Python build backend
- **torch.utils.cpp_extension** for compiling C++/CUDA code
- **Automatic CUDA detection** — CUDA sources are included when `nvcc` is available
- **SIMD multi-versioning** — AVX2 and AVX-512 translation units on x86_64

## Quick Start

```bash
# Install build dependencies (C++ compiler, optionally CUDA)
./scripts/install_build_deps.sh          # Linux / macOS
./scripts/install_build_deps.sh --cuda   # Linux with CUDA
.\scripts\install_build_deps.ps1         # Windows (PowerShell, Admin)
.\scripts\install_build_deps.ps1 -Cuda   # Windows with CUDA

# Install pygrog (builds the extension automatically)
pip install -e .
```

## How It Works

When you run `pip install .`, setuptools invokes `setup.py` which:

1. Detects whether CUDA is available (checks `nvcc` on PATH + `torch.cuda.is_available()`)
2. Selects `CUDAExtension` or `CppExtension` from `torch.utils.cpp_extension`
3. Compiles the C++ sources under `csrc/torch/`
4. On MSVC, injects per-file `/arch:AVX2` and `/arch:AVX512` flags for SIMD TUs
5. Installs the resulting `.so` / `.pyd` as `pygrog._pygrog_torch`

If compilation fails (missing compiler, wrong torch version, etc.), the package
still installs. At runtime, Python code falls back to:
- **JIT compilation** via `torch.utils.cpp_extension.load()` (first use is slow)
- **Pure-torch ATen operations** if JIT also fails

## Source Layout

```
csrc/torch/
├── module.cpp                  # pybind11 entry point
├── grog_interp.cpp             # GROG kernel interpolation (CPU)
├── grog_interp_cuda.cu         # GROG kernel interpolation (CUDA)
├── sparse_ops.cpp              # scatter_add / gather (CPU dispatch)
├── sparse_ops_cpu_impl.inl     # Templated CPU implementation
├── sparse_ops_avx2.cpp         # AVX2 multi-version TU
├── sparse_ops_avx512.cpp       # AVX-512 multi-version TU
├── sparse_ops_cuda.cu          # CUDA scatter_add / gather kernels
```

## Environment Variables

| Variable | Effect |
|----------|--------|
| `PYGROG_NO_CUDA=1` | Skip CUDA even if `nvcc` is available |
| `PYGROG_FORCE_CUDA=1` | Require CUDA (fail if `nvcc` missing) |

## CUDA Compute Capabilities

Pre-built extensions target these GPU architectures:

| Arch | GPUs |
|------|------|
| sm_70 | V100 |
| sm_80 | A100 |
| sm_86 | RTX 3090 |
| sm_89 | RTX 4090 |
| sm_90 | H100 |

## Troubleshooting

### Extension not built during pip install
Check for warnings in the install output. Common causes:
- No C++ compiler available → install one via `scripts/install_build_deps.sh`
- `torch` not in build environment → it should be pulled automatically

### JIT compilation slow on first use
This is expected. The JIT-compiled extension is cached in `~/.cache/torch_extensions/`.
To pre-build, run `pip install -e .` which compiles at install time.

### MSVC "internal compiler error" on AVX-512
Some older MSVC versions don't support AVX-512. Set `PYGROG_NO_AVX512=1` or
update Visual Studio.
