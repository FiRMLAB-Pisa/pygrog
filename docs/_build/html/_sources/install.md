# Installation

## Requirements

- Python ≥ 3.10 (3.10 – 3.14 supported)
- PyTorch ≥ 2.0

---

## Quick install (recommended)

Precompiled CPU wheels are published to PyPI for Linux (x86\_64, aarch64),
macOS (Intel and Apple Silicon), and Windows (x86\_64):

```bash
pip install pygrog
```

Verify the installation:

```bash
python -c "import pygrog._pygrog_torch as e; assert hasattr(e, 'scatter_add'); import pygrog; print('pygrog', pygrog.__version__, '-- OK')"
```

---

## CUDA wheels

CUDA wheels are **not** on PyPI (manylinux rules forbid CUDA runtime
libraries in standard wheels).  They are attached as assets to every
[GitHub Release](https://github.com/FiRMLAB-Pisa/pygrog/releases) and can
be installed with `--find-links`:

| CUDA version | Matches PyTorch index | Label |
|---|---|---|
| 12.6 | `cu126` | `+cu126` |
| 12.8 | `cu128` | `+cu128` |
| 13.0 | `cu130` | `+cu130` |

Replace `<version>` and `<cu>` (e.g. `1.0.0` / `cu126`):

```bash
pip install "pygrog==<version>+<cu>" \
  --find-links https://github.com/FiRMLAB-Pisa/pygrog/releases/expanded_assets/v<version>
```

Example for CUDA 12.6:

```bash
pip install "pygrog==1.0.0+cu126" \
  --find-links https://github.com/FiRMLAB-Pisa/pygrog/releases/expanded_assets/v1.0.0
```

:::{tip}
Match the CUDA version to the PyTorch wheel you already have
installed: `python -c "import torch; print(torch.version.cuda)"`.
:::

---

## Build from source

If no precompiled wheel matches your platform (e.g. exotic Linux distro,
custom Python build), pip falls back to a source build automatically.
A C++17 compiler and CMake are required.

### 1. Install the C++ toolchain

**Linux (requires sudo):**

```bash
sudo ./scripts/install_build_deps.sh                              # C++ only
sudo ./scripts/install_build_deps.sh --cuda                       # C++ + CUDA 12.6
sudo ./scripts/install_build_deps.sh --cuda --cuda-version=12.8   # CUDA 12.8
sudo ./scripts/install_build_deps.sh --cuda --cuda-version=13.0   # CUDA 13.0
```

Supported distributions: Ubuntu/Debian, Fedora/RHEL/Rocky, openSUSE/SLES, Arch.

**macOS (no sudo):**

```bash
./scripts/install_build_deps.sh
```

Requires [Homebrew](https://brew.sh) and Xcode Command Line Tools.
CUDA is not supported on macOS.

**Windows (Administrator PowerShell):**

```powershell
.\scripts\install_build_deps.ps1                          # C++ only
.\scripts\install_build_deps.ps1 -Cuda                    # C++ + CUDA 12.6
.\scripts\install_build_deps.ps1 -Cuda -CudaVersion 12.8  # CUDA 12.8
.\scripts\install_build_deps.ps1 -Cuda -CudaVersion 13.0  # CUDA 13.0
```

### 2. Install PyGROG

First install PyTorch and the build tools into the active environment:

```bash
pip install torch numpy
# or for a CUDA build:
# pip install torch numpy --index-url https://download.pytorch.org/whl/cu126

# Build tools (required with --no-build-isolation)
pip install scikit-build-core setuptools-scm
```

Then install PyGROG:

```bash
pip install --no-build-isolation pygrog                           # auto-detects CUDA
PYGROG_NO_CUDA=1 pip install --no-build-isolation pygrog          # force CPU-only
PYGROG_FORCE_CUDA=1 pip install --no-build-isolation pygrog       # fail if no CUDA found
```

Or pass the flag through pip's CMake interface:

```bash
pip install --no-build-isolation pygrog -C cmake.define.PYGROG_NO_CUDA=ON
```

:::{note}
`--no-build-isolation` is required whenever torch is installed in a conda or
virtualenv environment.  With build isolation disabled, pip no longer
auto-installs `[build-system].requires`, so the `pip install scikit-build-core
...` step above is mandatory.
:::

### 3. Verify

```bash
python scripts/verify_install.py
```

---

## Precompiled wheel matrix

| Platform | Architectures | Python | Notes |
|---|---|---|---|
| Linux | x86\_64, aarch64 | 3.10 – 3.14 | manylinux2014, PyPI |
| macOS | x86\_64 (Intel), arm64 (Apple Silicon) | 3.10 – 3.14 | macOS 12+, PyPI |
| Windows | x86\_64 | 3.10 – 3.14 | PyPI |
| Linux (CUDA 12.6) | x86\_64 | 3.10 – 3.14 | GitHub Releases |
| Linux (CUDA 12.8) | x86\_64 | 3.10 – 3.14 | GitHub Releases |
| Linux (CUDA 13.0) | x86\_64 | 3.10 – 3.14 | GitHub Releases |

---

## Troubleshooting

### Build fails — no C++ compiler

```text
error: command 'gcc' failed
```

Install the C++ toolchain first:

```bash
# Linux
sudo ./scripts/install_build_deps.sh

# macOS
./scripts/install_build_deps.sh

# Windows (Admin PowerShell)
.\scripts\install_build_deps.ps1
```

### Build fails — CUDA not found

```text
CMake Error: PYGROG_FORCE_CUDA is set but no CUDA compiler was found.
```

Either install the CUDA toolkit (`--cuda` flag above) or build CPU-only:

```bash
PYGROG_NO_CUDA=1 pip install pygrog
```

### Wrong CUDA version

Check the CUDA version that matches your installed PyTorch:

```bash
python -c "import torch; print(torch.version.cuda)"
```

Then pick the corresponding CUDA wheel (`cu126`, `cu128`, or `cu130`).

### Verify the installation

```bash
python scripts/verify_install.py
```

The script checks the import, the pre-built C++ extension, CUDA
availability, and runs a CPU (and optionally CUDA) scatter/gather smoke
test.  Exit code 0 means everything is working.
