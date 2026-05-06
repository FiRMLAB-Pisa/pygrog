# PyGROG

**GPU-accelerated GROG (GRAPPA Operator Gridding) for non-Cartesian MRI reconstruction.**

[![PyPI version](https://badge.fury.io/py/pygrog.svg)](https://badge.fury.io/py/pygrog)
[![Read the Docs](https://readthedocs.org/projects/pygrog/badge/?version=latest)](https://pygrog.readthedocs.io/en/latest/)
[![Tests](https://github.com/FiRMLAB-Pisa/pygrog/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/FiRMLAB-Pisa/pygrog/actions/workflows/ci.yml)
[![Docs Check](https://github.com/FiRMLAB-Pisa/pygrog/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/FiRMLAB-Pisa/pygrog/actions/workflows/docs.yml)
[![Coverage](https://codecov.io/gh/FiRMLAB-Pisa/pygrog/branch/main/graph/badge.svg)](https://codecov.io/gh/FiRMLAB-Pisa/pygrog)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

PyGROG implements the GROG algorithm — a data-driven, parallel-imaging-aware
gridding method that maps non-Cartesian k-space samples onto a Cartesian grid
using GRAPPA kernels trained from an auto-calibration region.  It provides:

- **GROG interpolator** (`pygrog.calib.GrogInterpolator`) — non-Cartesian to
  Cartesian gridding via fractional GRAPPA operators.
- **SparseFFT operator** (`pygrog.operator.SparseFFT`) — fast sparse forward
  and adjoint Cartesian FFT for undersampled MRI, with optional sensitivity
  maps and GPU pipelining.
- **MaskedFFT operator** (`pygrog.operator.MaskedFFT`) — dense-grid masked FFT
  companion for GROG gridded paths.
- **Reconstruction gadgets** (`pygrog.gadgets`) — off-resonance correction and
  low-rank subspace projection, stackable on any base operator.
  Public aliases include `SubspaceGadget`, `OffResonanceGadget`,
  `with_subspace`, and `with_offresonance`.
- **Calibration utilities** (`pygrog.utils`) — NLINV coil sensitivity
  estimation and PCA coil compression.
- **Iterative solvers** (`op.solve(...)`, `pygrog.solve`) — CG/LSMR interfaces
  that accept torch, NumPy, and CuPy arrays (CuPy via DLPack, no CPU copy).
- **Interoperability** (`pygrog.interop`) — drop-in adapters for
  [mri-nufft](https://mind-inria.github.io/mri-nufft/),
  [sigpy](https://sigpy.readthedocs.io/),
  [mrpro](https://mrpro.rocks/), and
  [deepinverse](https://deepinv.github.io/).

## Quick Start

```bash
pip install pygrog
```

```python
import numpy as np
from pygrog.calib import GrogInterpolator
from pygrog.operator import SparseFFT

# 1. Build the GROG plan from the non-Cartesian trajectory
grog = GrogInterpolator(shape=(256, 256), coords=coords)

# 2. Fit GRAPPA kernels from the auto-calibration region (ACR)
grog.calc_interp_table(acr_data)

# 3. Grid and reconstruct in one call
image = grog.interpolate(kspace_nc, ret_image=True)
```

See the [documentation](https://pygrog.readthedocs.io/en/latest/) for full
examples, API reference, and theory.

## Documentation

Full documentation (installation, examples, API, theory) lives at
<https://pygrog.readthedocs.io/en/latest/>.

## Installation

```bash
# CPU (from PyPI)
pip install pygrog

# Development install with all optional dependencies
pip install --no-build-isolation -e ".[dev]"
```

CUDA wheels are attached to each
[GitHub Release](https://github.com/FiRMLAB-Pisa/pygrog/releases).

## Development Style

For contributors, formatting and linting are Ruff-only:

```bash
ruff format .
ruff check .
```

`ruff check` is configured to apply safe auto-fixes by default.

## Related Projects

- [mri-nufft](https://mind-inria.github.io/mri-nufft/) — Non-uniform FFT for MRI
- [mrpro](https://mrpro.rocks/) — MRI reconstruction in PyTorch
- [sigpy](https://sigpy.readthedocs.io/) — Signal processing for inverse problems
- [deepinverse](https://deepinv.github.io/) — Deep learning for inverse problems

## License

MIT — see [LICENSE](LICENSE) for details.
