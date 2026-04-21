# PyGROG

**GPU-accelerated GROG (GRAPPA Operator Gridding) for non-Cartesian MRI reconstruction.**

[![PyPI version](https://badge.fury.io/py/pygrog.svg)](https://badge.fury.io/py/pygrog)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

PyGROG implements the GROG algorithm — a data-driven, parallel-imaging-aware
gridding method that maps non-Cartesian k-space samples onto a Cartesian grid
using GRAPPA kernels trained from an auto-calibration region.  It provides:

- **GROG interpolator** (`pygrog.calib.GrogInterpolator`) — non-Cartesian to
  Cartesian gridding via fractional GRAPPA operators.
- **SparseFFT operator** (`pygrog.operator.SparseFFT`) — fast sparse forward
  and adjoint Cartesian FFT for undersampled MRI, with optional sensitivity
  maps and GPU pipelining.
- **Reconstruction gadgets** (`pygrog.gadgets`) — off-resonance correction and
  low-rank subspace projection, stackable on any base operator.
- **Calibration utilities** (`pygrog.utils`) — NLINV coil sensitivity
  estimation and PCA coil compression.
- **Interoperability** (`pygrog.interop`) — drop-in adapters for
  [mri-nufft](https://mind-inria.github.io/mri-nufft/),
  [sigpy](https://sigpy.readthedocs.io/),
  [mrpro](https://mrpro.readthedocs.io/), and
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

See the [documentation](https://firnlab-pisa.github.io/pygrog/) for full
examples, API reference, and theory.

## Documentation

Full documentation (installation, examples, API, theory) lives at
<https://firnlab-pisa.github.io/pygrog/>.

## Installation

```bash
# CPU (from PyPI)
pip install pygrog

# Development install with all optional dependencies
pip install --no-build-isolation -e ".[dev]"
```

CUDA wheels are attached to each
[GitHub Release](https://github.com/FiRMLAB-Pisa/pygrog/releases).

## Related Projects

- [mri-nufft](https://mind-inria.github.io/mri-nufft/) — Non-uniform FFT for MRI
- [mrpro](https://mrpro.readthedocs.io/) — MRI reconstruction in PyTorch
- [sigpy](https://sigpy.readthedocs.io/) — Signal processing for inverse problems
- [deepinverse](https://deepinv.github.io/) — Deep learning for inverse problems

## License

MIT — see [LICENSE](LICENSE) for details.
