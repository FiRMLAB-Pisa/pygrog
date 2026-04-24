"""
==================================================
Interoperability with mrpro, sigpy and deepinverse
==================================================

PyGROG ships three thin adapter layers so that
:class:`~pygrog.operator.SparseFFT` (and any gadget stacked on top of it)
can be used directly inside third-party reconstruction frameworks:

* **mrpro** — :class:`~pygrog.interop.GrogLinearOp` wraps SparseFFT as an
  ``mrpro.operators.LinearOperator`` with explicit autograd support.
* **sigpy** — :class:`~pygrog.interop.GrogLinop` wraps SparseFFT as a
  ``sigpy.linop.Linop`` with a working ``.H`` adjoint property.
* **deepinverse** — :class:`~pygrog.interop.GrogLinearPhysics` subclasses
  ``deepinv.physics.LinearPhysics``, enabling all deepinv algorithms
  (unrolled networks, PnP, RED, …) to be used directly.

For differentiable use in plain PyTorch (without deepinv) the lower-level
:func:`~pygrog.interop.grog_measure` and
:func:`~pygrog.interop.grog_backproject` autograd functions are also
available.

Each adapter is imported lazily so that missing optional dependencies raise
an informative error only when the adapter is actually instantiated.

All data are **synthetic** — no scanner files are required.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

# %%
# Shared SparseFFT operator
# ==========================
#
# A single :class:`~pygrog.operator.SparseFFT` is used throughout the
# example so every adapter wraps the same underlying computation.

from pygrog.operator import SparseFFT

image_shape = (32, 32)
grid_shape = (40, 40)
rng = np.random.default_rng(7)
grid_size = grid_shape[0] * grid_shape[1]
n_samples = grid_size // 4  # 25 % k-space coverage
indices = rng.integers(0, grid_size, size=n_samples).astype(np.int64)
weights = np.ones(n_samples, dtype=np.float32)

op = SparseFFT(grid_shape, image_shape, indices, weights)


def _phantom(shape):
    ny, nx = shape
    yy, xx = np.mgrid[-1 : 1 : ny * 1j, -1 : 1 : nx * 1j]
    img = np.zeros(shape, dtype=np.float32)
    img += 1.0 * ((xx / 0.9) ** 2 + (yy / 0.9) ** 2 < 1)
    img += 0.4 * ((xx / 0.6) ** 2 + ((yy - 0.1) / 0.7) ** 2 < 1)
    img -= 0.6 * ((xx / 0.15) ** 2 + ((yy + 0.2) / 0.2) ** 2 < 1)
    return img.clip(0, 1)


image_ref = torch.as_tensor(_phantom(image_shape)).to(torch.complex64)
# Add coil dimension: operator expects (n_coils, *image_shape) when smaps are absent
image_1coil = image_ref.unsqueeze(0)  # (1, 32, 32)
kspace_ref = op.adjoint(image_1coil)  # (1, n_samples)

print(
    f"Operator   : grid={op.grid_shape}, image={op.image_shape}, n={op.indices.shape[0]}"
)
print(f"K-space    : {kspace_ref.shape}")

# %%
# deepinverse adapter — GrogLinearPhysics
# ========================================
#
# :class:`~pygrog.interop.GrogLinearPhysics` subclasses
# ``deepinv.physics.LinearPhysics`` so that SparseFFT slots into any deepinv
# reconstruction algorithm (unrolled networks, PnP, RED, …).  Gradients are
# provided by explicit :class:`torch.autograd.Function` subclasses — not by
# automatic differentiation through the GROG kernels.
#
# .. note::
#    ``deepinv`` must be installed separately (``pip install deepinv``).
#    When not available the lazy class factory raises an informative
#    ``ImportError``.

from pygrog.interop import GrogLinearPhysics

try:
    physics = GrogLinearPhysics(op)

    # Forward measurement: image → k-space
    ksp_deepinv = physics.A(image_1coil)  # grad provided by _torch.py autograd fn
    print(f"\nGrogLinearPhysics.A   output shape: {ksp_deepinv.shape}")

    # Adjoint: k-space → image
    img_deepinv = physics.A_adjoint(ksp_deepinv)
    print(f"GrogLinearPhysics.A_adjoint output: {img_deepinv.shape}")

    # Gradient flows through A
    ksp_grad = kspace_ref.clone().requires_grad_(True)
    img_out = physics.A_adjoint(ksp_grad)
    img_out.abs().sum().backward()
    print(f"Gradient populated on ksp_grad     : {ksp_grad.grad is not None}")

except ImportError as exc:
    print(f"\ndeepinv not available — skipping GrogLinearPhysics demo: {exc}")

# %%
# Plain-PyTorch autograd functions
# ---------------------------------
#
# For use without deepinv, :func:`~pygrog.interop.grog_measure` (image → k-space)
# and :func:`~pygrog.interop.grog_backproject` (k-space → image) are the
# underlying autograd functions.

from pygrog.interop import grog_measure

img_grad = image_1coil.clone().requires_grad_(True)
ksp_out = grog_measure(img_grad, op)
ksp_out.abs().sum().backward()
print(f"\ngrog_measure output shape : {ksp_out.shape}")
print(f"Gradient on img_grad      : {img_grad.grad is not None}")

# %%
# sigpy adapter — GrogLinop
# ==========================
#
# :class:`~pygrog.interop.GrogLinop` returns a ``sigpy.linop.Linop`` with a
# working ``.H`` property, so it can be composed with other sigpy operators
# using ``*``, ``+``, and scalar multiplication.
#
# .. note::
#    ``sigpy`` must be installed separately (``pip install sigpy``).  When
#    sigpy is not available this block prints an informative message and
#    continues.

from pygrog.interop import GrogLinop

try:
    linop = GrogLinop(op)
    print(f"\nGrogLinop    ishape={linop.ishape}, oshape={linop.oshape}")
    print(f"GrogLinop.H  ishape={linop.H.ishape}, oshape={linop.H.oshape}")

    # Forward: k-space -> image (sigpy forward direction)
    ksp_np_sigpy = kspace_ref.numpy()
    img_sigpy = linop * ksp_np_sigpy  # uses __mul__ (apply)
    print(f"linop output shape   : {img_sigpy.shape}")

    # Adjoint: image -> k-space
    img_np_sigpy = np.abs(img_sigpy)
    ksp_sigpy = linop.H * img_np_sigpy.astype(np.complex64)
    print(f"linop.H output shape : {ksp_sigpy.shape}")

except ImportError as exc:
    print(f"\nsigpy not available — skipping GrogLinop demo: {exc}")

# %%
# mrpro adapter — GrogLinearOp
# =============================
#
# :class:`~pygrog.interop.GrogLinearOp` returns an
# ``mrpro.operators.LinearOperator`` with autograd support via explicit
# :class:`torch.autograd.Function` wrappers (no ``adjoint_as_backward``).
# It participates in mrpro operator algebra (``@``, ``+``, adjoint via ``.H``)
# and can be passed directly to mrpro solvers (e.g. ``ConjugateGradient``).

from pygrog.interop import GrogLinearOp

try:
    mrpro_op = GrogLinearOp(op)
    print(f"\nGrogLinearOp ready: {type(mrpro_op).__mro__[1].__name__}")

    # Forward (backprojection): k-space -> image
    (img_mrpro,) = mrpro_op.forward(kspace_ref)
    print(f"mrpro forward output shape: {img_mrpro.shape}")

    # Adjoint (measurement): image -> k-space
    (ksp_mrpro,) = mrpro_op.adjoint(img_mrpro)
    print(f"mrpro adjoint output shape: {ksp_mrpro.shape}")

    # Normal operator A^H A
    AHA = mrpro_op.H @ mrpro_op
    (img_aha,) = AHA.forward(kspace_ref)
    print(f"Normal operator (A^H A) output shape: {img_aha.shape}")

except ImportError as exc:
    print(f"\nmrpro not available — skipping GrogLinearOp demo: {exc}")

# %%
# Summary
# =======
#
# .. list-table:: Adapter summary
#    :header-rows: 1
#    :widths: 25 40 35
#
#    * - Adapter
#      - Interface
#      - Use case
#    * - ``GrogLinearPhysics``
#      - ``deepinv.physics.LinearPhysics``
#      - deepinv algorithms (PnP, …)
#    * - ``grog_measure`` / ``grog_backproject``
#      - ``torch.autograd.Function``
#      - plain PyTorch gradient recon
#    * - ``GrogLinop``
#      - ``sigpy.linop.Linop``
#      - sigpy CG / PDHG solvers
#    * - ``GrogLinearOp``
#      - ``mrpro.operators.LinearOperator``
#      - mrpro CG / PDHG solvers

plt.show()
