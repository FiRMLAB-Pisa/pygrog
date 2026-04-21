"""
==================================================
Interoperability with mrpro, sigpy and deepinverse
==================================================

PyGROG ships four thin adapter layers so that
:class:`~pygrog.operator.SparseFFT` (and any gadget stacked on top of it)
can be used directly inside third-party reconstruction frameworks:

* **mrpro** — :class:`~pygrog.interop.GrogLinearOp` wraps SparseFFT as an
  ``mrpro.operators.LinearOperator`` with autograd support.
* **sigpy** — :class:`~pygrog.interop.GrogLinop` wraps SparseFFT as a
  ``sigpy.linop.Linop`` with a working ``.H`` adjoint property.
* **mri-nufft** — :class:`~pygrog.interop.GrogFourierOp` exposes the
  ``op`` / ``adj_op`` interface expected by mri-nufft reconstruction
  pipelines.
* **deepinverse** — :func:`~pygrog.interop.sparse_fft_forward` and
  :func:`~pygrog.interop.sparse_fft_adjoint` are autograd-compatible
  wrappers for gradient-based reconstruction (e.g. with deepinverse or
  plain PyTorch).

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
kspace_ref = op.adjoint(image_ref)  # (1, n_samples)

print(
    f"Operator   : grid={op.grid_shape}, image={op.image_shape}, n={op.indices.shape[0]}"
)
print(f"K-space    : {kspace_ref.shape}")

# %%
# mri-nufft adapter — GrogFourierOp
# ===================================
#
# :class:`~pygrog.interop.GrogFourierOp` exposes the ``op`` / ``adj_op``
# interface expected by mri-nufft density-compensated reconstructors and
# trajectory-display utilities.

from pygrog.interop import GrogFourierOp

mrinufft_op = GrogFourierOp(op)

# Forward: image (numpy) -> k-space (numpy)
ksp_np = mrinufft_op.op(image_ref.numpy())
print(f"\nGrogFourierOp.op   output shape: {ksp_np.shape}")

# Adjoint: k-space (numpy) -> image (numpy)
img_np = mrinufft_op.adj_op(ksp_np)
print(f"GrogFourierOp.adj_op output shape: {img_np.shape}")

# %%

fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
axes[0].imshow(np.abs(image_ref.numpy()), cmap="gray", origin="lower")
axes[0].set_title("Reference")
axes[0].axis("off")
axes[1].imshow(np.abs(img_np), cmap="gray", origin="lower")
axes[1].set_title("GrogFourierOp round-trip")
axes[1].axis("off")
plt.tight_layout()
plt.show()

# %%
# deepinverse adapter — autograd wrappers
# =======================================
#
# :func:`~pygrog.interop.sparse_fft_forward` and
# :func:`~pygrog.interop.sparse_fft_adjoint` wrap SparseFFT as
# ``torch.autograd.Function`` objects, enabling gradient flow for
# gradient-based reconstruction (e.g. unrolled networks, deepinverse).

from pygrog.interop import sparse_fft_forward, sparse_fft_adjoint

# Forward with gradient tracking
ksp_grad = kspace_ref.clone().requires_grad_(True)
img_out = sparse_fft_forward(ksp_grad, op)  # (image_shape,)
loss = img_out.abs().sum()
loss.backward()

print(f"\nsparse_fft_forward output shape : {img_out.shape}")
print(f"Gradient populated on ksp_grad  : {ksp_grad.grad is not None}")
print(f"Gradient shape                   : {ksp_grad.grad.shape}")

# Adjoint with gradient tracking
img_grad = image_ref.clone().requires_grad_(True)
ksp_out = sparse_fft_adjoint(img_grad, op)
ksp_out.abs().sum().backward()

print(f"\nsparse_fft_adjoint output shape : {ksp_out.shape}")
print(f"Gradient populated on img_grad  : {img_grad.grad is not None}")

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
# ``mrpro.operators.LinearOperator`` with autograd support via mrpro's base
# class.  It participates in mrpro operator algebra (``@``, ``+``, adjoint
# via ``.H``) and can be passed directly to mrpro solvers (e.g.
# ``ConjugateGradient``).
#
# .. note::
#    ``mrpro`` must be installed separately (``pip install mrpro``).  When
#    mrpro is not available this block prints an informative message and
#    continues.

from pygrog.interop import GrogLinearOp

try:
    mrpro_op = GrogLinearOp(op)
    print(f"\nGrogLinearOp ready: {type(mrpro_op).__mro__[1].__name__}")

    # Forward (adjoint NUFFT direction): k-space -> image
    # mrpro operators return tuples
    (img_mrpro,) = mrpro_op.forward(kspace_ref)
    print(f"mrpro forward output shape: {img_mrpro.shape}")

    # Adjoint: image -> k-space
    (ksp_mrpro,) = mrpro_op.adjoint(img_mrpro)
    print(f"mrpro adjoint output shape: {ksp_mrpro.shape}")

    # Compose: normal operator A^H A (mrpro uses .H for adjoint)
    AHA = mrpro_op.H @ mrpro_op
    (img_aha,) = AHA.forward(kspace_ref)
    print(f"Normal operator (A^H A) output shape: {img_aha.shape}")

except ImportError as exc:
    print(f"\nmrpro not available — skipping GrogLinearOp demo: {exc}")

# %%
# Summary
# =======
#
# The table below summarises which adapter to choose:
#
# +-----------------+----------------------------+-------------------------------+
# | Adapter         | Interface                  | Use case                      |
# +=================+============================+===============================+
# | GrogFourierOp   | ``op`` / ``adj_op``        | mri-nufft pipelines           |
# +-----------------+----------------------------+-------------------------------+
# | sparse_fft_*    | ``torch.autograd.Function``| PyTorch / deepinverse          |
# +-----------------+----------------------------+-------------------------------+
# | GrogLinop       | ``sigpy.linop.Linop``      | sigpy CG / PDHG solvers       |
# +-----------------+----------------------------+-------------------------------+
# | GrogLinearOp    | ``mrpro.LinearOperator``   | mrpro CG / PDHG solvers       |
# +-----------------+----------------------------+-------------------------------+

plt.show()
