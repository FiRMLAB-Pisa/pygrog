"""Framework-agnostic autograd Functions for pygrog SparseFFT.

For a linear operator A (measurement):
  - *measure*    A:   image   → k-space   (SparseFFT.adjoint)
  - *backproject* A^H: k-space → image    (SparseFFT.forward)

Because SparseFFT is implemented with custom kernels, automatic
differentiation through its internals is not meaningful.  These explicit
``torch.autograd.Function`` subclasses attach the correct backward pass:
the gradient of a linear operator is its adjoint, so

  backward(measure)      = backproject  (A^H)
  backward(backproject)  = measure      (A)

These functions are the single source of truth for gradient propagation
used by both the deepinv and mrpro adapters.
"""

__all__ = ["grog_backproject", "grog_measure"]

import torch
from torch.autograd import Function


class _GrogMeasureFn(Function):
    """A: image → k-space.  Backward = A^H (SparseFFT.forward)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, op) -> torch.Tensor:
        ctx.op = op
        return op.adjoint(x)  # SparseFFT.adjoint  ≡  forward NUFFT  ≡  A

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        # gradient w.r.t. x; no gradient for op (non-tensor)
        return ctx.op.forward(grad), None  # A^H = SparseFFT.forward


class _GrogBackprojectFn(Function):
    """A^H: k-space → image.  Backward = A (SparseFFT.adjoint)."""

    @staticmethod
    def forward(ctx, y: torch.Tensor, op) -> torch.Tensor:
        ctx.op = op
        return op.forward(y)  # SparseFFT.forward  ≡  adjoint NUFFT  ≡  A^H

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        # gradient w.r.t. y; no gradient for op (non-tensor)
        return ctx.op.adjoint(grad), None  # A = SparseFFT.adjoint


def grog_measure(x: torch.Tensor, op) -> torch.Tensor:
    """Apply measurement operator A (image → k-space) with correct gradient."""
    return _GrogMeasureFn.apply(x, op)


def grog_backproject(y: torch.Tensor, op) -> torch.Tensor:
    """Apply backprojection A^H (k-space → image) with correct gradient."""
    return _GrogBackprojectFn.apply(y, op)
