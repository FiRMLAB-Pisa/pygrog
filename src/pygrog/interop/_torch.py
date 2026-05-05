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

The functions are also compatible with the :mod:`torch.func` transforms
(``grad``, ``jacrev``, ``vmap``) that mrpro's ``pgd`` and similar
algorithms rely on.  Because our forward path bottoms out in C++ kernels
that cannot read from functorch wrapper tensors (``GradTrackingTensor``,
``BatchedTensor``), the backward unwraps incoming cotangents to plain
tensors before invoking the kernel and re-wraps the result so the calling
transform sees the expected layered tensor.
"""

__all__ = ["grog_backproject", "grog_measure"]

import torch
from torch.autograd import Function


# --- functorch wrapper helpers --------------------------------------------
def _unwrap_functorch(t: torch.Tensor):
    """Strip ``GradTrackingTensor`` / ``BatchedTensor`` wrappers from ``t``.

    Returns ``(plain_tensor, layers)`` where ``layers`` is the ordered list
    of wrappers that were peeled off, suitable for :func:`_rewrap_functorch`.
    """
    from torch._C import _functorch as _ft

    layers: list[tuple[str, int, int | None]] = []
    while True:
        if _ft.is_gradtrackingtensor(t):
            level = _ft.maybe_get_level(t)
            t = _ft._unwrap_for_grad(t, level)
            layers.append(("grad", level, None))
        elif _ft.is_batchedtensor(t):
            level = _ft.maybe_get_level(t)
            bdim = _ft.maybe_get_bdim(t)
            t = _ft.get_unwrapped(t)
            layers.append(("vmap", level, bdim))
        else:
            return t, layers


def _rewrap_functorch(t: torch.Tensor, layers) -> torch.Tensor:
    """Re-apply the wrapper stack returned by :func:`_unwrap_functorch`."""
    from torch._C import _functorch as _ft

    for kind, level, bdim in reversed(layers):
        if kind == "grad":
            t = _ft._wrap_for_grad(t, level)
        else:
            t = _ft._add_batch_dim(t, bdim, level)
    return t


class _GrogMeasureFn(Function):
    """A: image → k-space.  Backward = A^H (SparseFFT.adjoint)."""

    generate_vmap_rule = True

    @staticmethod
    def forward(x: torch.Tensor, op) -> torch.Tensor:
        return op.forward(x)  # SparseFFT.forward  ≡  forward NUFFT  ≡  A

    @staticmethod
    def setup_context(ctx, inputs, _output):
        _, op = inputs
        ctx.op = op

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        # Unwrap any functorch wrappers (torch.func.grad / vmap) and pop the
        # interpreter stack so that buffers allocated inside the C++ kernel
        # are not lifted to the active functorch level.
        from torch._functorch.pyfunctorch import temporarily_clear_interpreter_stack

        plain, _ = _unwrap_functorch(grad)
        with temporarily_clear_interpreter_stack():
            out = ctx.op.adjoint(plain.contiguous())  # A^H = SparseFFT.adjoint
        return out, None


class _GrogBackprojectFn(Function):
    """A^H: k-space → image.  Backward = A (SparseFFT.forward)."""

    generate_vmap_rule = True

    @staticmethod
    def forward(y: torch.Tensor, op) -> torch.Tensor:
        return op.adjoint(y)  # SparseFFT.adjoint  ≡  adjoint NUFFT  ≡  A^H

    @staticmethod
    def setup_context(ctx, inputs, _output):
        y, op = inputs
        ctx.op = op
        ctx.y_shape = y.shape

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        from torch._functorch.pyfunctorch import temporarily_clear_interpreter_stack

        plain, _ = _unwrap_functorch(grad)
        with temporarily_clear_interpreter_stack():
            out = ctx.op._adjoint_flat(
                plain.contiguous()
            )  # always (..., n_samples) flat
            out = out.reshape(ctx.y_shape)  # A = SparseFFT.forward
        return out, None


def grog_measure(x: torch.Tensor, op) -> torch.Tensor:
    """Apply measurement operator A (image → k-space) with correct gradient."""
    return _GrogMeasureFn.apply(x, op)


def grog_backproject(y: torch.Tensor, op) -> torch.Tensor:
    """Apply backprojection A^H (k-space → image) with correct gradient."""
    return _GrogBackprojectFn.apply(y, op)
