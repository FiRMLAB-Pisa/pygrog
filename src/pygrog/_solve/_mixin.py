"""Mixin providing :meth:`solve` to PyGROG operators.

Any operator (or decorator) that exposes ``.forward(x)``, ``.adjoint(y)``,
``.normal(x)``, and a ``.toeplitz`` attribute can inherit
:class:`SolveMixin` to gain the same ``op.solve(b, ...)`` entry point.
"""

from __future__ import annotations

__all__ = ["SolveMixin"]

from typing import Any

import numpy as np
import torch


def _is_cupy_array(x: Any) -> bool:
    """Return True when *x* is a CuPy ndarray without importing CuPy eagerly."""
    return x.__class__.__module__.startswith("cupy") and hasattr(x, "__dlpack__")


def _to_torch_no_host_transfer(x: Any) -> torch.Tensor:
    """Convert numpy / cupy / torch inputs to torch tensors.

    - NumPy: ``torch.as_tensor`` (zero-copy view when contiguous/compatible).
    - CuPy: ``torch.from_dlpack`` (zero-copy on-device, no CPU transfer).
    - Torch: returned as-is.
    """
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, np.ndarray):
        return torch.as_tensor(x)
    if _is_cupy_array(x):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "solve() received a CuPy array but CUDA is unavailable in torch; "
                "refusing CPU fallback to avoid host transfer/copy."
            )
        return torch.from_dlpack(x)
    return torch.as_tensor(x)


def _from_torch_to_backend(x: torch.Tensor, backend: str):
    """Convert torch result back to caller backend (torch/numpy/cupy)."""
    if backend == "torch":
        return x
    if backend == "numpy":
        return x.detach().cpu().numpy()
    if backend == "cupy":
        import cupy as cp

        if x.device.type != "cuda":
            raise RuntimeError(
                "solve() produced a CPU tensor from CuPy input; this would require "
                "a host/device transfer, which is not allowed."
            )
        return cp.from_dlpack(x.detach())
    return x


def _detect_backend(x: Any) -> str:
    """Return input backend label among {'torch', 'numpy', 'cupy'}.

    Defaults to ``'torch'`` for generic array-like inputs.
    """
    if isinstance(x, torch.Tensor):
        return "torch"
    if isinstance(x, np.ndarray):
        return "numpy"
    if _is_cupy_array(x):
        return "cupy"
    return "torch"


class SolveMixin:
    r"""Provides :meth:`solve` dispatching to CG / LSMR.

    Requires the host class to expose at minimum ``forward``, ``adjoint``,
    and (optionally) ``normal`` methods plus a ``toeplitz`` boolean
    attribute (used to pick the default solver).
    """

    def solve(
        self,
        b,
        *,
        method: str | None = None,
        damp: float = 0.0,
        x0=None,
        max_iter: int = 20,
        tol: float = 1e-6,
        preconditioner=None,
        callback=None,
        **kwargs,
    ) -> torch.Tensor:
        r"""Solve :math:`\arg\min_x \|A x - b\|^2 + \lambda^2 \|x\|^2`.

        Parameters
        ----------
        b : array-like
            Right-hand side. NumPy inputs are accepted and converted to
            torch tensors internally.
        method : {'cg', 'lsmr', None}
            Iterative solver.  ``None`` (default) → ``'cg'`` if
            ``self.toeplitz`` is True, otherwise ``'lsmr'``.
        damp : float, optional
            Tikhonov regulariser :math:`\lambda`.
        x0 : array-like | None
            Initial guess. NumPy inputs are accepted and converted
            internally. Default zero.
        max_iter, tol : int, float
            Iteration limits.
        preconditioner : optional
            CG preconditioner (e.g.
            :class:`pygrog.PolynomialPreconditioner`).  Ignored by LSMR.
        callback : optional
            Called as ``callback(k, x, residual)``.

        Returns
        -------
        torch.Tensor
            Solution image.
        """
        from . import cg, lsmr

        backend = _detect_backend(b)
        b_t = _to_torch_no_host_transfer(b)
        x0_t = None if x0 is None else _to_torch_no_host_transfer(x0)

        if method is None:
            method = "cg" if getattr(self, "toeplitz", False) else "lsmr"
        method = method.lower()
        if method == "cg":
            out = cg(
                self,
                b_t,
                damp=damp,
                x0=x0_t,
                max_iter=max_iter,
                tol=tol,
                preconditioner=preconditioner,
                callback=callback,
                **kwargs,
            )
            return _from_torch_to_backend(out, backend)
        if method == "lsmr":
            out = lsmr(
                self,
                b_t,
                damp=damp,
                x0=x0_t,
                max_iter=max_iter,
                callback=callback,
                **kwargs,
            )
            return _from_torch_to_backend(out, backend)
        raise ValueError(f"Unknown method: {method!r}")
