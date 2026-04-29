"""Mixin providing :meth:`solve` to PyGROG operators.

Any operator (or decorator) that exposes ``.forward(x)``, ``.adjoint(y)``,
``.normal(x)``, and a ``.toeplitz`` attribute can inherit
:class:`SolveMixin` to gain the same ``op.solve(b, ...)`` entry point.
"""

from __future__ import annotations

__all__ = ["SolveMixin"]

import torch


class SolveMixin:
    r"""Provides :meth:`solve` dispatching to CG / LSMR.

    Requires the host class to expose at minimum ``forward``, ``adjoint``,
    and (optionally) ``normal`` methods plus a ``toeplitz`` boolean
    attribute (used to pick the default solver).
    """

    def solve(
        self,
        b: torch.Tensor,
        *,
        method: str | None = None,
        damp: float = 0.0,
        x0: torch.Tensor | None = None,
        max_iter: int = 20,
        tol: float = 1e-6,
        preconditioner=None,
        callback=None,
        **kwargs,
    ) -> torch.Tensor:
        r"""Solve :math:`\arg\min_x \|A x - b\|^2 + \lambda^2 \|x\|^2`.

        Parameters
        ----------
        b : torch.Tensor
            Right-hand side.
        method : {'cg', 'lsmr', None}
            Iterative solver.  ``None`` (default) → ``'cg'`` if
            ``self.toeplitz`` is True, otherwise ``'lsmr'``.
        damp : float, optional
            Tikhonov regulariser :math:`\lambda`.
        x0 : torch.Tensor | None
            Initial guess.  Default zero.
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

        if method is None:
            method = "cg" if getattr(self, "toeplitz", False) else "lsmr"
        method = method.lower()
        if method == "cg":
            return cg(
                self,
                b,
                damp=damp,
                x0=x0,
                max_iter=max_iter,
                tol=tol,
                preconditioner=preconditioner,
                callback=callback,
                **kwargs,
            )
        if method == "lsmr":
            return lsmr(
                self,
                b,
                damp=damp,
                x0=x0,
                max_iter=max_iter,
                callback=callback,
                **kwargs,
            )
        raise ValueError(f"Unknown method: {method!r}")
