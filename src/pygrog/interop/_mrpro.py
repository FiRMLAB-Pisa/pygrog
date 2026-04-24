"""mrpro LinearOperator adapter for SparseFFT.

Wraps :class:`~pygrog.operator.SparseFFT` (or any pygrog operator with
``forward`` / ``adjoint`` methods) as an ``mrpro.operators.LinearOperator``
so it plugs into mrpro reconstruction pipelines and algorithms
(conjugate gradient, PDHG, etc.) without modification.

Gradients are computed via :mod:`pygrog.interop._torch` — explicit
``torch.autograd.Function`` subclasses whose backward is the adjoint of the
measurement operator — rather than relying on ``adjoint_as_backward=True``
or automatic differentiation through the GROG kernels.

mrpro ``LinearOperator`` contract (this adapter's convention):
  - ``forward(kspace)``  → image   (backprojection, A^H)
  - ``adjoint(image)``   → kspace  (measurement, A)
"""

__all__ = ["GrogLinearOp"]

import torch

from ._torch import grog_backproject, grog_measure


class GrogLinearOp:
    """Wrap a pygrog operator as an ``mrpro.operators.LinearOperator``.

    Because mrpro is an optional dependency, the class is built lazily on
    first instantiation so the import of this module does not fail when mrpro
    is not installed.

    Parameters
    ----------
    op : SparseFFT-like
        Any operator with ``forward(kspace) -> image`` and
        ``adjoint(image) -> kspace`` methods.

    Raises
    ------
    ImportError
        If ``mrpro`` is not installed.

    Examples
    --------
    ::

        from pygrog.operator import SparseFFT
        from pygrog.interop import GrogLinearOp

        base = SparseFFT(plan=grog.plan, smaps=smaps)
        mrpro_op = GrogLinearOp(base)

        # Use inside mrpro CG reconstruction:
        from mrpro.algorithms import ConjugateGradient
        x0 = torch.zeros(image_shape, dtype=torch.complex64)
        result, = ConjugateGradient(mrpro_op.H @ mrpro_op, mrpro_op.H(kspace)[0], x0)
    """

    # --- class factory -----------------------------------------------------
    _mrpro_class = None  # cached at class level

    def __new__(cls, op):
        """Return an instance of the lazily-built mrpro subclass."""
        if cls._mrpro_class is None:
            cls._mrpro_class = cls._build_mrpro_class()
        return cls._mrpro_class(op)

    @staticmethod
    def _build_mrpro_class():
        try:
            from mrpro.operators import LinearOperator
        except ImportError as exc:
            raise ImportError(
                "mrpro is required for GrogLinearOp.  "
                "Install it with: pip install mrpro"
            ) from exc

        class _GrogLinearOpImpl(LinearOperator):
            """mrpro LinearOperator wrapping a pygrog SparseFFT-like operator.

            Gradients are provided by explicit autograd Functions in
            ``pygrog.interop._torch``; ``adjoint_as_backward`` is intentionally
            *not* used to avoid double-wrapping the backward pass.
            """

            def __init__(self, op):
                super().__init__()
                self._op = op

            def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
                """Backprojection: k-space → image (A^H)."""
                return (grog_backproject(x, self._op),)

            def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
                """Measurement: image → k-space (A)."""
                return (grog_measure(x, self._op),)

        return _GrogLinearOpImpl
