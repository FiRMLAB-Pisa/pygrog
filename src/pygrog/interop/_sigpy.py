"""sigpy Linop adapter for SparseFFT.

Wraps a pygrog operator (with ``forward`` / ``adjoint`` methods) as a
``sigpy.linop.Linop`` so it plugs into sigpy reconstruction algorithms
(conjugate gradient, primal-dual hybrid gradient, etc.) without modification.

sigpy ``Linop`` contract:
  - Subclass ``sigpy.linop.Linop``
  - Implement ``_apply(self, input)`` → applies the forward direction
  - Implement ``_adjoint_linop(self)`` → returns a ``Linop`` for the adjoint
  - Inputs/outputs are numpy (CPU) or cupy (GPU) arrays

Array conversion (numpy/cupy ↔ torch) is handled transparently by
:class:`~pygrog.operator.SparseFFT`, which is decorated with
:func:`mrinufft._array_compat.with_torch` on its ``forward`` and ``adjoint``
methods.  The sigpy ``_apply`` method therefore passes arrays directly to
the operator and receives numpy/cupy back.
"""

__all__ = ["GrogLinop"]


class GrogLinop:
    """Wrap a pygrog SparseFFT-like operator as a ``sigpy.linop.Linop``.

    The returned object is a real ``sigpy.linop.Linop`` with a working
    ``.H`` (adjoint) property, and therefore participates in all sigpy
    operator algebra (composition via ``*``, addition, scaling, etc.).

    Parameters
    ----------
    op : SparseFFT-like
        Any pygrog operator with ``forward(kspace) -> image`` and
        ``adjoint(image) -> kspace`` methods.

    Raises
    ------
    ImportError
        If ``sigpy`` is not installed.

    Examples
    --------
    ::

        from pygrog.operator import SparseFFT
        from pygrog.interop import GrogLinop

        base = SparseFFT(plan=grog.plan, smaps=smaps)
        A = GrogLinop(base)

        # Use inside sigpy CG reconstruction:
        import sigpy.alg as alg
        AHA = A.H * A
        # ... set up CG solver using AHA

    """

    # --- class factory (lazy) ---------------------------------------------
    _sigpy_class = None

    def __new__(cls, op):
        if cls._sigpy_class is None:
            cls._sigpy_class = cls._build_sigpy_class()
        return cls._sigpy_class(op)

    @staticmethod
    def _build_sigpy_class():
        try:
            from sigpy.linop import Linop
        except ImportError as exc:
            raise ImportError(
                "sigpy is required for GrogLinop.  "
                "Install it with: pip install sigpy"
            ) from exc

        class _GrogLinopImpl(Linop):
            """sigpy Linop wrapping a pygrog SparseFFT-like operator."""

            def __init__(self, op, *, _adjoint=False):
                self._op = op
                self._is_adjoint = _adjoint

                # Shapes for sigpy: oshape, ishape
                # Convention: forward = kspace → image (adjoint NUFFT direction)
                #   ishape = (n_coils, n_samples)
                #   oshape = image_shape (or (n_coils, *image_shape) without smaps)
                n_samples = op.indices.shape[0]
                n_coils = op.smaps.shape[0] if op.smaps is not None else 1

                if op.smaps is not None:
                    # forward: (n_coils, n_samples) → (*image_shape)
                    ksp_shape = [n_coils, n_samples]
                    img_shape = list(op.image_shape)
                else:
                    ksp_shape = [n_coils, n_samples]
                    img_shape = [n_coils, *list(op.image_shape)]

                if not _adjoint:
                    super().__init__(img_shape, ksp_shape)
                else:
                    super().__init__(ksp_shape, img_shape)

            def _apply(self, input):  # noqa: A002
                # SparseFFT.forward / .adjoint are decorated with with_torch,
                # so they accept numpy/cupy arrays and return the same type.
                if self._is_adjoint:
                    return self._op.adjoint(input)
                return self._op.forward(input)

            def _adjoint_linop(self):
                return _GrogLinopImpl(self._op, _adjoint=not self._is_adjoint)

        return _GrogLinopImpl
