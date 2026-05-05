"""Polynomial preconditioner for self-adjoint operators.

Given a normal operator ``T = A^H A`` with eigenvalue spectrum bracketed by
``[l, L]``, builds a polynomial ``P(T) = sum_k c_k T^k`` such that
``P(T) ~ T^{-1}`` in the L2 sense over ``[l, L]``.

The polynomial coefficients are obtained by minimising

.. math::

    \\int_l^L w(x) \\, (1 - x \\, p(x))^2 \\, dx

with respect to ``p`` of degree ``degree``, following Johnson, Micchelli &
Paul (1983) — the same construction used in the MRF reconstruction code.

The constructed preconditioner exposes ``.apply(x) -> P @ x`` and acts as a
drop-in for the ``preconditioner`` argument of :func:`pygrog._solve.cg`.
"""

from __future__ import annotations

__all__ = ["PolynomialPreconditioner"]

from typing import Any

import torch


def _l2_optimal_coeffs(degree: int, lo: float, hi: float) -> list[float]:
    """Solve for L2-optimal preconditioner polynomial coefficients.

    Returns the ``degree+1`` real coefficients ``c_0 ... c_d`` of
    ``p(x) = sum_k c_k x^k``.  Uses sympy for symbolic integration so the
    result is exact up to floating-point conversion at the end.
    """
    try:
        from sympy import diff, integrate, symbols, Poly
    except ImportError as exc:  # pragma: no cover - explicit dependency hint
        raise ImportError(
            "PolynomialPreconditioner requires 'sympy' (install with "
            "`pip install sympy`)."
        ) from exc
    import numpy as np

    c = symbols(f"c0:{degree + 1}")
    x = symbols("x")
    p = sum(c[k] * x**k for k in range(degree + 1))
    f = (1 - x * p) ** 2
    J = integrate(f, (x, lo, hi))

    mat = [[0.0] * (degree + 1) for _ in range(degree + 1)]
    vec = [0.0] * (degree + 1)
    for edx in range(degree + 1):
        eqn = diff(J, c[edx])
        tmp = eqn
        for cdx in range(degree + 1):
            mat[edx][cdx] = float(Poly(eqn, c[cdx]).coeffs()[0])
            tmp = tmp.subs(c[cdx], 0)
        vec[edx] = float(-tmp)

    mat = np.asarray(mat, dtype=np.float64)
    vec = np.asarray(vec, dtype=np.float64)
    coeffs = np.linalg.pinv(mat) @ vec
    return [float(v) for v in coeffs]


class PolynomialPreconditioner:
    """Polynomial preconditioner :math:`P(T) \\approx T^{-1}` over ``[l, L]``.

    Parameters
    ----------
    op : object
        Operator that exposes a self-adjoint normal operator.  Must provide
        either ``.normal(x)`` (preferred — uses Toeplitz embedding when
        available) or a ``.forward``/``.adjoint`` pair (fallback to
        :math:`A^H (A x)`).
    degree : int
        Polynomial degree.  Higher → better preconditioner, but each
        ``apply()`` evaluates ``T`` ``degree`` times.  Typical values: 2-5.
    spectrum : tuple[float, float], optional
        ``(l, L)`` bracket on the eigenvalues of ``T``.  When ``None`` the
        upper bound is estimated by power iteration; the lower bound
        defaults to 0.
    n_power_iter : int, optional
        Power iterations to estimate ``L`` when ``spectrum`` is None.
        Default 10.
    sample_shape : tuple[int, ...], optional
        Explicit input shape to use for power iteration.  Required when
        ``op`` does not expose ``image_shape`` *and* lacks SENSE coils
        (e.g. multi-coil :class:`SparseFFT` without ``smaps``).  Defaults
        to ``op.image_shape`` (with ``n_coils`` prepended when applicable).

    Notes
    -----
    The preconditioner is *constant* (no iterative inner loop): ``apply(x)``
    evaluates ``P(T) x`` via Horner's scheme:

    .. math::

        P(T) x = c_d T^{d-1} (T x) + ... + c_1 (T x) + c_0 x

    so the cost is exactly ``degree`` evaluations of ``T`` plus ``degree``
    AXPY operations, regardless of whether ``T`` is dense or Toeplitz.
    """

    def __init__(
        self,
        op: Any,
        *,
        degree: int = 3,
        spectrum: tuple[float, float] | None = None,
        n_power_iter: int = 10,
        sample_shape: tuple[int, ...] | None = None,
    ):
        if degree < 0:
            raise ValueError("degree must be >= 0")

        self._op = op
        self._sample_shape = (
            tuple(int(s) for s in sample_shape) if sample_shape is not None else None
        )

        # Pick T = AHA callable. PyGROG convention: forward = A, adjoint = A^H.
        normal_fn = getattr(op, "normal", None)
        if normal_fn is None:

            def normal_fn(x):
                return op.adjoint(op.forward(x))

        self._T = normal_fn

        # Estimate spectrum if not provided.
        if spectrum is None:
            lo = 0.0
            hi = self._estimate_largest_eigenvalue(n_power_iter)
        else:
            lo, hi = float(spectrum[0]), float(spectrum[1])

        self.spectrum = (lo, hi)
        self.degree = int(degree)
        self.coeffs = _l2_optimal_coeffs(self.degree, lo, hi)

    # ------------------------------------------------------------------
    def _estimate_largest_eigenvalue(self, n_iter: int) -> float:
        """Power-iteration estimate of ``||T||`` (largest singular value)."""
        if self._sample_shape is not None:
            full_shape = self._sample_shape
        else:
            image_shape = getattr(self._op, "image_shape", None)
            if image_shape is None:
                image_shape = getattr(self._op, "natural_shape", None)
            if image_shape is None:
                raise RuntimeError(
                    "Cannot estimate spectrum: pass `sample_shape=...` "
                    "explicitly when `op` has no `image_shape`."
                )
            full_shape = tuple(image_shape)
            smaps = getattr(self._op, "smaps", None)
            if smaps is None:
                n_coils = getattr(self._op, "n_coils", None)
                if n_coils is None:
                    raise RuntimeError(
                        "Cannot infer coil dimension for spectrum estimate: "
                        "pass `sample_shape=(n_coils, *image_shape)` explicitly."
                    )
                full_shape = (int(n_coils), *full_shape)

        device = getattr(self._op, "device", None)
        if device is None:
            device = torch.device("cpu")

        x = torch.randn(*full_shape, dtype=torch.complex64, device=device)
        x = x / (torch.linalg.vector_norm(x) + 1e-30)
        eig = 0.0
        for _ in range(n_iter):
            y = self._T(x)
            eig = torch.linalg.vector_norm(y).item()
            x = y / (eig + 1e-30)
        # Safety margin: bound the spectrum slightly above estimate.
        return 1.05 * eig

    # ------------------------------------------------------------------
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate ``P(T) x`` via Horner's scheme.

        Cost: ``degree`` evaluations of ``T`` plus ``degree`` AXPYs.
        """
        c = self.coeffs
        if self.degree == 0:
            return c[0] * x

        # Horner: starts with the deepest coefficient and works outwards.
        # P(T) x = c_0 x + T (c_1 x + T (c_2 x + ... + T (c_d x)))
        out = c[-1] * x
        for k in range(self.degree - 1, -1, -1):
            out = self._T(out) + c[k] * x
        return out

    # Convenience: callable form.
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.apply(x)

    def __repr__(self) -> str:
        return (
            f"PolynomialPreconditioner(degree={self.degree}, "
            f"spectrum=({self.spectrum[0]:.3g}, {self.spectrum[1]:.3g}))"
        )
