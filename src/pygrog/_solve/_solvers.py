"""Pure-torch CG and LSMR solvers.

Both solvers operate on any duck-typed operator with ``.forward(x)`` and
``.adjoint(y)`` methods.  CG additionally requires ``.normal(x)`` (or the
adjoint+forward composition is used as a fallback).

The signatures intentionally mirror :func:`mrinufft.extras.optim.cg` /
:func:`mrinufft.extras.optim.lsmr` so callers familiar with mri-nufft can
swap them in one-for-one.
"""

from __future__ import annotations

__all__ = ["cg", "lsmr", "solve"]

from collections.abc import Callable
from typing import Any

import torch


# ---------------------------------------------------------------------
# Shape-normalisation helper
# ---------------------------------------------------------------------
def _normalize_kspace_shape(op: Any, b: torch.Tensor) -> torch.Tensor:
    """Reshape flat ``b`` into the natural k-space shape that ``op.adjoint``
    returns, so that subsequent ``b - op.adjoint(...)`` operations align.

    SparseFFT and friends accept both flat ``(*B, *S, n_coils, n_samples)``
    and natural ``(*B, *S, n_coils, *natural_shape)`` inputs to ``forward``,
    but ``adjoint`` always emits the natural form.  We expand ``b`` once
    upfront so its shape stays invariant across iterations.
    """
    nat = getattr(op, "natural_shape", None)
    if nat is None or len(nat) <= 1:
        return b
    nat = tuple(int(s) for s in nat)
    n_samples = 1
    for s in nat:
        n_samples *= s
    if b.shape[-1] != n_samples:
        return b  # already natural (or some other layout)
    return b.reshape(*b.shape[:-1], *nat)


# =====================================================================
# Conjugate Gradient — solves A^H A x = A^H b (or normal equations of LS)
# =====================================================================
def cg(
    op: Any,
    b: torch.Tensor,
    *,
    damp: float = 0.0,
    x0: torch.Tensor | None = None,
    max_iter: int = 20,
    tol: float = 1e-6,
    preconditioner: Any | None = None,
    callback: Callable | None = None,
) -> torch.Tensor:
    r"""Conjugate-gradient solver for :math:`(A^H A + \lambda I) x = A^H b`.

    Solves the regularised normal equation system

    .. math::

        (A^H A + \lambda I) \, x \;=\; A^H b

    using PyGROG operator convention (``forward = A`` image→kspace,
    ``adjoint = A^H`` kspace→image).  When ``op.normal`` is unavailable
    (or for non-square ``op``), the explicit composition
    ``op.forward(op.adjoint(x))`` is used.

    Parameters
    ----------
    op : object
        Operator with ``.forward(x)``, ``.adjoint(y)``, optionally
        ``.normal(x)``.
    b : torch.Tensor
        Right-hand side k-space data.
    damp : float, optional
        L2 regularisation :math:`\lambda` on ``x``.  Default 0.
    x0 : torch.Tensor | None, optional
        Initial guess.  Default zero (image-shape inferred from
        ``op.adjoint(b)``).
    max_iter : int
        Maximum CG iterations.
    tol : float
        Stop when ``||r||/||r0|| < tol`` (relative residual).
    preconditioner : object | None
        Optional preconditioner with ``.apply(r) -> z`` performing
        ``z = M r`` where ``M ~ (A^H A)^{-1}``.  Activates preconditioned CG.
    callback : callable, optional
        Called as ``callback(k, x, residual)`` at each iteration.

    Returns
    -------
    torch.Tensor
        Solution image, same shape as ``x0`` (or ``op.adjoint(b)`` when
        ``x0`` is None).
    """
    # Normal-equation right-hand side: A^H b = adjoint(b) in pygrog convention.
    b = _normalize_kspace_shape(op, b)
    AHb = op.adjoint(b)

    # Pick a normal-op callable: T = A^H A = adjoint o forward.
    AHA = getattr(op, "normal", None)
    if AHA is None:

        def AHA(x):
            return op.adjoint(op.forward(x))

    x = torch.zeros_like(AHb) if x0 is None else x0.clone()

    r = AHb - AHA(x) - damp * x
    z = preconditioner.apply(r) if preconditioner is not None else r
    p = z.clone()
    rz_old = torch.vdot(r.reshape(-1), z.reshape(-1)).real

    r0_norm = torch.linalg.vector_norm(r) + 1e-30

    for k in range(max_iter):
        Ap = AHA(p) + damp * p
        alpha = rz_old / (torch.vdot(p.reshape(-1), Ap.reshape(-1)).real + 1e-30)
        x = x + alpha * p
        r = r - alpha * Ap

        rel = (torch.linalg.vector_norm(r) / r0_norm).item()
        if callback is not None:
            callback(k, x, rel)
        if rel < tol:
            break

        z = preconditioner.apply(r) if preconditioner is not None else r
        rz_new = torch.vdot(r.reshape(-1), z.reshape(-1)).real
        beta = rz_new / (rz_old + 1e-30)
        p = z + beta * p
        rz_old = rz_new

    return x


# =====================================================================
# LSMR — least-squares Krylov solver for argmin ||Ax - b||² + λ²||x||²
# =====================================================================
def lsmr(
    op: Any,
    b: torch.Tensor,
    *,
    damp: float = 0.0,
    x0: torch.Tensor | None = None,
    max_iter: int = 20,
    atol: float = 1e-6,
    btol: float = 1e-6,
    callback: Callable | None = None,
) -> torch.Tensor:
    r"""LSMR solver for :math:`\arg\min \| A x - b \|_2^2 + \lambda^2 \| x \|_2^2`.

    Implementation follows Fong & Saunders (2011); operates directly on the
    rectangular operator ``op`` (no normal equations, better conditioning
    than CG for ill-conditioned A).

    Parameters
    ----------
    op : object
        Operator with ``.forward(x)`` and ``.adjoint(y)``.
    b : torch.Tensor
        Right-hand side k-space data.
    damp : float, optional
        Tikhonov regulariser :math:`\lambda`.  Default 0.
    x0 : torch.Tensor | None, optional
        Initial guess.  Default zero.
    max_iter : int
        Maximum LSMR iterations.
    atol, btol : float
        Stopping tolerances (see Fong & Saunders).
    callback : callable, optional
        Called as ``callback(k, x, residual)`` at each iteration.

    Returns
    -------
    torch.Tensor
        Solution image.
    """
    # Initialise via Golub–Kahan bidiagonalisation.
    # Pygrog convention: A = forward (image→kspace), A^H = adjoint (kspace→image).
    b = _normalize_kspace_shape(op, b)
    if x0 is not None:
        b = b - op.forward(x0)

    beta = torch.linalg.vector_norm(b)
    if beta.item() == 0:
        # Trivial: b = 0 → x = x0 (or zero-image).
        return x0.clone() if x0 is not None else op.adjoint(b)

    u = b / beta
    v_full = op.adjoint(u)  # A^H u
    alpha = torch.linalg.vector_norm(v_full)
    if alpha.item() == 0:
        return x0.clone() if x0 is not None else torch.zeros_like(v_full)
    v = v_full / alpha

    # LSMR bookkeeping (see Fong & Saunders Table 5.1).
    zetabar = alpha * beta
    alphabar = alpha
    rho = torch.tensor(1.0, dtype=alpha.dtype, device=alpha.device)
    rhobar = torch.tensor(1.0, dtype=alpha.dtype, device=alpha.device)
    cbar = torch.tensor(1.0, dtype=alpha.dtype, device=alpha.device)
    sbar = torch.tensor(0.0, dtype=alpha.dtype, device=alpha.device)

    h = v.clone()
    hbar = torch.zeros_like(v)
    x = torch.zeros_like(v) if x0 is None else x0.clone()

    normb = beta.clone()

    for k in range(max_iter):
        # --- Bidiagonalisation step --------------------------------------
        u = op.forward(v) - alpha * u  # A v
        beta = torch.linalg.vector_norm(u)
        if beta.item() > 0:
            u = u / beta
            v_full = op.adjoint(u) - beta * v  # A^H u
            alpha = torch.linalg.vector_norm(v_full)
            if alpha.item() > 0:
                v = v_full / alpha

        # --- Construct rotations Q_k -------------------------------------
        alphahat = torch.sqrt(alphabar * alphabar + damp * damp)
        alphabar / alphahat
        damp / alphahat

        rhoold = rho.clone()
        rho = torch.sqrt(alphahat * alphahat + beta * beta)
        c = alphahat / rho
        s = beta / rho

        thetanew = s * alpha
        alphabar = c * alpha

        # --- Construct rotations Qbar_k -----------------------------------
        rhobarold = rhobar.clone()
        zetabar.clone() if k == 0 else zeta.clone()  # noqa: F821
        thetabar = sbar * rho
        rhotemp = cbar * rho
        rhobar = torch.sqrt(rhotemp * rhotemp + thetanew * thetanew)
        cbar = rhotemp / rhobar
        sbar = thetanew / rhobar
        zeta = cbar * zetabar
        zetabar = -sbar * zetabar

        # --- Update h, hbar, x -------------------------------------------
        hbar = h - (thetabar * rho / (rhoold * rhobarold)) * hbar
        x = x + (zeta / (rho * rhobar)) * hbar
        h = v - (thetanew / rho) * h

        rel = (torch.abs(zetabar) / (normb + 1e-30)).item()
        if callback is not None:
            callback(k, x, rel)
        if rel < atol + btol:
            break

    return x


# =====================================================================
# Top-level dispatcher
# =====================================================================
def solve(
    op: Any,
    b: torch.Tensor,
    *,
    method: str = "cg",
    **kwargs,
) -> torch.Tensor:
    """Dispatch to :func:`cg` or :func:`lsmr`.

    Parameters
    ----------
    op : object
        Operator (:class:`~pygrog.operator.SparseFFT`,
        :class:`~pygrog.operator.MaskedFFT`, decorator, etc.).
    b : torch.Tensor
        Right-hand side data.
    method : {'cg', 'lsmr'}
        Iterative solver.  ``'cg'`` solves the normal equations, ``'lsmr'``
        solves the rectangular least-squares directly.  Use ``'cg'`` when
        Toeplitz acceleration is on; ``'lsmr'`` when the problem is
        ill-conditioned and Toeplitz is off.
    **kwargs
        Forwarded to the chosen solver.

    Returns
    -------
    torch.Tensor
        Solution image.
    """
    method = method.lower()
    if method == "cg":
        return cg(op, b, **kwargs)
    if method == "lsmr":
        return lsmr(op, b, **kwargs)
    raise ValueError(f"Unknown solver method: {method!r}. Use 'cg' or 'lsmr'.")
