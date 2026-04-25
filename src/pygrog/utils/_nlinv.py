"""NLINV coil sensitivity estimation.

Clean rewrite following the MATLAB reference implementation, extended
for non-Cartesian acquisitions via mri-nufft.
"""

__all__ = ["nlinv_calib"]

import torch

from numpy.typing import NDArray
from mrinufft._array_compat import with_torch

from .._base._fftc import fft, ifft
from .._base._nufft import nufft, nufft_adjoint
from .._toep._toep_op import ToeplitzOp
from .._utils import resize, estimate_shape


@with_torch
def nlinv_calib(
    y: NDArray[complex],
    cal_width: int = 24,
    ndim: int | None = None,
    mask: NDArray[bool] | None = None,
    shape: list[int] | tuple[int] | None = None,
    coords: NDArray[float] | None = None,
    weights: NDArray[float] | None = None,
    oversamp: float = 1.25,
    eps: float = 1e-3,
    sobolev_width: float = 200.0,
    sobolev_deg: int = 32,
    max_iter: int = 10,
    cg_iter: int = 10,
    cg_tol: float = 1e-2,
    alpha0: float = 1.0,
    q: float = 2 / 3,
    toeplitz: bool | None = None,
    ret_cal: bool = True,
    ret_image: bool = False,
) -> tuple[NDArray[complex], ...]:
    """
    Estimate coil sensitivity maps using NLINV.

    Parameters
    ----------
    y : NDArray[complex]
        Measured k-space data of shape ``(n_coils, ...)``.
    cal_width : int, optional
        Calibration region size.  The default is ``24``.
    ndim : int, optional
        Acquisition dimensionality (2 or 3).  Required for Cartesian.
    mask : NDArray[bool], optional
        Cartesian sampling pattern.
    shape : list[int] | tuple[int], optional
        Image dimensions.  Required for non-Cartesian.
    coords : NDArray[float], optional
        Fourier domain coordinates of shape ``(..., ndim)``.
    weights : NDArray[float], optional
        Density compensation weights.
    oversamp : float, optional
        NUFFT oversampling factor.  The default is ``1.25``.
    eps : float, optional
        NUFFT precision.  The default is ``1e-3``.
    sobolev_width : float, optional
        Sobolev kernel width.  The default is ``220``.
    sobolev_deg : int, optional
        Sobolev norm order.  The default is ``16``.
    max_iter : int, optional
        Gauss-Newton iterations.  The default is ``10``.
    cg_iter : int, optional
        Conjugate gradient iterations per Gauss-Newton step.  The default is ``10``.
    cg_tol : float, optional
        CG relative tolerance.  The default is ``1e-2``.
    alpha0 : float, optional
        Initial regularization parameter.  The default is ``1.0``.
    q : float, optional
        Regularization decay factor per iteration.  The default is ``1/3``.
    toeplitz : bool | None, optional
        Use Toeplitz acceleration for the CG normal equation in the
        non-Cartesian case.  Precomputes the NUFFT PSF kernel and replaces
        each ``nufft_adjoint(nufft(·))`` pair with an FFT-based multiplication,
        significantly reducing per-iteration cost.  ``None`` (default) auto-selects:
        ``True`` on CPU (where NUFFT is expensive), ``False`` on GPU.
    ret_cal : bool, optional
        Return synthesized calibration data.  The default is ``True``.
    ret_image : bool, optional
        Return reconstructed image.  The default is ``False``.

    Returns
    -------
    smaps : NDArray[complex]
        Coil sensitivity maps of shape ``(n_coils, *shape)``.
    grappa_train : NDArray[complex], optional
        Synthesized calibration k-space of shape ``(n_coils, *cal_shape)``.
        Only if ``ret_cal=True``.
    image : NDArray[complex], optional
        Reconstructed image.  Only if ``ret_image=True``.

    """
    noncart = coords is not None
    n_coils = y.shape[0]
    device = y.device

    # Auto-select Toeplitz: use PSF embedding on CPU (avoids repeated NUFFTs),
    # disable on GPU where NUFFT is fast enough.
    if toeplitz is None:
        toeplitz = noncart and (device.type == "cpu")

    # --- Setup ----------------------------------------------------------
    if noncart:
        oshape, cshape, y, coords, weights, W = _setup_noncartesian(
            y,
            shape,
            coords,
            weights,
            cal_width,
            oversamp,
            eps,
            sobolev_width,
            sobolev_deg,
        )
    else:
        oshape, cshape, y, mask, W = _setup_cartesian(
            y,
            ndim,
            mask,
            cal_width,
            sobolev_width,
            sobolev_deg,
        )

    # --- Build operator closures ----------------------------------------
    if noncart:
        # Pre-compute sqrt(DCF) for consistent data pre-conditioning:
        # y was multiplied by sqrt(W) in _setup_noncartesian, so the forward
        # model must also include sqrt(W) so residuals are in the same units.
        _sqrt_w = weights**0.5 if weights is not None else None

        def fwd(x):
            k = _op_noncart(x, coords, weights, cshape, oversamp, eps)
            return k * _sqrt_w if _sqrt_w is not None else k

        def deriv(x0, dx):
            k = _der_noncart(x0, dx, W, coords, weights, cshape, oversamp, eps)
            return k * _sqrt_w if _sqrt_w is not None else k

        def derivH(x0, dk):
            dk_w = dk * _sqrt_w if _sqrt_w is not None else dk
            return _derH_noncart(x0, dk_w, W, coords, weights, cshape, oversamp, eps)

        if toeplitz:
            toep_op = ToeplitzOp(cshape, coords, weights, oversamp, eps)
            normal_eq = lambda x0, dx: _derH_der_noncart_toeplitz(
                x0, dx, W, toep_op, cshape
            )
        else:
            normal_eq = lambda x0, dx: derivH(x0, deriv(x0, dx))
    else:
        fwd = lambda x: _op_cart(mask, x)
        deriv = lambda x0, dx: _der_cart(mask, W, x0, dx)
        derivH = lambda x0, dk: _derH_cart(mask, W, x0, dk)
        normal_eq = lambda x0, dx: derivH(x0, deriv(x0, dx))

    # --- Initialize -----------------------------------------------------
    XN = torch.zeros((n_coils + 1, *cshape), dtype=y.dtype, device=device)
    XN[0] = 1.0
    X0 = XN.clone()

    # Normalize data
    yscale = 100.0 / (y.conj().ravel() * y.ravel()).sum().real.sqrt()

    YS = y * yscale

    # --- IRGNM loop -----------------------------------------------------
    alpha = alpha0
    for _ in range(max_iter):
        # Apply weights to get XT
        XT = _apply_W(XN, W)

        # Residual
        RES = YS - fwd(XT)

        # RHS = derH(RES) + alpha * (X0 - XN)
        r = derivH(XT, RES) + alpha * (X0 - XN)

        # CG solve
        z = torch.zeros_like(r)
        d = r.clone()
        dnew = (r.conj().ravel() * r.ravel()).sum().real
        dnot = dnew.sqrt()

        for _ in range(cg_iter):
            q_vec = normal_eq(XT, d) + alpha * d
            dq = (d.conj().ravel() * q_vec.ravel()).sum().real
            if dq == 0:
                break
            a = dnew / dq
            z = z + a * d
            r = r - a * q_vec
            dold = dnew
            dnew = (r.conj().ravel() * r.ravel()).sum().real
            if dnew == 0 or dold == 0:
                break
            d = (dnew / dold) * d + r
            if dnew.sqrt() < cg_tol * dnot:
                break

        XN = XN + z
        alpha *= q

    # --- Post-process ---------------------------------------------------
    return _postprocess(
        XN,
        W,
        yscale,
        cshape,
        oshape,
        noncart,
        ret_cal,
        ret_image,
    )


# -----------------------------------------------------------------------
# Setup helpers
# -----------------------------------------------------------------------


def _setup_cartesian(y, ndim, mask, cal_width, sobolev_width, sobolev_deg):
    """Setup for Cartesian acquisition."""
    if ndim is None or ndim not in {2, 3}:
        raise ValueError("ndim must be 2 or 3 for Cartesian acquisition.")

    oshape = tuple(y.shape[1:])

    mask = (y.abs().pow(2).sum(dim=0) > 0).float() if mask is None else mask.float()

    if cal_width is not None:
        # Extract calibration region
        cal_shape = list(y.shape[:1]) + [min(cal_width, s) for s in oshape]
        y = resize(y, cal_shape)
        mask = resize(mask, cal_shape[1:])
    cshape = tuple(y.shape[1:])

    W = _sobolev_weights(cshape, oshape, sobolev_width, sobolev_deg, device=y.device)

    return oshape, cshape, y, mask, W


def _setup_noncartesian(
    y,
    shape,
    coords,
    weights,
    cal_width,
    oversamp,  # noqa: ARG001
    eps,  # noqa: ARG001
    sobolev_width,
    sobolev_deg,
):
    """Setup for non-Cartesian acquisition."""
    ndim = coords.shape[-1]
    oshape = tuple(shape) if shape is not None else estimate_shape(coords)

    # Filter to calibration samples: keep only those within cal_width/2 radius
    # in scaled k-space units (coords scaled so each axis spans oshape).
    # rescale_coords expects (ndim, npts); coords is (..., ndim) → transpose
    from .._utils import rescale_coords

    flat_coords = coords.reshape(-1, ndim)
    flat_y = y.reshape(y.shape[0], -1)
    flat_w = weights.reshape(-1) if weights is not None else None

    scaled = rescale_coords(flat_coords.movedim(-1, 0), list(oshape[-ndim:])).movedim(0, -1)
    radius = 0.5 * cal_width
    mask = (scaled**2).sum(dim=-1).sqrt() <= radius

    flat_coords = flat_coords[mask].contiguous()
    flat_y = flat_y[:, mask].contiguous()
    flat_w = flat_w[mask].contiguous() if flat_w is not None else None

    # Apply DCF
    if flat_w is not None:
        flat_y = flat_y * flat_w**0.5

    cshape = tuple([cal_width] * ndim)
    W = _sobolev_weights(cshape, oshape, sobolev_width, sobolev_deg, device=y.device)

    return oshape, cshape, flat_y, flat_coords, flat_w, W


# -----------------------------------------------------------------------
# Forward model operators (Cartesian)
# -----------------------------------------------------------------------


def _apply_W(XN, W):
    """Apply Sobolev weights to coil components."""
    XT = XN.clone()
    for s in range(1, XN.shape[0]):
        XT[s] = ifft(W * XN[s])
    return XT


def _apply_WH(x, W):
    """Adjoint Sobolev weighting."""
    return W.conj() * fft(x)


def _op_cart(P, X):
    """Forward: F(x) = P * FFT(rho * c_i)."""
    K = torch.zeros_like(X[1:])
    for i in range(X.shape[0] - 1):
        K[i] = P * fft(X[0] * X[i + 1])
    return K


def _der_cart(P, W, X0, DX):
    """Derivative: dF(x)[dx]."""
    K = torch.zeros_like(DX[1:])
    for i in range(DX.shape[0] - 1):
        K[i] = P * fft(X0[0] * ifft(W * DX[i + 1]) + DX[0] * X0[i + 1])
    return K


def _derH_cart(P, W, X0, DK):
    """Adjoint derivative: dF(x)^H[dk]."""
    DX = torch.zeros_like(X0)
    for i in range(DK.shape[0]):
        K = ifft(P * DK[i])
        DX[0] = DX[0] + K * X0[i + 1].conj()
        DX[i + 1] = _apply_WH(K * X0[0].conj(), W)
    return DX


# -----------------------------------------------------------------------
# Forward model operators (Non-Cartesian)
# -----------------------------------------------------------------------


def _op_noncart(X, coords, _weights, _shape, oversamp, eps):
    """Forward: F(x) = NUFFT(rho * c_i)."""
    K = torch.zeros(
        (X.shape[0] - 1, *coords.shape[:-1]), dtype=X.dtype, device=X.device
    )
    for i in range(X.shape[0] - 1):
        K[i] = nufft(X[0] * X[i + 1], coords, oversamp, eps)
    return K


def _der_noncart(X0, DX, W, coords, _weights, _shape, oversamp, eps):
    """Derivative for non-Cartesian."""
    K = torch.zeros(
        (DX.shape[0] - 1, *coords.shape[:-1]), dtype=DX.dtype, device=DX.device
    )
    for i in range(DX.shape[0] - 1):
        K[i] = nufft(
            X0[0] * ifft(W * DX[i + 1]) + DX[0] * X0[i + 1],
            coords,
            oversamp,
            eps,
        )
    return K


def _derH_noncart(X0, DK, W, coords, _weights, shape, oversamp, eps):
    """Adjoint derivative for non-Cartesian."""
    DX = torch.zeros_like(X0)
    for i in range(DK.shape[0]):
        K = nufft_adjoint(DK[i], coords, shape, oversamp, eps)
        DX[0] = DX[0] + K * X0[i + 1].conj()
        DX[i + 1] = _apply_WH(K * X0[0].conj(), W)
    return DX


def _derH_der_noncart_toeplitz(X0, DX, W, toep_op, _shape):
    """Fused derivH @ deriv for non-Cartesian using Toeplitz normal operator.

    Replaces ``nufft_adjoint(nufft(f_i))`` for each coil with
    ``toep_op(f_i)`` (FFT-based PSF multiplication), avoiding 2x n_coils
    NUFFT calls per CG iteration.
    """
    out = torch.zeros_like(X0)
    for i in range(DX.shape[0] - 1):
        # Same image as _der_noncart would NUFFT-forward
        f_i = X0[0] * ifft(W * DX[i + 1]) + DX[0] * X0[i + 1]
        # Toeplitz normal operator: A^H A f_i via FFT PSF multiplication
        T_i = toep_op(f_i)
        # Same accumulation as _derH_noncart
        out[0] = out[0] + T_i * X0[i + 1].conj()
        out[i + 1] = _apply_WH(T_i * X0[0].conj(), W)
    return out


# -----------------------------------------------------------------------
# Postprocessing
# -----------------------------------------------------------------------


def _postprocess(XN, W, yscale, cshape, oshape, _noncart, ret_cal, ret_image):
    """Post-process NLINV result to extract smaps and optional outputs."""
    # Apply weights and undo normalization
    X = _apply_W(XN, W) / yscale

    rho = X[0]
    smaps = X[1:]

    # RSS normalization
    rss = (smaps.conj() * smaps).sum(dim=0).real.sqrt()
    rss = rss.clamp(min=1e-10)
    rho = rho * rss
    smaps = smaps / rss

    results = []

    # Interpolate smaps to original matrix size
    spatial_ndim = len(cshape)
    spatial_axes = tuple(range(-spatial_ndim, 0))
    smaps_k = fft(smaps, axes=spatial_axes)
    smaps_full_shape = list(smaps.shape[:-spatial_ndim]) + list(oshape[-spatial_ndim:])
    smaps = ifft(resize(smaps_k, smaps_full_shape), axes=spatial_axes)

    results.append(smaps)

    if ret_cal:
        grappa_train = fft(X[1:] * X[0], axes=spatial_axes)
        cal_shape = list(grappa_train.shape[:-spatial_ndim]) + [
            min(s, c)
            for s, c in zip(cshape, grappa_train.shape[-spatial_ndim:], strict=False)
        ]
        grappa_train = resize(grappa_train, cal_shape)
        results.append(grappa_train)

    if ret_image:
        results.append(rho)

    return tuple(results) if len(results) > 1 else results[0]


# -----------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------


def _sobolev_weights(cshape, oshape, width, degree, device=None):
    """Compute Sobolev regularization weights in k-space.

    Matches the mrops / MATLAB reference: pixel-unit coordinates
    ``arange(-n//2, n//2)`` scaled by ``width / max(oshape)^2``.
    This ensures the filter half-power point is independent of
    the calibration grid size and consistent with the reference.
    """
    n = max(oshape)
    kw = width / (n**2)
    grids = torch.meshgrid(
        *[
            torch.arange(-s // 2, s // 2, dtype=torch.float32, device=device)
            for s in cshape
        ],
        indexing="ij",
    )
    d = sum(g**2 for g in grids)
    return (1.0 / (1.0 + kw * d) ** (degree / 2)).float()
