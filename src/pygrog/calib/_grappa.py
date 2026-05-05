"""GRAPPA kernel table calculation subroutines."""

__all__ = ["KernelTable"]

import warnings
from types import SimpleNamespace

from numpy.typing import NDArray

import numpy as np
import torch

from mrinufft._array_compat import with_torch


def lstsq(A, b, lamda=0.0):
    """Tikhonov-regularized least squares via augmented system + torch.linalg.lstsq.

    Solves ``min_X ||A @ X - B||_F^2 + lamda * ||X||_F^2`` by forming the
    augmented system ``[A; sqrt(lamda)*I] @ X = [B; 0]`` and delegating to
    ``torch.linalg.lstsq``, which operates on the rectangular matrix directly
    (no normal-equation squaring of the condition number).

    Parameters
    ----------
    A : NDArray
        Source matrix of shape ``(*, M, N)``.
    b : NDArray
        Target matrix of shape ``(*, M, K)``.
    lamda: float
        Tikhonov regularization parameter.

    Returns
    -------
    NDArray
        Solution of shape ``(*, N, K)``.
    """
    A_t = torch.as_tensor(A)
    b_t = torch.as_tensor(b)
    N = A_t.shape[-1]

    if lamda > 0:
        sqrt_lamda = lamda**0.5
        I_reg = sqrt_lamda * torch.eye(N, dtype=A_t.dtype, device=A_t.device)
        if A_t.dim() > 2:
            I_reg = I_reg.expand(*A_t.shape[:-2], N, N)
        zeros = torch.zeros(
            *A_t.shape[:-2], N, b_t.shape[-1], dtype=b_t.dtype, device=b_t.device
        )
        A_aug = torch.cat([A_t, I_reg], dim=-2)
        b_aug = torch.cat([b_t, zeros], dim=-2)
    else:
        A_aug, b_aug = A_t, b_t

    sol = torch.linalg.lstsq(A_aug, b_aug, driver="gelsd").solution

    if isinstance(A, np.ndarray):
        return sol.numpy()
    return sol


def _matrix_logm(A: torch.Tensor) -> torch.Tensor:
    """Batched matrix logarithm via eigendecomposition: logm(A) = V diag(log(λ)) V⁻¹.

    Promotes to complex128 for numerical accuracy, then casts back.
    Works for 2-D ``(N, N)`` and batched ``(*, N, N)`` inputs.
    """
    dtype_out = A.dtype if A.is_complex() else torch.complex64
    A128 = A.to(torch.complex128)
    vals, vecs = torch.linalg.eig(A128)
    result = vecs @ torch.diag_embed(torch.log(vals)) @ torch.linalg.inv(vecs)
    return result.to(dtype_out)


def KernelTable(
    train_data: NDArray[complex], radius: float, precision: int = 1, lamda: float = 0.01
) -> tuple[NDArray[complex], int, int]:
    """
    Set the GRAPPA kernels and compute the GROG table for interpolation.

    Parameters
    ----------
    train_data : NDArray[complex]
        Calibration k-space region of shape ``(coils, z_cal, y_cal, x_cal)``.
    radius : float
        Desired interpolation radius in k-space grid units (radius = 1 -> distance between two Cartesian samples).
    precision : int
        Number of decimal digits to round shifts.
    lamda: float
        L2 regularization for GRAPPA kernel estimation.

    Returns
    -------
    NDArray
        GRAPPA kernel table for fractional shifts ``(dx, dy, dz)`` of shape
        ``(nsteps**ndim, coils, coils)``.
    int
        Number of fractional steps.
    int
        Number of spatial dimensions.

    """
    grappa_kernels = train_grappa(train_data, lamda)

    # Calculate displacements steps.
    # stepsize is the quantisation step in standard k-space units.
    # nsteps covers the full range [-radius, +radius] with step stepsize,
    # so nsteps - 1 = 2 * radius / stepsize.
    # delta[i] must equal the actual shift d such that
    #   interp_idx = (radius + d) / stepsize → d = (i - (nsteps-1)/2) * stepsize.
    # Dividing by (nsteps-1) instead (old code) gave delta = d/(2*radius),
    # which is correct only for radius=0.5 (kernel_width=1).
    stepsize = 10 ** (-precision)
    nsteps = int(2 * radius / stepsize + 1)
    deltas = torch.arange(nsteps).float()
    deltas = (deltas - (nsteps - 1) / 2) * stepsize

    # Pre-compute partial operators
    Gx = grappa_power(grappa_kernels.x, deltas)  # (nsteps, nc, nc)
    Gy = grappa_power(grappa_kernels.y, deltas)  # (nsteps, nc, nc)
    if grappa_kernels.z is not None:
        Gz = grappa_power(grappa_kernels.z, deltas)  # (nsteps, nc, nc), 3D only
        ndim = 3
    else:
        Gz = None
        ndim = 2

    return prepare_grappa_table(Gx, Gy, Gz, nsteps, ndim).numpy(), nsteps, ndim


def prepare_grappa_table(
    Gx: NDArray[complex],
    Gy: NDArray[complex],
    Gz: NDArray[complex] | None,
    nsteps: int,  # noqa: ARG001
    ndim: int,
) -> NDArray[complex]:
    """
    Prepare the GRAPPA operator table.

    Parameters
    ----------
    Gx : NDArray
        GRAPPA kernel for shifting alonx ``x``, shaped ``(nsteps, ncoils, ncoils)``.
    Gy : NDArray
        GRAPPA kernel for shifting alonx ``y``, shaped ``(nsteps, ncoils, ncoils)``.
    Gz : NDArray
        GRAPPA kernel for shifting alonx ``z``, shaped ``(nsteps, ncoils, ncoils)``.
        Not provided for 2D imaging.
    nsteps : int
        Number of fractional steps between ``-0.5`` and ``0.5``.
    ndim : int
        Acquisition dimensionality (``2`` or ``3``).

    Returns
    -------
    NDArray
        GRAPPA kernel table for fractional shifts ``(dx, dy, dz)`` of shape
        ``(nsteps**ndim, ncoils, ncoils)``.

    """
    Gx = torch.as_tensor(Gx)
    Gy = torch.as_tensor(Gy)

    if ndim == 2:
        # table[i*nsteps+j] = Gx[j] @ Gy[i]  via broadcasting matmul
        grappa_table = (Gx[None] @ Gy[:, None]).reshape(-1, *Gx.shape[-2:])

    elif ndim == 3:
        if Gz is None:
            raise ValueError("3D interpolation requires Z operator")
        Gz = torch.as_tensor(Gz)
        # table[i*nsteps^2 + j*nsteps + k] = Gx[k] @ Gy[j] @ Gz[i]
        grappa_table = (Gx[None, None] @ Gy[None, :, None] @ Gz[:, None, None]).reshape(
            -1, *Gx.shape[-2:]
        )

    else:
        raise ValueError(f"GROG interpolation only supports 2D or 3D data, got {ndim}D")

    return grappa_table


def grappa_power(G_unit: torch.Tensor, exponents: torch.Tensor) -> torch.Tensor:
    """Compute batched fractional matrix powers of a GRAPPA operator.

    Uses the identity ``G^p = exp(p * logm(G))`` to handle fractional and
    negative exponents without ever forming the normal equations.  All powers
    are computed in a single batched ``matrix_exp`` call.

    Parameters
    ----------
    G_unit : torch.Tensor
        GRAPPA kernel for unit shifts, shape ``(nc, nc)``.
    exponents : torch.Tensor
        1-D tensor of exponents (e.g. fractional shifts).

    Returns
    -------
    torch.Tensor
        Stacked operators of shape ``(len(exponents), nc, nc)``.
    """
    G_unit = torch.as_tensor(G_unit)
    exponents = torch.as_tensor(exponents, dtype=torch.float32)
    lG = _matrix_logm(G_unit)  # (nc, nc)
    # Broadcast: (nsteps, 1, 1) * (nc, nc) -> (nsteps, nc, nc)
    lG_batch = exponents.to(lG.dtype).view(-1, 1, 1) * lG.unsqueeze(0)
    return torch.linalg.matrix_exp(lG_batch).to(G_unit.dtype)


def train_grappa(
    train_data: NDArray[complex],
    lamda: float | None = None,
    coords: NDArray[float] | None = None,
) -> SimpleNamespace:
    """
    Train GRAPPA Operator Gridding (GROG) interpolator.

    Parameters
    ----------
    train_data : NDArray[complex]
        Calibration region data of shape ``(nc, nz, ny, nx)`` or ``(nc, ny, nx)``.
        Usually a small portion from the center of kspace.
    lamda: float | None, optional
        Tikhonov regularization parameter.  Set to 0 for no
        regularization. Defaults to ``0.01`` for standard GROG.
        and ``0.0`` for self-calibrating GROG.
    coords : NDArray[float], optional
        Fourier domain coordinate array of shape ``(..., ndim)``.
        ``ndim`` determines the number of dimensions to apply the NUFFT
        (``None`` for Cartesian).

    Returns
    -------
    SimpleNamespace
        Output grog interpolator with attributes ``x``, ``y`` (single slice 2D)
        or ``x``, ``y``, ``z`` (multi-slice or 3D).

    Notes
    -----
    Produces the unit operator described in [1]_.

    This seems to only work well when coil sensitivities are very
    well separated/distinct.  If coil sensitivities are similar,
    operators perform poorly.

    References
    ----------
    .. [1] Griswold, Mark A., et al. "Parallel magnetic resonance
           imaging using the GRAPPA operator formalism." Magnetic
           resonance in medicine 54.6 (2005): 1553-1556.

    """
    if lamda is None:
        lamda = 0.01 if coords is None else 0.0
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        return _train(train_data, lamda, coords)


# %% subroutines
@with_torch
def _train(train_data, lamda, coords):
    ndim = len(train_data.shape) - 1 if coords is None else coords.shape[-1]

    # get grappa operator
    kern = _calc_grappaop(ndim, train_data, lamda, coords)
    Gx = kern.Gx
    Gy = kern.Gy
    Gz = kern.Gz if ndim == 3 else None

    return SimpleNamespace(x=Gx, y=Gy, z=Gz)


def _calc_grappaop(ndim, train_data, lamda, coords):
    train_data = train_data / torch.linalg.norm(train_data)
    if coords is not None:
        gz = None
        gy, gx = _radial_grappa_op(train_data, lamda, coords)
    else:
        if ndim == 2:
            gz = None
            gy, gx = _grappa_op_2d(train_data, lamda)
        elif ndim == 3:
            gz, gy, gx = _grappa_op_3d(train_data, lamda)

    return SimpleNamespace(Gx=gx, Gy=gy, Gz=gz)


def _radial_grappa_op(calib, lamda, coords):
    """Return a 2D GROG operators from radial data."""
    calib = calib.movedim(0, -1)
    nr, _ns, nc = calib.shape

    # extract x and y components of trajectory
    xcoord = coords[..., 0].T
    ycoord = coords[..., 1].T

    # we need sources (last source has no target!)
    S = calib[:, :-1, ...]

    # and targets (first target has no associated source!)
    T = calib[:, 1:, ...]

    # train one operator per readout: (nr, nc, nc)
    Gtheta = lstsq(S, T, lamda)
    # batched logm: (nr, nc, nc); F-order flatten -> (nr, nc^2)
    lGtheta = _matrix_logm(Gtheta).permute(0, 2, 1).reshape(nr, nc * nc).contiguous()

    # we now need Gx, Gy.
    dx = torch.diff(xcoord, dim=0).mean(dim=0)  # (nr,)
    dy = torch.diff(ycoord, dim=0).mean(dim=0)  # (nr,)
    dxy = torch.stack([dx, dy], dim=-1).to(lGtheta.dtype)  # (nr, 2)

    # solve: dxy (nr, 2) @ lG (2, nc^2) ≈ lGtheta (nr, nc^2)
    lG = lstsq(dxy, lGtheta, lamda)

    # extract components (C-order unflatten, consistent with original)
    lGx = lG[0].reshape(nc, nc)
    lGy = lG[1].reshape(nc, nc)

    # take matrix exponential to get from (lGx, lGy) -> (Gx, Gy)
    return torch.linalg.matrix_exp(lGy), torch.linalg.matrix_exp(lGx)


def _grappa_op_2d(calib, lamda):
    """Return a 2D GROG operators."""
    calib = calib.movedim(0, -1)
    _cx, _cy, nc = calib.shape[:]

    # we need sources (last source has no target!)
    Sy = calib[:, :-1, :].reshape(-1, nc)
    Sx = calib[:-1, ...].reshape(-1, nc)

    # and we need targets for an operator along each axis (first
    # target has no associated source!)
    Ty = calib[:, 1:, :].reshape(-1, nc)
    Tx = calib[1:, ...].reshape(-1, nc)

    # train the operators: S @ G.T ≈ T → lstsq gives G.T → .T gives G
    Gy = lstsq(Sy, Ty, lamda).mT
    Gx = lstsq(Sx, Tx, lamda).mT

    return Gy, Gx


def _grappa_op_3d(calib, lamda):
    """Return 3D GROG operator."""
    calib = calib.movedim(0, -1)
    _, _, _, nc = calib.shape[:]

    # we need sources (last source has no target!)
    Sz = calib[:-1, :, :, :].reshape(-1, nc)
    Sy = calib[:, :-1, :, :].reshape(-1, nc)
    Sx = calib[:, :, :-1, :].reshape(-1, nc)

    # and we need targets for an operator along each axis (first
    # target has no associated source!)
    Tz = calib[1:, :, :, :].reshape(-1, nc)
    Ty = calib[:, 1:, :, :].reshape(-1, nc)
    Tx = calib[:, :, 1:, :].reshape(-1, nc)

    # train the operators: S @ G.T ≈ T → lstsq gives G.T → .T gives G
    Gz = lstsq(Sz, Tz, lamda).mT
    Gy = lstsq(Sy, Ty, lamda).mT
    Gx = lstsq(Sx, Tx, lamda).mT

    return Gz, Gy, Gx
