"""GRAPPA kernel table calculation subroutines."""

__all__ = ["KernelTable"]

import warnings

from numpy.typing import NDArray

import numpy as np

from scipy.linalg import expm, logm
from scipy.linalg import fractional_matrix_power as fmp

from mrinufft._array_compat import with_numpy

from .._linalg import lstsq

def KernelTable(
        train_data: NDArray[complex], 
        radius: float, 
        precision: int = 1, 
        lamda: float = 0.01
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
    lamda : float
        L2 regularization for GRAPPA kernel estimation.
        
    Returns
    -------
    NDArray
        GRAPPA kernel table for fractional shifts ``(dz, dy, dx)`` of shape
        ``(nsteps**ndim, coils, coils)``.
    int
        Number of fractional steps.
    int
        Number of spatial dimensions.
        
    """
    grappa_kernels = train_grappa(train_data, lamda)
    
    # Calculate displacements steps
    nsteps = 2 * radius / 10 ** (-precision) + 1
    nsteps = int(nsteps)
    deltas = (np.arange(nsteps) - (nsteps - 1) // 2) / (nsteps - 1)
    
    # Pre-compute partial operators
    Gx = grappa_power(grappa_kernels["x"], deltas)  # (nsteps, nc, nc)
    Gy = grappa_power(grappa_kernels["y"], deltas)  # (nsteps, nc, nc)
    if "z" in grappa_kernels and grappa_kernels["z"] is not None:
        Gz = grappa_power(grappa_kernels["z"], deltas)  # (nsteps, nc, nc), 3D only
        ndim = 3
    else:
        Gz = None
        ndim = 2
        
    return prepare_grappa_table(Gx, Gy, Gz, nsteps, ndim), nsteps, ndim

def prepare_grappa_table(
        Gx: NDArray[complex], 
        Gy: NDArray[complex], 
        Gz: NDArray[complex] | None, 
        nsteps: int, ndim: int
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
        GRAPPA kernel table for fractional shifts ``(dz, dy, dx)`` of shape
        ``(nsteps**ndim, ncoils, ncoils)``.
        
    """
    # Convert to numpy arrays
    Gx = np.asarray(Gx)
    Gy = np.asarray(Gy)
    
    if ndim == 2:
        # 2D case
        Gx = Gx[None, :, ...]  # (1, nsteps, nc, nc)
        Gy = Gy[:, None, ...]  # (nsteps, 1, nc, nc)
        Gx = np.repeat(Gx, nsteps, axis=0)  # (nsteps, nsteps, nc, nc)
        Gy = np.repeat(Gy, nsteps, axis=1)  # (nsteps, nsteps, nc, nc)
        Gx = Gx.reshape(-1, *Gx.shape[-2:])  # (nsteps**2, nc, nc)
        Gy = Gy.reshape(-1, *Gy.shape[-2:])  # (nsteps**2, nc, nc)
        grappa_table = Gx @ Gy  # (nsteps**2, nc, nc)
        
    elif ndim == 3:
        # 3D case
        if Gz is None:
            raise ValueError("3D interpolation requires Z operator")
        
        Gz = np.asarray(Gz)
        Gx = Gx[None, None, :, ...]  # (1, 1, nsteps, nc, nc)
        Gy = Gy[None, :, None, ...]  # (1, nsteps, 1, nc, nc)
        Gz = Gz[:, None, None, ...]  # (nsteps, 1, 1, nc, nc)
        
        # Repeat to create a grid of all combinations
        Gx = np.repeat(Gx, nsteps, axis=0)  # (nsteps, 1, nsteps, nc, nc)
        Gx = np.repeat(Gx, nsteps, axis=1)  # (nsteps, nsteps, nsteps, nc, nc)
        Gy = np.repeat(Gy, nsteps, axis=0)  # (nsteps, nsteps, 1, nc, nc)
        Gy = np.repeat(Gy, nsteps, axis=2)  # (nsteps, nsteps, nsteps, nc, nc)
        Gz = np.repeat(Gz, nsteps, axis=1)  # (nsteps, nsteps, 1, nc, nc)
        Gz = np.repeat(Gz, nsteps, axis=2)  # (nsteps, nsteps, nsteps, nc, nc)
        
        # Reshape to flat combinations
        Gz = Gz.reshape(-1, *Gz.shape[-2:])  # (nsteps**3, nc, nc)
        Gy = Gy.reshape(-1, *Gy.shape[-2:])  # (nsteps**3, nc, nc)
        Gx = Gx.reshape(-1, *Gx.shape[-2:])  # (nsteps**3, nc, nc)
        
        # Combine all operators
        grappa_table = Gx @ Gy @ Gz  # (nsteps**3, nc, nc)
    
    else:
        raise ValueError(f"GROG interpolation only supports 2D or 3D data, got {ndim}D")
        
    return grappa_table


def grappa_power(G_unit: NDArray[complex], exponents: NDArray[float]) -> NDArray[complex]:
    """
    Compute matrix powers of GRAPPA operators.
    
    Parameters
    ----------
    G_unit : NDArray
        GRAPPA kernel for unit shifts.
    exponents : NDArray
        List of exponents to obtain fractional shifts.
        
    Returns
    -------
    NDArray
        GRAPPA operators for fractional shifts.
    
    """
    G_frac, idx = [], 0
    for exp in exponents:
        if np.isclose(exp, 0.0):
            _G = np.eye(G_unit.shape[0], dtype=G_unit.dtype)
        else:
            _G = fmp(G_unit, np.abs(exp)).astype(G_unit.dtype)
            if np.sign(exp) < 0:
                _G = np.linalg.pinv(_G).astype(G_unit.dtype)
        G_frac.append(_G)
        idx += 1

    return np.stack(G_frac, axis=0)


def train_grappa(
    train_data: NDArray[complex],
    lamda: float | None = None,
    coords: NDArray[float] | None = None,
) -> dict:
    """
    Train GRAPPA Operator Gridding (GROG) interpolator.

    Parameters
    ----------
    train_data : NDArray[complex]
        Calibration region data of shape ``(nc, nz, ny, nx)`` or ``(nc, ny, nx)``.
        Usually a small portion from the center of kspace.
    lamda : float | None, optional
        Tikhonov regularization parameter.  Set to 0 for no
        regularization. Defaults to ``0.01`` for standard GROG.
        and ``0.0`` for self-calibrating GROG.
    coords : NDArray[float], optional
        Fourier domain coordinate array of shape ``(..., ndim)``.
        ``ndim`` determines the number of dimensions to apply the NUFFT
        (``None`` for Cartesian).

    Returns
    -------
    dict
        Output grog interpolator with keys ``(y, x)`` (single slice 2D)
        or ``(z, y, x)`` (multi-slice or 3D).

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
@with_numpy
def _train(train_data, lamda, coords):
    if coords is None:
        ndim = len(train_data.shape) - 1
    else:
        ndim = coords.shape[-1]

    # get grappa operator
    kern = _calc_grappaop(ndim, train_data, lamda, coords)
    Gx = kern["Gx"]
    Gy = kern["Gy"]
    if ndim == 3:
        Gz = kern["Gz"]
    else:
        Gz = None

    return {"x": Gx, "y": Gy, "z": Gz}


def _calc_grappaop(ndim, train_data, lamda, coords):
    train_data = train_data / np.linalg.norm(train_data)
    if coords is not None:
        gz = None
        gy, gx = _radial_grappa_op(train_data, lamda, coords)
    else:
        if ndim == 2:
            gz = None
            gy, gx = _grappa_op_2d(train_data, lamda)
        elif ndim == 3:
            gz, gy, gx = _grappa_op_3d(train_data, lamda)

    return {"Gx": gx, "Gy": gy, "Gz": gz}


def _radial_grappa_op(calib, lamda, coords):
    """Return a 2D GROG operators from radial data."""
    calib = np.moveaxis(calib, 0, -1)
    nr, ns, nc = calib.shape

    # extract x and y components of trajectory
    xcoord = coords[..., 0].T
    ycoord = coords[..., 1].T

    # we need sources (last source has no target!)
    S = calib[:, :-1, ...]

    # and targets (first target has no associated source!)
    T = calib[:, 1:, ...]

    # train the operator
    Gtheta = lstsq(S, T, lamda)
    lGtheta = np.stack([logm(G) for G in Gtheta])
    lGtheta = np.reshape(lGtheta, (nr, nc**2), "F")

    # we now need Gx, Gy.
    dx = np.mean(np.diff(xcoord, axis=0), axis=0)
    dy = np.mean(np.diff(ycoord, axis=0), axis=0)
    dxy = np.concatenate((dx[:, None], dy[:, None]), axis=1)
    dxy = dxy.astype(lGtheta.dtype)

    # solve
    lG = lstsq(dxy, lGtheta.T, lamda).T

    # extract components
    lGx = np.reshape(lG[0, :], (nc, nc))
    lGy = np.reshape(lG[1, :], (nc, nc))

    # take matrix exponential to get from (lGx, lGy) -> (Gx, Gy)
    return expm(lGy), expm(lGx)


def _grappa_op_2d(calib, lamda):
    """Return a 2D GROG operators."""
    calib = np.moveaxis(calib, 0, -1)
    _, _, nc = calib.shape[:]

    # we need sources (last source has no target!)
    Sx = np.reshape(calib[:, :-1, :], (-1, nc))
    Sy = np.reshape(calib[:-1, ...], (-1, nc))

    # and we need targets for an operator along each axis (first
    # target has no associated source!)
    Tx = np.reshape(calib[:, 1:, :], (-1, nc))
    Ty = np.reshape(calib[1:, ...], (-1, nc))

    # train the operators:
    Gx = lstsq(Sx, Tx.T, lamda).T
    Gy = lstsq(Sy, Ty.T, lamda).T

    return Gy, Gx


def _grappa_op_3d(calib, lamda):
    """Return 3D GROG operator."""
    calib = np.moveaxis(calib, 0, -1)
    _, _, _, nc = calib.shape[:]

    # we need sources (last source has no target!)
    Sx = np.reshape(calib[:-1, :, :, :], (-1, nc))
    Sy = np.reshape(calib[:, :-1, :, :], (-1, nc))
    Sz = np.reshape(calib[:, :, :-1, :], (-1, nc))

    # and we need targets for an operator along each axis (first
    # target has no associated source!)
    Tx = np.reshape(calib[1:, :, :, :], (-1, nc))
    Ty = np.reshape(calib[:, 1:, :, :], (-1, nc))
    Tz = np.reshape(calib[:, :, 1:, :], (-1, nc))

    # train the operators:
    Gx = lstsq(Sx, Tx.T, lamda).T
    Gy = lstsq(Sy, Ty.T, lamda).T
    Gz = lstsq(Sz, Tz.T, lamda).T

    return Gz.T, Gy.T, Gx.T

