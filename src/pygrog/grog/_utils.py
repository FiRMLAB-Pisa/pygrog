"""Coordinates scaling helper."""

__all__ = ["rescale_coords", "prepare_grog_table", "grog_power"]

import warnings

import numpy as np

from scipy.linalg import fractional_matrix_power as fmp
from numpy.typing import NDArray

from mrinufft._array_compat import with_numpy_cupy

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from sigpy import get_device


@with_numpy_cupy
def rescale_coords(coords: NDArray, amp: float | NDArray) -> NDArray:
    """
    Rescale Fourier domain coordinates to desired amplitude.

    Parameters
    ----------
    coords : NDArray
        Fourier domain coordinates array of shape ``(..., ndims)``.
    amp : float | NDArray
        Output scale. This represent the full dynamic range ``2 * kmax``,
        i.e., output coordinates will be scaled between ``(-0.5 * amp, 0.5 * amp)``.
        If array, must have ``ndim`` elements.

    Returns
    -------
    NDArray
        Scaled domain coordinate array of shape ``(..., ndim)``.

    """
    device = get_device(coords)
    cmax = abs(coords).reshape(-1, coords.shape[-1]).max(axis=0)
    if np.isscalar(amp):
        amp = coords.shape[-1] * [amp]
    with device:
        return 0.5 * device.xp.asarray(amp, dtype=coords.dtype) * coords / cmax
    
def prepare_grog_table(Dx: NDArray, Dy: NDArray, Dz: NDArray | None, nsteps: int, ndim: int) -> NDArray:
    """
    Prepare the GROG operator table.
    
    Parameters
    ----------
    Dx : NDArray
        GRAPPA kernel for shifting alonx ``x``, shaped ``(nsteps, ncoils, ncoils)``.
    Dy : NDArray
        GRAPPA kernel for shifting alonx ``y``, shaped ``(nsteps, ncoils, ncoils)``.
    Dz : NDArray
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
    # Convert to numpy arrays
    Dx = np.asarray(Dx)
    Dy = np.asarray(Dy)
    
    if ndim == 2:
        # 2D case
        Dx = Dx[None, :, ...]  # (1, nsteps, nc, nc)
        Dy = Dy[:, None, ...]  # (nsteps, 1, nc, nc)
        Dx = np.repeat(Dx, nsteps, axis=0)  # (nsteps, nsteps, nc, nc)
        Dy = np.repeat(Dy, nsteps, axis=1)  # (nsteps, nsteps, nc, nc)
        Dx = Dx.reshape(-1, *Dx.shape[-2:])  # (nsteps**2, nc, nc)
        Dy = Dy.reshape(-1, *Dy.shape[-2:])  # (nsteps**2, nc, nc)
        grog_table = Dx @ Dy  # (nsteps**2, nc, nc)
        
    elif ndim == 3:
        # 3D case
        if Dz is None:
            raise ValueError("3D interpolation requires Z operator")
        
        Dz = np.asarray(Dz)
        Dx = Dx[None, None, :, ...]  # (1, 1, nsteps, nc, nc)
        Dy = Dy[None, :, None, ...]  # (1, nsteps, 1, nc, nc)
        Dz = Dz[:, None, None, ...]  # (nsteps, 1, 1, nc, nc)
        
        # Repeat to create a grid of all combinations
        Dx = np.repeat(Dx, nsteps, axis=0)  # (nsteps, 1, nsteps, nc, nc)
        Dx = np.repeat(Dx, nsteps, axis=1)  # (nsteps, nsteps, nsteps, nc, nc)
        Dy = np.repeat(Dy, nsteps, axis=0)  # (nsteps, nsteps, 1, nc, nc)
        Dy = np.repeat(Dy, nsteps, axis=2)  # (nsteps, nsteps, nsteps, nc, nc)
        Dz = np.repeat(Dz, nsteps, axis=1)  # (nsteps, nsteps, 1, nc, nc)
        Dz = np.repeat(Dz, nsteps, axis=2)  # (nsteps, nsteps, nsteps, nc, nc)
        
        # Reshape to flat combinations
        Dx = Dx.reshape(-1, *Dx.shape[-2:])  # (nsteps**3, nc, nc)
        Dy = Dy.reshape(-1, *Dy.shape[-2:])  # (nsteps**3, nc, nc)
        Dz = Dz.reshape(-1, *Dz.shape[-2:])  # (nsteps**3, nc, nc)
        
        # Combine all operators
        grog_table = Dx @ Dy @ Dz  # (nsteps**3, nc, nc)
    
    else:
        raise ValueError(f"GROG interpolation only supports 2D or 3D data, got {ndim}D")
        
    return grog_table


def grog_power(D_unit: NDArray, exponents: NDArray) -> NDArray:
    """
    Compute matrix powers of GROG operators.
    
    Parameters
    ----------
    D_unit : NDArray
        GRAPPA kernel for unit shifts.
    exponents : NDArray
        List of exponents to obtain fractional shifts.
        
    Returns
    -------
    NDArray
        GRAPPA operators for fractional shifts.
    
    """
    D_frac, idx = [], 0
    for exp in exponents:
        if np.isclose(exp, 0.0):
            _D = np.eye(D_unit.shape[0], dtype=D_unit.dtype)
        else:
            _D = fmp(D_unit, np.abs(exp)).astype(D_unit.dtype)
            if np.sign(exp) < 0:
                _D = np.linalg.pinv(_D).astype(D_unit.dtype)
        D_frac.append(_D)
        idx += 1

    return np.stack(D_frac, axis=0)
