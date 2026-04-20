"""Toeplitz kernel calculators."""

__all__ = ["calc_toeplitz_kernel"]

import torch
from numpy.typing import ArrayLike
from mrinufft._array_compat import with_torch

from .._base._fftc import fft
from .._base._nufft import nufft, nufft_adjoint


@with_torch
def calc_toeplitz_kernel(
    coords: ArrayLike,
    shape: ArrayLike,
    weights: ArrayLike = None,
    oversamp: float = 1.25,
    eps: float = 1e-6,
    normalize_coords: bool = True,
):
    """
    Toeplitz PSF for fast Normal non-uniform Fast Fourier Transform.

    While fast, this is more memory intensive.

    Parameters
    ----------
    coord : ArrayLike
        Fourier domain coordinate array of shape ``(..., ndim)``.
        ``ndim`` determines the number of dimensions to apply the NUFFT.
    shape : ArrayLike[int] | None, optional
        Shape of the form ``(..., n_{ndim - 1}, ..., n_1, n_0)``.
        The default is ``None`` (estimated from ``coord``).
    oversamp : float, optional
        Oversampling factor. The default is ``1.25``.
    eps : float, optional
        Desired numerical precision. The default is ``1e-6``.
    normalize_coords : bool, optional
        Normalize coordinates between -pi and pi. If ``False``,
        assume they are correctly normalized already. The default
        is ``True``.

    Returns
    -------
    ArrayLike
        Signal domain data of shape
        ``input.shape[:-ndim] + coord.shape[:-1]``.

    """
    ndim = coords.shape[-1]
    shape_list = list(shape[-ndim:])

    # Determine per-axis oversampling: 1 for Cartesian, 2 for non-uniform
    _coords = torch.stack(
        [
            0.5 * coords[..., n] / torch.max(torch.abs(coords[..., n]))
            for n in range(ndim)
        ],
        dim=-1,
    )
    _scale = torch.tensor(shape_list, dtype=_coords.dtype, device=_coords.device)
    _coords = _coords * _scale
    _coords = torch.stack(
        [_coords[..., n] - torch.min(_coords[..., n]) for n in range(ndim)], dim=-1
    )
    osf = [
        1 if torch.allclose(_coords[..., n].round(), _coords[..., n], atol=1e-4) else 2
        for n in range(ndim)
    ]
    os_shape = [osf[n] * shape_list[n] for n in range(ndim)]

    # Delta image at center of oversampled grid
    idx = tuple(s // 2 for s in os_shape)
    d = torch.zeros(os_shape, dtype=torch.complex64, device=coords.device)
    d[idx] = 1.0

    # Default DCF: uniform weights
    if weights is None:
        weights = torch.ones_like(coords[..., 0])

    # Get Point Spread Function via forward + adjoint NUFFT
    psf = nufft(d, coords, oversamp, eps, normalize_coords)
    psf = nufft_adjoint(
        weights * psf, coords, os_shape, oversamp, eps, normalize_coords
    )

    # Kernel is the FFT of the PSF
    fft_axes = tuple(range(-1, -(ndim + 1), -1))
    psf = fft(psf, axes=fft_axes, norm=None) * (2**ndim)

    return psf
