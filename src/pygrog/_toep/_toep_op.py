"""Toeplitz Operator."""

__all__ = ["ToeplitzOp"]

import torch
from numpy.typing import ArrayLike

from .._utils import resize
from .._base._fftc import fft, ifft
from ._toep import calc_toeplitz_kernel


class ToeplitzOp:
    """
    Single coil Toeplitz Normal operator (A^H A).

    Implements resize -> FFT -> multiply PSF -> IFFT -> resize, all torch-native.

    Parameters
    ----------
    shape : list[int] | tuple[int]
        Input spatial shape.
    coords : ArrayLike
        Fourier domain coordinate array of shape ``(..., ndim)``.
    weights : ArrayLike | None, optional
        Density compensation weights.
    oversamp : float, optional
        Oversampling factor. The default is ``1.25``.
    eps : float, optional
        Desired numerical precision. The default is ``1e-3``.
    normalize_coords : bool, optional
        Normalize coordinates between -pi and pi. The default is ``True``.

    """

    def __init__(
        self,
        shape: list[int] | tuple[int],
        coords: ArrayLike,
        weights: ArrayLike | None = None,
        oversamp: float = 1.25,
        eps: float = 1e-3,
        normalize_coords: bool = True,
    ):
        self.shape = tuple(shape)
        ndim = coords.shape[-1]
        self.fft_axes = tuple(range(-1, -(ndim + 1), -1))

        # Generate PSF kernel
        psf = calc_toeplitz_kernel(
            coords, shape, weights, oversamp, eps, normalize_coords
        )
        self.psf = torch.as_tensor(psf)
        self.os_shape = tuple(self.psf.shape)

    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        """Apply Toeplitz normal operator: resize -> FFT -> PSF multiply -> IFFT -> resize."""
        os_shape = list(input.shape)
        for ax in self.fft_axes:
            os_shape[ax] = self.os_shape[ax]
        x = resize(input, os_shape)
        x = fft(x, axes=self.fft_axes, norm=None)
        psf = self.psf.to(x.device, x.dtype)
        x = x * psf
        x = ifft(x, axes=self.fft_axes, norm=None)
        out_shape = list(x.shape)
        for ax in self.fft_axes:
            out_shape[ax] = self.shape[ax]
        return resize(x, out_shape)
