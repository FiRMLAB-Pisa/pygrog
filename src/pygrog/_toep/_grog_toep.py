"""GROG-Toeplitz normal operator for plain SparseFFT.

For Cartesian gridding (which GROG produces) the normal operator
``A^H A`` reduces to:

    sum_c S_c^* * IFFT( PSF * FFT( pad( S_c * x ) ) ) (cropped)

where the PSF is a real grid of weights:

    PSF[j] = sum_i w_i * delta(j == idx_i)

built once via the ``psf_scatter_scalar`` C++/CUDA kernel.

No 2x oversampling is needed — pygrog's gridding is exact, so the PSF
lives on ``plan.grid_shape``.  When ``grid_shape == image_shape`` the
pad/crop are no-ops.
"""

from __future__ import annotations

__all__ = ["GrogToeplitzOp"]

import torch

from .._utils import resize
from .._base._fftc import fft, ifft
from ..operator._sparse_fft import _get_torch_ext


class GrogToeplitzOp:
    """Self-adjoint operator ``A^H A`` for :class:`SparseFFT`.

    Parameters
    ----------
    sparse_fft : SparseFFT
        The forward/adjoint operator whose normal we precompute.
        Plan metadata (indices, sqrt_weights, grid/image shape, smaps)
        is read directly; no copy is taken.
    device : str | torch.device | None
        Device to build the PSF on.  Defaults to ``sparse_fft.device``
        if set, otherwise CPU.
    """

    def __init__(self, sparse_fft, device=None):
        self.sparse_fft = sparse_fft

        if device is not None:
            self.device = torch.device(device)
        elif sparse_fft.device is not None:
            self.device = torch.device(sparse_fft.device)
        else:
            self.device = torch.device("cpu")

        self.grid_shape = sparse_fft.grid_shape
        self.image_shape = sparse_fft.image_shape
        self.fft_axes = sparse_fft.fft_axes
        self._pad_slices = sparse_fft._pad_slices

        # Build PSF once: scatter w (= sqrt_w**2) onto the flat grid.
        sqrt_w = sparse_fft.sqrt_weights.to(self.device)
        indices = sparse_fft.indices.to(self.device)
        w_sq = (sqrt_w * sqrt_w).contiguous()
        psf_flat = torch.zeros(sparse_fft.grid_size,
                               dtype=w_sq.dtype, device=self.device)
        ext = _get_torch_ext()
        ext.psf_scatter_scalar(psf_flat, indices, w_sq)
        # Reshape to grid_shape so it broadcasts against (..., *grid_shape).
        self.psf = psf_flat.reshape(self.grid_shape)

    # ------------------------------------------------------------------
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Apply ``A^H A x``.

        Parameters
        ----------
        image : torch.Tensor
            ``(*image_shape,)`` (with smaps) or ``(n_coils, *image_shape)``
            (no smaps) complex.  Optional leading batch dim ``T``.

        Returns
        -------
        torch.Tensor
            Same shape as input.
        """
        sf = self.sparse_fft
        # Batch detection (mirror of SparseFFT._adjoint_flat).
        single_frame_ndim = (
            len(self.image_shape) if sf.smaps is not None
            else len(self.image_shape) + 1
        )
        if image.ndim == single_frame_ndim + 1:
            return torch.stack(
                [self(image[t]) for t in range(image.shape[0])], dim=0
            )

        src_device = image.device
        comp_device = self.device
        dtype = image.dtype
        psf = self.psf.to(comp_device, dtype=torch.promote_types(
            self.psf.dtype, torch.float32))
        # PSF is real; we'll multiply by it in complex space via broadcast.
        psf_c = psf.to(_real_to_complex_dtype(dtype))

        if sf.smaps is not None:
            smaps = sf.smaps.to(comp_device, dtype=dtype)
            conj_smaps = sf._conj_smaps.to(comp_device, dtype=dtype)
            n_coils = smaps.shape[0]
            image_d = image.to(comp_device)
            accum = torch.zeros(self.image_shape, dtype=dtype, device=comp_device)
            padded = torch.empty(self.grid_shape, dtype=dtype, device=comp_device)
            for c in range(n_coils):
                coil_img = image_d * smaps[c]
                out_c = self._apply_one(coil_img, padded, psf_c)
                accum.addcmul_(out_c, conj_smaps[c])
            return accum.to(src_device)

        # No smaps — coil dim is leading.
        n_coils = image.shape[0]
        image_d = image.to(comp_device)
        out = torch.empty_like(image_d)
        padded = torch.empty(self.grid_shape, dtype=dtype, device=comp_device)
        for c in range(n_coils):
            out[c] = self._apply_one(image_d[c], padded, psf_c)
        return out.to(src_device)

    # ------------------------------------------------------------------
    def _apply_one(self, coil_img, padded, psf_c):
        """Pad -> FFT -> PSF multiply -> IFFT -> crop for one coil image."""
        if self.grid_shape == self.image_shape:
            k = fft(coil_img, axes=self.fft_axes)
            k = k * psf_c
            return ifft(k, axes=self.fft_axes)
        padded.zero_()
        padded[self._pad_slices] = coil_img
        k = fft(padded, axes=self.fft_axes)
        k = k * psf_c
        full_img = ifft(k, axes=self.fft_axes)
        return resize(full_img, self.image_shape)


def _real_to_complex_dtype(complex_dtype: torch.dtype) -> torch.dtype:
    if complex_dtype in (torch.complex64, torch.complex32):
        return torch.complex64
    return torch.complex128
