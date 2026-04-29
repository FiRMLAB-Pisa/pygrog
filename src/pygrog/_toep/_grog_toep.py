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
        self.stack_shape = tuple(getattr(sparse_fft, "stack_shape", ()) or ())

        # Build PSF(s): for MaskedFFT use the pre-computed density grid
        # directly (no scatter needed); for SparseFFT scatter w²  onto the
        # flat grid via the C++ kernel.
        _is_masked = getattr(sparse_fft, "density", None) is not None

        if _is_masked:
            # density is already sum-of-squared-weights on the grid.
            density = sparse_fft.density.to(self.device)
            if not self.stack_shape:
                self.psf = density.clone()
            else:
                # density shape: (*stack_shape, *grid_shape)
                S_total = int(torch.tensor(self.stack_shape).prod().item())
                self.psf = density.reshape(*self.stack_shape, *self.grid_shape).clone()
        else:
            # SparseFFT path: scatter sqrt_w² onto the grid.
            sqrt_w_full = sparse_fft.sqrt_weights.to(self.device)
            indices_full = sparse_fft.indices.to(self.device)
            ext = _get_torch_ext()

            if not self.stack_shape:
                w_sq = (sqrt_w_full * sqrt_w_full).contiguous()
                psf_flat = torch.zeros(sparse_fft.grid_size,
                                       dtype=w_sq.dtype, device=self.device)
                ext.psf_scatter_scalar(psf_flat, indices_full, w_sq)
                self.psf = psf_flat.reshape(self.grid_shape)
            else:
                S_total = int(torch.tensor(self.stack_shape).prod().item())
                n_per = int(sparse_fft.n_samples)
                sqrt_w_v = sqrt_w_full.reshape(S_total, n_per)
                indices_v = indices_full.reshape(S_total, n_per)
                grid_off = (
                    torch.arange(S_total, device=self.device, dtype=indices_v.dtype)
                    * sparse_fft.grid_size
                ).unsqueeze(-1)
                indices_packed = (indices_v + grid_off).reshape(-1).contiguous()
                w_sq_packed = (sqrt_w_v * sqrt_w_v).reshape(-1).contiguous()
                psf_super = torch.zeros(
                    S_total * sparse_fft.grid_size,
                    dtype=w_sq_packed.dtype, device=self.device,
                )
                ext.psf_scatter_scalar(psf_super, indices_packed, w_sq_packed)
                self.psf = psf_super.reshape(*self.stack_shape, *self.grid_shape)
        self._psf_complex = self.psf.to(
            torch.complex64 if self.psf.dtype == torch.float32
            else torch.complex128
        )

    # ------------------------------------------------------------------
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Apply ``A^H A x``.

        Accepted layouts (with optional leading ``*B`` batch and, for
        stacked operators, leading ``*S`` stack axes):

        - ``(*B, *S, *image_shape)`` if smaps are set
        - ``(*B, *S, n_coils, *image_shape)`` otherwise

        Returns
        -------
        torch.Tensor
            Same shape as input.
        """
        sf = self.sparse_fft
        s_shape = self.stack_shape
        s_ndim = len(s_shape)
        single_ndim = (
            len(self.image_shape) if sf.smaps is not None
            else len(self.image_shape) + 1
        )
        prefix = tuple(int(s) for s in image.shape[: image.ndim - single_ndim])
        if s_ndim:
            if len(prefix) < s_ndim or tuple(prefix[-s_ndim:]) != s_shape:
                raise ValueError(
                    f"image prefix {prefix} must end with stack_shape {s_shape}"
                )
            B_shape = prefix[:-s_ndim]
        else:
            B_shape = prefix

        if not prefix:
            return self._apply_single(image, 0)

        S_total = int(torch.tensor(s_shape).prod().item()) if s_shape else 1
        B_total = int(torch.tensor(B_shape).prod().item()) if B_shape else 1
        single_shape = tuple(image.shape[image.ndim - single_ndim:])
        flat = image.reshape(B_total, S_total, *single_shape)
        outs = []
        for b in range(B_total):
            for s in range(S_total):
                outs.append(self._apply_single(flat[b, s], s))
        stacked = torch.stack(outs, dim=0)
        return stacked.reshape(*B_shape, *s_shape, *single_shape)

    def _apply_single(self, image: torch.Tensor, s_flat_idx: int = 0) -> torch.Tensor:
        """Single-frame application for one stack element."""
        sf = self.sparse_fft
        src_device = image.device
        comp_device = self.device
        dtype = image.dtype
        psf_full = self._psf_complex.to(comp_device, dtype=dtype)
        if self.stack_shape:
            psf_c = psf_full.reshape(-1, *self.grid_shape)[s_flat_idx]
        else:
            psf_c = psf_full
        image_d = image.to(comp_device)

        if sf.smaps is not None:
            smaps = sf.smaps.to(comp_device, dtype=dtype)
            conj_smaps = sf._conj_smaps.to(comp_device, dtype=dtype)
            coil_img = image_d.unsqueeze(0) * smaps
            full = self._apply_batched(coil_img, psf_c)
            return (full * conj_smaps).sum(0).to(src_device)

        # No smaps — coil dim is leading.
        full = self._apply_batched(image_d, psf_c)
        return full.to(src_device)

    # ------------------------------------------------------------------
    def _apply_batched(self, batch_img, psf_c):
        """Pad → batched FFT → PSF multiply → batched IFFT → crop.

        Parameters
        ----------
        batch_img : torch.Tensor
            ``(B, *image_shape)`` complex.
        psf_c : torch.Tensor
            ``(*grid_shape,)`` complex (broadcasts over B).

        Returns
        -------
        torch.Tensor
            ``(B, *image_shape)`` complex.
        """
        if self.grid_shape == self.image_shape:
            k = fft(batch_img, axes=self.fft_axes)
            k = k * psf_c
            return ifft(k, axes=self.fft_axes)
        B = batch_img.shape[0]
        padded = torch.zeros(
            B, *self.grid_shape,
            dtype=batch_img.dtype, device=batch_img.device,
        )
        padded[(slice(None), *self._pad_slices)] = batch_img
        k = fft(padded, axes=self.fft_axes)
        k = k * psf_c
        full = ifft(k, axes=self.fft_axes)
        return resize(full, (B, *self.image_shape))
