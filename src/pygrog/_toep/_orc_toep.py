"""GROG-Toeplitz normal operator for off-resonance-corrected SparseFFT.

For Cartesian gridding (which GROG produces) the normal operator
``A^H A`` of an off-resonance-corrected SparseFFT factorises into:

    img = sum_{l'} conj(C_{l'}) * IFFT( PSF[..., l', l] @ FFT( C_l * S_c * x ) )

per coil, where the PSF lives on `grid_shape` with shape
``(*grid_shape, L, L)`` and is built once via the ``psf_scatter_outer``
C++/CUDA kernel:

    PSF[idx, l', l] = sum_{n: idx_n = idx} w_n * conj(B[n, l']) * B[n, l]

Application uses ``torch.matmul`` for the per-grid-cell (L, L) @ (L,)
matvec, which dispatches BLAS / cuBLAS over the spatial batch.
"""

from __future__ import annotations

__all__ = ["OffResonanceToeplitzOp"]

import torch

from .._utils import resize
from .._base._fftc import fft, ifft
from ..operator._sparse_fft import _get_torch_ext


class OffResonanceToeplitzOp:
    """Self-adjoint operator ``A^H A`` for :class:`OffResonanceSparseFFT`.

    Parameters
    ----------
    orc_op : OffResonanceSparseFFT
        The off-resonance-corrected sparse FFT operator.
    device : str | torch.device | None
        Device to build the PSF on.  Defaults to base operator's device
        (or CPU).
    """

    def __init__(self, orc_op, device=None):
        base = orc_op._base
        self._base = base
        self._orc = orc_op

        if device is not None:
            self.device = torch.device(device)
        elif base.device is not None:
            self.device = torch.device(base.device)
        else:
            self.device = torch.device("cpu")

        self.grid_shape = base.grid_shape
        self.image_shape = base.image_shape
        self.fft_axes = base.fft_axes
        self._pad_slices = base._pad_slices
        self.L = orc_op.L

        # ---- Build PSF: (grid_size, L, L) complex --------------------------
        sort_perm = base.sort_perm.to(self.device)
        sorted_B = orc_op.B.to(self.device).contiguous()[sort_perm].contiguous()
        # Match dtype: PSF is complex64 by default (B stored as complex64).
        psf_dtype = sorted_B.dtype
        real_dtype = torch.float32 if psf_dtype == torch.complex64 else torch.float64

        sqrt_w = base.sqrt_weights.to(self.device, dtype=real_dtype)
        w_sq = (sqrt_w * sqrt_w).contiguous()
        indices = base.indices.to(self.device)

        psf_flat = torch.zeros(
            base.grid_size, self.L, self.L,
            dtype=psf_dtype, device=self.device,
        )
        ext = _get_torch_ext()
        ext.psf_scatter_outer(psf_flat, indices, w_sq, sorted_B)

        # Reshape to (*grid_shape, L, L) for spatial-batch matmul.
        self.psf = psf_flat.reshape(*self.grid_shape, self.L, self.L)
        # Cache C on device.
        self.C = orc_op.C.to(self.device)

    # ------------------------------------------------------------------
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Apply ``A^H A x``.

        Parameters
        ----------
        image : torch.Tensor
            ``(*image_shape,)`` (with smaps) or
            ``(n_coils, *image_shape)`` (no smaps) complex.
            Optional leading batch dim.

        Returns
        -------
        torch.Tensor
            Same shape as input.
        """
        base = self._base
        # Batch detection.
        single_frame_ndim = (
            len(self.image_shape) if base.smaps is not None
            else len(self.image_shape) + 1
        )
        if image.ndim == single_frame_ndim + 1:
            return torch.stack(
                [self(image[t]) for t in range(image.shape[0])], dim=0
            )

        src_device = image.device
        comp_device = self.device
        image_d = image.to(comp_device)
        dtype = image_d.dtype
        psf = self.psf.to(dtype)
        C = self.C.to(dtype)        # (L, *image_shape)

        if base.smaps is not None:
            smaps = base.smaps.to(comp_device, dtype=dtype)
            conj_smaps = base._conj_smaps.to(comp_device, dtype=dtype)
            n_coils = smaps.shape[0]
            accum = torch.zeros(self.image_shape, dtype=dtype, device=comp_device)
            for c in range(n_coils):
                coil_img = image_d * smaps[c]
                accum = accum + self._apply_one(coil_img, psf, C) \
                    * conj_smaps[c]
            return accum.to(src_device)

        # No smaps — coil dim is leading.
        n_coils = image_d.shape[0]
        out = torch.empty_like(image_d)
        for c in range(n_coils):
            out[c] = self._apply_one(image_d[c], psf, C)
        return out.to(src_device)

    # ------------------------------------------------------------------
    def _apply_one(self, coil_img, psf, C):
        """Per-coil core: pad → FFT → PSF matvec → IFFT → crop → C-combine.

        Parameters
        ----------
        coil_img : torch.Tensor
            ``(*image_shape,)`` complex.
        psf : torch.Tensor
            ``(*grid_shape, L, L)`` complex.
        C : torch.Tensor
            ``(L, *image_shape)`` complex.
        """
        L = self.L
        # Per-component weighting in image space: (L, *image_shape).
        weighted = C * coil_img.unsqueeze(0)

        # Pad (L, *image_shape) → (L, *grid_shape).
        if self.grid_shape == self.image_shape:
            padded = weighted
        else:
            padded = torch.zeros(L, *self.grid_shape,
                                 dtype=coil_img.dtype, device=coil_img.device)
            padded[(slice(None), *self._pad_slices)] = weighted

        # FFT over spatial axes → (L, *grid_shape).
        K = fft(padded, axes=self.fft_axes)

        # Per-grid-cell (L, L) @ (L,) matvec.
        # Move L to the last axis: (*grid_shape, L) and add column dim.
        K_perm = K.movedim(0, -1).unsqueeze(-1)          # (*G, L, 1)
        # psf @ K_perm: broadcasts over *grid_shape batch.
        out_perm = torch.matmul(psf, K_perm).squeeze(-1) # (*G, L)
        out = out_perm.movedim(-1, 0)                    # (L, *G)

        # IFFT and crop.
        spatial = ifft(out, axes=self.fft_axes)
        if self.grid_shape != self.image_shape:
            spatial = resize(spatial, (L, *self.image_shape))

        # Combine with conj(C): (L, *image_shape) → (*image_shape).
        return (C.conj() * spatial).sum(0)
