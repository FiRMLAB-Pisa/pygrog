"""GROG-Toeplitz normal operator for SubspaceSparseFFT.

For Cartesian gridding, the normal operator ``A^H A`` of a subspace-decorated
SparseFFT factorises into a per-grid-cell (K, K) matvec on the FFT of the
zero-padded SENSE-weighted coefficient images.

PSF construction (built once via the C++/CUDA ``psf_scatter_outer_basis``
kernel):

    PSF[idx, k', k] = sum_{n: idx_n = idx}
                          w_eff_n * conj(B[k', t_n]) * B[k, t_n]

where ``w_eff_n = w_n**2`` and ``w_n = sqrt_w_n**2`` is the original GROG
weight (symmetric ``pre_w * sqrt_w`` factor on each side gives ``w_n`` per
direction → ``w_n**2`` for the normal).
"""

from __future__ import annotations

__all__ = ["SubspaceToeplitzOp"]

import numpy as np
import torch

from .._utils import resize
from .._base._fftc import fft, ifft
from ..operator._sparse_fft import _get_torch_ext


class SubspaceToeplitzOp:
    """Self-adjoint operator ``A^H A`` for :class:`SubspaceSparseFFT`.

    Parameters
    ----------
    sub_op : SubspaceSparseFFT
        The subspace-decorated sparse FFT operator (must have smaps).
    device : str | torch.device | None
        Device to build the PSF on.  Defaults to base operator's device.
    """

    def __init__(self, sub_op, device=None):
        base = sub_op._base
        if base.smaps is None:
            raise NotImplementedError(
                "SubspaceToeplitzOp requires base_op.smaps"
            )
        self._base = base
        self._sub = sub_op

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
        self.K = sub_op.K
        self.T = sub_op.T

        # ---- Build per-sample time index in sorted order ------------------
        nat = tuple(int(s) for s in base.natural_shape)
        N = base.n_samples
        # Unravel natural-flat indices [0, N) → multi-dim natural coords, take T axis.
        raw_natural_indices = np.arange(N, dtype=np.int64)
        unraveled = np.unravel_index(raw_natural_indices, nat)
        time_index_natural = torch.from_numpy(
            np.asarray(unraveled[sub_op._t_axis_in_nat], dtype=np.int64)
        )
        sort_perm = base.sort_perm.to("cpu")
        sorted_time_index = time_index_natural[sort_perm].to(self.device).contiguous()

        # ---- Build PSF: (grid_size, K, K) complex -------------------------
        basis = sub_op.basis.to(self.device).contiguous()  # (K, T)
        psf_dtype = basis.dtype
        real_dtype = (torch.float32 if psf_dtype == torch.complex64
                      else torch.float64)

        sqrt_w = base.sqrt_weights.to(self.device, dtype=real_dtype)
        # Effective per-sample weight w_eff = w_n**2 = sqrt_w_n**4
        # (symmetric pre_w * scatter sqrt_w → w_n per direction).
        w_n = sqrt_w * sqrt_w
        w_eff = (w_n * w_n).contiguous()
        indices = base.indices.to(self.device)

        psf_flat = torch.zeros(
            base.grid_size, self.K, self.K,
            dtype=psf_dtype, device=self.device,
        )
        ext = _get_torch_ext()
        ext.psf_scatter_outer_basis(
            psf_flat, indices, w_eff, basis, sorted_time_index,
        )
        self.psf = psf_flat.reshape(*self.grid_shape, self.K, self.K)

    # ------------------------------------------------------------------
    def __call__(self, coeffs: torch.Tensor) -> torch.Tensor:
        """Apply ``A^H A coeffs``.

        Parameters
        ----------
        coeffs : torch.Tensor
            ``(K, *image_shape)`` complex.

        Returns
        -------
        torch.Tensor
            Same shape as input.
        """
        if coeffs.shape[0] != self.K:
            raise ValueError(
                f"coeffs.shape[0]={coeffs.shape[0]} != K={self.K}"
            )

        base = self._base
        src_device = coeffs.device
        comp_device = self.device
        coeffs_d = coeffs.to(comp_device)
        dtype = coeffs_d.dtype
        psf = self.psf.to(dtype)

        smaps = base.smaps.to(comp_device, dtype=dtype)
        conj_smaps = base._conj_smaps.to(comp_device, dtype=dtype)
        n_coils = smaps.shape[0]

        accum = torch.zeros(
            self.K, *self.image_shape, dtype=dtype, device=comp_device,
        )
        for c in range(n_coils):
            coil = coeffs_d * smaps[c].unsqueeze(0)  # (K, *image)
            out = self._apply_one(coil, psf)         # (K, *image)
            accum.addcmul_(out, conj_smaps[c].unsqueeze(0))

        return accum.to(src_device)

    # ------------------------------------------------------------------
    def _apply_one(self, coil_coeffs, psf):
        """Per-coil core: pad → FFT → PSF matvec → IFFT → crop.

        Parameters
        ----------
        coil_coeffs : torch.Tensor
            ``(K, *image_shape)`` complex.
        psf : torch.Tensor
            ``(*grid_shape, K, K)`` complex.

        Returns
        -------
        torch.Tensor
            ``(K, *image_shape)`` complex.
        """
        K = self.K
        # Pad (K, *image_shape) → (K, *grid_shape).
        if self.grid_shape == self.image_shape:
            padded = coil_coeffs
        else:
            padded = torch.zeros(
                K, *self.grid_shape,
                dtype=coil_coeffs.dtype, device=coil_coeffs.device,
            )
            padded[(slice(None), *self._pad_slices)] = coil_coeffs

        # FFT over spatial axes → (K, *grid_shape).
        Kk = fft(padded, axes=self.fft_axes)

        # Per-grid-cell (K, K) @ (K,) matvec.
        # The C++ `psf_scatter_outer_basis` kernel produces
        #   PSF_C[j, a, b] = sum_n w_n * conj(B[a, t_n]) * B[b, t_n],
        # but the SubspaceSparseFFT scatter-side basis is *un-conjugated*
        #   PSF_match[j, k', k] = sum_n w_n^2 * B[k', t_n] * conj(B[k, t_n]),
        # so PSF_match = PSF_C.transpose(-2, -1).  We apply the transpose at
        # matvec time (cheap view) instead of materialising a transposed copy.
        Kk_perm = Kk.movedim(0, -1).unsqueeze(-1)         # (*G, K, 1)
        psf_T = psf.transpose(-2, -1)                     # (*G, K, K)
        out_perm = torch.matmul(psf_T, Kk_perm).squeeze(-1)  # (*G, K)
        out = out_perm.movedim(-1, 0)                     # (K, *G)

        spatial = ifft(out, axes=self.fft_axes)
        if self.grid_shape != self.image_shape:
            spatial = resize(spatial, (K, *self.image_shape))
        return spatial
