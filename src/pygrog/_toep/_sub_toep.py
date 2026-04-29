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
            raise NotImplementedError("SubspaceToeplitzOp requires base_op.smaps")
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
        self.stack_shape = tuple(getattr(base, "stack_shape", ()) or ())

        # ---- Build PSF: (*S, grid_size, K, K) complex ---------------------
        _is_masked = getattr(base, "density", None) is not None

        if _is_masked:
            # MaskedFFT path: each grid cell j has a unique time index t_j
            # (the _t_axis_in_nat-th axis of grid_shape).
            # PSF[j, k', k] = density[j] * conj(basis[k', t_j]) * basis[k, t_j]
            basis = sub_op.basis.to(self.device).contiguous()  # (K, T)
            psf_dtype = basis.dtype
            density = base.density.to(
                self.device
            )  # (*S, *grid_shape) or (*grid_shape,)

            # Build per-grid-cell time index: shape (*grid_shape,) int64
            # The t-axis position within grid_shape (== natural_shape for MaskedFFT).
            t_axis = sub_op._t_axis_in_nat
            t_size = int(base.grid_shape[t_axis])
            # Create a broadcast-able time index grid
            t_idx = torch.arange(t_size, device=self.device, dtype=torch.int64)
            expand_shape = [1] * len(base.grid_shape)
            expand_shape[t_axis] = t_size
            t_idx = t_idx.reshape(expand_shape).expand(base.grid_shape)
            # t_idx: (*grid_shape,), int64

            # Gather basis vectors at each grid cell's time index
            # basis_at_j[j] = basis[:, t_idx[j]] — shape (*grid_shape, K, K) for PSF
            # We want: PSF[..., k', k] = density * conj(basis[k', t]) * basis[k, t]
            # Reshape t_idx for gather: (*grid_shape,) → needed for indexing basis
            t_flat = t_idx.reshape(-1)  # (G,)
            # basis: (K, T) → per-cell: (G, K)
            basis_at_j = basis[:, t_flat].T.contiguous()  # (G, K)
            # PSF_flat[j, k', k] = conj(basis_at_j[j, k']) * basis_at_j[j, k]
            psf_flat_nodens = basis_at_j.unsqueeze(1).conj() * basis_at_j.unsqueeze(
                2
            )  # (G, K, K)

            if not self.stack_shape:
                dens_flat = density.reshape(-1)  # (G,)
                psf_flat = (
                    dens_flat.to(dtype=psf_dtype).unsqueeze(-1).unsqueeze(-1)
                    * psf_flat_nodens
                )
                # Transpose to match the SparseFFT PSF layout (see transpose at end of SparseFFT path)
                self.psf = (
                    psf_flat.transpose(-2, -1)
                    .contiguous()
                    .reshape(*self.grid_shape, self.K, self.K)
                )
            else:
                S_total = int(np.prod(self.stack_shape))
                dens_v = density.reshape(S_total, -1)  # (S, G)
                psf_v = dens_v.to(dtype=psf_dtype).unsqueeze(-1).unsqueeze(
                    -1
                ) * psf_flat_nodens.unsqueeze(0)
                self.psf = (
                    psf_v.transpose(-2, -1)
                    .contiguous()
                    .reshape(*self.stack_shape, *self.grid_shape, self.K, self.K)
                )
        else:
            # SparseFFT path: build per-sample time index in sorted order
            nat = tuple(int(s) for s in base.natural_shape)
            N = base.n_samples
            raw_natural_indices = np.arange(N, dtype=np.int64)
            unraveled = np.unravel_index(raw_natural_indices, nat)
            time_index_natural = torch.from_numpy(
                np.asarray(unraveled[sub_op._t_axis_in_nat], dtype=np.int64)
            )
            sort_perm_full = base.sort_perm.to("cpu")
            sorted_time_index_full = (
                time_index_natural[sort_perm_full].to(self.device).contiguous()
            )

            basis = sub_op.basis.to(self.device).contiguous()  # (K, T)
            psf_dtype = basis.dtype
            real_dtype = (
                torch.float32 if psf_dtype == torch.complex64 else torch.float64
            )

            sqrt_w_full = base.sqrt_weights.to(self.device, dtype=real_dtype)
            indices_full = base.indices.to(self.device)
            ext = _get_torch_ext()

            if not self.stack_shape:
                w_n = sqrt_w_full * sqrt_w_full
                w_eff = (w_n * w_n).contiguous()
                psf_flat = torch.zeros(
                    base.grid_size,
                    self.K,
                    self.K,
                    dtype=psf_dtype,
                    device=self.device,
                )
                ext.psf_scatter_outer_basis(
                    psf_flat,
                    indices_full,
                    w_eff,
                    basis,
                    sorted_time_index_full,
                )
                self.psf = (
                    psf_flat.transpose(-2, -1)
                    .contiguous()
                    .reshape(*self.grid_shape, self.K, self.K)
                )
            else:
                S_total = int(np.prod(self.stack_shape))
                n_per = N // S_total
                sqrt_w_v = sqrt_w_full.reshape(S_total, n_per)
                indices_v = indices_full.reshape(S_total, n_per)
                time_v = sorted_time_index_full.reshape(S_total, n_per)
                grid_off = (
                    torch.arange(S_total, device=self.device, dtype=indices_v.dtype)
                    * base.grid_size
                ).unsqueeze(-1)
                indices_packed = (indices_v + grid_off).reshape(-1).contiguous()
                w_n = sqrt_w_v * sqrt_w_v
                w_eff_packed = (w_n * w_n).reshape(-1).contiguous()
                time_packed = time_v.reshape(-1).contiguous()
                psf_super = torch.zeros(
                    S_total * base.grid_size,
                    self.K,
                    self.K,
                    dtype=psf_dtype,
                    device=self.device,
                )
                ext.psf_scatter_outer_basis(
                    psf_super,
                    indices_packed,
                    w_eff_packed,
                    basis,
                    time_packed,
                )
                self.psf = (
                    psf_super.transpose(-2, -1)
                    .contiguous()
                    .reshape(*self.stack_shape, *self.grid_shape, self.K, self.K)
                )

    # ------------------------------------------------------------------
    def __call__(self, coeffs: torch.Tensor) -> torch.Tensor:
        """Apply ``A^H A coeffs``.

        Accepted layouts (with optional leading ``*B`` and, for stacked
        operators, leading ``*S`` axes):

        - ``(*B, *S, K, *image_shape)`` complex.
        """
        s_shape = self.stack_shape
        s_ndim = len(s_shape)
        single_ndim = 1 + len(self.image_shape)  # K, *image
        prefix = tuple(int(s) for s in coeffs.shape[: coeffs.ndim - single_ndim])
        if coeffs.shape[coeffs.ndim - single_ndim] != self.K:
            raise ValueError(
                f"coeffs K-dim {coeffs.shape[coeffs.ndim - single_ndim]} != K={self.K}"
            )
        if s_ndim:
            if len(prefix) < s_ndim or tuple(prefix[-s_ndim:]) != s_shape:
                raise ValueError(
                    f"coeffs prefix {prefix} must end with stack_shape {s_shape}"
                )
            B_shape = prefix[:-s_ndim]
        else:
            B_shape = prefix

        if not prefix:
            return self._apply_single(coeffs, 0)

        S_total = int(np.prod(s_shape)) if s_shape else 1
        B_total = int(np.prod(B_shape)) if B_shape else 1
        single_shape = tuple(coeffs.shape[coeffs.ndim - single_ndim :])
        flat = coeffs.reshape(B_total, S_total, *single_shape)
        outs = []
        for b in range(B_total):
            for s in range(S_total):
                outs.append(self._apply_single(flat[b, s], s))
        stacked = torch.stack(outs, dim=0)
        return stacked.reshape(*B_shape, *s_shape, *single_shape)

    def _apply_single(self, coeffs: torch.Tensor, s_flat_idx: int = 0) -> torch.Tensor:
        base = self._base
        src_device = coeffs.device
        comp_device = self.device
        coeffs_d = coeffs.to(comp_device)
        dtype = coeffs_d.dtype
        psf_full = self.psf.to(dtype)
        if self.stack_shape:
            psf = psf_full.reshape(-1, *self.grid_shape, self.K, self.K)[s_flat_idx]
        else:
            psf = psf_full

        smaps = base.smaps.to(comp_device, dtype=dtype)
        conj_smaps = base._conj_smaps.to(comp_device, dtype=dtype)

        coil = coeffs_d.unsqueeze(0) * smaps.unsqueeze(1)
        out = self._apply_batched(coil, psf)
        return (out * conj_smaps.unsqueeze(1)).sum(0).to(src_device)

    # ------------------------------------------------------------------
    def _apply_batched(self, coil_coeffs, psf):
        """Per-batch core: pad → batched FFT → PSF matvec → IFFT → crop.

        Parameters
        ----------
        coil_coeffs : torch.Tensor
            ``(B, K, *image_shape)`` complex.
        psf : torch.Tensor
            ``(*grid_shape, K, K)`` complex (already pre-transposed at init).

        Returns
        -------
        torch.Tensor
            ``(B, K, *image_shape)`` complex.
        """
        K = self.K
        B = coil_coeffs.shape[0]
        # Pad (B, K, *image_shape) → (B, K, *grid_shape).
        if self.grid_shape == self.image_shape:
            padded = coil_coeffs
        else:
            padded = torch.zeros(
                B,
                K,
                *self.grid_shape,
                dtype=coil_coeffs.dtype,
                device=coil_coeffs.device,
            )
            padded[(slice(None), slice(None), *self._pad_slices)] = coil_coeffs

        # Batched FFT over spatial axes → (B, K, *grid_shape).
        Kk = fft(padded, axes=self.fft_axes)

        # Per-grid-cell (K, K) @ (K,) matvec, batched over B and spatial.
        Kk_perm = Kk.movedim(1, -1).unsqueeze(-1)  # (B, *G, K, 1)
        out_perm = torch.matmul(psf, Kk_perm).squeeze(-1)  # (B, *G, K)
        out = out_perm.movedim(-1, 1)  # (B, K, *G)

        spatial = ifft(out, axes=self.fft_axes)
        if self.grid_shape != self.image_shape:
            spatial = resize(spatial, (B, K, *self.image_shape))
        return spatial
