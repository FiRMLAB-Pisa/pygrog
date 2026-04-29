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
        self.stack_shape = tuple(getattr(base, "stack_shape", ()) or ())

        # ---- Build PSF: (*S, *grid_shape, L, L) complex --------------------
        _is_masked = getattr(base, "density", None) is not None

        if _is_masked:
            # MaskedFFT path: B is already gridded to (*grid_shape, L).
            # PSF[j, l', l] = density[j] * conj(B_grid[j, l']) * B_grid[j, l]
            # where B is the gridded ORC basis from OffResonanceMaskedFFT.
            B_grid = orc_op.B.to(
                self.device
            )  # (*grid_shape, L) or (*S, *grid_shape, L)
            density = base.density.to(
                self.device
            )  # (*S, *grid_shape) or (*grid_shape,)
            psf_dtype = B_grid.dtype
            if not self.stack_shape:
                # density: (*grid_shape,), B_grid: (*grid_shape, L)
                psf = (
                    density.unsqueeze(-1).unsqueeze(-1)
                    * B_grid.unsqueeze(-2).conj()
                    * B_grid.unsqueeze(-1)
                )  # (*grid_shape, L, L)
                self.psf = psf
            else:
                S_total = int(torch.tensor(self.stack_shape).prod().item())
                # density: (*stack_shape, *grid_shape) → (S, *grid_shape)
                dens_v = density.reshape(S_total, *self.grid_shape)
                # B_grid: may be (*grid_shape, L) shared or (S, *grid_shape, L)
                if B_grid.ndim == len(self.grid_shape) + 1:
                    B_v = B_grid.unsqueeze(0).expand(S_total, *self.grid_shape, self.L)
                else:
                    B_v = B_grid.reshape(S_total, *self.grid_shape, self.L)
                psf = (
                    dens_v.unsqueeze(-1).unsqueeze(-1)
                    * B_v.unsqueeze(-2).conj()
                    * B_v.unsqueeze(-1)
                )  # (S, *grid_shape, L, L)
                self.psf = psf.reshape(
                    *self.stack_shape, *self.grid_shape, self.L, self.L
                )
        else:
            sort_perm = base.sort_perm.to(self.device)
            sorted_B_full = (
                orc_op.B.to(self.device).contiguous()[sort_perm].contiguous()
            )
            psf_dtype = sorted_B_full.dtype
            real_dtype = (
                torch.float32 if psf_dtype == torch.complex64 else torch.float64
            )

            sqrt_w_full = base.sqrt_weights.to(self.device, dtype=real_dtype)
            indices_full = base.indices.to(self.device)
            ext = _get_torch_ext()

            if not self.stack_shape:
                w_sq = (sqrt_w_full * sqrt_w_full).contiguous()
                psf_flat = torch.zeros(
                    base.grid_size,
                    self.L,
                    self.L,
                    dtype=psf_dtype,
                    device=self.device,
                )
                ext.psf_scatter_outer(psf_flat, indices_full, w_sq, sorted_B_full)
                self.psf = psf_flat.reshape(*self.grid_shape, self.L, self.L)
            else:
                S_total = int(torch.tensor(self.stack_shape).prod().item())
                n_per = int(base.n_samples)
                sqrt_w_v = sqrt_w_full.reshape(S_total, n_per)
                indices_v = indices_full.reshape(S_total, n_per)
                sorted_B_v = sorted_B_full.reshape(S_total, n_per, self.L)
                grid_off = (
                    torch.arange(S_total, device=self.device, dtype=indices_v.dtype)
                    * base.grid_size
                ).unsqueeze(-1)
                indices_packed = (indices_v + grid_off).reshape(-1).contiguous()
                w_sq_packed = (sqrt_w_v * sqrt_w_v).reshape(-1).contiguous()
                sorted_B_packed = sorted_B_v.reshape(-1, self.L).contiguous()
                psf_super = torch.zeros(
                    S_total * base.grid_size,
                    self.L,
                    self.L,
                    dtype=psf_dtype,
                    device=self.device,
                )
                ext.psf_scatter_outer(
                    psf_super,
                    indices_packed,
                    w_sq_packed,
                    sorted_B_packed,
                )
                self.psf = psf_super.reshape(
                    *self.stack_shape, *self.grid_shape, self.L, self.L
                )
        # Cache C on device.
        self.C = orc_op.C.to(self.device)

    # ------------------------------------------------------------------
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Apply ``A^H A x``.

        Accepted layouts (with optional leading ``*B`` and, for stacked
        operators, leading ``*S`` axes):

        - ``(*B, *S, *image_shape)`` if smaps are set
        - ``(*B, *S, n_coils, *image_shape)`` otherwise
        """
        base = self._base
        s_shape = self.stack_shape
        s_ndim = len(s_shape)
        single_ndim = (
            len(self.image_shape)
            if base.smaps is not None
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
        single_shape = tuple(image.shape[image.ndim - single_ndim :])
        flat = image.reshape(B_total, S_total, *single_shape)
        outs = []
        for b in range(B_total):
            for s in range(S_total):
                outs.append(self._apply_single(flat[b, s], s))
        stacked = torch.stack(outs, dim=0)
        return stacked.reshape(*B_shape, *s_shape, *single_shape)

    def _apply_single(self, image: torch.Tensor, s_flat_idx: int = 0) -> torch.Tensor:
        base = self._base
        src_device = image.device
        comp_device = self.device
        image_d = image.to(comp_device)
        dtype = image_d.dtype
        psf_full = self.psf.to(dtype)
        if self.stack_shape:
            psf = psf_full.reshape(-1, *self.grid_shape, self.L, self.L)[s_flat_idx]
        else:
            psf = psf_full
        C_full = self.C.to(dtype)
        if C_full.ndim > 1 + len(self.image_shape):
            C = C_full.reshape(-1, self.L, *self.image_shape)[s_flat_idx]
        else:
            C = C_full

        if base.smaps is not None:
            smaps = base.smaps.to(comp_device, dtype=dtype)
            conj_smaps = base._conj_smaps.to(comp_device, dtype=dtype)
            coil_img = image_d.unsqueeze(0) * smaps  # (n_coils, *image)
            weighted = C.unsqueeze(0) * coil_img.unsqueeze(1)
            spatial = self._apply_batched(weighted, psf)
            combined = (C.conj().unsqueeze(0) * spatial).sum(1)
            return (combined * conj_smaps).sum(0).to(src_device)

        weighted = C.unsqueeze(0) * image_d.unsqueeze(1)
        spatial = self._apply_batched(weighted, psf)
        out = (C.conj().unsqueeze(0) * spatial).sum(1)
        return out.to(src_device)

    # ------------------------------------------------------------------
    def _apply_batched(self, weighted, psf):
        """Per-batch core: pad → batched FFT → PSF matvec → IFFT → crop.

        Parameters
        ----------
        weighted : torch.Tensor
            ``(B, L, *image_shape)`` complex (already C-weighted).
        psf : torch.Tensor
            ``(*grid_shape, L, L)`` complex.

        Returns
        -------
        torch.Tensor
            ``(B, L, *image_shape)`` complex.
        """
        L = self.L
        B = weighted.shape[0]
        # Pad (B, L, *image_shape) → (B, L, *grid_shape).
        if self.grid_shape == self.image_shape:
            padded = weighted
        else:
            padded = torch.zeros(
                B,
                L,
                *self.grid_shape,
                dtype=weighted.dtype,
                device=weighted.device,
            )
            padded[(slice(None), slice(None), *self._pad_slices)] = weighted

        # Batched FFT over spatial axes → (B, L, *grid_shape).
        Kk = fft(padded, axes=self.fft_axes)

        # Per-grid-cell (L, L) @ (L,) matvec, batched over B and spatial.
        # Move L to the last-but-one axis: (B, *grid_shape, L, 1).
        Kk_perm = Kk.movedim(1, -1).unsqueeze(-1)  # (B, *G, L, 1)
        # psf broadcasts over B; matmul over (L, L) @ (L, 1).
        out_perm = torch.matmul(psf, Kk_perm).squeeze(-1)  # (B, *G, L)
        out = out_perm.movedim(-1, 1)  # (B, L, *G)

        # IFFT and crop.
        spatial = ifft(out, axes=self.fft_axes)
        if self.grid_shape != self.image_shape:
            spatial = resize(spatial, (B, L, *self.image_shape))
        return spatial
