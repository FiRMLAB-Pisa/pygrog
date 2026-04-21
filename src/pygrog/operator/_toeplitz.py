"""Toeplitz-style fast normal operator (A^H A).

Precomputes the sampling density on the k-space grid so that
``A^H A x`` reduces to element-wise multiplication in k-space,
avoiding the per-call scatter + gather entirely.

The key identity: for gridding operator A with sorted indices and
density-compensation weights w, the normal operator is

    A^H A x = crop(IFFT(D · zeropad(FFT(x))))

where D[k] = sum_{i: idx_i = k} w_i is the gridded sampling density.
Since ``zeropad`` is zero outside the center and the subsequent crop
keeps only the center, the product ``D · zeropad`` is non-zero only
in the center (image_shape) region, so we can work entirely at image
resolution: ``D_c = D[pad_slices]``, then ``A^H A x = IFFT(D_c · FFT(x))``.

Supports:

- Base ``SparseFFT`` (with optional SENSE smaps)
- ``OffResonanceSparseFFT`` (cross-frequency density terms)
- ``SubspaceSparseFFT`` (Gram-weighted density)
- Combined off-resonance + subspace

Usage
-----
::

    from pygrog.operator import SparseFFT, toeplitz_normal

    base = SparseFFT(plan=grog.plan, smaps=smaps)
    normal = toeplitz_normal(base)
    AHA_x = normal(x)
"""

__all__ = ["toeplitz_normal"]

import torch

from .._base._fftc import fft, ifft


def toeplitz_normal(op):
    """Build a fast normal operator from *op*.

    Parameters
    ----------
    op : SparseFFT | OffResonanceSparseFFT | SubspaceSparseFFT
        Any operator produced by the pygrog.operator module.

    Returns
    -------
    callable
        ``normal(x) → A^H A x`` implemented via k-space density
        multiplication (no scatter / gather).
    """
    from ._off_resonance import OffResonanceSparseFFT
    from ._subspace import SubspaceSparseFFT

    if isinstance(op, SubspaceSparseFFT):
        inner = op._base
        if isinstance(inner, OffResonanceSparseFFT):
            return _ToeplitzSubspaceORC(op)
        return _ToeplitzSubspace(op)
    elif isinstance(op, OffResonanceSparseFFT):
        return _ToeplitzORC(op)
    else:
        return _ToeplitzBase(op)


# -- helpers ---------------------------------------------------------------


def _gridded_density(indices, sqrt_weights, grid_size, grid_shape, pad_slices):
    """Compute D[k] = sum_{i: idx_i=k} w_i on grid, cropped to center."""
    w = (sqrt_weights * sqrt_weights).to(torch.float32)
    D = torch.zeros(grid_size, dtype=torch.float32)
    D.index_add_(0, indices, w)
    return D.reshape(grid_shape)[pad_slices].clone()


# =====================================================================
# Base Toeplitz (plain SparseFFT)
# =====================================================================
class _ToeplitzBase:
    """Toeplitz normal for plain SparseFFT (with optional SENSE).

    Precomputes the real-valued sampling density D on the grid, cropped
    to the image-shape center region.  Application is O(n_coils) FFTs
    with no scatter/gather.
    """

    def __init__(self, op):
        if op.smaps is None:
            raise ValueError(
                "toeplitz_normal requires sensitivity maps (smaps). "
                "RSS coil combination is nonlinear and cannot be "
                "expressed as a Toeplitz normal operator."
            )
        self._op = op
        self._D = _gridded_density(
            op.indices,
            op.sqrt_weights,
            op.grid_size,
            op.grid_shape,
            op._pad_slices,
        )
        self._fft_axes = op.fft_axes
        self._smaps = op.smaps
        self._conj_smaps = op._conj_smaps

    def __call__(self, x):
        device = x.device
        dtype = x.dtype
        D = self._D.to(device, dtype=dtype)
        axes = self._fft_axes
        smaps = self._smaps.to(device, dtype=dtype)
        conj_smaps = self._conj_smaps.to(device, dtype=dtype)

        accum = torch.zeros_like(x)
        for c in range(smaps.shape[0]):
            fft_c = fft(smaps[c] * x, axes=axes)
            accum.addcmul_(ifft(D * fft_c, axes=axes), conj_smaps[c])
        return accum


# =====================================================================
# Off-resonance Toeplitz
# =====================================================================
class _ToeplitzORC:
    r"""Toeplitz normal for OffResonanceSparseFFT.

    The normal operator expands to:

    .. math::
        (A_\mathrm{orc})^H A_\mathrm{orc}\, x
        = \sum_{l,l'} \bar C_l \sum_c \bar S_c\,
          \mathrm{IFFT}\!\bigl(D_{l,l'}\, \mathrm{FFT}(S_c\, C_{l'}\, x)\bigr)

    where :math:`D_{l,l'}[k] = \sum_{i:\mathrm{idx}_i=k} w_i\,
    \bar B_{i,l}\, B_{i,l'}` are *L × L* complex density matrices
    on the (cropped) grid.
    """

    def __init__(self, op):
        base = op._base
        L = op.L
        self.L = L
        self.C = op.C

        # B in sorted order to match indices / sqrt_weights
        B_s = op.B[base.sort_perm].to(torch.complex64)
        w = (base.sqrt_weights * base.sqrt_weights).to(torch.complex64)

        D_ll = torch.zeros(L, L, *base.image_shape, dtype=torch.complex64)
        for l1 in range(L):
            for l2 in range(L):
                coeff = w * B_s[:, l1].conj() * B_s[:, l2]
                D_flat = torch.zeros(base.grid_size, dtype=torch.complex64)
                D_flat.index_add_(0, base.indices, coeff)
                D_ll[l1, l2] = D_flat.reshape(base.grid_shape)[base._pad_slices]
        self._D_ll = D_ll

        self._fft_axes = base.fft_axes
        if base.smaps is not None:
            self._smaps = base.smaps
            self._conj_smaps = base._conj_smaps
        else:
            self._smaps = None
            self._conj_smaps = None

    def __call__(self, x):
        device = x.device
        dtype = x.dtype
        D_ll = self._D_ll.to(device, dtype=dtype)
        C = self.C.to(device, dtype=dtype)
        axes = self._fft_axes
        accum = torch.zeros_like(x)

        if self._smaps is not None:
            smaps = self._smaps.to(device, dtype=dtype)
            conj_smaps = self._conj_smaps.to(device, dtype=dtype)
            n_coils = smaps.shape[0]
            for l2 in range(self.L):
                cx = C[l2] * x
                for c in range(n_coils):
                    fft_c = fft(smaps[c] * cx, axes=axes)
                    for l1 in range(self.L):
                        img = ifft(D_ll[l1, l2] * fft_c, axes=axes)
                        accum += C[l1].conj() * conj_smaps[c] * img
        else:
            for l2 in range(self.L):
                fft_cx = fft(C[l2] * x, axes=axes)
                for l1 in range(self.L):
                    accum += C[l1].conj() * ifft(D_ll[l1, l2] * fft_cx, axes=axes)

        return accum


# =====================================================================
# Subspace Toeplitz
# =====================================================================
class _ToeplitzSubspace:
    r"""Toeplitz normal for SubspaceSparseFFT (no off-resonance).

    .. math::
        \text{normal}(\mathbf c)[k_1]
        = \sum_{k_2} G_{k_1 k_2}\, \text{base\_normal}(\mathbf c[k_2])

    where :math:`G = \Phi^H \Phi` is the Gram matrix of the basis.
    """

    def __init__(self, op):
        self._base_toep = _ToeplitzBase(op._base)
        self._gram = op.basis @ op.basis.conj().T
        self.K = op.K

    def __call__(self, coeffs):
        gram = self._gram.to(coeffs.device, dtype=coeffs.dtype)
        normals = torch.stack([self._base_toep(coeffs[k]) for k in range(self.K)])
        return torch.einsum("ij,j...->i...", gram, normals)


# =====================================================================
# Combined Subspace + Off-Resonance Toeplitz
# =====================================================================
class _ToeplitzSubspaceORC:
    """Toeplitz normal for SubspaceSparseFFT wrapping OffResonanceSparseFFT."""

    def __init__(self, op):
        self._orc_toep = _ToeplitzORC(op._base)
        self._gram = op.basis @ op.basis.conj().T
        self.K = op.K

    def __call__(self, coeffs):
        gram = self._gram.to(coeffs.device, dtype=coeffs.dtype)
        normals = torch.stack([self._orc_toep(coeffs[k]) for k in range(self.K)])
        return torch.einsum("ij,j...->i...", gram, normals)
