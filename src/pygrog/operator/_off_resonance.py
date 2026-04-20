"""Off-resonance correction decorator for SparseFFT.

Wraps a :class:`SparseFFT` operator with B0/R2* field-map compensation
using the bilinear factorisation ``E ≈ B @ C`` (time × space).

Three factorisation methods are provided, matching mri-nufft:

- **svd** (default): Truncated SVD of the weighted phase-evolution matrix.
- **mti**: Mixed-Time Interpolation — fixed temporal grid, least-squares fit.
- **mfi**: Multi-Frequency Interpolation — k-means clustering of field-map values.

Usage
-----
::

    from pygrog.operator import SparseFFT, with_off_resonance

    base = SparseFFT(plan=grog.plan, smaps=smaps)
    corrected = with_off_resonance(
        base,
        b0_map=b0,               # Hz, (*image_shape)
        readout_time=t_readout,   # s, (n_samples,) or (n_shots, n_pts)
        method="svd",
        L=6,
    )
    img = corrected.forward(kspace)
"""

__all__ = ["with_off_resonance"]

from types import SimpleNamespace

import numpy as np
import torch


# =====================================================================
# Public decorator
# =====================================================================
def with_off_resonance(
    base_op,
    b0_map,
    readout_time,
    r2star_map=None,
    mask=None,
    method="svd",
    L=-1,
    n_bins=1024,
):
    """Wrap a SparseFFT operator with B0 inhomogeneity compensation.

    Parameters
    ----------
    base_op : SparseFFT
        Base sparse FFT operator.
    b0_map : array-like, float32
        Static B0 field map in **Hz**, shape ``(*image_shape)``.
    readout_time : array-like, float32
        Per-sample readout time in **seconds**.
        Shape ``(n_samples,)`` or ``(n_shots, n_pts_per_shot)``.
    r2star_map : array-like | None
        R2* map in Hz (effective transverse relaxation).  Same shape as
        *b0_map*.  Default *None* (pure B0 only).
    mask : array-like | None
        Boolean mask of the object support, shape ``(*image_shape)``.
        Used to restrict histogram computation.  Default *None* (full FOV).
    method : str
        ``'svd'``, ``'mti'``, or ``'mfi'``.
    L : int
        Number of basis functions / interpolators.  ``-1`` = auto.
    n_bins : int
        Number of histogram bins for field-map quantisation.

    Returns
    -------
    OffResonanceSparseFFT
        Wrapped operator with the same ``forward`` / ``adjoint`` / ``normal``
        API as the base operator.
    """
    b0_map = np.asarray(b0_map, dtype=np.float32)
    readout_time = np.asarray(readout_time, dtype=np.float32).ravel()

    field_map = _complex_fieldmap_rad(b0_map, r2star_map)

    if mask is None:
        mask = np.ones(b0_map.shape, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)

    compute = _METHODS[method]
    B, C = compute(field_map, readout_time, mask, L=L, n_bins=n_bins)

    return OffResonanceSparseFFT(base_op, B, C)


# =====================================================================
# Wrapped operator
# =====================================================================
class OffResonanceSparseFFT:
    """SparseFFT with multi-frequency B0 correction.

    Implements:

    - **forward** (k-space → image):
      ``img = sum_l  conj(C_l) * base.forward(conj(B_l) * kspace)``

    - **adjoint** (image → k-space):
      ``ksp = sum_l  B_l * base.adjoint(C_l * img)``

    Parameters
    ----------
    base_op : SparseFFT
        The underlying sparse FFT operator (with smaps, etc.).
    B : torch.Tensor, complex64
        Temporal basis, shape ``(n_samples, L)``.
    C : torch.Tensor, complex64
        Spatial interpolators, shape ``(L, *image_shape)``.
    """

    def __init__(self, base_op, B, C):
        self._base = base_op
        self.B = torch.as_tensor(B)   # (n_samples, L)
        self.C = torch.as_tensor(C)   # (L, *image_shape)
        self.L = self.B.shape[1]

        # Expose base attributes
        self.grid_shape = base_op.grid_shape
        self.image_shape = base_op.image_shape
        self.smaps = base_op.smaps

    def forward(self, sparse_kspace):
        """B0-corrected sparse k-space → image.

        Parameters
        ----------
        sparse_kspace : torch.Tensor
            ``(n_coils, n_samples)`` complex.

        Returns
        -------
        torch.Tensor
            ``(*image_shape,)`` combined image.
        """
        device = sparse_kspace.device
        B = self.B.to(device, dtype=sparse_kspace.dtype)
        C = self.C.to(device, dtype=sparse_kspace.dtype)

        accum = torch.zeros(self.image_shape, dtype=sparse_kspace.dtype, device=device)
        for ll in range(self.L):
            # Apply temporal basis conjugate to k-space
            b_conj = B[:, ll].conj()  # (n_samples,)
            weighted_ksp = sparse_kspace * b_conj.unsqueeze(0)  # (coils, n_samples)
            img_l = self._base.forward(weighted_ksp)
            # Apply spatial interpolator conjugate
            accum += C[ll].conj() * img_l

        return accum

    def adjoint(self, image):
        """B0-corrected image → sparse k-space.

        Parameters
        ----------
        image : torch.Tensor
            ``(*image_shape,)`` complex.

        Returns
        -------
        torch.Tensor
            ``(n_coils, n_samples)`` complex.
        """
        device = image.device
        B = self.B.to(device, dtype=image.dtype)
        C = self.C.to(device, dtype=image.dtype)

        n_coils = self.smaps.shape[0] if self.smaps is not None else 1
        n_samples = self.B.shape[0]
        accum = torch.zeros(n_coils, n_samples, dtype=image.dtype, device=device)

        for ll in range(self.L):
            # Apply spatial interpolator
            coil_img = C[ll] * image
            ksp_l = self._base.adjoint(coil_img)  # (coils, n_samples)
            # Apply temporal basis
            accum += B[:, ll].unsqueeze(0) * ksp_l

        return accum

    def normal(self, image):
        """Normal operator: ``A^H A x``.

        Parameters
        ----------
        image : torch.Tensor
            ``(*image_shape,)`` complex.

        Returns
        -------
        torch.Tensor
            ``(*image_shape,)`` complex.
        """
        return self.forward(self.adjoint(image))

    def __call__(self, x, adjoint=False):
        if adjoint:
            return self.adjoint(x)
        return self.forward(x)


# =====================================================================
# Field map → complex field in rad/s
# =====================================================================
def _complex_fieldmap_rad(b0_map, r2star_map=None):
    """Convert B0 (Hz) + R2* (Hz) → complex field map in rad/s.

    ``field = R2* + 2j*pi*B0``
    """
    field = np.complex64(2j * np.pi) * b0_map.astype(np.float32)
    if r2star_map is not None:
        field += np.asarray(r2star_map, dtype=np.float32)
    return field


# =====================================================================
# Histogram helpers
# =====================================================================
def _create_histogram(field_map, mask, n_bins=1024):
    """Quantise the complex field map into histogram bins.

    Returns
    -------
    h_counts : ndarray, shape ``(n_bins_r, n_bins_i)``
    w_k : ndarray, complex64, shape ``(n_bins_r, n_bins_i)``
        Bin centres in the complex plane.
    """
    masked = field_map[mask]

    delta_r = np.ptp(masked.real)
    delta_i = np.ptp(masked.imag)

    if delta_i == 0:
        nb = (n_bins, 1)
    elif delta_r == 0:
        nb = (1, n_bins)
    else:
        nb_r = max(1, int(np.around(n_bins * delta_r / delta_i)))
        nb_i = max(1, int(np.around(n_bins / nb_r)))
        nb = (nb_r, nb_i)

    z = masked.view(np.float32).reshape(-1, 2)
    h_counts, h_edges = np.histogramdd(z, bins=nb)

    centres = [e[1:] - (e[1] - e[0]) / 2 for e in h_edges]
    if len(centres) == 2:
        w_k = np.add.outer(centres[0], 1j * centres[1]).astype(np.complex64)
    else:
        w_k = (1j * centres[0]).astype(np.complex64)

    return h_counts, w_k


def _full_C(field_map, C_small, hist_shape):
    """Expand ``(L, *hist_shape) → (L, *field_map.shape)``."""
    fr = field_map.real.ravel()
    fi = field_map.imag.ravel()
    min_r, max_r = fr.min(), fr.max()
    min_i, max_i = fi.min(), fi.max()

    dr = (max_r - min_r) / hist_shape[0] if hist_shape[0] > 1 else 1.0
    di = (max_i - min_i) / hist_shape[1] if hist_shape[1] > 1 else 1.0

    idx_r = np.clip(np.around((fr - min_r) / dr).astype(int), 0, hist_shape[0] - 1) if dr != 0 else np.zeros_like(fr, dtype=int)
    idx_i = np.clip(np.around((fi - min_i) / di).astype(int), 0, hist_shape[1] - 1) if di != 0 else np.zeros_like(fi, dtype=int)

    C_sr = C_small.reshape(-1, *hist_shape)
    C_big = C_sr[:, idx_r, idx_i]
    return np.ascontiguousarray(C_big.reshape(-1, *field_map.shape))


# =====================================================================
# SVD factorisation
# =====================================================================
def _compute_svd(field_map, readout_time, mask, L=-1, n_bins=1024):
    """SVD-based off-resonance factorisation.

    References
    ----------
    Sutton BP, Noll DC, Fessler JA. Fast, iterative image reconstruction
    for MRI in the presence of field inhomogeneities. IEEE TMI 2003.
    """
    h_k, w_k = _create_histogram(field_map, mask, n_bins)
    hist_shape = h_k.shape
    h_flat = h_k.ravel()
    w_flat = w_k.ravel()

    E = np.exp(np.outer(readout_time, w_flat))  # (n_times, n_bins)
    Ew = np.sqrt(h_flat) * E

    if L == -1:
        L = max(1, int(np.ceil(
            abs(w_flat[-1] - w_flat[0]) * np.max(readout_time) / (2 * np.pi)
        )))

    from scipy.sparse.linalg import svds
    B, S, D = svds(Ew, L)

    C_small, _, _, _ = np.linalg.lstsq(B, E, rcond=None)
    C = _full_C(field_map, C_small, hist_shape)

    return B.astype(np.complex64), C.astype(np.complex64)


# =====================================================================
# MTI factorisation
# =====================================================================
def _compute_mti(field_map, readout_time, mask, L=-1, n_bins=1024):
    """Mixed-Time Interpolation.

    References
    ----------
    Noll DC, Meyer CH, Pauly JM, Nishimura DG, Macovski A. IEEE TMI 1991.
    """
    h_k, w_k = _create_histogram(field_map, mask, n_bins)
    hist_shape = h_k.shape
    h_flat = h_k.ravel()
    w_flat = w_k.ravel()

    if L == -1:
        L = max(1, int(np.ceil(
            2 * abs(w_flat[-1] - w_flat[0]) * np.max(readout_time) / np.pi
        )))

    t_l = np.linspace(readout_time.min(), readout_time.max(), L, dtype=np.float32)

    C_hist = np.exp(np.outer(t_l, w_flat))  # (L, n_bins)
    E = np.exp(np.outer(readout_time, w_flat))

    Ch = np.sqrt(h_flat) * C_hist
    Eh = np.sqrt(h_flat) * E

    B, _, _, _ = np.linalg.lstsq(Ch.T, Eh.T, rcond=None)  # (L, n_times)

    C = _full_C(field_map, C_hist, hist_shape)
    return B.T.astype(np.complex64), C.astype(np.complex64)


# =====================================================================
# MFI factorisation
# =====================================================================
def _compute_mfi(field_map, readout_time, mask, L=9, n_bins=1024):
    """Multi-Frequency Interpolation.

    References
    ----------
    Man L-C, Pauly JM, Macovski A. MRM 1997.
    """
    h_k, w_k = _create_histogram(field_map, mask, n_bins)
    hist_shape = h_k.shape
    h_flat = h_k.ravel()
    w_flat = w_k.ravel()

    if L == -1:
        L = max(1, int(np.ceil(
            4 * abs(w_flat[-1] - w_flat[0]) * np.max(readout_time) / np.pi
        )))

    from sklearn.cluster import KMeans
    z = w_flat.view(np.float32).reshape(-1, 2)
    w_l = (
        KMeans(n_clusters=L)
        .fit(z, sample_weight=h_flat)
        .cluster_centers_.view(np.complex64)
        .ravel()
    )

    B = np.exp(np.outer(readout_time, w_l))  # (n_times, L)
    E = np.exp(np.outer(readout_time, w_flat))

    C_small, _, _, _ = np.linalg.lstsq(B, E, rcond=None)
    C = _full_C(field_map, C_small, hist_shape)

    return B.astype(np.complex64), C.astype(np.complex64)


_METHODS = {
    "svd": _compute_svd,
    "mti": _compute_mti,
    "mfi": _compute_mfi,
}
