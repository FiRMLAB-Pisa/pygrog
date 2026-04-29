"""Off-resonance correction gadget and SparseFFT decorator.

Provides two complementary approaches to B0/R2* field-map correction:

* :class:`OffResonanceCorrection` — standalone gadget wrapping any
  ``SparseFFT`` instance; suitable for custom reconstruction loops.
* :func:`with_off_resonance` / :class:`OffResonanceSparseFFT` — decorator
  that bakes the correction into the operator itself, enabling clean
  composition with :func:`~pygrog.gadgets.with_subspace` and
  :func:`~pygrog.operator.toeplitz_normal`.

Both paths use the same mri-nufft field-map factorisation
(``E ≈ B @ C``, time x space) and support the same three methods:
``'svd'`` (default), ``'mti'``, and ``'mfi'``.
"""

__all__ = [
    "OffResonanceCorrection",
    "OffResonanceMaskedFFT",
    "OffResonanceSparseFFT",
    "with_off_resonance",
]

import numpy as np
import torch

from mrinufft._array_compat import with_torch

from .._solve._mixin import SolveMixin


class OffResonanceCorrection:
    """
    Off-resonance correction via field-map factorization (SVD / MFI / MTI).

    Approximates the off-resonance phase accumulation as a low-rank sum

    .. math::

        e^{\\Delta(\\mathbf{r})\\,t} \\approx \\sum_{l=1}^{L} B_l(t)\\,C_l(\\mathbf{r})

    where :math:`\\Delta(\\mathbf{r}) = R_2^*(\\mathbf{r}) + j\\,2\\pi\\,\\Delta f(\\mathbf{r})`
    (rad/s) and the :math:`B` / :math:`C` matrices are computed by
    ``mri-nufft``.

    Parameters
    ----------
    sparse_fft : SparseFFT
        Pre-configured sparse FFT operator (holds grid/image shape, indices,
        weights).
    field_map : torch.Tensor | np.ndarray
        B0 field map **in Hz**, shape ``(*spatial)``.
    readout_time : torch.Tensor | np.ndarray
        Per-sample readout times **in seconds**, shape ``(n_samples,)``.
    r2star_map : torch.Tensor | np.ndarray | None, optional
        R2* map in Hz (same shape as ``field_map``).  Default ``None``
        (purely imaginary field).
    mask : np.ndarray | None, optional
        Boolean spatial mask used for histogram computation.  Default: all-True
        (use every voxel).
    n_components : int, optional
        Number of basis components ``L``.  ``-1`` lets the factorization
        auto-select.  Default ``-1``.
    n_bins : int, optional
        Number of histogram bins for field-map quantization.  Default ``1024``.
    method : str, optional
        Factorization method: ``'svd'`` (default, no extra deps),
        ``'mfi'`` (requires ``scikit-learn``), or ``'mti'``.

    References
    ----------
    Sutton et al., IEEE Trans Med Imaging, 2003. (SVD)
    Man et al., Magn Reson Med, 1997. (MFI)
    Noll et al., IEEE Trans Med Imaging, 1991. (MTI)
    """

    def __init__(
        self,
        sparse_fft,
        field_map,
        readout_time,
        r2star_map=None,
        mask=None,
        n_components: int = -1,
        n_bins: int = 1024,
        method: str = "svd",
    ):
        from mrinufft.extras.field_map import (
            get_orc_factorization,
            get_complex_fieldmap_rad,
        )

        self.sparse_fft = sparse_fft

        # Convert inputs to numpy (mri-nufft operates on numpy/cupy arrays)
        def _to_np(x, dtype):
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
            return np.asarray(x, dtype=dtype)

        field_map_np = _to_np(field_map, np.float32)
        readout_time_np = _to_np(readout_time, np.float32)
        r2star_np = _to_np(r2star_map, np.float32) if r2star_map is not None else None

        # Preserve the original time-axis layout so we can broadcast B against
        # ``base.natural_shape`` at apply time without materialising it.
        self._time_shape = tuple(int(s) for s in readout_time_np.shape)

        if mask is None:
            mask = np.ones(field_map_np.shape, dtype=bool)

        complex_fmap = get_complex_fieldmap_rad(field_map_np, r2star_np)

        factorize = get_orc_factorization(method)
        # The temporal LS problem depends only on the per-sample time *values*,
        # not on their tensor shape, so we ravel before factorising.
        B_np, C_np, _ = factorize(
            field_map=complex_fmap,
            readout_time=readout_time_np.ravel(),
            mask=mask,
            L=n_components,
            n_bins=n_bins,
            lazy=False,
        )

        # B: (prod(time_shape), L) → reshape to (*time_shape, L) for natural-shape
        #    broadcasting against ``base.natural_shape``.
        # C: (L, *spatial) complex64 — spatial coefficients
        B_t = torch.as_tensor(np.asarray(B_np, dtype=np.complex64))
        L = int(B_t.shape[-1])
        self._B = B_t.reshape(*self._time_shape, L)
        self._C = torch.as_tensor(np.asarray(C_np, dtype=np.complex64))

    @property
    def n_components(self) -> int:
        """Number of basis components ``L``."""
        return self._B.shape[-1]

    def _b_view(self, B: torch.Tensor) -> torch.Tensor:
        """Reshape ``B`` to broadcast against ``base.natural_shape``.

        Returns a strided view of shape ``(*nat_with_singletons, L)`` where
        ``time_shape`` occupies a contiguous slice of ``natural_shape`` and the
        other axes are singletons.  No memory is materialised.
        """
        nat = tuple(int(s) for s in self.sparse_fft.natural_shape)
        ts = self._time_shape
        L = int(B.shape[-1])
        # Find the first contiguous slice of nat that matches ts.
        start = -1
        for i in range(len(nat) - len(ts) + 1):
            if nat[i : i + len(ts)] == ts:
                start = i
                break
        if start < 0:
            raise ValueError(
                f"readout_time shape {ts} is not a contiguous slice of "
                f"base.natural_shape {nat}; cannot broadcast B."
            )
        view_shape = (1,) * start + ts + (1,) * (len(nat) - start - len(ts)) + (L,)
        return B.view(view_shape)

    @with_torch
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Forward operator: image → sparse k-space (with off-resonance).

        Computes :math:`s \\approx \\sum_l B_l(t) \\cdot \\mathcal{F}\\{C_l(\\mathbf{r})\\,\\rho\\}`.

        Parameters
        ----------
        image : torch.Tensor
            Shape ``(n_coils, *spatial)``.

        Returns
        -------
        torch.Tensor
            Shape ``(n_coils, *natural_shape)``.
        """
        device = image.device
        B = self._B.to(device, dtype=image.dtype)
        C = self._C.to(device, dtype=image.dtype)  # (L, *spatial)

        # Strided view of B → (*nat_with_singletons, L); move L to the front.
        Bv = self._b_view(B).movedim(-1, 0)  # (L, *nat_with_singletons)

        # Pre-weight image per component: (L, n_coils, *spatial)
        weighted = C.unsqueeze(1) * image.unsqueeze(0)
        # NUFFT per component → (n_coils, *natural_shape); stack over L.
        ksps = torch.stack(
            [self.sparse_fft.forward(weighted[l]) for l in range(self.n_components)]
        )  # (L, n_coils, *nat)
        # Broadcast-multiply by B (strided) and sum over L.
        return (Bv.unsqueeze(1) * ksps).sum(0)

    @with_torch
    def adjoint(self, kspace: torch.Tensor) -> torch.Tensor:
        """Adjoint operator: sparse k-space → image (with off-resonance correction).

        Computes :math:`\\hat{\\rho} \\approx \\sum_l C_l^*(\\mathbf{r})\\,\\mathcal{F}^H\\{B_l^*(t)\\,s\\}`.

        Parameters
        ----------
        kspace : torch.Tensor
            Shape ``(n_coils, *natural_shape)`` (or any shape broadcastable to it
            after reshape).

        Returns
        -------
        torch.Tensor
            Shape ``(n_coils, *spatial)``.
        """
        device = kspace.device
        nat = tuple(int(s) for s in self.sparse_fft.natural_shape)
        kspace_nat = kspace.reshape(kspace.shape[0], *nat)

        B = self._B.to(device, dtype=kspace.dtype)
        C = self._C.to(device, dtype=kspace.dtype)  # (L, *spatial)
        Bv = self._b_view(B).movedim(-1, 0)  # (L, *nat_with_singletons)

        # Pre-weight k-space per component via broadcasting (no expand+contiguous).
        weighted = kspace_nat.unsqueeze(0) * Bv.conj().unsqueeze(
            1
        )  # (L, n_coils, *nat)
        # NUFFT per component → (L, n_coils, *spatial)
        imgs = torch.stack(
            [self.sparse_fft.adjoint(weighted[l]) for l in range(self.n_components)]
        )
        # Weighted sum: C.conj() is (L, *spatial), broadcast over coils
        return (C.conj().unsqueeze(1) * imgs).sum(0)


# =====================================================================
# SparseFFT decorator
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
    *,
    toeplitz=None,
):
    """Wrap a SparseFFT or MaskedFFT operator with B0 inhomogeneity compensation.

    Parameters
    ----------
    base_op : SparseFFT | MaskedFFT
        Base sparse or gridded FFT operator.
    b0_map : array-like, float32
        Static B0 field map in **Hz**, shape ``(*image_shape)``.
    readout_time : array-like, float32
        Per-sample readout time in **seconds**.
        Shape ``(n_samples,)`` or ``(n_shots, n_pts_per_shot)``.
    r2star_map : array-like | None
        R2* map in Hz.  Same shape as *b0_map*.  Default *None*.
    mask : array-like | None
        Boolean mask of the object support, shape ``(*image_shape)``.
        Default *None* (full FOV).
    method : str
        ``'svd'``, ``'mti'``, or ``'mfi'``.
    L : int
        Number of basis functions / interpolators.  ``-1`` = auto.
    n_bins : int
        Number of histogram bins for field-map quantisation.

    Returns
    -------
    OffResonanceSparseFFT | OffResonanceMaskedFFT
    """
    from mrinufft.extras.field_map import (
        get_orc_factorization,
        get_complex_fieldmap_rad,
    )

    b0_map = np.asarray(b0_map, dtype=np.float32)
    readout_time = np.asarray(readout_time, dtype=np.float32).ravel()
    r2star_np = (
        np.asarray(r2star_map, dtype=np.float32) if r2star_map is not None else None
    )

    field_map = get_complex_fieldmap_rad(b0_map, r2star_np)

    if mask is None:
        mask = np.ones(b0_map.shape, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)

    factorize = get_orc_factorization(method)
    B, C, _ = factorize(
        field_map=field_map,
        readout_time=readout_time,
        mask=mask,
        L=L,
        n_bins=n_bins,
        lazy=False,
    )
    B = np.asarray(B, dtype=np.complex64)
    C = np.asarray(C, dtype=np.complex64)

    from ..operator._masked_fft import MaskedFFT

    if isinstance(base_op, MaskedFFT):
        return OffResonanceMaskedFFT(base_op, B, C, toeplitz=toeplitz)
    return OffResonanceSparseFFT(base_op, B, C, toeplitz=toeplitz)


class OffResonanceSparseFFT(SolveMixin):
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
    toeplitz : bool | None, optional
        Use Toeplitz embedding for :meth:`normal`.  ``None`` inherits
        from ``base_op.toeplitz``.
    """

    def __init__(self, base_op, B, C, *, toeplitz=None):
        self._base = base_op
        self.B = torch.as_tensor(B)  # (n_samples, L)
        self.C = torch.as_tensor(C)  # (L, *image_shape)
        self.L = self.B.shape[1]

        # Expose base attributes
        self.grid_shape = base_op.grid_shape
        self.image_shape = base_op.image_shape
        self.smaps = base_op.smaps

        # Toeplitz flag inherits from base unless overridden.
        if toeplitz is None:
            toeplitz = bool(getattr(base_op, "toeplitz", False))
        self.toeplitz = bool(toeplitz)
        self._toep_op = None  # lazily built

    @property
    def n_samples(self):
        return self.B.shape[0]

    @with_torch
    def adjoint(self, sparse_kspace):
        """B0-corrected sparse k-space → image.

        Accepted layouts:

        - ``(*B, *S, n_coils, n_samples)`` with optional batch ``*B`` and,
          for stacked plans, leading ``*S`` axes.
        - ``(n_coils, n_samples)`` (single frame, no batch / stack)

        Returns
        -------
        torch.Tensor
            ``(*B, *S, *image_shape)`` (or ``(*image_shape,)``).
        """
        s_shape = tuple(getattr(self._base, "stack_shape", ()) or ())
        s_ndim = len(s_shape)
        prefix = tuple(int(s) for s in sparse_kspace.shape[:-2])
        if s_ndim:
            if len(prefix) < s_ndim or tuple(prefix[-s_ndim:]) != s_shape:
                raise ValueError(
                    f"sparse_kspace prefix {prefix} must end with stack_shape {s_shape}"
                )
            B_shape = prefix[:-s_ndim]
        else:
            B_shape = prefix
        if not prefix:
            return self._forward_single(sparse_kspace, 0)
        B_total = int(np.prod(B_shape)) if B_shape else 1
        S_total = int(np.prod(s_shape)) if s_shape else 1
        flat = sparse_kspace.reshape(B_total, S_total, *sparse_kspace.shape[-2:])
        outs = [
            self._forward_single(flat[b, s], s)
            for b in range(B_total)
            for s in range(S_total)
        ]
        stacked = torch.stack(outs, dim=0)
        return stacked.reshape(*B_shape, *s_shape, *self.image_shape)

    def _forward_single(self, sparse_kspace, s_flat_idx: int = 0):
        """Single-frame ORC forward.  ``sparse_kspace`` shape: ``(n_coils, n_samples)``."""
        device = sparse_kspace.device
        B = self.B.to(device, dtype=sparse_kspace.dtype)
        C = self.C.to(device, dtype=sparse_kspace.dtype)
        # Allow per-stack B/C: shape (*S, n_samples, L) or (*S, L, *image).
        if B.ndim > 2:
            B = B.reshape(-1, B.shape[-2], B.shape[-1])[s_flat_idx]
        if C.ndim > 1 + len(self.image_shape):
            C = C.reshape(-1, C.shape[-1 - len(self.image_shape)], *self.image_shape)[
                s_flat_idx
            ]
        n_coils = sparse_kspace.shape[0]

        use_batch = self._base.smaps is not None and hasattr(
            self._base, "_scatter_ifft_crop_batch"
        )

        if use_batch:
            conj_smaps = self._base._conj_smaps.to(device, dtype=sparse_kspace.dtype)
            weighted = (sparse_kspace.unsqueeze(0) * B.conj().T.unsqueeze(1)).reshape(
                -1, self.n_samples
            )
            imgs_flat = self._base._scatter_ifft_crop_batch(
                weighted,
                s_flat_idx=s_flat_idx,
            )
            imgs = (
                imgs_flat.reshape(self.L, n_coils, *self.image_shape)
                * conj_smaps.unsqueeze(0)
            ).sum(1)
        else:
            weighted = sparse_kspace.unsqueeze(0) * B.conj().T.unsqueeze(1)
            imgs = torch.stack(
                [
                    self._base._forward_single(weighted[ll], s_flat_idx)
                    for ll in range(self.L)
                ]
            )

        n_extra = imgs.ndim - C.ndim
        c = C.conj().view(*C.shape[:1], *([1] * n_extra), *C.shape[1:])
        return (c * imgs).sum(0)

    @with_torch
    def forward(self, image):
        """B0-corrected image → sparse k-space.

        Accepted layouts:

        - ``(*B, *S, *image_shape)`` (smaps path)
        - ``(*B, *S, n_coils, *image_shape)`` (no-smaps path)
        - single-frame variants without leading prefix.

        Returns
        -------
        torch.Tensor
            ``(*B, *S, n_coils, n_samples)``.
        """
        s_shape = tuple(getattr(self._base, "stack_shape", ()) or ())
        s_ndim = len(s_shape)
        single_ndim = len(self.image_shape) + (0 if self._base.smaps is not None else 1)
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
            return self._adjoint_single(image, 0)
        B_total = int(np.prod(B_shape)) if B_shape else 1
        S_total = int(np.prod(s_shape)) if s_shape else 1
        flat = image.reshape(B_total, S_total, *image.shape[image.ndim - single_ndim :])
        outs = [
            self._adjoint_single(flat[b, s], s)
            for b in range(B_total)
            for s in range(S_total)
        ]
        n_coils = outs[0].shape[0]
        stacked = torch.stack(outs, dim=0)
        return stacked.reshape(*B_shape, *s_shape, n_coils, self.n_samples)

    def _adjoint_single(self, image, s_flat_idx: int = 0):
        """Single-frame ORC adjoint."""
        device = image.device
        B = self.B.to(device, dtype=image.dtype)
        C = self.C.to(device, dtype=image.dtype)
        if B.ndim > 2:
            B = B.reshape(-1, B.shape[-2], B.shape[-1])[s_flat_idx]
        if C.ndim > 1 + len(self.image_shape):
            C = C.reshape(-1, C.shape[-1 - len(self.image_shape)], *self.image_shape)[
                s_flat_idx
            ]

        n_extra = image.ndim - len(self.image_shape)
        c = C.view(self.L, *([1] * n_extra), *self.image_shape)
        weighted = c * image.unsqueeze(0)

        use_batch = self._base.smaps is not None and hasattr(
            self._base, "_fft_pad_gather_batch"
        )

        if use_batch:
            smaps = self._base.smaps.to(device, dtype=image.dtype)
            n_coils = smaps.shape[0]
            all_imgs = (weighted.unsqueeze(1) * smaps.unsqueeze(0)).reshape(
                -1, *self.image_shape
            )
            all_ksps = self._base._fft_pad_gather_batch(
                all_imgs,
                s_flat_idx=s_flat_idx,
            )
            ksps = all_ksps.reshape(self.L, n_coils, self.n_samples)
        else:
            ksps = torch.stack(
                [
                    self._base._adjoint_single(weighted[ll], s_flat_idx)
                    for ll in range(self.L)
                ]
            )

        return (B.T.unsqueeze(1) * ksps).sum(0)

    @with_torch
    def normal(self, image):
        """Normal operator: ``A^H A x``."""
        if self.toeplitz:
            if self._toep_op is None:
                from .._toep._orc_toep import OffResonanceToeplitzOp

                self._toep_op = OffResonanceToeplitzOp(
                    self,
                    device=self._base.device,
                )
            return self._toep_op(image)
        return self.adjoint(self.forward(image))

    def __call__(self, x, adjoint=False):
        if adjoint:
            return self.adjoint(x)
        return self.forward(x)


# =====================================================================
# MaskedFFT decorator
# =====================================================================
class OffResonanceMaskedFFT(SolveMixin):
    """MaskedFFT with multi-frequency B0 correction.

    Mirrors :class:`OffResonanceSparseFFT` but operates on pre-gridded
    k-space data via :class:`~pygrog.operator.MaskedFFT`.

    Implements:

    - **forward** (gridded k-space → image):
      ``img = sum_l  conj(C_l) * base.forward(conj(B_l[T_grid]) * kspace_grid)``

    - **adjoint** (image → gridded k-space):
      ``kgrid = sum_l  B_l[T_grid] * base.adjoint(C_l * img)``

    where ``B_l[T_grid]`` is the temporal basis vector reshaped to broadcast
    over the temporal axis of the gridded k-space ``(*grid_shape)``.

    Parameters
    ----------
    base_op : MaskedFFT
        The underlying gridded FFT operator (with smaps, etc.).
    B : torch.Tensor, complex64
        Temporal basis, shape ``(n_samples, L)`` or ``(*grid_shape, L)``
        if already gridded.
    C : torch.Tensor, complex64
        Spatial interpolators, shape ``(L, *image_shape)``.
    toeplitz : bool | None, optional
        Use Toeplitz embedding for :meth:`normal`.  ``None`` inherits
        from ``base_op.toeplitz``.
    """

    def __init__(self, base_op, B, C, *, toeplitz=None):
        self._base = base_op
        # B may arrive as (n_samples, L) — reshape to (*grid_shape, L) for
        # broadcasting against gridded k-space axes.
        B_t = torch.as_tensor(B)
        grid_size = int(np.prod(base_op.grid_shape))
        if B_t.shape[0] == grid_size and B_t.ndim == 2:
            B_t = B_t.reshape(*base_op.grid_shape, B_t.shape[-1])
        self.B = B_t  # (*grid_shape, L)
        self.C = torch.as_tensor(C)  # (L, *image_shape)
        self.L = int(self.B.shape[-1])

        self.grid_shape = base_op.grid_shape
        self.image_shape = base_op.image_shape
        self.smaps = base_op.smaps

        if toeplitz is None:
            toeplitz = bool(getattr(base_op, "toeplitz", False))
        self.toeplitz = bool(toeplitz)
        self._toep_op = None

    @property
    def n_samples(self):
        return int(np.prod(self.grid_shape))

    @with_torch
    def adjoint(self, kspace_grid):
        """B0-corrected gridded k-space → image.

        Parameters
        ----------
        kspace_grid : torch.Tensor
            ``(*B, *S, n_coils, *grid_shape)``

        Returns
        -------
        torch.Tensor
            ``(*B, *S, *image_shape)`` (SENSE combined).
        """
        s_shape = tuple(getattr(self._base, "stack_shape", ()) or ())
        s_ndim = len(s_shape)
        grid_ndim = len(self.grid_shape)
        prefix = tuple(
            int(s) for s in kspace_grid.shape[: kspace_grid.ndim - (1 + grid_ndim)]
        )
        if s_ndim:
            if len(prefix) < s_ndim or tuple(prefix[-s_ndim:]) != s_shape:
                raise ValueError(
                    f"kspace_grid prefix {prefix} must end with stack_shape {s_shape}"
                )
            B_shape = prefix[:-s_ndim]
        else:
            B_shape = prefix
        if not prefix:
            return self._forward_single(kspace_grid, 0)
        B_total = int(np.prod(B_shape)) if B_shape else 1
        S_total = int(np.prod(s_shape)) if s_shape else 1
        n_coils = int(kspace_grid.shape[-grid_ndim - 1])
        flat = kspace_grid.reshape(B_total, S_total, n_coils, *self.grid_shape)
        outs = [
            self._forward_single(flat[b, s], s)
            for b in range(B_total)
            for s in range(S_total)
        ]
        stacked = torch.stack(outs, dim=0)
        return stacked.reshape(*B_shape, *s_shape, *self.image_shape)

    def _forward_single(self, kspace_grid, s_flat_idx: int = 0):
        """Single-frame ORC forward.  Input: ``(n_coils, *grid_shape)``."""
        device = kspace_grid.device
        B = self.B.to(device, dtype=kspace_grid.dtype)  # (*grid_shape, L)
        C = self.C.to(device, dtype=kspace_grid.dtype)  # (L, *image_shape)

        use_batch = self._base.smaps is not None and hasattr(
            self._base, "_mask_ifft_crop_batch"
        )

        if use_batch:
            conj_smaps = self._base._conj_smaps.to(device, dtype=kspace_grid.dtype)
            n_coils = kspace_grid.shape[0]
            # Expand kspace_grid over L: (L, n_coils, *grid_shape)
            # B.conj(): (*grid_shape, L) → movedim → (L, *grid_shape)
            B_conj = B.conj().movedim(-1, 0)  # (L, *grid_shape)
            # (L, n_coils, *grid_shape)
            weighted = B_conj.unsqueeze(1) * kspace_grid.unsqueeze(0)
            # Flatten L and coils: (L*n_coils, *grid_shape)
            weighted_flat = weighted.reshape(-1, *self.grid_shape)
            imgs_flat = self._base._mask_ifft_crop_batch(
                weighted_flat, s_flat_idx=s_flat_idx
            )
            # (L, n_coils, *image_shape)
            imgs = imgs_flat.reshape(self.L, n_coils, *self.image_shape)
            # Apply C.conj() (L,*image) and smaps (C,*image), sum over L
            # result: (n_coils, *image)  then SENSE combine via smaps.conj()
            imgs_weighted = (C.conj().unsqueeze(1) * imgs).sum(0)  # (n_coils, *image)
            # SENSE combine
            return (imgs_weighted * conj_smaps).sum(0)
        else:
            B_conj = B.conj().movedim(-1, 0)  # (L, *grid_shape)
            weighted = B_conj.unsqueeze(1) * kspace_grid.unsqueeze(
                0
            )  # (L, C, *grid_shape)
            imgs = torch.stack(
                [
                    self._base._forward_single(weighted[ll], s_flat_idx)
                    for ll in range(self.L)
                ]
            )
            n_extra = imgs.ndim - C.ndim
            c = C.conj().view(*C.shape[:1], *([1] * n_extra), *C.shape[1:])
            return (c * imgs).sum(0)

    @with_torch
    def forward(self, image):
        """B0-corrected image → gridded k-space.

        Parameters
        ----------
        image : torch.Tensor
            ``(*B, *S, *image_shape)`` (smaps path) or
            ``(*B, *S, n_coils, *image_shape)``.

        Returns
        -------
        torch.Tensor
            ``(*B, *S, n_coils, *grid_shape)``.
        """
        s_shape = tuple(getattr(self._base, "stack_shape", ()) or ())
        s_ndim = len(s_shape)
        single_ndim = len(self.image_shape) + (0 if self._base.smaps is not None else 1)
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
            return self._adjoint_single(image, 0)
        B_total = int(np.prod(B_shape)) if B_shape else 1
        S_total = int(np.prod(s_shape)) if s_shape else 1
        flat = image.reshape(B_total, S_total, *image.shape[image.ndim - single_ndim :])
        outs = [
            self._adjoint_single(flat[b, s], s)
            for b in range(B_total)
            for s in range(S_total)
        ]
        n_coils = outs[0].shape[0]
        stacked = torch.stack(outs, dim=0)
        return stacked.reshape(*B_shape, *s_shape, n_coils, *self.grid_shape)

    def _adjoint_single(self, image, s_flat_idx: int = 0):
        """Single-frame ORC adjoint."""
        device = image.device
        B = self.B.to(device, dtype=image.dtype)  # (*grid_shape, L)
        C = self.C.to(device, dtype=image.dtype)  # (L, *image_shape)
        B_moved = B.movedim(-1, 0)  # (L, *grid_shape)

        n_extra = image.ndim - len(self.image_shape)
        c = C.view(self.L, *([1] * n_extra), *self.image_shape)
        weighted = c * image.unsqueeze(0)  # (L, *image)

        use_batch = self._base.smaps is not None and hasattr(
            self._base, "_fft_pad_mask_batch"
        )

        if use_batch:
            smaps = self._base.smaps.to(device, dtype=image.dtype)
            n_coils = smaps.shape[0]
            # (L, n_coils, *image_shape)
            all_imgs = weighted.unsqueeze(1) * smaps.unsqueeze(0)
            # Flatten to (L*n_coils, *image_shape)
            all_imgs_flat = all_imgs.reshape(-1, *self.image_shape)
            all_kgrids = self._base._fft_pad_mask_batch(
                all_imgs_flat,
                s_flat_idx=s_flat_idx,
            )  # (L*n_coils, *grid_shape)
            kgrids = all_kgrids.reshape(self.L, n_coils, *self.grid_shape)
            # Multiply by B and sum over L: (n_coils, *grid_shape)
            return (B_moved.unsqueeze(1) * kgrids).sum(0)
        else:
            kgrids = torch.stack(
                [
                    self._base._adjoint_single(weighted[ll], s_flat_idx)
                    for ll in range(self.L)
                ]
            )  # (L, n_coils, *grid_shape)
            return (B_moved.unsqueeze(1) * kgrids).sum(0)

    @with_torch
    def normal(self, image):
        """Normal operator: ``A^H A x``."""
        if self.toeplitz:
            if self._toep_op is None:
                from .._toep._orc_toep import OffResonanceToeplitzOp

                self._toep_op = OffResonanceToeplitzOp(
                    self,
                    device=self._base.device,
                )
            return self._toep_op(image)
        return self.adjoint(self.forward(image))

    def __call__(self, x, adjoint=False):
        if adjoint:
            return self.adjoint(x)
        return self.forward(x)
