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

__all__ = ["OffResonanceCorrection", "OffResonanceSparseFFT", "with_off_resonance"]

import numpy as np
import torch


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
        view_shape = (
            (1,) * start
            + ts
            + (1,) * (len(nat) - start - len(ts))
            + (L,)
        )
        return B.view(view_shape)

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
        Bv = self._b_view(B).movedim(-1, 0)         # (L, *nat_with_singletons)

        # Pre-weight image per component: (L, n_coils, *spatial)
        weighted = C.unsqueeze(1) * image.unsqueeze(0)
        # NUFFT per component → (n_coils, *natural_shape); stack over L.
        ksps = torch.stack(
            [self.sparse_fft.adjoint(weighted[l]) for l in range(self.n_components)]
        )                                            # (L, n_coils, *nat)
        # Broadcast-multiply by B (strided) and sum over L.
        return (Bv.unsqueeze(1) * ksps).sum(0)

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
        Bv = self._b_view(B).movedim(-1, 0)         # (L, *nat_with_singletons)

        # Pre-weight k-space per component via broadcasting (no expand+contiguous).
        weighted = kspace_nat.unsqueeze(0) * Bv.conj().unsqueeze(1)  # (L, n_coils, *nat)
        # NUFFT per component → (L, n_coils, *spatial)
        imgs = torch.stack(
            [self.sparse_fft.forward(weighted[l]) for l in range(self.n_components)]
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
    OffResonanceSparseFFT
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

    return OffResonanceSparseFFT(base_op, B, C, toeplitz=toeplitz)


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
        n_coils = sparse_kspace.shape[0]

        use_batch = self._base.smaps is not None and hasattr(
            self._base, "_scatter_ifft_crop_batch"
        )

        if use_batch:
            conj_smaps = self._base._conj_smaps.to(device, dtype=sparse_kspace.dtype)
            weighted = (sparse_kspace.unsqueeze(0) * B.conj().T.unsqueeze(1)).reshape(
                -1, self.n_samples
            )
            imgs_flat = self._base._scatter_ifft_crop_batch(weighted)
            imgs = (
                imgs_flat.reshape(self.L, n_coils, *self.image_shape)
                * conj_smaps.unsqueeze(0)
            ).sum(1)
        else:
            weighted = sparse_kspace.unsqueeze(0) * B.conj().T.unsqueeze(1)
            imgs = torch.stack(
                [self._base.forward(weighted[ll]) for ll in range(self.L)]
            )

        # C: (L, *image_shape).  imgs may be (L, *image_shape) [smaps path]
        # or (L, n_coils, *image_shape) [no-smaps loop path].  Insert
        # singleton coil dims on C so broadcasting always works.
        n_extra = imgs.ndim - C.ndim
        c = C.conj().view(*C.shape[:1], *([1] * n_extra), *C.shape[1:])
        return (c * imgs).sum(0)

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

        # C: (L, *image_shape).  image may be (*image_shape,) [smaps path]
        # or (n_coils, *image_shape) [no-smaps loop path].  Reshape C so it
        # broadcasts correctly against the leading coil dimension if present.
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
            all_ksps = self._base._fft_pad_gather_batch(all_imgs)
            ksps = all_ksps.reshape(self.L, n_coils, self.n_samples)
        else:
            ksps = torch.stack(
                [self._base.adjoint(weighted[ll]) for ll in range(self.L)]
            )

        return (B.T.unsqueeze(1) * ksps).sum(0)

    def normal(self, image):
        """Normal operator: ``A^H A x``."""
        if self.toeplitz:
            if self._toep_op is None:
                from .._toep._orc_toep import OffResonanceToeplitzOp
                self._toep_op = OffResonanceToeplitzOp(
                    self, device=self._base.device,
                )
            return self._toep_op(image)
        return self.forward(self.adjoint(image))

    def __call__(self, x, adjoint=False):
        if adjoint:
            return self.adjoint(x)
        return self.forward(x)
