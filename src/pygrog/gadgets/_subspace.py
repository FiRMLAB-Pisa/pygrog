"""Subspace projection gadget and SparseFFT decorator.

Provides two complementary views of low-rank temporal/contrast subspace
compression:

* :class:`SubspaceProjection` — standalone projection via truncated SVD,
  operates on dense (n_frames, *spatial) tensors.
* :func:`with_subspace` / :class:`SubspaceSparseFFT` — decorator that wraps
  a :class:`~pygrog.operator.SparseFFT` and fuses the subspace projection
  directly into the k-space ↔ image transform.

The subspace basis ``Phi`` has shape ``(K, T)`` where ``K`` is the subspace
rank and ``T`` is the number of temporal frames or contrasts.

Data conventions::

    Sparse k-space: (*batch, n_coils, *natural_shape) — natural_shape comes
        from the GROG plan, e.g. (T, k1, k0, kw) for 3D MRF.
    Image space:    (*batch_image, K, *image_shape) — K subspace coefficients.

The ``encoding_axis`` argument identifies which axis of the sparse tensor
carries the temporal/contrast dimension ``T``; the gadget broadcasts the
basis along that axis.
"""

__all__ = ["SubspaceProjection", "SubspaceSparseFFT", "with_subspace"]

import torch


# =====================================================================
# Standalone gadget
# =====================================================================
class SubspaceProjection:
    """Low-rank temporal subspace projection via truncated SVD.

    Given multi-frame data ``(n_frames, *spatial)``, projects onto the
    leading ``n_components`` left singular vectors.

    Parameters
    ----------
    n_components : int
        Number of subspace components to retain.
    """

    def __init__(self, n_components: int):
        self.n_components = n_components
        self._basis = None

    def fit(self, calib_data: torch.Tensor) -> "SubspaceProjection":
        U, _S, _Vh = torch.linalg.svd(calib_data, full_matrices=False)
        self._basis = U[:, : self.n_components].T.conj()
        return self

    @property
    def basis(self) -> torch.Tensor:
        if self._basis is None:
            raise RuntimeError("Call fit() first.")
        return self._basis

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        spatial_shape = data.shape[1:]
        flat = data.reshape(data.shape[0], -1)
        coeff = self.basis @ flat
        return coeff.reshape(self.n_components, *spatial_shape)

    def adjoint(self, coefficients: torch.Tensor) -> torch.Tensor:
        spatial_shape = coefficients.shape[1:]
        flat = coefficients.reshape(self.n_components, -1)
        frames = self.basis.conj().T @ flat
        return frames.reshape(-1, *spatial_shape)


# =====================================================================
# SparseFFT decorator
# =====================================================================
def with_subspace(base_op, subspace_basis, encoding_axis: int = -4,
                  *, toeplitz=None):
    """Wrap a SparseFFT operator with subspace projection.

    Parameters
    ----------
    base_op : SparseFFT
        Underlying operator with a multi-dim ``natural_shape`` containing
        the temporal axis.
    subspace_basis : array-like, complex
        ``(K, T)`` subspace basis matrix.
    encoding_axis : int
        Axis (in the full sparse-tensor layout) carrying ``T``.  Default
        ``-4`` matches ``(*batch, C, T, k1, k0, kw)``.
    toeplitz : bool | None, optional
        Use Toeplitz embedding for :meth:`normal`.  ``None`` inherits
        from ``base_op.toeplitz``.
    """
    return SubspaceSparseFFT(
        base_op, subspace_basis, encoding_axis=encoding_axis, toeplitz=toeplitz,
    )


class SubspaceSparseFFT:
    """SparseFFT with low-rank subspace projection (loop-fused).

    Adjoint (sparse → image), per coil:
        1. weight by ``sqrt_w`` once on the input;
        2. for each ``k``: multiply by ``basis[k]`` along the T axis,
           scatter into the per-K oversampled grid;
        3. ONE batched K-IFFT + center-crop;
        4. fused FMA with ``smaps[c].conj()`` into the ``(K, *image)`` accumulator.

    Forward (image → sparse), per coil:
        1. multiply ``coeffs`` by ``smaps[c]``;
        2. ONE batched K-FFT + center-pad;
        3. for each ``k``: gather; accumulate ``basis.conj()[k] * gathered``
           into the per-coil ``(*natural)`` accumulator;
        4. write into the output coil slot.

    Parameters
    ----------
    base_op : SparseFFT
        Must have a multi-dim ``natural_shape`` covering the sparse layout
        (e.g. ``(T, k1, k0, kw)``) and SENSE maps (``smaps``) attached.
    subspace_basis : torch.Tensor
        ``(K, T)`` complex basis.
    encoding_axis : int
        Axis (in full sparse layout) of the temporal dimension ``T``.
        Default ``-4`` (last four axes are natural ``(T, k1, k0, kw)``).
    """

    def __init__(self, base_op, subspace_basis, encoding_axis: int = -4,
                 *, toeplitz=None):
        self._base = base_op
        self.basis = torch.as_tensor(subspace_basis)  # (K, T)
        self.K, self.T = self.basis.shape
        self.encoding_axis = encoding_axis

        self.grid_shape = base_op.grid_shape
        self.image_shape = base_op.image_shape
        self.smaps = getattr(base_op, "smaps", None)

        # Position of T inside `natural_shape` (positive index).
        nat_ndim = len(base_op.natural_shape)
        # Full sparse layout: (*batch, C, *natural).  encoding_axis is given
        # relative to that layout; we need the position inside `natural`.
        # E.g. encoding_axis=-4, nat_ndim=4 → axis_in_nat = -4 + nat_ndim = 0 ✓.
        ax = encoding_axis if encoding_axis >= 0 else encoding_axis + (1 + nat_ndim)
        # `ax` now indexes (C, *natural); subtract the leading C dim.
        self._t_axis_in_nat = ax - 1
        if not (0 <= self._t_axis_in_nat < nat_ndim):
            raise ValueError(
                f"encoding_axis={encoding_axis} does not land inside natural_shape "
                f"{base_op.natural_shape} (computed nat-axis {self._t_axis_in_nat})"
            )
        if base_op.natural_shape[self._t_axis_in_nat] != self.T:
            raise ValueError(
                f"basis T={self.T} does not match natural_shape"
                f"[{self._t_axis_in_nat}]={base_op.natural_shape[self._t_axis_in_nat]}"
            )

        # Toeplitz flag inherits from base unless overridden.
        if toeplitz is None:
            toeplitz = bool(getattr(base_op, "toeplitz", False))
        self.toeplitz = bool(toeplitz)
        self._toep_op = None  # lazily built

    # ------------------------------------------------------------------
    # forward: sparse k-space → subspace coefficient images
    # ------------------------------------------------------------------
    def forward(self, sparse_kspace: torch.Tensor) -> torch.Tensor:
        """Sparse → subspace coefficient images. (named ``forward`` to match
        the SparseFFT convention where ``forward`` is sparse → image.)"""
        return self._adjoint_impl(sparse_kspace)

    # ------------------------------------------------------------------
    # adjoint: subspace coefficient images → sparse k-space
    # ------------------------------------------------------------------
    def adjoint(self, coeffs: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(coeffs)

    # ==================================================================
    # implementation
    # ==================================================================
    def _adjoint_impl(self, sparse_kspace: torch.Tensor) -> torch.Tensor:
        """Sparse (..., C, *natural) → coefficients (K, *image_shape).

        Input may have any number of leading batch dims with product 1
        (we squeeze them).  Output has no batch dim.
        """
        base = self._base
        nat = base.natural_shape
        nat_ndim = len(nat)

        # Squeeze leading batch dims to simplify (assume product=1, typical use).
        if sparse_kspace.ndim > 1 + nat_ndim:
            lead = sparse_kspace.shape[:-(1 + nat_ndim)]
            if any(int(s) != 1 for s in lead):
                raise NotImplementedError(
                    "Multi-element leading batch not supported in "
                    f"SubspaceSparseFFT (got lead={lead})"
                )
            sparse_kspace = sparse_kspace.reshape(*sparse_kspace.shape[-(1 + nat_ndim):])
        if sparse_kspace.ndim != 1 + nat_ndim:
            raise ValueError(
                f"Expected (C, *natural)={('C',) + tuple(nat)}; got {tuple(sparse_kspace.shape)}"
            )

        device = sparse_kspace.device
        dtype = sparse_kspace.dtype
        n_coils = int(sparse_kspace.shape[0])

        if base.smaps is None:
            raise NotImplementedError("SubspaceSparseFFT requires base_op.smaps")
        smaps = base.smaps.to(device, dtype=dtype)

        basis = self.basis.to(device, dtype=dtype)  # (K, T)
        T = self.T
        K = self.K

        # Broadcast view of basis along T axis inside natural space.
        # basis (K, T) → (K, *phi_shape) so it lines up with (K, *nat).
        phi_shape = [1] * nat_ndim
        phi_shape[self._t_axis_in_nat] = T

        output = torch.zeros(K, *base.image_shape, dtype=dtype, device=device)

        # Full density compensation w = sqrt_w**2:
        # _scatter_ifft_crop_batch applies sqrt_w internally, so we apply
        # the matching pre_w (= sqrt_w in natural / interpolate-output order)
        # on the sparse side.
        pre_w = base.sqrt_weights[base.inv_perm].to(device=device, dtype=dtype).view(*nat)

        for c in range(n_coils):
            sw_c = sparse_kspace[c] * pre_w  # (*nat) pre-weighted
            # Build (K, *nat): basis[k, t] * sw_c[..., t, ...]
            weighted = basis.view(K, *phi_shape) * sw_c.unsqueeze(0)
            weighted_flat = weighted.reshape(K, -1)  # (K, n_samples)

            # ONE batched K-IFFT + center-crop (uses scatter with sqrt_w internally).
            imgs = base._scatter_ifft_crop_batch(weighted_flat)  # (K, *image)

            # Fused multiply-accumulate into output.
            output.addcmul_(imgs, smaps[c].conj().unsqueeze(0))

        return output

    def _forward_impl(self, coeffs: torch.Tensor) -> torch.Tensor:
        """Coefficients (K, *image_shape) → sparse (1, C, *natural)."""
        base = self._base
        nat = base.natural_shape
        nat_ndim = len(nat)

        if coeffs.shape[0] != self.K:
            raise ValueError(f"coeffs.shape[0]={coeffs.shape[0]} != K={self.K}")
        if tuple(int(s) for s in coeffs.shape[1:]) != tuple(base.image_shape):
            raise ValueError(
                f"coeffs spatial {tuple(coeffs.shape[1:])} != image_shape {base.image_shape}"
            )

        device = coeffs.device
        dtype = coeffs.dtype

        if base.smaps is None:
            raise NotImplementedError("SubspaceSparseFFT requires base_op.smaps")
        smaps = base.smaps.to(device, dtype=dtype)
        n_coils = int(smaps.shape[0])

        basis_conj = self.basis.conj().to(device, dtype=dtype)  # (K, T)
        T = self.T
        K = self.K

        phi_shape = [1] * nat_ndim
        phi_shape[self._t_axis_in_nat] = T

        # Output: (1, C, *natural) — leading batch dim of size 1 to match
        # the (1, C, T, k1, k0, kw) input convention used by the adjoint.
        output = torch.empty(1, n_coils, *nat, dtype=dtype, device=device)

        # Symmetric density compensation (matches _adjoint_impl).
        pre_w = base.sqrt_weights[base.inv_perm].to(device=device, dtype=dtype).view(*nat)

        for c in range(n_coils):
            coil_imgs = coeffs * smaps[c].unsqueeze(0)  # (K, *image)

            # ONE batched K-FFT + pad + gather (sqrt_w applied internally).
            gathered = base._fft_pad_gather_batch(coil_imgs)  # (K, n_samples)
            gathered_nat = gathered.reshape(K, *nat)

            # Reduce K dim weighted by basis_conj broadcast over T axis.
            ksp_c = (basis_conj.view(K, *phi_shape) * gathered_nat).sum(dim=0)
            output[0, c] = ksp_c * pre_w

        return output

    def normal(self, coeffs):
        if self.toeplitz:
            if self._toep_op is None:
                from .._toep._sub_toep import SubspaceToeplitzOp
                self._toep_op = SubspaceToeplitzOp(
                    self, device=self._base.device,
                )
            return self._toep_op(coeffs)
        return self._adjoint_impl(self._forward_impl(coeffs))

    def __call__(self, x, adjoint=False):
        if adjoint:
            return self.adjoint(x)
        return self.forward(x)
