"""Subspace projection gadget and SparseFFT decorator.

Provides two complementary views of low-rank temporal/contrast subspace
compression:

* :class:`SubspaceProjection` — standalone projection via truncated SVD,
  operates on dense (n_frames, *spatial) tensors.
* :func:`with_subspace` / :class:`SubspaceSparseFFT` — decorator that wraps
  a :class:`~pygrog.operator.SparseFFT` (or
  :class:`~pygrog.gadgets.OffResonanceSparseFFT`) and fuses the subspace
  projection directly into the k-space ↔ image transform.

The subspace basis ``Phi`` has shape ``(K, T)`` where ``K`` is the subspace
rank and ``T`` is the number of temporal frames or contrasts.

Data conventions (matching mri-nufft)::

    Image space:  (K, *image_shape)    — subspace coefficients
    K-space:      (T, n_coils, n_samples) — time-domain k-space
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
        """Compute temporal basis from calibration data.

        Parameters
        ----------
        calib_data : torch.Tensor
            Calibration time-series of shape ``(n_frames, n_spatial)``.

        Returns
        -------
        SubspaceProjection
            Self, with fitted basis.
        """
        U, _S, _Vh = torch.linalg.svd(calib_data, full_matrices=False)
        # basis: (n_components, n_frames) — rows are temporal basis vectors
        self._basis = U[:, : self.n_components].T.conj()
        return self

    @property
    def basis(self) -> torch.Tensor:
        """Temporal basis matrix of shape ``(n_components, n_frames)``."""
        if self._basis is None:
            raise RuntimeError("Call fit() first.")
        return self._basis

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Project multi-frame data onto subspace.

        Parameters
        ----------
        data : torch.Tensor
            Shape ``(n_frames, *spatial)``.

        Returns
        -------
        torch.Tensor
            Shape ``(n_components, *spatial)``.
        """
        spatial_shape = data.shape[1:]
        flat = data.reshape(data.shape[0], -1)
        coeff = self.basis @ flat  # (n_components, n_spatial)
        return coeff.reshape(self.n_components, *spatial_shape)

    def adjoint(self, coefficients: torch.Tensor) -> torch.Tensor:
        """Expand subspace coefficients back to frame domain.

        Parameters
        ----------
        coefficients : torch.Tensor
            Shape ``(n_components, *spatial)``.

        Returns
        -------
        torch.Tensor
            Shape ``(n_frames, *spatial)``.
        """
        spatial_shape = coefficients.shape[1:]
        flat = coefficients.reshape(self.n_components, -1)
        frames = self.basis.conj().T @ flat
        return frames.reshape(-1, *spatial_shape)


# =====================================================================
# SparseFFT decorator
# =====================================================================
def with_subspace(base_op, subspace_basis):
    """Wrap a SparseFFT operator with subspace projection.

    Parameters
    ----------
    base_op : SparseFFT (or OffResonanceSparseFFT)
        Base operator with ``forward`` / ``adjoint`` API.
    subspace_basis : array-like, complex
        ``(K, T)`` subspace basis matrix.

    Returns
    -------
    SubspaceSparseFFT
    """
    return SubspaceSparseFFT(base_op, subspace_basis)


class SubspaceSparseFFT:
    """SparseFFT with low-rank subspace projection.

    Parameters
    ----------
    base_op : SparseFFT-like
        Underlying operator.
    subspace_basis : torch.Tensor
        ``(K, T)`` basis.  Columns span the temporal subspace.
    """

    def __init__(self, base_op, subspace_basis):
        self._base = base_op
        self.basis = torch.as_tensor(subspace_basis)  # (K, T)
        self.K, self.T = self.basis.shape

        self.grid_shape = base_op.grid_shape
        self.image_shape = base_op.image_shape
        self.smaps = getattr(base_op, "smaps", None)

    def forward(self, sparse_kspace):
        """Time-domain k-space → subspace coefficient images.

        Correct algorithm: loop over K (not T).  For each coefficient i,
        weight all T frames' k-space by ``phi[i,:]^*`` and apply one
        IFFT/SparseFFT call.  This costs K transforms instead of T, giving
        a ~T/K speedup (×100 for T=500, K=5).

        Parameters
        ----------
        sparse_kspace : torch.Tensor
            ``(T, n_coils, n_samples)`` complex.

        Returns
        -------
        torch.Tensor
            ``(K, *image_shape)`` subspace coefficient images.
        """
        device = sparse_kspace.device
        dtype = sparse_kspace.dtype
        basis = self.basis.to(device, dtype=dtype)  # (K, T)

        coeff_imgs = []
        for i in range(self.K):
            # Weight all frames by phi[i,:]^* → (T,1,1) broadcast → sum over T
            phi_i = basis[i].conj()  # (T,)
            y_i = (phi_i[:, None, None] * sparse_kspace).sum(0)  # (n_coils, n_samples)
            img_i = self._base.forward(y_i)  # (*image_shape,)
            coeff_imgs.append(img_i)

        return torch.stack(coeff_imgs, dim=0)  # (K, *image_shape)

    def adjoint(self, coeffs):
        """Subspace coefficient images → time-domain k-space.

        Correct algorithm: loop over K (not T).  For each coefficient i,
        apply one FFT/SparseFFT call and broadcast the result across all T
        frames scaled by ``phi[i,:]``.  This costs K transforms instead of T.

        Parameters
        ----------
        coeffs : torch.Tensor
            ``(K, *image_shape)`` complex.

        Returns
        -------
        torch.Tensor
            ``(T, n_coils, n_samples)`` complex.
        """
        device = coeffs.device
        dtype = coeffs.dtype
        basis = self.basis.to(device, dtype=dtype)  # (K, T)

        kspace_out: torch.Tensor | None = None
        for i in range(self.K):
            ksp_i = self._base.adjoint(coeffs[i])  # (n_coils, n_samples)
            phi_i = basis[i]  # (T,) — no conjugate for forward/synthesis
            contrib = phi_i[:, None, None] * ksp_i[None, :, :]  # (T, n_coils, n_samples)
            kspace_out = contrib if kspace_out is None else kspace_out + contrib

        assert kspace_out is not None
        return kspace_out  # (T, n_coils, n_samples)

    def normal(self, coeffs):
        """Normal operator: ``A^H A x``."""
        return self.forward(self.adjoint(coeffs))

    def __call__(self, x, adjoint=False):
        if adjoint:
            return self.adjoint(x)
        return self.forward(x)
