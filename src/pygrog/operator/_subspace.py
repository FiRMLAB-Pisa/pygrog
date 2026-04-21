"""Subspace projection decorator for SparseFFT.

Wraps a :class:`SparseFFT` operator with low-rank temporal/contrast
subspace projection, following the mri-nufft convention.

The subspace basis ``Phi`` has shape ``(K, T)`` where ``K`` is the
subspace rank and ``T`` is the number of temporal frames or contrasts.

Data conventions (matching mri-nufft)::

    Image space:  (K, *image_shape)    — subspace coefficients
    K-space:      (T, n_coils, n_samples) — time-domain k-space

Usage
-----
::

    from pygrog.operator import SparseFFT, with_subspace

    base = SparseFFT(plan=grog.plan, smaps=smaps)
    sub_op = with_subspace(base, subspace_basis=phi)
    coeffs = sub_op.forward(kspace)   # (T, C, N) -> (K, *img)
    kspace = sub_op.adjoint(coeffs)   # (K, *img) -> (T, C, N)
"""

__all__ = ["with_subspace"]

import torch


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

        coeffs = torch.zeros(self.K, *self.image_shape, dtype=dtype, device=device)
        for t in range(self.T):
            img_t = self._base.forward(sparse_kspace[t])  # (*image_shape)
            # Project onto subspace: coeffs[k] += conj(basis[k, t]) * img_t
            for k in range(self.K):
                coeffs[k] += basis[k, t].conj() * img_t

        return coeffs

    def adjoint(self, coeffs):
        """Subspace coefficient images → time-domain k-space.

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

        n_coils = self.smaps.shape[0] if self.smaps is not None else 1
        n_samples = (
            self._base.indices.shape[0]
            if hasattr(self._base, "indices")
            else self._base._base.indices.shape[0]
        )

        output = torch.zeros(self.T, n_coils, n_samples, dtype=dtype, device=device)
        for k in range(self.K):
            ksp_k = self._base.adjoint(coeffs[k])  # (n_coils, n_samples)
            # Back-project: output[t] += basis[k, t] * ksp_k
            for t in range(self.T):
                output[t] += basis[k, t] * ksp_k

        return output

    def normal(self, coeffs):
        """Normal operator: ``A^H A x``."""
        return self.forward(self.adjoint(coeffs))

    def __call__(self, x, adjoint=False):
        if adjoint:
            return self.adjoint(x)
        return self.forward(x)
