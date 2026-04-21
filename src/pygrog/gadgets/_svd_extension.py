"""SVD-based subspace projection for temporal acceleration."""

__all__ = ["SubspaceProjection"]

import torch


class SubspaceProjection:
    """
    Low-rank temporal subspace projection via truncated SVD.

    Given multi-frame data ``(n_coils, n_frames, *spatial)``, projects
    onto the leading ``n_components`` right singular vectors.

    Parameters
    ----------
    n_components : int
        Number of subspace components to retain.

    """

    def __init__(self, n_components: int):
        self.n_components = n_components
        self._basis = None

    def fit(self, calib_data: torch.Tensor) -> "SubspaceProjection":
        """
        Compute temporal basis from calibration data.

        Parameters
        ----------
        calib_data : torch.Tensor
            Calibration time-series of shape ``(n_frames, n_spatial)``.

        Returns
        -------
        SubspaceProjection
            Self, with fitted basis.

        """
        # calib_data: (n_frames, n_spatial)
        U, S, Vh = torch.linalg.svd(calib_data, full_matrices=False)
        # basis: (n_components, n_frames) — rows are temporal basis vectors
        # U has shape (n_frames, min(n_frames, n_spatial)); take leading columns
        self._basis = U[:, : self.n_components].T.conj()
        return self

    @property
    def basis(self) -> torch.Tensor:
        """Temporal basis matrix of shape ``(n_components, n_frames)``."""
        if self._basis is None:
            raise RuntimeError("Call fit() first.")
        return self._basis

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Project multi-frame data onto subspace.

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
        flat = data.reshape(data.shape[0], -1)  # (n_frames, n_spatial)
        coeff = self.basis @ flat  # (n_components, n_spatial)
        return coeff.reshape(self.n_components, *spatial_shape)

    def adjoint(self, coefficients: torch.Tensor) -> torch.Tensor:
        """
        Expand subspace coefficients back to frame domain.

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
        # basis.H: (n_frames, n_components)
        frames = self.basis.conj().T @ flat
        return frames.reshape(-1, *spatial_shape)
