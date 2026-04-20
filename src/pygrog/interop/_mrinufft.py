"""mri-nufft FourierOperatorBase adapter for SparseFFT."""

__all__ = ["GrogFourierOp"]

import numpy as np
import torch

from ..operator._sparse_fft import SparseFFT


class GrogFourierOp:
    """
    Adapter exposing SparseFFT with the mri-nufft FourierOperator interface.

    Parameters
    ----------
    sparse_fft : SparseFFT
        A configured SparseFFT operator instance.

    """

    def __init__(self, sparse_fft: SparseFFT):
        self._op = sparse_fft
        self.shape = sparse_fft.image_shape
        self.n_samples = sparse_fft.indices.shape[0]

    def op(self, image: np.ndarray) -> np.ndarray:
        """Forward: image -> k-space."""
        img_t = torch.as_tensor(image).unsqueeze(0)  # add coil dim
        ksp = self._op.adjoint(img_t)
        return ksp.squeeze(0).numpy(force=True)

    def adj_op(self, kspace: np.ndarray) -> np.ndarray:
        """Adjoint: k-space -> image."""
        ksp_t = torch.as_tensor(kspace).unsqueeze(0)
        img = self._op.forward(ksp_t)
        return img.squeeze(0).numpy(force=True)
