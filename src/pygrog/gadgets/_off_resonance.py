"""Multi-frequency interpolation (MFI) off-resonance correction."""

__all__ = ["OffResonanceCorrection"]

import torch
import numpy as np


class OffResonanceCorrection:
    """
    Off-resonance correction via multi-frequency interpolation (MFI).

    Decomposes an off-resonance field map into ``n_bins`` discrete
    frequency bins and applies per-bin phase correction during reconstruction.

    Parameters
    ----------
    field_map : torch.Tensor
        B0 field map in Hz, shape ``(*spatial)``.
    readout_time : torch.Tensor
        Per-sample readout times in seconds, shape ``(n_samples,)``.
    n_bins : int, optional
        Number of frequency bins. Default is ``15``.

    References
    ----------
    Man, L.C., Pauly, J.M., Macovski, A. (1997).
    Multi‐frequency interpolation for fast off‐resonance correction.
    Magnetic Resonance in Medicine, 37(5), 785-792.
    """

    def __init__(
        self,
        field_map: torch.Tensor,
        readout_time: torch.Tensor,
        n_bins: int = 15,
    ):
        self.field_map = field_map
        self.readout_time = readout_time
        self.n_bins = n_bins

        # Compute bin centers and interpolation coefficients
        self._setup_bins()

    def _setup_bins(self):
        """Compute MFI bin centers and spatial interpolation weights."""
        fmap = self.field_map
        f_min = fmap.min().item()
        f_max = fmap.max().item()

        # Equally spaced bin centers
        self.bin_centers = torch.linspace(
            f_min, f_max, self.n_bins, device=fmap.device, dtype=fmap.dtype
        )

        # Spatial weights: histogram-style soft assignment (nearest 2 bins)
        # shape: (n_bins, *spatial)
        diffs = fmap.unsqueeze(0) - self.bin_centers.reshape(-1, *([1] * fmap.ndim))
        bin_spacing = (f_max - f_min) / max(self.n_bins - 1, 1)
        if bin_spacing > 0:
            weights = 1.0 - torch.abs(diffs) / bin_spacing
            weights = torch.clamp(weights, min=0.0)
        else:
            weights = torch.ones_like(diffs) / self.n_bins
        self.spatial_weights = weights

    def apply(self, recon_fn, kspace: torch.Tensor) -> torch.Tensor:
        """
        Apply off-resonance-corrected reconstruction.

        Parameters
        ----------
        recon_fn : callable
            Reconstruction function mapping ``(kspace, phase)`` -> image.
            ``phase`` is a per-sample complex phase vector of shape
            ``(n_samples,)`` that should be multiplied into the k-space
            data before gridding.
        kspace : torch.Tensor
            Non-Cartesian k-space data, shape ``(n_coils, n_samples)``.

        Returns
        -------
        torch.Tensor
            Corrected image, shape ``(n_coils, *spatial)``.

        """
        device = kspace.device
        t = self.readout_time.to(device)
        result = None

        for b in range(self.n_bins):
            freq = self.bin_centers[b]
            # Phase modulation: exp(-j * 2π * freq * t)
            phase = torch.exp(-2j * np.pi * freq * t)
            img_bin = recon_fn(kspace, phase)

            # Weight by spatial coefficient
            w = self.spatial_weights[b].to(device)
            contribution = w * img_bin

            if result is None:
                result = contribution
            else:
                result = result + contribution

        return result
