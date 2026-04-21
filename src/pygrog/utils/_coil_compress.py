"""Preprocessing utilities for k-space data."""

__all__ = ["coil_compress"]

from numpy.typing import NDArray

from mrinufft.extras.smaps import coil_compression


def coil_compress(
    kspace_data: NDArray,
    n_coils: int | float,
    traj: NDArray | None = None,
    krad_thresh: float | None = None,
) -> tuple[NDArray, NDArray]:
    """
    Coil compression using principal component analysis on k-space data.

    Thin wrapper around :func:`mrinufft.extras.smaps.coil_compression`.

    Parameters
    ----------
    kspace_data : NDArray
        Multi-coil k-space data of shape ``(n_coils, n_samples)``.
    n_coils : int | float
        Number of virtual coils to retain (if ``int``), or energy
        threshold (if ``float`` between 0 and 1).
    traj : NDArray, optional
        Sampling trajectory of shape ``(n_samples, n_dims)``.
    krad_thresh : float, optional
        Relative k-space radius threshold for calibration region selection.

    Returns
    -------
    compressed_data : NDArray
        Coil-compressed data of shape ``(n_virtual_coils, n_samples)``.
    compression_matrix : NDArray
        Compression matrix of shape ``(n_virtual_coils, n_coils)``.

    """
    return coil_compression(kspace_data, n_coils, traj=traj, krad_thresh=krad_thresh)
