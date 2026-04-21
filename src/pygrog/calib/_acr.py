"""Autocalibration region extraction subroutines."""

__all__ = ["extract_acr"]

import torch
from numpy.typing import NDArray
from mrinufft._array_compat import with_torch

from .._utils import rescale_coords


@with_torch
def extract_acr(
    data: NDArray[complex],
    cal_width: int = 20,
    coords: NDArray[float] | None = None,
    weights: NDArray[float] | None = None,
    shape: int | None = None,
    ndim: int | None = None,
    mask: NDArray[bool] | None = None,
) -> (
    tuple[NDArray[complex], NDArray[bool] | None]
    | tuple[NDArray[complex], NDArray[float], NDArray[float] | None]
):
    """
    Extract calibration region from input dataset.

    Parameters
    ----------
    data : NDArray
        Input k-space dataset of shape ``(*others, coils, k2, k1, k0)``.
    cal_width : int, optional
        Calibration region size. The default is ``24``.
    coords : NDArray[float], optional
        Fourier domain coordinate array of shape ``(*others, 1, k2, k1, k0, ndim)``.
        Required for Non Cartesian datasets.
        The default is ``None``.
    weights : NDArray[float], optional
        K-space density compensation of shape ``(*others, 1, k2, k1, k0)``.
        The default is ``None``.
    shape : int, optional
        Matrix size of shape ``(ndim,)``.
        Required for Non Cartesian datasets. The default is ``None``.
    ndim : int, optional
        Number of spatial dimensions. Required for Cartesian datasets.
        The default is ``None``.
    mask : NDArray[bool], optional
        Sampling mask for Cartesian datasets of shape ``(*others, 1, k2, k1, k0)``.

    Raises
    ------
    ValueError
        If ``ndim`` is not provided for Cartesian datasets (``trajectory = None``) or
        ``shape`` is not provided for Non Cartesian datasets (``trajectory != None``).

    Returns
    -------
    cal_data : NDArray[complex]
        Calibration dataset of shape ``(*others_cal, coils, k2_cal, k1_cal, k0_cal)``
    cal_mask : NDArray[bool], optional
        Sampling mask for calibration dataset of shape ``(*others_cal, 1, k2_cal, k1_cal, k0_cal)`` (Cartesian).
    cal_coords : NDArray[float], optional
        Trajectory for calibration dataset of shape ``(*others_cal, coi1ls, k2_cal, k1_cal, k0_cal, ndim)`` (Non Cartesian).
    cal_weights : NDArray[float], optional
        Density compensation for calibration dataset of shape ``(*others_cal, 1, k2_cal, k1_cal, k0_cal)`` (Non Cartesian).

    """
    if coords is None:
        if ndim is None:
            raise ValueError(
                "Please provide number of spatial dimensions for Cartesian datasets"
            )
        shape = list(data.shape[-ndim:])
        cal_shape = list(data.shape[:-ndim]) + ndim * [cal_width]
        _data = _center_crop(data, cal_shape)
        if mask is not None:
            _mask = _center_crop(mask, ndim * [cal_width])
            return _data, _mask
        return _data
    else:
        if shape is None:
            raise ValueError("Please provide matrix size for Non Cartesian datasets")

        # get indexes for calibration samples
        coords = rescale_coords(
            coords, shape
        )  # enforce scaling between (-0.5 * npix, 0.5 * npix)
        if torch.allclose(coords[..., -1], torch.round(coords[..., -1])):
            stack = True
            cal_idx = (coords[..., :2] ** 2).sum(dim=-1) ** 0.5 <= (0.5 * cal_width)
            cal_idx = cal_idx.reshape(-1, cal_idx.shape[-1])
            cal_idx = cal_idx.prod(dim=0)
            cal_idx_z = coords[..., -1].abs() <= (0.5 * cal_width)
            cal_idx_z = cal_idx_z.reshape(-1, cal_idx.shape[-1])
            cal_idx_z = cal_idx_z.prod(dim=-1).bool()
        else:
            stack = False
            cal_idx = (coords**2).sum(dim=-1) ** 0.5 <= (0.5 * cal_width)
            cal_idx = cal_idx.reshape(-1, cal_idx.shape[-1])
            cal_idx = cal_idx.prod(dim=0)
        cal_idx = cal_idx.bool()

        # select data
        _data = data[..., cal_idx]
        _coords = coords[..., cal_idx, :]
        if weights is not None:
            _weights = weights[..., cal_idx]
        else:
            _weights = None

        if stack:
            _data = _data[..., cal_idx_z, :]
            _coords = _coords[..., cal_idx_z, :, :]
            if weights is not None:
                _weights = _weights[..., cal_idx_z, :]
            else:
                _weights = None

        return _data, _coords, _weights


def _center_crop(arr, oshape):
    """Center crop an array to the target shape (works for numpy and torch)."""
    oshape = list(oshape)
    # Pad oshape to match arr ndim (keep leading dims unchanged)
    if len(oshape) < arr.ndim:
        oshape = list(arr.shape[: arr.ndim - len(oshape)]) + oshape
    slices = []
    for i_len, o_len in zip(arr.shape, oshape):
        if o_len <= i_len:
            start = (i_len - o_len) // 2
            slices.append(slice(start, start + o_len))
        else:
            slices.append(slice(None))
    return arr[tuple(slices)]
