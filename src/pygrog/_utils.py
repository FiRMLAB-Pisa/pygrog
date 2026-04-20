"""Core utilities for pygrog — torch-native replacements for sigpy helpers."""

__all__ = ["resize", "rescale_coords", "normalize_axes", "estimate_shape"]

import math

import numpy as np
from numpy.typing import NDArray

import torch


def resize(input: torch.Tensor, oshape: list[int] | tuple[int]) -> torch.Tensor:
    """
    Resize tensor via center crop / zero-pad to target shape.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor.
    oshape : list[int] | tuple[int]
        Target shape.  Must have the same number of dimensions as ``input``.

    Returns
    -------
    torch.Tensor
        Resized tensor.

    """
    ishape = input.shape
    if len(ishape) != len(oshape):
        raise ValueError(
            f"Input ndim ({len(ishape)}) != output ndim ({len(oshape)})"
        )
    result = input
    for axis in range(len(oshape)):
        i_len = result.shape[axis]
        o_len = oshape[axis]
        if o_len < i_len:
            start = (i_len - o_len) // 2
            result = result.narrow(axis, start, o_len)
        elif o_len > i_len:
            pad_total = o_len - i_len
            pad_before = pad_total // 2
            pad_after = pad_total - pad_before
            pad_widths = [0, 0] * len(result.shape)
            idx = (len(result.shape) - 1 - axis) * 2
            pad_widths[idx] = pad_before
            pad_widths[idx + 1] = pad_after
            result = torch.nn.functional.pad(result, pad_widths)
    return result


def normalize_axes(
    axes: int | list[int] | tuple[int] | None, ndim: int
) -> tuple[int, ...]:
    """
    Normalize FFT axes specification.

    Parameters
    ----------
    axes : int | list[int] | tuple[int] | None
        Axes specification. ``None`` means all axes.
    ndim : int
        Number of dimensions.

    Returns
    -------
    tuple[int, ...]
        Normalized axes as positive indices.

    """
    if axes is None:
        return tuple(range(ndim))
    if np.isscalar(axes):
        axes = (axes,)
    return tuple(a % ndim for a in axes)


def rescale_coords(coords: NDArray, amp: float | NDArray) -> NDArray:
    """
    Rescale Fourier domain coordinates to desired amplitude.

    Parameters
    ----------
    coords : NDArray
        Fourier domain coordinates array of shape ``(..., ndims)``.
    amp : float | NDArray
        Output scale.  Full dynamic range ``2 * kmax``, i.e. output
        coordinates will be in ``(-0.5 * amp, 0.5 * amp)``.

    Returns
    -------
    NDArray
        Scaled coordinate array.

    """
    if isinstance(coords, torch.Tensor):
        cmax = coords.abs().reshape(-1, coords.shape[-1]).max(dim=0).values
        if np.isscalar(amp):
            amp_t = torch.full(
                (coords.shape[-1],), amp, dtype=coords.dtype, device=coords.device
            )
        else:
            amp_t = torch.as_tensor(amp, dtype=coords.dtype, device=coords.device)
        return 0.5 * amp_t * coords / cmax
    else:
        cmax = np.abs(coords).reshape(-1, coords.shape[-1]).max(axis=0)
        if np.isscalar(amp):
            amp = np.asarray([amp] * coords.shape[-1], dtype=coords.dtype)
        else:
            amp = np.asarray(amp, dtype=coords.dtype)
        return 0.5 * amp * coords / cmax


def estimate_shape(coords: NDArray) -> tuple[int, ...]:
    """
    Estimate grid shape from non-uniform coordinates.

    Parameters
    ----------
    coords : NDArray
        Coordinate array of shape ``(..., ndim)``.

    Returns
    -------
    tuple[int, ...]
        Estimated grid shape.

    """
    ndim = coords.shape[-1]
    shape = []
    for i in range(ndim):
        c = coords[..., i].ravel()
        max_val = float(np.max(np.abs(c)))
        n = max(int(2 * math.ceil(max_val)), 1)
        shape.append(n)
    return tuple(shape)