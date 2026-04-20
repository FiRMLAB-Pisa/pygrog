"""
Fast scatter-add and gather operations for sparse FFT.

Uses C++ extension with OpenMP when available, falls back to numpy.
"""

__all__ = ["scatter_add", "gather"]

import warnings

import numpy as np
from numpy.typing import NDArray

try:
    from .. import _pygrog_cpp
    _has_cpp = True
except ImportError:
    _has_cpp = False


def scatter_add(
    grid: NDArray[np.complexfloating],
    data: NDArray[np.complexfloating],
    indices: NDArray[np.intp],
    weights: NDArray[np.floating],
) -> None:
    """
    Scatter-add weighted data into a flat grid: grid[indices[i]] += weights[i] * data[i].

    Parameters
    ----------
    grid : NDArray
        Output flat grid, modified in-place.
    data : NDArray
        Input data values.
    indices : NDArray
        Target grid indices.
    weights : NDArray
        Per-sample real weights.

    """
    if _has_cpp and data.dtype == np.complex64:
        _pygrog_cpp.scatter_add_f32(
            np.ascontiguousarray(grid),
            np.ascontiguousarray(data),
            np.ascontiguousarray(indices, dtype=np.int64),
            np.ascontiguousarray(weights, dtype=np.float32),
        )
    elif _has_cpp and data.dtype == np.complex128:
        _pygrog_cpp.scatter_add_f64(
            np.ascontiguousarray(grid),
            np.ascontiguousarray(data),
            np.ascontiguousarray(indices, dtype=np.int64),
            np.ascontiguousarray(weights, dtype=np.float64),
        )
    else:
        # numpy fallback
        np.add.at(grid, indices, weights * data)


def gather(
    grid: NDArray[np.complexfloating],
    indices: NDArray[np.intp],
) -> NDArray[np.complexfloating]:
    """
    Gather values from grid at given indices.

    Parameters
    ----------
    grid : NDArray
        Input flat grid.
    indices : NDArray
        Source indices.

    Returns
    -------
    NDArray
        Gathered values.

    """
    if _has_cpp and grid.dtype == np.complex64:
        return _pygrog_cpp.gather_f32(
            np.ascontiguousarray(grid),
            np.ascontiguousarray(indices, dtype=np.int64),
        )
    elif _has_cpp and grid.dtype == np.complex128:
        return _pygrog_cpp.gather_f64(
            np.ascontiguousarray(grid),
            np.ascontiguousarray(indices, dtype=np.int64),
        )
    else:
        return grid[indices]
