"""
Fast SIMD-accelerated binning operations for pygrog.

This module provides high-performance binning operations using platform-specific
SIMD instructions (AVX-512, AVX, SSE) with automatic fallback to scalar operations.
"""

__all__ = ["fast_binning_add_at", "detect_simd_level"]

import warnings

import numpy as np
from numpy.typing import NDArray

try:
    from . import _fast_binning
    _has_fast_binning = True
except ImportError:
    _has_fast_binning = False
    warnings.warn(
        "Fast binning C++ extension not available. "
        "Using pure Python implementation with reduced performance. "
        "To build the C++ extension, ensure you have: "
        "cmake, pybind11, and a C++ compiler installed.",
        RuntimeWarning,
        stacklevel=2
    )


def fast_binning_add_at(
    bins: NDArray[np.complex64],
    points: NDArray[np.complex64],
    indices: NDArray[np.uint32],
    thread_mask: NDArray[np.uint32],
) -> None:
    """
    Perform fast binning operation: bins[indices] += points
    
    This function implements a high-performance version of numpy's add.at
    operation for complex arrays, using SIMD instructions and multithreading.
    
    Parameters
    ----------
    bins : NDArray[np.complex64]
        Output array of complex numbers (modified in-place)
        Shape: (n_bins,), dtype: complex64
    points : NDArray[np.complex64]
        Input complex values to be binned
        Shape: (n_points,), dtype: complex64
    indices : NDArray[np.uint32]
        Bin indices for each point
        Shape: (n_points,), dtype: uint32
    thread_mask : NDArray[np.uint32]
        Thread mask for chunking of shape (n_threads, 2)
        
    Notes
    -----
    - All arrays must be C-contiguous
    - This function modifies `bins` in-place
    - Uses platform-specific SIMD instructions (AVX-512, AVX, SSE) when available
    - Uses thread-safe chunking to avoid race conditions
    
    """
    if not _has_fast_binning:
        # Pure Python fallback
        _fallback_binning_add_at(bins, points, indices)
        return
    
    # Validate and convert input arrays
    bins = _prepare_array(bins, np.complex64, "bins")
    points = _prepare_array(points, np.complex64, "points")
    indices = _prepare_array(indices, np.uint32, "indices")
    
    # Validate shapes
    if points.shape != indices.shape:
        raise ValueError("points and indices must have the same shape")

    if indices.max() >= len(bins):
        raise ValueError("indices contain values outside bins array bounds")
        
    # Call C++ implementation
    _fast_binning.fast_binning_add_at(bins, points, indices, thread_mask)


def detect_simd_level() -> str:
    """
    Detect the highest SIMD instruction set available on this machine.
    
    Returns
    -------
    str
        One of: "AVX512", "AVX", "SSE", "Scalar", "Unavailable"
        
    Examples
    --------
    >>> from pygrog.operator import detect_simd_level
    >>> print(f"SIMD level: {detect_simd_level()}")
    SIMD level: AVX512
    """
    if not _has_fast_binning:
        return "Unavailable (C++ extension not built)"
    return _fast_binning.detect_simd_level()


def _prepare_array(arr: np.ndarray, dtype: np.dtype, name: str) -> np.ndarray:
    """Prepare array for C++ function call."""
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")
    
    # Convert to required dtype if needed
    if arr.dtype != dtype:
        arr = arr.astype(dtype)
    
    # Ensure C-contiguous
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    
    return arr


def _fallback_binning_add_at(
    bins: NDArray[np.complex64],
    points: NDArray[np.complex64], 
    indices: NDArray[np.uint32],
) -> None:
    """Pure Python fallback implementation."""
    # Simple numpy implementation (not optimized)
    np.add.at(bins, indices, points)
