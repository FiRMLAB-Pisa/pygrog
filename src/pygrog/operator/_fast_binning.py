"""
Fast SIMD-accelerated binning operations for pygrog.

This module provides high-performance binning operations using platform-specific
SIMD instructions (AVX-512, AVX, SSE) with automatic fallback to scalar operations.
"""

import numpy as np
from typing import Optional, Tuple, Union
import warnings

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
    bins: np.ndarray,
    points: np.ndarray,
    weights: np.ndarray,
    indices: np.ndarray,
    num_threads: Optional[int] = None,
    min_chunk_size: int = 1000
) -> None:
    """
    Perform fast binning operation: bins[indices] += points * weights
    
    This function implements a high-performance version of numpy's add.at
    operation for complex arrays, using SIMD instructions and multithreading.
    
    Parameters
    ----------
    bins : np.ndarray
        Output array of complex numbers (modified in-place)
        Shape: (n_bins,), dtype: complex64
    points : np.ndarray
        Input complex values to be binned
        Shape: (n_points,), dtype: complex64
    weights : np.ndarray
        Real-valued weights for each point
        Shape: (n_points,), dtype: float32
    indices : np.ndarray
        Bin indices for each point
        Shape: (n_points,), dtype: uint64 or int64
    num_threads : int, optional
        Number of threads to use. If None, uses optimal number based on data size
    min_chunk_size : int, default=1000
        Minimum chunk size for multithreading
        
    Notes
    -----
    - All arrays must be C-contiguous
    - This function modifies `bins` in-place
    - Uses platform-specific SIMD instructions (AVX-512, AVX, SSE) when available
    - Automatically handles thread-safe chunking to avoid race conditions
    
    Examples
    --------
    >>> import numpy as np
    >>> from pygrog.operator import fast_binning_add_at
    >>> 
    >>> # Create test data
    >>> n_points = 10000
    >>> n_bins = 1000
    >>> points = np.random.complex64(np.random.randn(n_points) + 1j*np.random.randn(n_points))
    >>> weights = np.random.float32(np.random.rand(n_points))
    >>> indices = np.random.randint(0, n_bins, n_points, dtype=np.uint64)
    >>> bins = np.zeros(n_bins, dtype=np.complex64)
    >>> 
    >>> # Perform fast binning
    >>> fast_binning_add_at(bins, points, weights, indices)
    """
    if not _has_fast_binning:
        # Pure Python fallback
        _fallback_binning_add_at(bins, points, weights, indices)
        return
    
    # Validate and convert input arrays
    bins = _prepare_array(bins, np.complex64, "bins")
    points = _prepare_array(points, np.complex64, "points")
    weights = _prepare_array(weights, np.float32, "weights")
    indices = _prepare_array(indices, np.uint64, "indices")
    
    # Validate shapes
    if points.shape != weights.shape or points.shape != indices.shape:
        raise ValueError("points, weights, and indices must have the same shape")
    
    if indices.max() >= len(bins):
        raise ValueError("indices contain values outside bins array bounds")
    
    # Create thread mask for optimal chunking
    thread_mask = _fast_binning.create_thread_mask(
        len(points), 
        num_chunks=0,  # Auto-determine
        min_chunk_size=min_chunk_size
    )
    
    # Call C++ implementation
    _fast_binning.fast_binning_add_at(bins, points, weights, indices, thread_mask)


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


def create_thread_mask(
    total_size: int, 
    num_chunks: int = 0,
    min_chunk_size: int = 1000
) -> np.ndarray:
    """
    Create thread mask for chunked processing.
    
    Parameters
    ----------
    total_size : int
        Total number of elements to process
    num_chunks : int, default=0
        Number of chunks. If 0, auto-determines optimal number
    min_chunk_size : int, default=1000
        Minimum chunk size
        
    Returns
    -------
    np.ndarray
        Array of (start, end) pairs, shape (2*num_chunks,)
    """
    if not _has_fast_binning:
        # Simple fallback chunking
        if num_chunks == 0:
            num_chunks = max(1, total_size // min_chunk_size)
        
        chunks = []
        chunk_size = total_size // num_chunks
        remainder = total_size % num_chunks
        
        start = 0
        for i in range(num_chunks):
            current_chunk_size = chunk_size + (1 if i < remainder else 0)
            end = start + current_chunk_size
            chunks.extend([start, end])
            start = end
            
        return np.array(chunks, dtype=np.uint64)
    
    return _fast_binning.create_thread_mask(total_size, num_chunks, min_chunk_size)


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
    bins: np.ndarray,
    points: np.ndarray, 
    weights: np.ndarray,
    indices: np.ndarray
) -> None:
    """Pure Python fallback implementation."""
    # Simple numpy implementation (not optimized)
    weighted_points = points * weights
    np.add.at(bins, indices, weighted_points)


# Performance benchmarking utility
def benchmark_binning(
    n_points: int = 100000,
    n_bins: int = 10000,
    num_runs: int = 10,
    compare_numpy: bool = True
) -> dict:
    """
    Benchmark fast binning performance.
    
    Parameters
    ----------
    n_points : int, default=100000
        Number of points to bin
    n_bins : int, default=10000
        Number of bins
    num_runs : int, default=10
        Number of benchmark runs
    compare_numpy : bool, default=True
        Whether to compare with numpy implementation
        
    Returns
    -------
    dict
        Benchmark results with timing information
    """
    import time
    
    # Generate test data
    points = np.random.randn(n_points).astype(np.complex64)
    points += 1j * np.random.randn(n_points).astype(np.float32)
    weights = np.random.rand(n_points).astype(np.float32)
    indices = np.random.randint(0, n_bins, n_points, dtype=np.uint64)
    
    results = {
        'simd_level': detect_simd_level(),
        'n_points': n_points,
        'n_bins': n_bins,
        'num_runs': num_runs
    }
    
    if _has_fast_binning:
        # Benchmark fast implementation
        times = []
        for _ in range(num_runs):
            bins = np.zeros(n_bins, dtype=np.complex64)
            start = time.perf_counter()
            fast_binning_add_at(bins, points, weights, indices)
            end = time.perf_counter()
            times.append(end - start)
        
        results['fast_binning_time'] = {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times)
        }
    
    if compare_numpy:
        # Benchmark numpy implementation
        times = []
        for _ in range(num_runs):
            bins = np.zeros(n_bins, dtype=np.complex64)
            start = time.perf_counter()
            weighted_points = points * weights
            np.add.at(bins, indices, weighted_points)
            end = time.perf_counter()
            times.append(end - start)
        
        results['numpy_time'] = {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times)
        }
        
        if _has_fast_binning:
            speedup = results['numpy_time']['mean'] / results['fast_binning_time']['mean']
            results['speedup'] = speedup
    
    return results