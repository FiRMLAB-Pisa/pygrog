"""
Test the fast binning implementation.
"""

import numpy as np
import pytest


def test_fast_binning_import():
    """Test that we can import the fast binning module."""
    try:
        from pygrog.operator import fast_binning_add_at, detect_simd_level
        return True
    except ImportError:
        pytest.skip("Fast binning C++ extension not available")


def test_fast_binning_functionality():
    """Test the fast binning functionality."""
    try:
        from pygrog.operator import fast_binning_add_at, detect_simd_level
    except ImportError:
        pytest.skip("Fast binning C++ extension not available")
    
    # Print SIMD level for debugging
    simd_level = detect_simd_level()
    print(f"SIMD level: {simd_level}")
    
    # Create test data
    n_points = 1000
    n_bins = 100
    
    points = np.random.randn(n_points).astype(np.complex64)
    points += 1j * np.random.randn(n_points).astype(np.float32)
    weights = np.random.rand(n_points).astype(np.float32)
    indices = np.random.randint(0, n_bins, n_points, dtype=np.uint64)
    
    # Test fast implementation
    bins_fast = np.zeros(n_bins, dtype=np.complex64)
    fast_binning_add_at(bins_fast, points, weights, indices)
    
    # Test against numpy reference
    bins_numpy = np.zeros(n_bins, dtype=np.complex64)
    weighted_points = points * weights
    np.add.at(bins_numpy, indices, weighted_points)
    
    # Compare results
    np.testing.assert_allclose(bins_fast, bins_numpy, rtol=1e-5, atol=1e-5)


def test_fast_binning_benchmark():
    """Test the benchmarking functionality."""
    try:
        from pygrog.operator import benchmark_binning
    except ImportError:
        pytest.skip("Fast binning C++ extension not available")
    
    results = benchmark_binning(n_points=10000, n_bins=1000, num_runs=3)
    
    assert 'simd_level' in results
    assert 'n_points' in results
    assert 'n_bins' in results
    
    if 'speedup' in results:
        print(f"Speedup: {results['speedup']:.2f}x")
        assert results['speedup'] > 0  # Should be positive


def test_edge_cases():
    """Test edge cases."""
    try:
        from pygrog.operator import fast_binning_add_at
    except ImportError:
        pytest.skip("Fast binning C++ extension not available")
    
    # Empty arrays
    bins = np.zeros(10, dtype=np.complex64)
    points = np.array([], dtype=np.complex64)
    weights = np.array([], dtype=np.float32)
    indices = np.array([], dtype=np.uint64)
    
    fast_binning_add_at(bins, points, weights, indices)
    assert np.allclose(bins, 0)
    
    # Single element
    bins = np.zeros(5, dtype=np.complex64)
    points = np.array([1.0 + 2.0j], dtype=np.complex64)
    weights = np.array([0.5], dtype=np.float32)
    indices = np.array([2], dtype=np.uint64)
    
    fast_binning_add_at(bins, points, weights, indices)
    expected = np.zeros(5, dtype=np.complex64)
    expected[2] = 0.5 + 1.0j
    
    np.testing.assert_allclose(bins, expected)


if __name__ == "__main__":
    test_fast_binning_import()
    test_fast_binning_functionality()
    test_fast_binning_benchmark()
    test_edge_cases()
    print("All tests passed!")