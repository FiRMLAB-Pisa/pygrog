"""
Fast Binning Operations for PyGROG
==================================

This example demonstrates how to use the fast SIMD-accelerated binning operations
in PyGROG for high-performance complex array accumulation.

The fast binning implementation provides significant speedups over numpy's add.at
by using:
- Platform-specific SIMD instructions (AVX-512, AVX, SSE)
- Multithreaded processing with automatic chunk management
- Optimized complex number arithmetic

Usage Examples
--------------
"""

import numpy as np
import time
from pygrog.operator import fast_binning_add_at, detect_simd_level, benchmark_binning


def basic_usage_example():
    """Basic usage of fast binning."""
    print("=== Basic Usage Example ===")

    # Detect SIMD capabilities
    simd_level = detect_simd_level()
    print(f"SIMD level: {simd_level}")

    # Create sample data
    n_points = 50000
    n_bins = 5000

    # Complex points to be binned
    points = np.random.randn(n_points).astype(np.complex64)
    points += 1j * np.random.randn(n_points).astype(np.float32)

    # Real weights
    weights = np.random.rand(n_points).astype(np.float32)

    # Bin indices (must be within [0, n_bins))
    indices = np.random.randint(0, n_bins, n_points, dtype=np.uint64)

    # Output bins (modified in-place)
    bins = np.zeros(n_bins, dtype=np.complex64)

    # Perform fast binning: bins[indices] += points * weights
    start_time = time.perf_counter()
    fast_binning_add_at(bins, points, weights, indices)
    end_time = time.perf_counter()

    print(f"Fast binning completed in {(end_time - start_time)*1000:.2f} ms")
    print(f"Processed {n_points:,} points into {n_bins:,} bins")
    print(f"Non-zero bins: {np.count_nonzero(bins):,}")

    return bins


def comparison_with_numpy():
    """Compare performance with numpy's add.at."""
    print("\n=== Performance Comparison ===")

    n_points = 100000
    n_bins = 10000

    # Generate test data
    points = np.random.randn(n_points).astype(np.complex64)
    points += 1j * np.random.randn(n_points).astype(np.float32)
    weights = np.random.rand(n_points).astype(np.float32)
    indices = np.random.randint(0, n_bins, n_points, dtype=np.uint64)

    # Test numpy implementation
    bins_numpy = np.zeros(n_bins, dtype=np.complex64)
    start_time = time.perf_counter()
    weighted_points = points * weights
    np.add.at(bins_numpy, indices, weighted_points)
    numpy_time = time.perf_counter() - start_time

    # Test fast implementation
    bins_fast = np.zeros(n_bins, dtype=np.complex64)
    start_time = time.perf_counter()
    fast_binning_add_at(bins_fast, points, weights, indices)
    fast_time = time.perf_counter() - start_time

    # Verify results are equivalent
    difference = np.abs(bins_fast - bins_numpy).max()
    print(f"Maximum difference: {difference:.2e}")

    # Performance comparison
    speedup = numpy_time / fast_time
    print(f"NumPy time: {numpy_time*1000:.2f} ms")
    print(f"Fast time: {fast_time*1000:.2f} ms")
    print(f"Speedup: {speedup:.1f}x")

    assert difference < 1e-5, "Results should be nearly identical"
    return speedup


def detailed_benchmark():
    """Run detailed benchmark with multiple sizes."""
    print("\n=== Detailed Benchmark ===")

    sizes = [1000, 10000, 100000, 500000]

    for n_points in sizes:
        n_bins = n_points // 10
        print(f"\nBenchmarking {n_points:,} points, {n_bins:,} bins...")

        results = benchmark_binning(
            n_points=n_points, n_bins=n_bins, num_runs=5, compare_numpy=True
        )

        if "speedup" in results:
            print(f"  Speedup: {results['speedup']:.1f}x")
            print(
                f"  Fast time: {results['fast_binning_time']['mean']*1000:.2f} ± "
                f"{results['fast_binning_time']['std']*1000:.2f} ms"
            )
            print(
                f"  NumPy time: {results['numpy_time']['mean']*1000:.2f} ± "
                f"{results['numpy_time']['std']*1000:.2f} ms"
            )


def threading_behavior():
    """Demonstrate threading behavior."""
    print("\n=== Threading Behavior ===")

    from pygrog.operator import create_thread_mask

    n_points = 50000

    # Auto-determined chunking
    mask_auto = create_thread_mask(n_points)
    num_chunks_auto = len(mask_auto) // 2
    print(f"Auto chunking: {num_chunks_auto} chunks for {n_points:,} points")

    # Manual chunking
    mask_manual = create_thread_mask(n_points, num_chunks=8)
    num_chunks_manual = len(mask_manual) // 2
    print(f"Manual chunking: {num_chunks_manual} chunks")

    # Show chunk sizes
    print("Chunk sizes (auto):", end=" ")
    for i in range(0, len(mask_auto), 2):
        start, end = mask_auto[i], mask_auto[i + 1]
        print(f"{end-start}", end=" ")
    print()


def memory_layout_tips():
    """Tips for optimal memory layout."""
    print("\n=== Memory Layout Tips ===")

    n_points = 10000

    # Non-contiguous arrays (slower)
    points_nc = np.random.randn(n_points, 2).astype(np.float32)
    points_nc = points_nc[:, 0] + 1j * points_nc[:, 1]  # Non-contiguous
    weights_nc = np.random.rand(n_points * 2)[::2].astype(np.float32)  # Non-contiguous

    print(f"Points contiguous: {points_nc.flags.c_contiguous}")
    print(f"Weights contiguous: {weights_nc.flags.c_contiguous}")

    # The fast_binning function will automatically make arrays contiguous
    # but it's more efficient to start with contiguous arrays

    # Optimal layout
    points_opt = np.random.randn(n_points).astype(np.complex64)
    points_opt += 1j * np.random.randn(n_points).astype(np.float32)
    weights_opt = np.random.rand(n_points).astype(np.float32)

    print(f"Optimized points contiguous: {points_opt.flags.c_contiguous}")
    print(f"Optimized weights contiguous: {weights_opt.flags.c_contiguous}")
    print("Use contiguous arrays for best performance!")


def main():
    """Run all examples."""
    try:
        basic_usage_example()
        speedup = comparison_with_numpy()

        if speedup > 1.0:
            detailed_benchmark()

        threading_behavior()
        memory_layout_tips()

        print(f"\n=== Summary ===")
        print(f"SIMD level: {detect_simd_level()}")
        print(f"Fast binning is working correctly!")

    except ImportError as e:
        print(f"Fast binning C++ extension not available: {e}")
        print("To build the extension, run: pip install -e .")
    except Exception as e:
        print(f"Error running examples: {e}")
        raise


if __name__ == "__main__":
    main()
