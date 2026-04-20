/**
 * scatter_add.h — SIMD-friendly scatter-add for sparse FFT gridding.
 *
 * Accumulates complex values into a flat grid using index arrays:
 *   grid[indices[i]] += weights[i] * data[i]
 *
 * Falls back to scalar code if OpenMP SIMD is unavailable.
 */
#pragma once

#include <complex>
#include <cstdint>
#include <vector>

namespace pygrog {

/**
 * Scatter-add weighted complex data into a grid.
 *
 * @param grid       Output flat grid of size grid_size (complex).
 * @param data       Input data of size n_samples (complex).
 * @param indices    Flat target indices of size n_samples.
 * @param weights    Per-sample real weights of size n_samples.
 * @param n_samples  Number of samples to scatter.
 * @param grid_size  Total number of grid points.
 */
template <typename T>
void scatter_add(
    std::complex<T>* grid,
    const std::complex<T>* data,
    const int64_t* indices,
    const T* weights,
    int64_t n_samples,
    int64_t grid_size)
{
    // Note: scatter-add is inherently non-parallelisable over samples
    // because multiple samples may map to the same grid point.
    // We use simple sequential accumulation which is cache-friendly
    // when indices are locally sorted.
    for (int64_t i = 0; i < n_samples; ++i) {
        int64_t idx = indices[i];
        if (idx >= 0 && idx < grid_size) {
            grid[idx] += weights[i] * data[i];
        }
    }
}

/**
 * Gather: extract values from grid at given indices.
 *
 * @param output     Output data of size n_samples (complex).
 * @param grid       Input flat grid of size grid_size (complex).
 * @param indices    Flat source indices of size n_samples.
 * @param n_samples  Number of samples.
 * @param grid_size  Total number of grid points.
 */
template <typename T>
void gather(
    std::complex<T>* output,
    const std::complex<T>* grid,
    const int64_t* indices,
    int64_t n_samples,
    int64_t grid_size)
{
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n_samples; ++i) {
        int64_t idx = indices[i];
        if (idx >= 0 && idx < grid_size) {
            output[i] = grid[idx];
        } else {
            output[i] = std::complex<T>(0, 0);
        }
    }
}

} // namespace pygrog
