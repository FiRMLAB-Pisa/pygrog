/**
 * grog_interp.h — GROG interpolation kernel (OpenMP).
 *
 * Applies pre-computed GRAPPA kernel table to perform batched
 * matrix-vector products for GROG gridding.
 *
 * For each target grid point, gathers source data from one or more
 * non-Cartesian samples, applies the corresponding GRAPPA kernel
 * (matrix multiply), and accumulates into the output.
 */
#pragma once

#include <complex>
#include <cstdint>

namespace pygrog {

/**
 * GROG interpolation: apply kernel table to source data.
 *
 * For target point t:
 *   output[:, t] = sum_s  kernel_table[kernel_idx[s], :, :] @ data[:, source_idx[s]]
 *
 * where the sum is over all sources assigned to target t.
 *
 * @param output        Output gridded data, shape (n_coils, n_targets), complex.
 * @param data          Input non-Cartesian data, shape (n_coils, n_sources), complex.
 * @param source_idx    Source sample indices, length n_pairs.
 * @param target_idx    Target grid indices, length n_pairs.
 * @param kernel_idx    Kernel table index for each pair, length n_pairs.
 * @param kernel_table  Pre-computed GRAPPA kernels, shape (n_kernels, n_coils, n_coils).
 * @param n_coils       Number of coils.
 * @param n_targets     Number of target grid points.
 * @param n_pairs       Number of (source, target) pairs.
 * @param n_kernels     Number of kernels in the table.
 */
template <typename T>
void grog_interpolate(
    std::complex<T>* output,
    const std::complex<T>* data,
    const int64_t* source_idx,
    const int64_t* target_idx,
    const int64_t* kernel_idx,
    const std::complex<T>* kernel_table,
    int64_t n_coils,
    int64_t n_targets,
    int64_t n_pairs,
    int64_t n_kernels)
{
    // Process each (source, target) pair
    // Note: multiple sources can map to same target, so we accumulate.
    // Parallelisation requires atomic adds (or per-thread buffers).
    // For now, sequential for correctness; OpenMP with reduction later.
    for (int64_t p = 0; p < n_pairs; ++p) {
        int64_t s = source_idx[p];
        int64_t t = target_idx[p];
        int64_t k = kernel_idx[p];

        if (k < 0 || k >= n_kernels) continue;

        // kernel_table layout: (n_kernels, n_coils, n_coils) row-major
        const std::complex<T>* K = kernel_table + k * n_coils * n_coils;

        // output[:, t] += K @ data[:, s]
        for (int64_t co = 0; co < n_coils; ++co) {
            std::complex<T> val(0, 0);
            for (int64_t ci = 0; ci < n_coils; ++ci) {
                val += K[co * n_coils + ci] * data[ci * n_targets + s];
                // Note: data layout is (n_coils, n_samples), so data[ci][s]
                // = data[ci * n_samples + s]. Caller passes n_samples as n_targets
                // when data is the full source array.
            }
            output[co * n_targets + t] += val;
        }
    }
}

/**
 * GROG interpolation with OpenMP parallelism over targets.
 *
 * Requires that pairs are sorted by target_idx so we can process
 * each target independently.
 *
 * @param pair_starts   Start offset in pairs for each target, length n_targets+1.
 */
template <typename T>
void grog_interpolate_sorted(
    std::complex<T>* output,
    const std::complex<T>* data,
    const int64_t* source_idx,
    const int64_t* target_idx,
    const int64_t* kernel_idx,
    const std::complex<T>* kernel_table,
    const int64_t* pair_starts,
    int64_t n_coils,
    int64_t n_targets,
    int64_t n_sources,
    int64_t n_kernels)
{
    #pragma omp parallel for schedule(dynamic, 64)
    for (int64_t t = 0; t < n_targets; ++t) {
        int64_t p_start = pair_starts[t];
        int64_t p_end = pair_starts[t + 1];

        for (int64_t p = p_start; p < p_end; ++p) {
            int64_t s = source_idx[p];
            int64_t k = kernel_idx[p];

            if (k < 0 || k >= n_kernels) continue;

            const std::complex<T>* K = kernel_table + k * n_coils * n_coils;

            for (int64_t co = 0; co < n_coils; ++co) {
                std::complex<T> val(0, 0);
                for (int64_t ci = 0; ci < n_coils; ++ci) {
                    val += K[co * n_coils + ci] * data[ci * n_sources + s];
                }
                output[co * n_targets + t] += val;
            }
        }
    }
}

} // namespace pygrog
