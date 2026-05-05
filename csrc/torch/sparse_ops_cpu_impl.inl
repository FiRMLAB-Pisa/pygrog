/**
 * sparse_ops_cpu_impl.inl — CPU scatter/gather implementations.
 *
 * Included from arch-specific translation units (baseline, AVX2, AVX-512).
 * The includer must  #define SPARSE_OPS_NS  to a unique namespace name
 * before including this file.
 *
 * scatter_add assumes **sorted indices** and partitions work across OMP
 * threads at "clean" boundaries (where the index changes) so that no two
 * threads ever write to the same grid cell — **zero extra memory and zero
 * synchronization**.
 *
 * gather is embarrassingly parallel.  With sorted indices the reads are
 * sequential → prefetch-friendly.
 *
 * Both loops operate on raw float* so the compiler can auto-vectorise at
 * whatever ISA level the TU targets.
 */

#ifndef SPARSE_OPS_NS
#error "Define SPARSE_OPS_NS before including sparse_ops_cpu_impl.inl"
#endif

#include <torch/extension.h>

#include <cstdint>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace SPARSE_OPS_NS {

// -------------------------------------------------------------------------
// scatter_add — sorted indices, clean-partition across threads
// -------------------------------------------------------------------------
void scatter_add_impl(
    torch::Tensor grid,              // (grid_size,) complex, modified in-place
    const torch::Tensor& data,       // (N,) complex
    const torch::Tensor& indices,    // (N,) int64 — assumed sorted ascending
    const torch::Tensor& weights)    // (N,) float
{
    const int64_t N = data.size(0);
    const int64_t grid_size = grid.size(0);

    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif
    if (N < 4096) n_threads = 1;

    // indices is always int64 — declare i_ptr here so the partition loop
    // (below, outside AT_DISPATCH) can use it without type dispatch.
    const auto* i_ptr = indices.data_ptr<int64_t>();

    if (n_threads <= 1) {
        AT_DISPATCH_COMPLEX_TYPES(data.scalar_type(), "scatter_add_cpu", [&] {
            using real_t = typename scalar_t::value_type;
            auto* g_ptr        = grid.data_ptr<scalar_t>();
            const auto* d_ptr  = data.data_ptr<scalar_t>();
            const auto* w_ptr  = weights.data_ptr<real_t>();
            for (int64_t n = 0; n < N; ++n) {
                const int64_t idx = i_ptr[n];
                if (idx >= 0 && idx < grid_size) {
                    auto* gp = reinterpret_cast<real_t*>(g_ptr + idx);
                    const auto* dp = reinterpret_cast<const real_t*>(d_ptr + n);
                    const real_t w = w_ptr[n];
                    gp[0] += w * dp[0];
                    gp[1] += w * dp[1];
                }
            }
        });
        return;
    }

    // Find clean partition boundaries — uses i_ptr declared above.
    // Each boundary lands where the index value changes so that disjoint
    // threads always write to disjoint grid cells (no locks needed).
    std::vector<int64_t> bounds(n_threads + 1);
    bounds[0] = 0;
    bounds[n_threads] = N;
    for (int t = 1; t < n_threads; ++t) {
        int64_t target = N * t / n_threads;
        while (target < N && target > 0 && i_ptr[target] == i_ptr[target - 1])
            ++target;
        bounds[t] = target;
    }

    // #pragma omp parallel MUST be outside AT_DISPATCH_COMPLEX_TYPES: the C
    // preprocessor strips #pragma from macro arguments and emits it before
    // the macro expansion, which would make n_threads / typed vars invisible.
#ifdef _OPENMP
    #pragma omp parallel num_threads(n_threads)
#endif
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        const int64_t start = bounds[tid];
        const int64_t end   = bounds[tid + 1];
        AT_DISPATCH_COMPLEX_TYPES(data.scalar_type(), "scatter_add_cpu", [&] {
            using real_t = typename scalar_t::value_type;
            auto* g_ptr        = grid.data_ptr<scalar_t>();
            const auto* d_ptr  = data.data_ptr<scalar_t>();
            const auto* w_ptr  = weights.data_ptr<real_t>();
            for (int64_t n = start; n < end; ++n) {
                const int64_t idx = i_ptr[n];
                if (idx >= 0 && idx < grid_size) {
                    auto* gp = reinterpret_cast<real_t*>(g_ptr + idx);
                    const auto* dp = reinterpret_cast<const real_t*>(d_ptr + n);
                    const real_t w = w_ptr[n];
                    gp[0] += w * dp[0];
                    gp[1] += w * dp[1];
                }
            }
        });
    }
}

// -------------------------------------------------------------------------
// gather — embarrassingly parallel + auto-vectorisable
// -------------------------------------------------------------------------
torch::Tensor gather_impl(
    const torch::Tensor& grid,       // (grid_size,) complex
    const torch::Tensor& indices,    // (N,) int64
    const torch::Tensor& weights)    // (N,) float
{
    const int64_t N = indices.size(0);
    const int64_t grid_size = grid.size(0);
    auto output = torch::zeros({N}, grid.options());

    // #pragma omp parallel for CANNOT go inside an AT_DISPATCH_COMPLEX_TYPES
    // lambda argument — the C preprocessor hoists it before the expansion,
    // leaving a dangling pragma with no matching for-loop.  Instead, use an
    // explicit parallel region outside the dispatch and compute sub-ranges per
    // thread so that each thread owns a contiguous slice [start, end).
#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
        int64_t start = 0, end = N;
#ifdef _OPENMP
        const int tid      = omp_get_thread_num();
        const int nthreads = omp_get_num_threads();
        start = N *  tid      / nthreads;
        end   = N * (tid + 1) / nthreads;
#endif
        AT_DISPATCH_COMPLEX_TYPES(grid.scalar_type(), "gather_cpu", [&] {
            using real_t = typename scalar_t::value_type;
            auto* o_ptr = reinterpret_cast<real_t*>(output.data_ptr<scalar_t>());
            const auto* g_ptr = reinterpret_cast<const real_t*>(grid.data_ptr<scalar_t>());
            const auto* i_ptr = indices.data_ptr<int64_t>();
            const auto* w_ptr = weights.data_ptr<real_t>();
            for (int64_t n = start; n < end; ++n) {
                const int64_t idx = i_ptr[n];
                if (idx >= 0 && idx < grid_size) {
                    const real_t w = w_ptr[n];
                    o_ptr[n * 2]     = w * g_ptr[idx * 2];
                    o_ptr[n * 2 + 1] = w * g_ptr[idx * 2 + 1];
                }
            }
        });
    }

    return output;
}

}  // namespace SPARSE_OPS_NS
