/**
 * toep_psf.cpp — PSF construction kernels for GROG-Toeplitz embedding.
 *
 * Three scatter-style kernels build a Toeplitz PSF on the oversampled
 * Cartesian grid from sorted GROG sample indices:
 *
 *   psf_scatter_scalar      (real PSF, plain SparseFFT)
 *      psf[indices[i]] += w_sq[i]
 *
 *   psf_scatter_outer       (M x M Hermitian PSF, off-resonance/subspace)
 *      psf[indices[i], l, l'] += w_sq[i] * conj(b[i, l]) * b[i, l']
 *
 *   psf_scatter_outer_basis (subspace, basis indexed by per-sample time)
 *      psf[indices[i], k, k'] += w_sq[i] * conj(B[k, t_i]) * B[k', t_i]
 *
 * All three assume `indices` are sorted ascending so that an OpenMP
 * thread partition aligned to clean boundaries (where index value
 * changes) writes to disjoint grid cells — no atomics, no per-thread
 * accumulators.  CUDA paths use global atomics (M is small enough that
 * shared-memory binning is overkill for the PSF use case).
 */

#include <torch/extension.h>

#include <cstdint>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// ---------------------------------------------------------------------------
// Forward declarations of CUDA launchers
// ---------------------------------------------------------------------------
#ifdef COMPILE_WITH_CUDA
void psf_scatter_scalar_cuda(
    torch::Tensor psf,
    const torch::Tensor& indices,
    const torch::Tensor& w_sq);

void psf_scatter_outer_cuda(
    torch::Tensor psf,
    const torch::Tensor& indices,
    const torch::Tensor& w_sq,
    const torch::Tensor& basis_per_sample);

void psf_scatter_outer_basis_cuda(
    torch::Tensor psf,
    const torch::Tensor& indices,
    const torch::Tensor& w_sq,
    const torch::Tensor& basis,
    const torch::Tensor& time_index);
#endif

// ---------------------------------------------------------------------------
// Helper: clean-partition boundaries for sorted indices
// ---------------------------------------------------------------------------
static std::vector<int64_t> clean_partition_bounds(
    const int64_t* i_ptr, int64_t N, int n_threads)
{
    std::vector<int64_t> bounds(n_threads + 1);
    bounds[0] = 0;
    bounds[n_threads] = N;
    for (int t = 1; t < n_threads; ++t) {
        int64_t target = N * t / n_threads;
        while (target < N && target > 0 && i_ptr[target] == i_ptr[target - 1])
            ++target;
        bounds[t] = target;
    }
    return bounds;
}

// ===========================================================================
// CPU kernels
// ===========================================================================

// -------------------------------------------------------------------------
// psf_scatter_scalar — real PSF for plain SparseFFT
// -------------------------------------------------------------------------
static void psf_scatter_scalar_cpu(
    torch::Tensor psf,                   // (grid_size,) real, in-place
    const torch::Tensor& indices,        // (N,) int64, sorted ascending
    const torch::Tensor& w_sq)           // (N,) real
{
    const int64_t N = indices.size(0);
    const int64_t grid_size = psf.size(0);
    if (N == 0) return;

    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif
    if (N < 4096) n_threads = 1;

    const auto* i_ptr = indices.data_ptr<int64_t>();
    auto bounds = clean_partition_bounds(i_ptr, N, n_threads);

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
        AT_DISPATCH_FLOATING_TYPES(psf.scalar_type(), "psf_scatter_scalar_cpu", [&] {
            auto* p_ptr        = psf.data_ptr<scalar_t>();
            const auto* w_ptr  = w_sq.data_ptr<scalar_t>();
            for (int64_t n = start; n < end; ++n) {
                const int64_t idx = i_ptr[n];
                if (idx >= 0 && idx < grid_size)
                    p_ptr[idx] += w_ptr[n];
            }
        });
    }
}

// -------------------------------------------------------------------------
// psf_scatter_outer — M x M PSF from per-sample basis
//   psf[idx, l, l'] += w_sq * conj(b[l]) * b[l']
// -------------------------------------------------------------------------
static void psf_scatter_outer_cpu(
    torch::Tensor psf,                   // (grid_size, M, M) complex, in-place
    const torch::Tensor& indices,        // (N,) int64, sorted
    const torch::Tensor& w_sq,           // (N,) real
    const torch::Tensor& basis)          // (N, M) complex
{
    const int64_t N = indices.size(0);
    const int64_t grid_size = psf.size(0);
    const int64_t M = psf.size(1);
    if (N == 0) return;

    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif
    if (N < 1024) n_threads = 1;

    const auto* i_ptr = indices.data_ptr<int64_t>();
    auto bounds = clean_partition_bounds(i_ptr, N, n_threads);

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
        AT_DISPATCH_COMPLEX_TYPES(psf.scalar_type(), "psf_scatter_outer_cpu", [&] {
            using real_t = typename scalar_t::value_type;
            auto* p_ptr       = psf.data_ptr<scalar_t>();
            const auto* b_ptr = basis.data_ptr<scalar_t>();
            const auto* w_ptr = w_sq.data_ptr<real_t>();
            const int64_t MM  = M * M;
            for (int64_t n = start; n < end; ++n) {
                const int64_t idx = i_ptr[n];
                if (idx < 0 || idx >= grid_size) continue;
                const real_t w  = w_ptr[n];
                const auto* bn  = b_ptr + n * M;
                auto* pn        = p_ptr + idx * MM;
                for (int64_t l = 0; l < M; ++l) {
                    const scalar_t cb_l = std::conj(bn[l]) * w;
                    for (int64_t lp = 0; lp < M; ++lp) {
                        pn[l * M + lp] += cb_l * bn[lp];
                    }
                }
            }
        });
    }
}

// -------------------------------------------------------------------------
// psf_scatter_outer_basis — K x K PSF from (K, T) basis + per-sample t
//   psf[idx, k, k'] += w_sq * conj(B[k, t]) * B[k', t]
// -------------------------------------------------------------------------
static void psf_scatter_outer_basis_cpu(
    torch::Tensor psf,                   // (grid_size, K, K) complex, in-place
    const torch::Tensor& indices,        // (N,) int64, sorted
    const torch::Tensor& w_sq,           // (N,) real
    const torch::Tensor& basis,          // (K, T) complex
    const torch::Tensor& time_index)     // (N,) int64
{
    const int64_t N = indices.size(0);
    const int64_t grid_size = psf.size(0);
    const int64_t K = psf.size(1);
    const int64_t T = basis.size(1);
    if (N == 0) return;

    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif
    if (N < 1024) n_threads = 1;

    const auto* i_ptr = indices.data_ptr<int64_t>();
    const auto* t_ptr = time_index.data_ptr<int64_t>();
    auto bounds = clean_partition_bounds(i_ptr, N, n_threads);

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
        AT_DISPATCH_COMPLEX_TYPES(psf.scalar_type(), "psf_scatter_outer_basis_cpu", [&] {
            using real_t = typename scalar_t::value_type;
            auto* p_ptr       = psf.data_ptr<scalar_t>();
            const auto* b_ptr = basis.data_ptr<scalar_t>();
            const auto* w_ptr = w_sq.data_ptr<real_t>();
            const int64_t KK  = K * K;
            for (int64_t n = start; n < end; ++n) {
                const int64_t idx = i_ptr[n];
                if (idx < 0 || idx >= grid_size) continue;
                const int64_t t = t_ptr[n];
                if (t < 0 || t >= T) continue;
                const real_t w = w_ptr[n];
                auto* pn = p_ptr + idx * KK;
                for (int64_t k = 0; k < K; ++k) {
                    // basis is (K, T) row-major: B[k, t] = b_ptr[k * T + t]
                    const scalar_t cb_k = std::conj(b_ptr[k * T + t]) * w;
                    for (int64_t kp = 0; kp < K; ++kp) {
                        pn[k * K + kp] += cb_k * b_ptr[kp * T + t];
                    }
                }
            }
        });
    }
}

// ===========================================================================
// Public entry points
// ===========================================================================
void psf_scatter_scalar(
    torch::Tensor psf,
    torch::Tensor indices,
    torch::Tensor w_sq)
{
    TORCH_CHECK(psf.dim() == 1, "psf must be 1-D, got ", psf.dim(), "-D");
    TORCH_CHECK(indices.dim() == 1, "indices must be 1-D");
    TORCH_CHECK(w_sq.dim() == 1, "w_sq must be 1-D");
    TORCH_CHECK(indices.size(0) == w_sq.size(0),
                "indices/w_sq size mismatch");
    TORCH_CHECK(indices.dtype() == torch::kInt64, "indices must be int64");
    TORCH_CHECK(psf.scalar_type() == w_sq.scalar_type(),
                "psf/w_sq dtype mismatch");

    psf     = psf.contiguous();
    indices = indices.contiguous();
    w_sq    = w_sq.contiguous();

#ifdef COMPILE_WITH_CUDA
    if (psf.is_cuda()) {
        psf_scatter_scalar_cuda(psf, indices, w_sq);
        return;
    }
#else
    if (psf.is_cuda()) {
        // ATen fallback
        psf.index_add_(0, indices, w_sq);
        return;
    }
#endif

    psf_scatter_scalar_cpu(psf, indices, w_sq);
}

void psf_scatter_outer(
    torch::Tensor psf,
    torch::Tensor indices,
    torch::Tensor w_sq,
    torch::Tensor basis_per_sample)
{
    TORCH_CHECK(psf.dim() == 3, "psf must be 3-D (grid_size, M, M)");
    TORCH_CHECK(psf.size(1) == psf.size(2), "psf last two dims must be equal");
    TORCH_CHECK(indices.dim() == 1 && w_sq.dim() == 1,
                "indices and w_sq must be 1-D");
    TORCH_CHECK(basis_per_sample.dim() == 2, "basis must be 2-D (N, M)");
    TORCH_CHECK(indices.size(0) == w_sq.size(0)
                && indices.size(0) == basis_per_sample.size(0),
                "indices/w_sq/basis sample-count mismatch");
    TORCH_CHECK(basis_per_sample.size(1) == psf.size(1),
                "basis M does not match psf M");
    TORCH_CHECK(indices.dtype() == torch::kInt64, "indices must be int64");

    psf              = psf.contiguous();
    indices          = indices.contiguous();
    w_sq             = w_sq.contiguous();
    basis_per_sample = basis_per_sample.contiguous();

#ifdef COMPILE_WITH_CUDA
    if (psf.is_cuda()) {
        psf_scatter_outer_cuda(psf, indices, w_sq, basis_per_sample);
        return;
    }
#else
    if (psf.is_cuda()) {
        // ATen fallback: per-sample outer product, then index_add_.
        const int64_t M = psf.size(1);
        auto bn   = basis_per_sample;                // (N, M)
        auto outer = torch::einsum(
            "ni,nj->nij",
            {bn.conj(), bn});                        // (N, M, M)
        outer = outer * w_sq.to(outer.dtype()).view({-1, 1, 1});
        auto flat = psf.view({psf.size(0), M * M});
        flat.index_add_(0, indices, outer.view({-1, M * M}));
        return;
    }
#endif

    psf_scatter_outer_cpu(psf, indices, w_sq, basis_per_sample);
}

void psf_scatter_outer_basis(
    torch::Tensor psf,
    torch::Tensor indices,
    torch::Tensor w_sq,
    torch::Tensor basis,
    torch::Tensor time_index)
{
    TORCH_CHECK(psf.dim() == 3, "psf must be 3-D (grid_size, K, K)");
    TORCH_CHECK(psf.size(1) == psf.size(2), "psf last two dims must be equal");
    TORCH_CHECK(basis.dim() == 2, "basis must be 2-D (K, T)");
    TORCH_CHECK(basis.size(0) == psf.size(1), "basis K does not match psf K");
    TORCH_CHECK(indices.dim() == 1 && w_sq.dim() == 1
                && time_index.dim() == 1,
                "indices, w_sq, time_index must be 1-D");
    TORCH_CHECK(indices.size(0) == w_sq.size(0)
                && indices.size(0) == time_index.size(0),
                "indices/w_sq/time_index size mismatch");
    TORCH_CHECK(indices.dtype() == torch::kInt64, "indices must be int64");
    TORCH_CHECK(time_index.dtype() == torch::kInt64, "time_index must be int64");

    psf        = psf.contiguous();
    indices    = indices.contiguous();
    w_sq       = w_sq.contiguous();
    basis      = basis.contiguous();
    time_index = time_index.contiguous();

#ifdef COMPILE_WITH_CUDA
    if (psf.is_cuda()) {
        psf_scatter_outer_basis_cuda(psf, indices, w_sq, basis, time_index);
        return;
    }
#else
    if (psf.is_cuda()) {
        // ATen fallback: gather basis rows by time, then outer-product path.
        auto bn = basis.t().index_select(0, time_index);  // (N, K)
        const int64_t K = psf.size(1);
        auto outer = torch::einsum("ni,nj->nij", {bn.conj(), bn});
        outer = outer * w_sq.to(outer.dtype()).view({-1, 1, 1});
        auto flat = psf.view({psf.size(0), K * K});
        flat.index_add_(0, indices, outer.view({-1, K * K}));
        return;
    }
#endif

    psf_scatter_outer_basis_cpu(psf, indices, w_sq, basis, time_index);
}
