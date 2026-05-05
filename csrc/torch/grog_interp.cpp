/**
 * grog_interp.cpp — PyTorch C++ extension for GROG interpolation.
 *
 * Applies precomputed GRAPPA kernel matrices to k-space samples, replacing
 * the Numba JIT implementation.  Works on CPU via ATen BLAS, and on CUDA
 * via either a custom kernel (if COMPILE_WITH_CUDA) or ATen cuBLAS fallback.
 *
 * Operation (per sample n, batch b):
 *   out[n, b, :] = kernel[indexes[n]] @ data[n, b, :]
 *
 *   data   : (N, B, C) complex tensor — input samples
 *   indexes: (N,)      int64 tensor   — kernel index per sample
 *   kernel : (K, C, C) complex tensor — precomputed GRAPPA kernel table
 *   return : (N, B, C) complex tensor — interpolated output
 */

#include <torch/extension.h>

// ---------------------------------------------------------------------------
// Forward declaration of CUDA launcher (defined in grog_interp_cuda.cu)
// ---------------------------------------------------------------------------
#ifdef COMPILE_WITH_CUDA
torch::Tensor grog_interpolate_cuda(
    const torch::Tensor& data,
    const torch::Tensor& indexes,
    const torch::Tensor& kernel);
#endif

// ---------------------------------------------------------------------------
// CPU path: parallel naive small-C matvec (no (N,C,C) gather, no BLAS call)
//
// For typical MRI workloads C ≤ 16 (often ≤ 8 after coil compression), so a
// single (C, C) × (C,) matvec fits in registers — far faster than calling a
// BLAS GEMM that pays its dispatch cost per sample.  Each output element
//   out[n, b, co] = Σ_ci kernel[indexes[n], co, ci] * data[n, b, ci]
// is computed in-place; no (N, C, C) intermediate from index_select is
// materialised.
// ---------------------------------------------------------------------------
template <typename T>
static void grog_interpolate_cpu_impl(
    c10::complex<T>*       out,        // (N, B, C)
    const c10::complex<T>* inp,        // (N, B, C)
    const int64_t*         indexes,    // (N,)
    const c10::complex<T>* kernel,     // (K, C, C)
    int64_t N, int64_t B, int64_t C, int64_t /*K_count*/)
{
    const int64_t CC = C * C;
    const int64_t NB = N * B;

    #pragma omp parallel for schedule(static)
    for (int64_t nb = 0; nb < NB; ++nb) {
        const int64_t n = nb / B;
        const int64_t k = indexes[n];
        // indexes are pre-clamped by _attach_fft_plan / calc_interp_table,
        // so 0 <= k < K_count is guaranteed by construction.
        const c10::complex<T>* Kmat = kernel + k * CC;       // (C, C) row-major
        const c10::complex<T>* src  = inp    + nb * C;       // (C,)
        c10::complex<T>*       dst  = out    + nb * C;       // (C,)

        // Naive (C, C) × (C,) matvec — fully unrolled by the compiler for
        // small C; for larger C this is still ≤ a few cache lines and beats
        // a BLAS dispatch per sample.
        for (int64_t co = 0; co < C; ++co) {
            const c10::complex<T>* Krow = Kmat + co * C;
            c10::complex<T> acc(T(0), T(0));
            #pragma omp simd
            for (int64_t ci = 0; ci < C; ++ci) {
                acc += Krow[ci] * src[ci];
            }
            dst[co] = acc;
        }
    }
}

static torch::Tensor grog_interpolate_cpu(
    const torch::Tensor& data,      // (N, B, C)
    const torch::Tensor& indexes,   // (N,) int64
    const torch::Tensor& kernel)    // (K, C, C)
{
    auto out = torch::empty_like(data);

    const int64_t N       = data.size(0);
    const int64_t B       = data.size(1);
    const int64_t C       = data.size(2);
    const int64_t K_count = kernel.size(0);

    const auto dtype = data.scalar_type();
    if (dtype == at::ScalarType::ComplexFloat) {
        grog_interpolate_cpu_impl<float>(
            reinterpret_cast<c10::complex<float>*>(out.data_ptr()),
            reinterpret_cast<const c10::complex<float>*>(data.data_ptr()),
            indexes.data_ptr<int64_t>(),
            reinterpret_cast<const c10::complex<float>*>(kernel.data_ptr()),
            N, B, C, K_count);
    } else if (dtype == at::ScalarType::ComplexDouble) {
        grog_interpolate_cpu_impl<double>(
            reinterpret_cast<c10::complex<double>*>(out.data_ptr()),
            reinterpret_cast<const c10::complex<double>*>(data.data_ptr()),
            indexes.data_ptr<int64_t>(),
            reinterpret_cast<const c10::complex<double>*>(kernel.data_ptr()),
            N, B, C, K_count);
    } else {
        TORCH_CHECK(false,
            "grog_interpolate_cpu: expected ComplexFloat or ComplexDouble, got ", dtype);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Public entry point — dispatches to CPU or CUDA
// ---------------------------------------------------------------------------
torch::Tensor grog_interpolate(
    torch::Tensor data,
    torch::Tensor indexes,
    torch::Tensor kernel)
{
    TORCH_CHECK(data.dim() == 3,
        "data must be 3-D (nsamples, nbatches, ncoils), got ", data.dim(), "-D");
    TORCH_CHECK(indexes.dim() == 1,
        "indexes must be 1-D (nsamples,), got ", indexes.dim(), "-D");
    TORCH_CHECK(kernel.dim() == 3,
        "kernel must be 3-D (n_kernels, ncoils, ncoils), got ", kernel.dim(), "-D");
    TORCH_CHECK(indexes.dtype() == torch::kInt64,
        "indexes must be int64, got ", indexes.dtype());
    TORCH_CHECK(data.size(0) == indexes.size(0),
        "data.size(0) (", data.size(0), ") != indexes.size(0) (", indexes.size(0), ")");
    TORCH_CHECK(data.size(2) == kernel.size(1) && kernel.size(1) == kernel.size(2),
        "ncoils mismatch between data (", data.size(2), ") and kernel (",
        kernel.size(1), "×", kernel.size(2), ")");

    // Ensure contiguous layout
    data    = data.contiguous();
    indexes = indexes.contiguous();
    kernel  = kernel.contiguous();

#ifdef COMPILE_WITH_CUDA
    if (data.is_cuda()) {
        return grog_interpolate_cuda(data, indexes, kernel);
    }
#else
    if (data.is_cuda()) {
        // ATen cuBLAS fallback when custom CUDA kernel was not compiled
        auto K = kernel.index_select(0, indexes);
        return torch::matmul(K.unsqueeze(1), data.unsqueeze(-1)).squeeze(-1);
    }
#endif

    return grog_interpolate_cpu(data, indexes, kernel);
}

// ---------------------------------------------------------------------------
// Module binding is in module.cpp
// ---------------------------------------------------------------------------
