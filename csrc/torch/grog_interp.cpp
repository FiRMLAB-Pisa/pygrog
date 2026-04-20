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
// CPU path: ATen batched matmul (dispatches to BLAS/MKL)
// ---------------------------------------------------------------------------
static torch::Tensor grog_interpolate_cpu(
    const torch::Tensor& data,      // (N, B, C)
    const torch::Tensor& indexes,   // (N,) int64
    const torch::Tensor& kernel)    // (K, C, C)
{
    // Gather per-sample kernels: (N, C, C)
    auto K = kernel.index_select(0, indexes);

    // Batched matmul with broadcasting over B:
    //   K.unsqueeze(1) : (N, 1, C, C)
    //   data.unsqueeze(-1) : (N, B, C, 1)
    //   result : (N, B, C, 1) → squeeze → (N, B, C)
    return torch::matmul(K.unsqueeze(1), data.unsqueeze(-1)).squeeze(-1);
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
