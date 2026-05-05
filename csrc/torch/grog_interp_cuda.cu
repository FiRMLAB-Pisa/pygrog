/**
 * grog_interp_cuda.cu — Custom CUDA kernel for GROG interpolation.
 *
 * For each (sample n, batch b) pair:
 *   out[n, b, co] = sum_{ci} kernel[indexes[n], co, ci] * data[n, b, ci]
 *
 * Grid/block layout:
 *   gridDim  = (N, B)  — one block per (sample, batch)
 *   blockDim = (C,)    — one thread per output coil
 *
 * Using separate input/output buffers avoids shared-memory sync while
 * remaining fully correct (no aliasing between reads and writes).
 * Each thread reads C elements from `inp` (coalesced within a warp for
 * typical C ≤ 32) and one row of the kernel matrix, then writes 1 element.
 *
 * Complexity per block: O(C²) flops, O(C) reads, O(1) writes.
 * For MRI: C ~ 8–64, easily fits in L1 / register file.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

// ---------------------------------------------------------------------------
// Kernel template
// ---------------------------------------------------------------------------

template <typename T>
__global__ void grog_interp_kernel(
    c10::complex<T>* __restrict__       out,       // (N, B, C) output
    const c10::complex<T>* __restrict__ inp,       // (N, B, C) input
    const int64_t*  __restrict__        indexes,   // (N,)
    const c10::complex<T>* __restrict__ kernel,    // (K, C, C)
    int64_t N, int64_t B, int64_t C, int64_t K_count)
{
    const int64_t n  = blockIdx.x;
    const int64_t b  = blockIdx.y;
    const int64_t co = threadIdx.x;

    if (n >= N || b >= B || co >= C) return;

    const int64_t k = indexes[n];
    if (k < 0 || k >= K_count) return;

    // Pointer to the co-th row of kernel matrix k: kernel[k, co, :]
    const c10::complex<T>* Krow = kernel + k * C * C + co * C;

    // Pointer to data[n, b, :] = inp + (n*B + b)*C
    const c10::complex<T>* src = inp + (n * B + b) * C;

    c10::complex<T> val(T(0), T(0));
    #pragma unroll 8
    for (int64_t ci = 0; ci < C; ++ci) {
        val += Krow[ci] * src[ci];
    }

    out[(n * B + b) * C + co] = val;
}

// ---------------------------------------------------------------------------
// Launcher (called from grog_interp.cpp)
// ---------------------------------------------------------------------------

torch::Tensor grog_interpolate_cuda(
    const torch::Tensor& data,      // (N, B, C) contiguous complex
    const torch::Tensor& indexes,   // (N,)       contiguous int64
    const torch::Tensor& kernel)    // (K, C, C)  contiguous complex
{
    const int64_t N       = data.size(0);
    const int64_t B       = data.size(1);
    const int64_t C       = data.size(2);
    const int64_t K_count = kernel.size(0);

    TORCH_CHECK(C <= 1024,
        "ncoils must be ≤ 1024 (CUDA block size limit), got ", C);

    auto out    = torch::empty_like(data);
    auto stream = at::cuda::getCurrentCUDAStream();

    const dim3 grid(static_cast<unsigned>(N), static_cast<unsigned>(B));
    const dim3 block(static_cast<unsigned>(C));

    const auto dtype = data.scalar_type();

    if (dtype == at::ScalarType::ComplexFloat) {
        grog_interp_kernel<float><<<grid, block, 0, stream>>>(
            reinterpret_cast<c10::complex<float>*>(out.data_ptr()),
            reinterpret_cast<const c10::complex<float>*>(data.data_ptr()),
            indexes.data_ptr<int64_t>(),
            reinterpret_cast<const c10::complex<float>*>(kernel.data_ptr()),
            N, B, C, K_count);
    } else if (dtype == at::ScalarType::ComplexDouble) {
        grog_interp_kernel<double><<<grid, block, 0, stream>>>(
            reinterpret_cast<c10::complex<double>*>(out.data_ptr()),
            reinterpret_cast<const c10::complex<double>*>(data.data_ptr()),
            indexes.data_ptr<int64_t>(),
            reinterpret_cast<const c10::complex<double>*>(kernel.data_ptr()),
            N, B, C, K_count);
    } else {
        TORCH_CHECK(false,
            "grog_interpolate_cuda: expected ComplexFloat or ComplexDouble, got ", dtype);
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
