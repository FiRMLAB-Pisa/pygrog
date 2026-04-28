/**
 * toep_psf_cuda.cu — CUDA PSF scatter kernels.
 *
 * Three kernels mirror the CPU impls in toep_psf.cpp:
 *   psf_scatter_scalar_cuda      — real PSF
 *   psf_scatter_outer_cuda       — M x M PSF from per-sample basis (N, M)
 *   psf_scatter_outer_basis_cuda — K x K PSF from (K, T) basis + per-sample t
 *
 * Strategy: 1 thread per (sample, l, l') tuple.  Grid points have low
 * collision probability for typical M ≤ 8, so a single global atomicAdd
 * per output cell is fine; shared-memory binning (à la sparse_ops_cuda)
 * is overkill here because PSF construction runs once per planning step.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

// ---------------------------------------------------------------------------
// Atomic helpers (same as sparse_ops_cuda.cu)
// ---------------------------------------------------------------------------
__device__ __forceinline__ void atomicAddComplex(
    c10::complex<float>* addr, c10::complex<float> val)
{
    float* fp = reinterpret_cast<float*>(addr);
    atomicAdd(fp,     val.real());
    atomicAdd(fp + 1, val.imag());
}

__device__ __forceinline__ void atomicAddComplex(
    c10::complex<double>* addr, c10::complex<double> val)
{
    double* dp = reinterpret_cast<double*>(addr);
    atomicAdd(dp,     val.real());
    atomicAdd(dp + 1, val.imag());
}

// ===========================================================================
// psf_scatter_scalar
// ===========================================================================
template <typename T>
__global__ void psf_scatter_scalar_kernel(
    T* __restrict__                   psf,
    const int64_t* __restrict__       indices,
    const T* __restrict__             w_sq,
    int64_t N, int64_t grid_size)
{
    int64_t n = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const int64_t idx = indices[n];
    if (idx < 0 || idx >= grid_size) return;
    atomicAdd(&psf[idx], w_sq[n]);
}

void psf_scatter_scalar_cuda(
    torch::Tensor psf,
    const torch::Tensor& indices,
    const torch::Tensor& w_sq)
{
    const int64_t N = indices.size(0);
    const int64_t grid_size = psf.size(0);
    if (N == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream();

    constexpr int BLOCK = 256;
    const int blocks = static_cast<int>((N + BLOCK - 1) / BLOCK);

    AT_DISPATCH_FLOATING_TYPES(psf.scalar_type(), "psf_scatter_scalar_cuda", [&] {
        psf_scatter_scalar_kernel<scalar_t><<<blocks, BLOCK, 0, stream>>>(
            psf.data_ptr<scalar_t>(),
            indices.data_ptr<int64_t>(),
            w_sq.data_ptr<scalar_t>(),
            N, grid_size);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ===========================================================================
// psf_scatter_outer
//   1 thread per sample.  Each thread loops over the M x M outer product
//   (M ≤ 8 in realistic use → trivial unroll).
// ===========================================================================
template <typename T>
__global__ void psf_scatter_outer_kernel(
    c10::complex<T>* __restrict__       psf,           // (grid_size, M, M) flat
    const int64_t* __restrict__         indices,
    const T* __restrict__               w_sq,
    const c10::complex<T>* __restrict__ basis,         // (N, M)
    int64_t N, int64_t grid_size, int64_t M)
{
    int64_t n = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const int64_t idx = indices[n];
    if (idx < 0 || idx >= grid_size) return;

    const T w = w_sq[n];
    const auto* bn = basis + n * M;
    auto* pn = psf + idx * (M * M);
    for (int64_t l = 0; l < M; ++l) {
        const c10::complex<T> cb_l =
            c10::complex<T>(bn[l].real(), -bn[l].imag()) * c10::complex<T>(w, T(0));
        for (int64_t lp = 0; lp < M; ++lp) {
            atomicAddComplex(&pn[l * M + lp], cb_l * bn[lp]);
        }
    }
}

void psf_scatter_outer_cuda(
    torch::Tensor psf,
    const torch::Tensor& indices,
    const torch::Tensor& w_sq,
    const torch::Tensor& basis_per_sample)
{
    const int64_t N = indices.size(0);
    const int64_t grid_size = psf.size(0);
    const int64_t M = psf.size(1);
    if (N == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream();

    constexpr int BLOCK = 128;
    const int blocks = static_cast<int>((N + BLOCK - 1) / BLOCK);

    AT_DISPATCH_FLOATING_TYPES(w_sq.scalar_type(), "psf_scatter_outer_cuda", [&] {
        psf_scatter_outer_kernel<scalar_t><<<blocks, BLOCK, 0, stream>>>(
            reinterpret_cast<c10::complex<scalar_t>*>(psf.data_ptr()),
            indices.data_ptr<int64_t>(),
            w_sq.data_ptr<scalar_t>(),
            reinterpret_cast<const c10::complex<scalar_t>*>(
                basis_per_sample.data_ptr()),
            N, grid_size, M);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ===========================================================================
// psf_scatter_outer_basis
// ===========================================================================
template <typename T>
__global__ void psf_scatter_outer_basis_kernel(
    c10::complex<T>* __restrict__       psf,
    const int64_t* __restrict__         indices,
    const T* __restrict__               w_sq,
    const c10::complex<T>* __restrict__ basis,         // (K, T_)
    const int64_t* __restrict__         time_index,
    int64_t N, int64_t grid_size, int64_t K, int64_t T_)
{
    int64_t n = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const int64_t idx = indices[n];
    if (idx < 0 || idx >= grid_size) return;
    const int64_t t = time_index[n];
    if (t < 0 || t >= T_) return;

    const T w = w_sq[n];
    auto* pn = psf + idx * (K * K);
    for (int64_t k = 0; k < K; ++k) {
        const auto bk = basis[k * T_ + t];
        const c10::complex<T> cb_k =
            c10::complex<T>(bk.real(), -bk.imag()) * c10::complex<T>(w, T(0));
        for (int64_t kp = 0; kp < K; ++kp) {
            atomicAddComplex(&pn[k * K + kp], cb_k * basis[kp * T_ + t]);
        }
    }
}

void psf_scatter_outer_basis_cuda(
    torch::Tensor psf,
    const torch::Tensor& indices,
    const torch::Tensor& w_sq,
    const torch::Tensor& basis,
    const torch::Tensor& time_index)
{
    const int64_t N = indices.size(0);
    const int64_t grid_size = psf.size(0);
    const int64_t K  = psf.size(1);
    const int64_t T_ = basis.size(1);
    if (N == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream();

    constexpr int BLOCK = 128;
    const int blocks = static_cast<int>((N + BLOCK - 1) / BLOCK);

    AT_DISPATCH_FLOATING_TYPES(w_sq.scalar_type(), "psf_scatter_outer_basis_cuda", [&] {
        psf_scatter_outer_basis_kernel<scalar_t><<<blocks, BLOCK, 0, stream>>>(
            reinterpret_cast<c10::complex<scalar_t>*>(psf.data_ptr()),
            indices.data_ptr<int64_t>(),
            w_sq.data_ptr<scalar_t>(),
            reinterpret_cast<const c10::complex<scalar_t>*>(basis.data_ptr()),
            time_index.data_ptr<int64_t>(),
            N, grid_size, K, T_);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
