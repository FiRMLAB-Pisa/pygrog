/**
 * sparse_ops_cuda.cu -- Custom CUDA kernels for scatter-add and gather.
 *
 * Two scatter strategies:
 *
 *   1. Simple (nupts-driven) -- 1 thread per NU point, global atomics.
 *      Good for uniformly distributed points or small N.
 *
 *   2. Binned (shared-memory) -- Grid divided into fixed-size bins.
 *      Each threadblock accumulates one bin in shared memory (fast ~1-cycle
 *      atomics), then writes once to global (conflict-free, since bins are
 *      disjoint).  Requires pre-sorted indices and bin_starts computed
 *      at plan time.  Dramatically reduces global atomic contention for
 *      non-uniform density (e.g., radial k-space center).
 *
 * Gather is embarrassingly parallel in both cases.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

// =========================================================================
// Atomic helpers for complex types
// =========================================================================

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

// =========================================================================
// Simple scatter-add kernel (Method 1: nupts-driven, global atomics)
// =========================================================================

template <typename T>
__global__ void scatter_add_kernel(
    c10::complex<T>* __restrict__       grid,
    const c10::complex<T>* __restrict__ data,
    const int64_t* __restrict__         indices,
    const T* __restrict__               weights,
    int64_t N, int64_t grid_size)
{
    int64_t n = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (n >= N) return;

    int64_t idx = indices[n];
    if (idx < 0 || idx >= grid_size) return;

    c10::complex<T> val = c10::complex<T>(weights[n], T(0)) * data[n];
    atomicAddComplex(&grid[idx], val);
}

// =========================================================================
// Binned scatter-add kernel (Method 2: shared-memory accumulation)
//
// One threadblock per grid bin.  Points are pre-sorted by grid index,
// so each bin's points are a contiguous slice [bin_starts[b], bin_starts[b+1]).
//
// Shared memory holds bin_size complex<T> values for the local grid region.
// Threads atomicAdd to shared (fast), then write shared -> global (conflict-
// free since bins tile the grid without overlap).
//
// For bins with many points (density hotspots), the loop naturally processes
// all points -- the threadblock stays busy.  Empty bins exit immediately.
// =========================================================================

template <typename T>
__global__ void scatter_add_binned_kernel(
    c10::complex<T>* __restrict__       grid,
    const c10::complex<T>* __restrict__ data,
    const int64_t* __restrict__         indices,
    const T* __restrict__               weights,
    const int64_t* __restrict__         bin_starts,
    int64_t bin_size,
    int64_t grid_size,
    int64_t n_bins)
{
    extern __shared__ char smem[];
    auto* local = reinterpret_cast<T*>(smem);

    const int64_t bin_id = blockIdx.x;
    if (bin_id >= n_bins) return;

    const int64_t grid_offset = bin_id * bin_size;
    const int64_t actual_bin  = min(bin_size, grid_size - grid_offset);

    // Zero shared memory
    for (int64_t i = threadIdx.x; i < actual_bin * 2; i += blockDim.x)
        local[i] = T(0);
    __syncthreads();

    // Scatter into shared memory (shared-memory atomics ~5x faster than global)
    const int64_t pt_start = bin_starts[bin_id];
    const int64_t pt_end   = bin_starts[bin_id + 1];
    for (int64_t n = pt_start + threadIdx.x; n < pt_end; n += blockDim.x) {
        const int64_t local_idx = indices[n] - grid_offset;
        const T w = weights[n];
        const T* dp = reinterpret_cast<const T*>(&data[n]);
        atomicAdd(&local[local_idx * 2],     w * dp[0]);
        atomicAdd(&local[local_idx * 2 + 1], w * dp[1]);
    }
    __syncthreads();

    // Write to global -- no atomics needed (bins are disjoint)
    T* g = reinterpret_cast<T*>(grid + grid_offset);
    for (int64_t i = threadIdx.x; i < actual_bin * 2; i += blockDim.x)
        g[i] += local[i];
}

// =========================================================================
// Gather kernel (unchanged -- embarrassingly parallel)
// =========================================================================

template <typename T>
__global__ void gather_kernel(
    c10::complex<T>* __restrict__       output,
    const c10::complex<T>* __restrict__ grid,
    const int64_t* __restrict__         indices,
    const T* __restrict__               weights,
    int64_t N, int64_t grid_size)
{
    int64_t n = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (n >= N) return;

    int64_t idx = indices[n];
    if (idx >= 0 && idx < grid_size) {
        output[n] = c10::complex<T>(weights[n], T(0)) * grid[idx];
    } else {
        output[n] = c10::complex<T>(T(0), T(0));
    }
}

// =========================================================================
// Launchers
// =========================================================================

void scatter_add_cuda(
    torch::Tensor grid,
    const torch::Tensor& data,
    const torch::Tensor& indices,
    const torch::Tensor& weights)
{
    const int64_t N         = data.size(0);
    const int64_t grid_size = grid.size(0);
    if (N == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream();

    constexpr int BLOCK = 256;
    const int blocks = static_cast<int>((N + BLOCK - 1) / BLOCK);

    AT_DISPATCH_FLOATING_TYPES(weights.scalar_type(), "scatter_add_cuda", [&] {
        scatter_add_kernel<scalar_t><<<blocks, BLOCK, 0, stream>>>(
            reinterpret_cast<c10::complex<scalar_t>*>(grid.data_ptr()),
            reinterpret_cast<const c10::complex<scalar_t>*>(data.data_ptr()),
            indices.data_ptr<int64_t>(),
            weights.data_ptr<scalar_t>(),
            N, grid_size);
    });

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void scatter_add_binned_cuda(
    torch::Tensor grid,
    const torch::Tensor& data,
    const torch::Tensor& indices,
    const torch::Tensor& weights,
    const torch::Tensor& bin_starts,
    int64_t bin_size)
{
    const int64_t grid_size = grid.size(0);
    const int64_t n_bins    = bin_starts.size(0) - 1;
    if (n_bins <= 0) return;
    auto stream = at::cuda::getCurrentCUDAStream();

    constexpr int BLOCK = 256;

    AT_DISPATCH_FLOATING_TYPES(weights.scalar_type(), "scatter_add_binned_cuda", [&] {
        const size_t smem = static_cast<size_t>(bin_size) * 2 * sizeof(scalar_t);

        scatter_add_binned_kernel<scalar_t><<<n_bins, BLOCK, smem, stream>>>(
            reinterpret_cast<c10::complex<scalar_t>*>(grid.data_ptr()),
            reinterpret_cast<const c10::complex<scalar_t>*>(data.data_ptr()),
            indices.data_ptr<int64_t>(),
            weights.data_ptr<scalar_t>(),
            bin_starts.data_ptr<int64_t>(),
            bin_size, grid_size, n_bins);
    });

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor gather_cuda(
    const torch::Tensor& grid,
    const torch::Tensor& indices,
    const torch::Tensor& weights)
{
    const int64_t N         = indices.size(0);
    const int64_t grid_size = grid.size(0);
    auto output = torch::zeros({N}, grid.options());
    if (N == 0) return output;
    auto stream = at::cuda::getCurrentCUDAStream();

    constexpr int BLOCK = 256;
    const int blocks = static_cast<int>((N + BLOCK - 1) / BLOCK);

    AT_DISPATCH_FLOATING_TYPES(weights.scalar_type(), "gather_cuda", [&] {
        gather_kernel<scalar_t><<<blocks, BLOCK, 0, stream>>>(
            reinterpret_cast<c10::complex<scalar_t>*>(output.data_ptr()),
            reinterpret_cast<const c10::complex<scalar_t>*>(grid.data_ptr()),
            indices.data_ptr<int64_t>(),
            weights.data_ptr<scalar_t>(),
            N, grid_size);
    });

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
