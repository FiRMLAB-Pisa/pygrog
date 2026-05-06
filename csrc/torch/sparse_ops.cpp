/**
 * sparse_ops.cpp — PyTorch C++ extension for scatter-add and gather.
 *
 * scatter_add:  grid[indices[i]] += weights[i] * data[i]
 * gather:       out[i] = weights[i] * grid[indices[i]]
 *
 * CPU multi-versioning strategy:
 *   JIT builds  (-DPYGROG_MARCH_NATIVE)  — single arch, -march=native.
 *   GCC / Clang ≥ 14 prebuilt            — target_clones → GNU ifunc.
 *   MSVC prebuilt (x86/x64)              — separate TUs + CPUID dispatch.
 *   Fallback (old Clang, ARM, …)         — baseline auto-vectorisation.
 *
 * CUDA path dispatches to custom kernels in sparse_ops_cuda.cu when
 * COMPILE_WITH_CUDA is defined.
 */

#include <torch/extension.h>

#include <algorithm>
#include <cstring>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// ---------------------------------------------------------------------------
// Forward declarations of CUDA launchers
// ---------------------------------------------------------------------------
#ifdef COMPILE_WITH_CUDA
void scatter_add_cuda(
    torch::Tensor grid,
    const torch::Tensor& data,
    const torch::Tensor& indices,
    const torch::Tensor& weights);

void scatter_add_binned_cuda(
    torch::Tensor grid,
    const torch::Tensor& data,
    const torch::Tensor& indices,
    const torch::Tensor& weights,
    const torch::Tensor& bin_starts,
    int64_t bin_size);

torch::Tensor gather_cuda(
    const torch::Tensor& grid,
    const torch::Tensor& indices,
    const torch::Tensor& weights);
#endif

// ===========================================================================
// CPU implementation — compile-time path selection
// ===========================================================================

#if defined(PYGROG_MARCH_NATIVE)
// -------------------------------------------------------------------
// Path A: JIT build — single arch compiled with -march=native.
//         No multi-versioning needed.
// -------------------------------------------------------------------
#define SPARSE_OPS_NS sparse_ops_native
#include "sparse_ops_cpu_impl.inl"

static void scatter_add_cpu(
    torch::Tensor grid, const torch::Tensor& data,
    const torch::Tensor& indices, const torch::Tensor& weights)
{
    sparse_ops_native::scatter_add_impl(grid, data, indices, weights);
}

static torch::Tensor gather_cpu(
    const torch::Tensor& grid,
    const torch::Tensor& indices, const torch::Tensor& weights)
{
    return sparse_ops_native::gather_impl(grid, indices, weights);
}

#elif defined(_MSC_VER)
// -------------------------------------------------------------------
// Path B: MSVC — separate TUs compiled with /arch:AVX2 & /arch:AVX512
//         plus a CPUID dispatcher that picks the best at startup.
// -------------------------------------------------------------------
#include <intrin.h>

// Baseline (SSE2 auto-vectorisation)
#define SPARSE_OPS_NS sparse_ops_baseline
#include "sparse_ops_cpu_impl.inl"

// Forward-declare arch-specific variants (defined in their own TUs).
namespace sparse_ops_avx2 {
void scatter_add_impl(torch::Tensor, const torch::Tensor&,
                      const torch::Tensor&, const torch::Tensor&);
torch::Tensor gather_impl(const torch::Tensor&,
                           const torch::Tensor&, const torch::Tensor&);
}
namespace sparse_ops_avx512 {
void scatter_add_impl(torch::Tensor, const torch::Tensor&,
                      const torch::Tensor&, const torch::Tensor&);
torch::Tensor gather_impl(const torch::Tensor&,
                           const torch::Tensor&, const torch::Tensor&);
}

#if defined(_M_X64) || defined(_M_IX86)
// CPUID-based runtime detection (x86/x64 only).
enum class SimdLevel { SSE2, AVX2, AVX512 };

static SimdLevel detect_simd() {
    int info[4] = {};

    // Check OSXSAVE — OS supports XSAVE/XRSTOR.
    __cpuid(info, 1);
    if (!(info[2] & (1 << 27)))
        return SimdLevel::SSE2;

    // Check XCR0 for OS-enabled state components.
    unsigned long long xcr0 = _xgetbv(0);
    bool os_avx    = (xcr0 & 0x6)  == 0x6;   // XMM + YMM
    bool os_avx512 = (xcr0 & 0xE6) == 0xE6;  // + opmask + ZMM

    // Check CPU feature flags (CPUID leaf 7, sub-leaf 0).
    __cpuidex(info, 7, 0);
    bool cpu_avx2    = (info[1] & (1 << 5))  != 0;
    bool cpu_avx512f = (info[1] & (1 << 16)) != 0;

    if (cpu_avx512f && os_avx512) return SimdLevel::AVX512;
    if (cpu_avx2    && os_avx)    return SimdLevel::AVX2;
    return SimdLevel::SSE2;
}

static void scatter_add_cpu(
    torch::Tensor grid, const torch::Tensor& data,
    const torch::Tensor& indices, const torch::Tensor& weights)
{
    static const SimdLevel level = detect_simd();
    switch (level) {
        case SimdLevel::AVX512:
            sparse_ops_avx512::scatter_add_impl(grid, data, indices, weights);
            return;
        case SimdLevel::AVX2:
            sparse_ops_avx2::scatter_add_impl(grid, data, indices, weights);
            return;
        default:
            sparse_ops_baseline::scatter_add_impl(grid, data, indices, weights);
            return;
    }
}

static torch::Tensor gather_cpu(
    const torch::Tensor& grid,
    const torch::Tensor& indices, const torch::Tensor& weights)
{
    static const SimdLevel level = detect_simd();
    switch (level) {
        case SimdLevel::AVX512:
            return sparse_ops_avx512::gather_impl(grid, indices, weights);
        case SimdLevel::AVX2:
            return sparse_ops_avx2::gather_impl(grid, indices, weights);
        default:
            return sparse_ops_baseline::gather_impl(grid, indices, weights);
    }
}

#else  // MSVC on ARM — baseline only
static void scatter_add_cpu(
    torch::Tensor grid, const torch::Tensor& data,
    const torch::Tensor& indices, const torch::Tensor& weights)
{
    sparse_ops_baseline::scatter_add_impl(grid, data, indices, weights);
}
static torch::Tensor gather_cpu(
    const torch::Tensor& grid,
    const torch::Tensor& indices, const torch::Tensor& weights)
{
    return sparse_ops_baseline::gather_impl(grid, indices, weights);
}
#endif  // _M_X64 || _M_IX86

#elif (defined(__GNUC__) && !defined(__clang__)) || \
      (defined(__clang__) && !defined(__apple_build_version__) && (__clang_major__ >= 14))
// -------------------------------------------------------------------
// Path C: GCC / Clang ≥ 14 (non-Apple) — target_clones → GNU ifunc resolver.
//         The compiler emits SSE2, AVX2+FMA, and AVX-512 clones.
//         The dynamic linker picks the best at load time (zero per-
//         call overhead).
// -------------------------------------------------------------------
#define SPARSE_OPS_NS sparse_ops_baseline
#include "sparse_ops_cpu_impl.inl"

__attribute__((target_clones("avx512f", "avx2,fma", "default")))
static void scatter_add_cpu(
    torch::Tensor grid, const torch::Tensor& data,
    const torch::Tensor& indices, const torch::Tensor& weights)
{
    sparse_ops_baseline::scatter_add_impl(grid, data, indices, weights);
}

__attribute__((target_clones("avx512f", "avx2,fma", "default")))
static torch::Tensor gather_cpu(
    const torch::Tensor& grid,
    const torch::Tensor& indices, const torch::Tensor& weights)
{
    return sparse_ops_baseline::gather_impl(grid, indices, weights);
}

#else
// -------------------------------------------------------------------
// Path D: Fallback — baseline auto-vectorisation only (old Clang,
//         non-x86 with GCC, or any other compiler).
// -------------------------------------------------------------------
#define SPARSE_OPS_NS sparse_ops_baseline
#include "sparse_ops_cpu_impl.inl"

static void scatter_add_cpu(
    torch::Tensor grid, const torch::Tensor& data,
    const torch::Tensor& indices, const torch::Tensor& weights)
{
    sparse_ops_baseline::scatter_add_impl(grid, data, indices, weights);
}

static torch::Tensor gather_cpu(
    const torch::Tensor& grid,
    const torch::Tensor& indices, const torch::Tensor& weights)
{
    return sparse_ops_baseline::gather_impl(grid, indices, weights);
}

#endif  // CPU dispatch selection

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------
void scatter_add(
    torch::Tensor grid,
    torch::Tensor data,
    torch::Tensor indices,
    torch::Tensor weights)
{
    TORCH_CHECK(grid.dim() == 1, "grid must be 1-D, got ", grid.dim(), "-D");
    TORCH_CHECK(data.dim() == 1, "data must be 1-D, got ", data.dim(), "-D");
    TORCH_CHECK(indices.dim() == 1, "indices must be 1-D");
    TORCH_CHECK(weights.dim() == 1, "weights must be 1-D");
    TORCH_CHECK(data.size(0) == indices.size(0), "data/indices size mismatch");
    TORCH_CHECK(data.size(0) == weights.size(0), "data/weights size mismatch");
    TORCH_CHECK(indices.dtype() == torch::kInt64, "indices must be int64");

    grid    = grid.contiguous();
    data    = data.contiguous();
    indices = indices.contiguous();
    weights = weights.contiguous();

#ifdef COMPILE_WITH_CUDA
    if (grid.is_cuda()) {
        scatter_add_cuda(grid, data, indices, weights);
        return;
    }
#else
    if (grid.is_cuda()) {
        // ATen fallback: use index_add with pre-weighted data
        auto w_cplx = weights.to(data.dtype());
        grid.index_add_(0, indices, w_cplx * data);
        return;
    }
#endif

    scatter_add_cpu(grid, data, indices, weights);
}

torch::Tensor gather(
    torch::Tensor grid,
    torch::Tensor indices,
    torch::Tensor weights)
{
    TORCH_CHECK(grid.dim() == 1, "grid must be 1-D, got ", grid.dim(), "-D");
    TORCH_CHECK(indices.dim() == 1, "indices must be 1-D");
    TORCH_CHECK(weights.dim() == 1, "weights must be 1-D");
    TORCH_CHECK(indices.size(0) == weights.size(0), "indices/weights size mismatch");
    TORCH_CHECK(indices.dtype() == torch::kInt64, "indices must be int64");

    grid    = grid.contiguous();
    indices = indices.contiguous();
    weights = weights.contiguous();

#ifdef COMPILE_WITH_CUDA
    if (grid.is_cuda()) {
        return gather_cuda(grid, indices, weights);
    }
#else
    if (grid.is_cuda()) {
        // ATen fallback
        auto w_cplx = weights.to(grid.dtype());
        return w_cplx * grid.index_select(0, indices);
    }
#endif

    return gather_cpu(grid, indices, weights);
}

void scatter_add_binned(
    torch::Tensor grid,
    torch::Tensor data,
    torch::Tensor indices,
    torch::Tensor weights,
    torch::Tensor bin_starts,
    int64_t bin_size)
{
    TORCH_CHECK(grid.dim() == 1, "grid must be 1-D, got ", grid.dim(), "-D");
    TORCH_CHECK(data.dim() == 1, "data must be 1-D, got ", data.dim(), "-D");
    TORCH_CHECK(indices.dim() == 1, "indices must be 1-D");
    TORCH_CHECK(weights.dim() == 1, "weights must be 1-D");
    TORCH_CHECK(bin_starts.dim() == 1, "bin_starts must be 1-D");
    TORCH_CHECK(data.size(0) == indices.size(0), "data/indices size mismatch");
    TORCH_CHECK(data.size(0) == weights.size(0), "data/weights size mismatch");
    TORCH_CHECK(indices.dtype() == torch::kInt64, "indices must be int64");
    TORCH_CHECK(bin_starts.dtype() == torch::kInt64, "bin_starts must be int64");
    TORCH_CHECK(bin_size > 0, "bin_size must be positive");

    grid       = grid.contiguous();
    data       = data.contiguous();
    indices    = indices.contiguous();
    weights    = weights.contiguous();
    bin_starts = bin_starts.contiguous();

#ifdef COMPILE_WITH_CUDA
    if (grid.is_cuda()) {
        scatter_add_binned_cuda(grid, data, indices, weights, bin_starts, bin_size);
        return;
    }
#else
    if (grid.is_cuda()) {
        // ATen fallback (ignores bin info)
        auto w_cplx = weights.to(data.dtype());
        grid.index_add_(0, indices, w_cplx * data);
        return;
    }
#endif

    // CPU: clean-partition scatter (same as scatter_add — indices are sorted)
    scatter_add_cpu(grid, data, indices, weights);
}
