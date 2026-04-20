/**
 * sparse_ops_avx2.cpp — AVX2+FMA translation unit for scatter/gather.
 *
 * On GCC/Clang the #pragma sets the ISA for this entire TU — no build
 * flag needed.  On MSVC the build system passes /arch:AVX2.
 *
 * For JIT builds (PYGROG_MARCH_NATIVE defined) this file compiles empty;
 * -march=native already targets the local CPU and sparse_ops.cpp takes the
 * native-only code path.
 */

#ifndef PYGROG_MARCH_NATIVE

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC target("avx2,fma")
#endif

#define SPARSE_OPS_NS sparse_ops_avx2
#include "sparse_ops_cpu_impl.inl"

#endif  // PYGROG_MARCH_NATIVE
