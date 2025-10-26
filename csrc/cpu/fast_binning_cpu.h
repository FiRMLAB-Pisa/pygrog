#ifndef FAST_BINNING_CPU
#define FAST_BINNING_CPU

#include <vector>
#include <complex>
#include <thread>
#include <iostream>
#include <utility>  // For std::pair
#include <algorithm>  // For std::min
#include <stdexcept>  // For std::invalid_argument

// SIMD intrinsics headers
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86_FP)
#include <xmmintrin.h>
#include <emmintrin.h>
#endif
#if defined(__AVX__) || defined(__AVX2__)
#include <immintrin.h>
#endif
#ifdef _MSC_VER
#include <intrin.h>
#endif

// Custom SIMD dispatch: runtime detection and selection
enum class SimdLevel { Scalar, SSE, AVX, AVX512 };

inline SimdLevel detect_simd_level() {
    #ifdef __GNUC__  // GCC/Clang
        if (__builtin_cpu_supports("avx512f")) return SimdLevel::AVX512;
        if (__builtin_cpu_supports("avx2")) return SimdLevel::AVX;
        if (__builtin_cpu_supports("sse4.2")) return SimdLevel::SSE;
    #elif defined(_MSC_VER)  // MSVC
        #ifdef __AVX512F__
        if (__isa_available >= __ISA_AVAILABLE_AVX512) return SimdLevel::AVX512;
        #endif
        #ifdef __AVX2__
        if (__isa_available >= __ISA_AVAILABLE_AVX2) return SimdLevel::AVX;
        #endif
        #ifdef __SSE4_2__
        if (__isa_available >= __ISA_AVAILABLE_SSE42) return SimdLevel::SSE;
        #endif
    #endif
    return SimdLevel::Scalar;  // Fallback
}

// SIMD-accelerated batches using intrinsics (custom dispatch)
inline void accumulate_batch(const std::complex<float>* points, const size_t* indices, size_t start, size_t end, std::complex<float>* bins, SimdLevel level) {
#if defined(__AVX512F__) || (defined(_MSC_VER) && defined(__AVX512F__))
    if (level == SimdLevel::AVX512 && (end - start) >= 8) {
        // AVX-512: 8 complex floats (16 float values)
        for (size_t i = start; i < end; i += 8) {
            size_t remaining = end - i;
            size_t batch_size = std::min(8UL, remaining);
            
            // Load 8 complex numbers (16 floats) - real0,imag0,real1,imag1,...,real7,imag7
            __m512 p_vec = _mm512_loadu_ps(reinterpret_cast<const float*>(&points[i]));
            
            // Extract and accumulate results (no weighting)
            float result_data[16];
            _mm512_storeu_ps(result_data, p_vec);
            for (size_t j = 0; j < batch_size; ++j) {
                size_t idx = indices[i + j];
                bins[idx] += std::complex<float>(result_data[j * 2], result_data[j * 2 + 1]);
            }
        }
    } else
#endif
#if defined(__AVX__) || defined(__AVX2__) || (defined(_MSC_VER) && defined(__AVX__))
    if (level == SimdLevel::AVX && (end - start) >= 4) {
        // AVX: 4 complex floats (8 float values)
        for (size_t i = start; i < end; i += 4) {
            size_t remaining = end - i;
            size_t batch_size = std::min(4UL, remaining);
            
            // Load 4 complex numbers (8 floats) - real0,imag0,real1,imag1,real2,imag2,real3,imag3
            __m256 p_vec = _mm256_loadu_ps(reinterpret_cast<const float*>(&points[i]));
            
            // Extract and accumulate results (no weighting)
            float result_data[8];
            _mm256_storeu_ps(result_data, p_vec);
            for (size_t j = 0; j < batch_size; ++j) {
                size_t idx = indices[i + j];
                bins[idx] += std::complex<float>(result_data[j * 2], result_data[j * 2 + 1]);
            }
        }
    } else
#endif
#if defined(__SSE__) || defined(__SSE2__) || (defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86_FP)))
    if (level == SimdLevel::SSE && (end - start) >= 2) {
        // SSE: 2 complex floats (4 float values)
        for (size_t i = start; i < end; i += 2) {
            size_t remaining = end - i;
            size_t batch_size = std::min(2UL, remaining);
            
            // Load 2 complex numbers (4 floats) - real0,imag0,real1,imag1
            __m128 p_vec = _mm_loadu_ps(reinterpret_cast<const float*>(&points[i]));
            
            // Extract and accumulate results (no weighting)
            float result_data[4];
            _mm_storeu_ps(result_data, p_vec);
            for (size_t j = 0; j < batch_size; ++j) {
                size_t idx = indices[i + j];
                bins[idx] += std::complex<float>(result_data[j * 2], result_data[j * 2 + 1]);
            }
        }
    } else
#endif
    {  // Scalar fallback
        for (size_t i = start; i < end; ++i) {
            size_t idx = indices[i];
            bins[idx] += points[i];
        }
    }
}

// Numpy add.at style: bins += points at indices, with presorted data and thread mask
inline void fast_binning_add_at(
    std::vector<std::complex<float>>& bins,  // Output (modified in-place)
    const std::vector<std::complex<float>>& points,
    const std::vector<size_t>& indices,  // Presorted
    const std::vector<std::pair<size_t, size_t>>& thread_mask  // Chunks: list of (start, end) tuples
) {
    size_t N = points.size();
    if (N != indices.size()) {
        throw std::invalid_argument("Input sizes mismatch");
    }
    
    // Validate thread_mask ranges
    for (const auto& chunk : thread_mask) {
        if (chunk.first > chunk.second || chunk.second > N) {
            throw std::invalid_argument("Invalid thread_mask range");
        }
    }
        
    // Detect SIMD level once
    SimdLevel level = detect_simd_level();
    
    // Multithreaded processing over chunks using std::thread (standard C++)
    size_t num_chunks = thread_mask.size();
    size_t num_threads = std::min(num_chunks, static_cast<size_t>(std::thread::hardware_concurrency()));
    std::vector<std::thread> threads;
    
    // Partition chunks among threads
    size_t chunks_per_thread = num_chunks / num_threads;
    size_t extra_chunks = num_chunks % num_threads;
    
    size_t chunk_start = 0;
    for (size_t t = 0; t < num_threads; ++t) {
        size_t chunk_end = chunk_start + chunks_per_thread + (t < extra_chunks ? 1 : 0);
        threads.emplace_back([&, chunk_start, chunk_end, level]() {
            for (size_t c = chunk_start; c < chunk_end; ++c) {
                size_t start = thread_mask[c].first;
                size_t end = thread_mask[c].second;
                accumulate_batch(points.data(), indices.data(), start, end, bins.data(), level);
            }
        });
        chunk_start = chunk_end;
    }
    
    // Join threads
    for (auto& th : threads) {
        th.join();
    }
}

#endif // FAST_BINNING_CPU