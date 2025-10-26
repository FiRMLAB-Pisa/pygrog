
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

#include <vector>
#include <complex>
#include <utility>

#include "fast_binning_cpu.h"

namespace py = pybind11;

// Python wrapper for fast_binning_add_at
void py_fast_binning_add_at(
    py::array_t<std::complex<float>> bins,
    py::array_t<std::complex<float>> points,
    py::array_t<float> weights,
    py::array_t<size_t> indices,
    py::array_t<size_t> thread_mask_flat
) {
    // Get buffer info for input validation
    auto bins_buf = bins.request();
    auto points_buf = points.request();
    auto weights_buf = weights.request();
    auto indices_buf = indices.request();
    auto mask_buf = thread_mask_flat.request();
    
    // Validate dimensions
    if (points_buf.ndim != 1 || weights_buf.ndim != 1 || 
        indices_buf.ndim != 1 || bins_buf.ndim != 1) {
        throw std::runtime_error("All arrays must be 1-dimensional");
    }
    
    if (points_buf.size != weights_buf.size || 
        points_buf.size != indices_buf.size) {
        throw std::runtime_error("points, weights, and indices must have the same size");
    }
    
    if (mask_buf.size % 2 != 0) {
        throw std::runtime_error("thread_mask must have even number of elements (pairs of start,end)");
    }
    
    // Convert numpy arrays to std::vectors
    std::vector<std::complex<float>> bins_vec(
        static_cast<std::complex<float>*>(bins_buf.ptr),
        static_cast<std::complex<float>*>(bins_buf.ptr) + bins_buf.size
    );
    
    std::vector<std::complex<float>> points_vec(
        static_cast<std::complex<float>*>(points_buf.ptr),
        static_cast<std::complex<float>*>(points_buf.ptr) + points_buf.size
    );
    
    std::vector<float> weights_vec(
        static_cast<float*>(weights_buf.ptr),
        static_cast<float*>(weights_buf.ptr) + weights_buf.size
    );
    
    std::vector<size_t> indices_vec(
        static_cast<size_t*>(indices_buf.ptr),
        static_cast<size_t*>(indices_buf.ptr) + indices_buf.size
    );
    
    // Convert flattened thread_mask to vector of pairs
    std::vector<std::pair<size_t, size_t>> thread_mask;
    size_t* mask_ptr = static_cast<size_t*>(mask_buf.ptr);
    for (size_t i = 0; i < static_cast<size_t>(mask_buf.size); i += 2) {
        thread_mask.emplace_back(mask_ptr[i], mask_ptr[i + 1]);
    }
    
    // Call the C++ function
    fast_binning_add_at(bins_vec, points_vec, weights_vec, indices_vec, thread_mask);
    
    // Copy results back to numpy array (bins is modified in-place)
    std::memcpy(bins_buf.ptr, bins_vec.data(), bins_vec.size() * sizeof(std::complex<float>));
}

// Utility function to detect SIMD level from Python
std::string py_detect_simd_level() {
    SimdLevel level = detect_simd_level();
    switch (level) {
        case SimdLevel::AVX512: return "AVX512";
        case SimdLevel::AVX: return "AVX";
        case SimdLevel::SSE: return "SSE";
        case SimdLevel::Scalar: return "Scalar";
        default: return "Unknown";
    }
}

// Create thread mask utility function
py::array_t<size_t> py_create_thread_mask(
    size_t total_size, 
    size_t num_chunks = 0,
    size_t min_chunk_size = 1000
) {
    if (num_chunks == 0) {
        // Auto-determine number of chunks based on size and available threads
        size_t max_threads = std::thread::hardware_concurrency();
        num_chunks = std::min(max_threads * 2, total_size / min_chunk_size);
        num_chunks = std::max(1UL, num_chunks);
    }
    
    std::vector<size_t> mask_flat;
    size_t chunk_size = total_size / num_chunks;
    size_t remainder = total_size % num_chunks;
    
    size_t start = 0;
    for (size_t i = 0; i < num_chunks; ++i) {
        size_t current_chunk_size = chunk_size + (i < remainder ? 1 : 0);
        size_t end = start + current_chunk_size;
        
        mask_flat.push_back(start);
        mask_flat.push_back(end);
        
        start = end;
    }
    
    // Convert to numpy array
    return py::array_t<size_t>(
        mask_flat.size(),
        mask_flat.data()
    );
}

PYBIND11_MODULE(_fast_binning, m) {
    m.doc() = "Fast SIMD-accelerated binning operations for complex arrays";
    
    m.def("fast_binning_add_at", &py_fast_binning_add_at,
          "Perform fast binning operation: bins[indices] += points * weights",
          py::arg("bins"), py::arg("points"), py::arg("weights"), 
          py::arg("indices"), py::arg("thread_mask"));
    
    m.def("detect_simd_level", &py_detect_simd_level,
          "Detect the highest SIMD instruction set available on this machine");
    
    m.def("create_thread_mask", &py_create_thread_mask,
          "Create thread mask for chunked processing",
          py::arg("total_size"), py::arg("num_chunks") = 0, 
          py::arg("min_chunk_size") = 1000);
    
    // Export version info
    m.attr("__version__") = "1.0.0";
}