/**
 * wrapper.cpp — pybind11 bindings for pygrog C++ kernels.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>

#include "scatter_add.h"
#include "grog_interp.h"

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Scatter / Gather
// ---------------------------------------------------------------------------

template <typename T>
void py_scatter_add(
    py::array_t<std::complex<T>, py::array::c_style> grid,
    py::array_t<std::complex<T>, py::array::c_style> data,
    py::array_t<int64_t, py::array::c_style> indices,
    py::array_t<T, py::array::c_style> weights)
{
    auto g = grid.mutable_unchecked<1>();
    auto d = data.unchecked<1>();
    auto idx = indices.unchecked<1>();
    auto w = weights.unchecked<1>();

    int64_t n_samples = d.shape(0);
    int64_t grid_size = g.shape(0);

    pygrog::scatter_add<T>(
        g.mutable_data(0), d.data(0), idx.data(0), w.data(0),
        n_samples, grid_size);
}

template <typename T>
py::array_t<std::complex<T>> py_gather(
    py::array_t<std::complex<T>, py::array::c_style> grid,
    py::array_t<int64_t, py::array::c_style> indices)
{
    auto g = grid.unchecked<1>();
    auto idx = indices.unchecked<1>();

    int64_t n_samples = idx.shape(0);
    int64_t grid_size = g.shape(0);

    auto output = py::array_t<std::complex<T>>(n_samples);
    auto out = output.mutable_unchecked<1>();

    pygrog::gather<T>(
        out.mutable_data(0), g.data(0), idx.data(0),
        n_samples, grid_size);

    return output;
}

// ---------------------------------------------------------------------------
// GROG interpolation
// ---------------------------------------------------------------------------

template <typename T>
void py_grog_interpolate(
    py::array_t<std::complex<T>, py::array::c_style> output,
    py::array_t<std::complex<T>, py::array::c_style> data,
    py::array_t<int64_t, py::array::c_style> source_idx,
    py::array_t<int64_t, py::array::c_style> target_idx,
    py::array_t<int64_t, py::array::c_style> kernel_idx,
    py::array_t<std::complex<T>, py::array::c_style> kernel_table,
    int64_t n_coils,
    int64_t n_targets)
{
    auto out_buf = output.mutable_unchecked<1>();
    auto data_buf = data.unchecked<1>();
    auto s_buf = source_idx.unchecked<1>();
    auto t_buf = target_idx.unchecked<1>();
    auto k_buf = kernel_idx.unchecked<1>();
    auto kt_buf = kernel_table.unchecked<1>();

    int64_t n_pairs = s_buf.shape(0);
    int64_t n_kernels = kt_buf.shape(0) / (n_coils * n_coils);

    pygrog::grog_interpolate<T>(
        out_buf.mutable_data(0), data_buf.data(0),
        s_buf.data(0), t_buf.data(0), k_buf.data(0), kt_buf.data(0),
        n_coils, n_targets, n_pairs, n_kernels);
}

template <typename T>
void py_grog_interpolate_sorted(
    py::array_t<std::complex<T>, py::array::c_style> output,
    py::array_t<std::complex<T>, py::array::c_style> data,
    py::array_t<int64_t, py::array::c_style> source_idx,
    py::array_t<int64_t, py::array::c_style> target_idx,
    py::array_t<int64_t, py::array::c_style> kernel_idx,
    py::array_t<std::complex<T>, py::array::c_style> kernel_table,
    py::array_t<int64_t, py::array::c_style> pair_starts,
    int64_t n_coils,
    int64_t n_targets,
    int64_t n_sources)
{
    auto out_buf = output.mutable_unchecked<1>();
    auto data_buf = data.unchecked<1>();
    auto s_buf = source_idx.unchecked<1>();
    auto t_buf = target_idx.unchecked<1>();
    auto k_buf = kernel_idx.unchecked<1>();
    auto kt_buf = kernel_table.unchecked<1>();
    auto ps_buf = pair_starts.unchecked<1>();

    int64_t n_kernels = kt_buf.shape(0) / (n_coils * n_coils);

    pygrog::grog_interpolate_sorted<T>(
        out_buf.mutable_data(0), data_buf.data(0),
        s_buf.data(0), t_buf.data(0), k_buf.data(0), kt_buf.data(0),
        ps_buf.data(0),
        n_coils, n_targets, n_sources, n_kernels);
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

PYBIND11_MODULE(_pygrog_cpp, m) {
    m.doc() = "pygrog C++ acceleration kernels";

    // Scatter / Gather (float32)
    m.def("scatter_add_f32", &py_scatter_add<float>,
        py::arg("grid"), py::arg("data"), py::arg("indices"), py::arg("weights"),
        "Scatter-add complex64 data into a flat grid.");

    m.def("gather_f32", &py_gather<float>,
        py::arg("grid"), py::arg("indices"),
        "Gather complex64 values from a flat grid.");

    // Scatter / Gather (float64)
    m.def("scatter_add_f64", &py_scatter_add<double>,
        py::arg("grid"), py::arg("data"), py::arg("indices"), py::arg("weights"),
        "Scatter-add complex128 data into a flat grid.");

    m.def("gather_f64", &py_gather<double>,
        py::arg("grid"), py::arg("indices"),
        "Gather complex128 values from a flat grid.");

    // GROG interpolation (float32)
    m.def("grog_interpolate_f32", &py_grog_interpolate<float>,
        py::arg("output"), py::arg("data"),
        py::arg("source_idx"), py::arg("target_idx"), py::arg("kernel_idx"),
        py::arg("kernel_table"), py::arg("n_coils"), py::arg("n_targets"),
        "GROG interpolation for complex64 data.");

    m.def("grog_interpolate_sorted_f32", &py_grog_interpolate_sorted<float>,
        py::arg("output"), py::arg("data"),
        py::arg("source_idx"), py::arg("target_idx"), py::arg("kernel_idx"),
        py::arg("kernel_table"), py::arg("pair_starts"),
        py::arg("n_coils"), py::arg("n_targets"), py::arg("n_sources"),
        "GROG interpolation (sorted by target) for complex64 data.");

    // GROG interpolation (float64)
    m.def("grog_interpolate_f64", &py_grog_interpolate<double>,
        py::arg("output"), py::arg("data"),
        py::arg("source_idx"), py::arg("target_idx"), py::arg("kernel_idx"),
        py::arg("kernel_table"), py::arg("n_coils"), py::arg("n_targets"),
        "GROG interpolation for complex128 data.");

    m.def("grog_interpolate_sorted_f64", &py_grog_interpolate_sorted<double>,
        py::arg("output"), py::arg("data"),
        py::arg("source_idx"), py::arg("target_idx"), py::arg("kernel_idx"),
        py::arg("kernel_table"), py::arg("pair_starts"),
        py::arg("n_coils"), py::arg("n_targets"), py::arg("n_sources"),
        "GROG interpolation (sorted by target) for complex128 data.");
}
