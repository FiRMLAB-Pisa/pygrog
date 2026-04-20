/**
 * module.cpp — Single pybind11 entry point for _pygrog_torch.
 *
 * Registers all C++ ops: grog_interpolate, scatter_add, gather.
 */

#include <torch/extension.h>

// Declarations from grog_interp.cpp
torch::Tensor grog_interpolate(
    torch::Tensor data,
    torch::Tensor indexes,
    torch::Tensor kernel);

// Declarations from sparse_ops.cpp
void scatter_add(
    torch::Tensor grid,
    torch::Tensor data,
    torch::Tensor indices,
    torch::Tensor weights);

void scatter_add_binned(
    torch::Tensor grid,
    torch::Tensor data,
    torch::Tensor indices,
    torch::Tensor weights,
    torch::Tensor bin_starts,
    int64_t bin_size);

torch::Tensor gather(
    torch::Tensor grid,
    torch::Tensor indices,
    torch::Tensor weights);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "PyGROG C++/CUDA extension: GROG interpolation + sparse scatter/gather";

    m.def("grog_interpolate", &grog_interpolate,
          py::arg("data"), py::arg("indexes"), py::arg("kernel"),
          "Apply per-sample GRAPPA kernels: out[n,b,:] = kernel[indexes[n]] @ data[n,b,:]");

    m.def("scatter_add", &scatter_add,
          py::arg("grid"), py::arg("data"), py::arg("indices"), py::arg("weights"),
          "Scatter-add: grid[indices[i]] += weights[i] * data[i] (in-place)");

    m.def("scatter_add_binned", &scatter_add_binned,
          py::arg("grid"), py::arg("data"), py::arg("indices"), py::arg("weights"),
          py::arg("bin_starts"), py::arg("bin_size"),
          "Binned scatter-add with shared-memory accumulation on GPU (in-place)");

    m.def("gather", &gather,
          py::arg("grid"), py::arg("indices"), py::arg("weights"),
          "Gather: out[i] = weights[i] * grid[indices[i]]");
}
