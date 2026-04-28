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

// Declarations from toep_psf.cpp
void psf_scatter_scalar(
    torch::Tensor psf,
    torch::Tensor indices,
    torch::Tensor w_sq);

void psf_scatter_outer(
    torch::Tensor psf,
    torch::Tensor indices,
    torch::Tensor w_sq,
    torch::Tensor basis_per_sample);

void psf_scatter_outer_basis(
    torch::Tensor psf,
    torch::Tensor indices,
    torch::Tensor w_sq,
    torch::Tensor basis,
    torch::Tensor time_index);

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

    m.def("psf_scatter_scalar", &psf_scatter_scalar,
          py::arg("psf"), py::arg("indices"), py::arg("w_sq"),
          "Scalar PSF accumulation: psf[indices[i]] += w_sq[i] (in-place)");

    m.def("psf_scatter_outer", &psf_scatter_outer,
          py::arg("psf"), py::arg("indices"), py::arg("w_sq"),
          py::arg("basis_per_sample"),
          "Outer-product PSF: psf[indices[i], l, l'] += w_sq[i] * conj(b[i,l]) * b[i,l']");

    m.def("psf_scatter_outer_basis", &psf_scatter_outer_basis,
          py::arg("psf"), py::arg("indices"), py::arg("w_sq"),
          py::arg("basis"), py::arg("time_index"),
          "Subspace PSF: psf[indices[i], k, k'] += w_sq[i] * conj(B[k,t_i]) * B[k',t_i]");
}
