"""Unit tests for the C++/CUDA PSF scatter kernels.

Each kernel is checked against a pure-Python reference using
``index_add_`` for a small random problem.  Tests run on CPU and (when
available) CUDA via the shared ``device`` fixture.

Kernels covered
---------------
* ``psf_scatter_scalar``       — real PSF (plain SparseFFT)
* ``psf_scatter_outer``        — (grid_size, M, M) PSF with per-sample
  Hermitian outer product (off-resonance / generic basis)
* ``psf_scatter_outer_basis``  — (grid_size, K, K) PSF with shared
  ``(K, T)`` basis and per-sample time index (subspace)
"""

import torch
import pytest

from pygrog.operator._sparse_fft import _get_torch_ext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ref_scalar(grid_size, indices, w_sq):
    out = torch.zeros(grid_size, dtype=w_sq.dtype, device=w_sq.device)
    out.index_add_(0, indices, w_sq)
    return out


def _ref_outer(grid_size, M, indices, w_sq, basis_per_sample):
    """Reference using einsum + index_add_."""
    outer = torch.einsum("ni,nj->nij", basis_per_sample.conj(), basis_per_sample)
    outer = outer * w_sq.to(outer.dtype).view(-1, 1, 1)
    out = torch.zeros(grid_size, M, M, dtype=outer.dtype, device=outer.device)
    out.view(grid_size, -1).index_add_(0, indices, outer.view(-1, M * M))
    return out


def _ref_outer_basis(grid_size, K, indices, w_sq, basis, time_index):
    """Reference for shared-basis variant.

    ``PSF[j, k, kp] = sum_{n in bin j} w_sq[n] * conj(basis[k, t_n]) * basis[kp, t_n]``
    """
    bn = basis[:, time_index].t().contiguous()  # (N, K)
    outer = torch.einsum("ni,nj->nij", bn.conj(), bn)
    outer = outer * w_sq.to(outer.dtype).view(-1, 1, 1)
    out = torch.zeros(grid_size, K, K, dtype=outer.dtype, device=outer.device)
    out.view(grid_size, -1).index_add_(0, indices, outer.view(-1, K * K))
    return out


# ---------------------------------------------------------------------------
# psf_scatter_scalar
# ---------------------------------------------------------------------------
def test_psf_scatter_scalar_matches_reference(device):
    ext = _get_torch_ext()
    torch.manual_seed(0)
    grid_size = 64
    n = 200
    indices = torch.randint(0, grid_size, (n,), dtype=torch.int64, device=device)
    indices, _ = torch.sort(indices)  # kernel assumes sorted
    w_sq = torch.rand(n, dtype=torch.float32, device=device) + 0.1

    psf = torch.zeros(grid_size, dtype=torch.float32, device=device)
    ext.psf_scatter_scalar(psf, indices, w_sq)
    expected = _ref_scalar(grid_size, indices, w_sq)
    torch.testing.assert_close(psf, expected, rtol=1e-6, atol=1e-6)


def test_psf_scatter_scalar_double_precision(device):
    ext = _get_torch_ext()
    torch.manual_seed(1)
    grid_size = 32
    n = 100
    indices = torch.randint(0, grid_size, (n,), dtype=torch.int64, device=device)
    indices, _ = torch.sort(indices)
    w_sq = torch.rand(n, dtype=torch.float64, device=device) + 0.1
    psf = torch.zeros(grid_size, dtype=torch.float64, device=device)
    ext.psf_scatter_scalar(psf, indices, w_sq)
    expected = _ref_scalar(grid_size, indices, w_sq)
    torch.testing.assert_close(psf, expected, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# psf_scatter_outer
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("M", [1, 3, 5])
def test_psf_scatter_outer_matches_reference(device, M):
    ext = _get_torch_ext()
    torch.manual_seed(M)
    grid_size = 32
    n = 150
    indices = torch.randint(0, grid_size, (n,), dtype=torch.int64, device=device)
    indices, _ = torch.sort(indices)
    w_sq = torch.rand(n, dtype=torch.float32, device=device) + 0.1
    basis = (torch.randn(n, M, dtype=torch.float32, device=device)
             + 1j * torch.randn(n, M, dtype=torch.float32, device=device))
    basis = basis.to(torch.complex64)

    psf = torch.zeros(grid_size, M, M, dtype=torch.complex64, device=device)
    ext.psf_scatter_outer(psf, indices, w_sq, basis)
    expected = _ref_outer(grid_size, M, indices, w_sq, basis)
    torch.testing.assert_close(psf, expected, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# psf_scatter_outer_basis
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("K,T", [(1, 8), (3, 16), (5, 24)])
def test_psf_scatter_outer_basis_matches_reference(device, K, T):
    ext = _get_torch_ext()
    torch.manual_seed(K * 100 + T)
    grid_size = 32
    n = 200
    indices = torch.randint(0, grid_size, (n,), dtype=torch.int64, device=device)
    indices, _ = torch.sort(indices)
    w_sq = torch.rand(n, dtype=torch.float32, device=device) + 0.1
    basis = (torch.randn(K, T, dtype=torch.float32, device=device)
             + 1j * torch.randn(K, T, dtype=torch.float32, device=device))
    basis = basis.to(torch.complex64)
    time_index = torch.randint(0, T, (n,), dtype=torch.int64, device=device)

    psf = torch.zeros(grid_size, K, K, dtype=torch.complex64, device=device)
    ext.psf_scatter_outer_basis(psf, indices, w_sq, basis, time_index)
    expected = _ref_outer_basis(grid_size, K, indices, w_sq, basis, time_index)
    torch.testing.assert_close(psf, expected, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Hermitian property (sanity)
# ---------------------------------------------------------------------------
def test_psf_scatter_outer_is_hermitian(device):
    ext = _get_torch_ext()
    torch.manual_seed(7)
    grid_size, M, n = 16, 4, 80
    indices = torch.randint(0, grid_size, (n,), dtype=torch.int64, device=device)
    indices, _ = torch.sort(indices)
    w_sq = torch.rand(n, dtype=torch.float32, device=device) + 0.1
    basis = (torch.randn(n, M, dtype=torch.complex64, device=device))
    psf = torch.zeros(grid_size, M, M, dtype=torch.complex64, device=device)
    ext.psf_scatter_outer(psf, indices, w_sq, basis)
    torch.testing.assert_close(psf, psf.conj().transpose(-2, -1),
                               rtol=1e-5, atol=1e-5)


def test_psf_scatter_outer_basis_is_hermitian(device):
    ext = _get_torch_ext()
    torch.manual_seed(8)
    grid_size, K, T, n = 16, 4, 12, 80
    indices = torch.randint(0, grid_size, (n,), dtype=torch.int64, device=device)
    indices, _ = torch.sort(indices)
    w_sq = torch.rand(n, dtype=torch.float32, device=device) + 0.1
    basis = torch.randn(K, T, dtype=torch.complex64, device=device)
    time_index = torch.randint(0, T, (n,), dtype=torch.int64, device=device)
    psf = torch.zeros(grid_size, K, K, dtype=torch.complex64, device=device)
    ext.psf_scatter_outer_basis(psf, indices, w_sq, basis, time_index)
    torch.testing.assert_close(psf, psf.conj().transpose(-2, -1),
                               rtol=1e-5, atol=1e-5)
