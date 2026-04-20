"""Tests for the SparseFFT operator."""

import numpy as np
import torch
import pytest

from pygrog.operator._sparse_fft import SparseFFT


@pytest.fixture
def sparse_fft_2d(grid_shape, out_shape, n_samples, rng):
    grid_size = int(np.prod(grid_shape))
    indices = rng.integers(0, grid_size, size=n_samples)
    weights = np.ones(n_samples, dtype=np.float32)
    return SparseFFT(grid_shape, out_shape, indices, weights)


class TestSparseFFT:
    def test_forward_shape(self, sparse_fft_2d, n_coils, n_samples, out_shape):
        ksp = torch.randn(n_coils, n_samples, dtype=torch.complex64)
        img = sparse_fft_2d.forward(ksp)
        assert img.shape == (n_coils, *out_shape)

    def test_adjoint_shape(self, sparse_fft_2d, n_coils, n_samples, out_shape):
        img = torch.randn(n_coils, *out_shape, dtype=torch.complex64)
        ksp = sparse_fft_2d.adjoint(img)
        assert ksp.shape == (n_coils, n_samples)

    def test_adjointness(self, sparse_fft_2d, n_coils, n_samples, out_shape):
        """Verify <Ax, y> == <x, A^H y> (dot product test)."""
        x = torch.randn(n_coils, n_samples, dtype=torch.complex64)
        y = torch.randn(n_coils, *out_shape, dtype=torch.complex64)

        Ax = sparse_fft_2d.forward(x)
        AHy = sparse_fft_2d.adjoint(y)

        lhs = torch.vdot(Ax.flatten(), y.flatten())
        rhs = torch.vdot(x.flatten(), AHy.flatten())

        assert torch.allclose(lhs, rhs, rtol=1e-4, atol=1e-5)
