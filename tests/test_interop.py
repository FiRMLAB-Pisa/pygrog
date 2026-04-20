"""Tests for interop modules."""

import torch
import numpy as np
import pytest

from pygrog.operator._sparse_fft import SparseFFT
from pygrog.interop._deepinverse import sparse_fft_forward, sparse_fft_adjoint


class TestAutograd:
    def test_forward_grad(self):
        grid_shape = (16, 16)
        out_shape = (12, 12)
        n_samples = 64
        indices = np.random.randint(0, 256, size=n_samples)
        weights = np.ones(n_samples, dtype=np.float32)
        op = SparseFFT(grid_shape, out_shape, indices, weights)

        ksp = torch.randn(2, n_samples, dtype=torch.complex64, requires_grad=True)
        img = sparse_fft_forward(ksp, op)
        loss = img.abs().sum()
        loss.backward()
        assert ksp.grad is not None
        assert ksp.grad.shape == ksp.shape

    def test_adjoint_grad(self):
        grid_shape = (16, 16)
        out_shape = (12, 12)
        n_samples = 64
        indices = np.random.randint(0, 256, size=n_samples)
        weights = np.ones(n_samples, dtype=np.float32)
        op = SparseFFT(grid_shape, out_shape, indices, weights)

        img = torch.randn(2, *out_shape, dtype=torch.complex64, requires_grad=True)
        ksp = sparse_fft_adjoint(img, op)
        loss = ksp.abs().sum()
        loss.backward()
        assert img.grad is not None
