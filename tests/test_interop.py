"""Tests for interop modules (autograd wrappers).

Exercises the autograd-compatible forward and adjoint wrappers from
``pygrog.interop._deepinverse``.  Tests parametrized on ``device`` run on
CPU always; CUDA is skipped when no GPU is available.
"""

import torch
import numpy as np

from pygrog.operator._sparse_fft import SparseFFT
from pygrog.interop._deepinverse import sparse_fft_forward, sparse_fft_adjoint


def _make_op(n_samples=64, seed=0):
    grid_shape = (16, 16)
    out_shape = (12, 12)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, 16 * 16, size=n_samples)
    weights = np.ones(n_samples, dtype=np.float32)
    return SparseFFT(grid_shape, out_shape, indices, weights)


# ---------------------------------------------------------------------------
# Autograd wrappers must pass gradients back to the input tensor
# ---------------------------------------------------------------------------


def test_autograd_forward_produces_grad(device):
    op = _make_op()
    ksp = torch.randn(2, 64, dtype=torch.complex64, device=device, requires_grad=True)
    img = sparse_fft_forward(ksp, op)
    img.abs().sum().backward()
    assert ksp.grad is not None
    assert ksp.grad.shape == ksp.shape


def test_autograd_adjoint_produces_grad(device):
    op = _make_op()
    img = torch.randn(
        2, 12, 12, dtype=torch.complex64, device=device, requires_grad=True
    )
    ksp = sparse_fft_adjoint(img, op)
    ksp.abs().sum().backward()
    assert img.grad is not None
    assert img.grad.shape == img.shape


def test_autograd_forward_grad_nonzero(device):
    """Gradient is not identically zero (the backward actually runs)."""
    op = _make_op(seed=1)
    ksp = torch.randn(2, 64, dtype=torch.complex64, device=device, requires_grad=True)
    sparse_fft_forward(ksp, op).abs().sum().backward()
    assert ksp.grad.abs().sum().item() > 0


def test_autograd_adjoint_grad_nonzero(device):
    op = _make_op(seed=2)
    img = torch.randn(
        2, 12, 12, dtype=torch.complex64, device=device, requires_grad=True
    )
    sparse_fft_adjoint(img, op).abs().sum().backward()
    assert img.grad.abs().sum().item() > 0
