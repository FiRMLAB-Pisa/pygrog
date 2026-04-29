"""Tests for the SparseFFT operator.

Covers:
  - scatter_add / gather primitives (C++ extension exercised directly)
  - SparseFFT output shapes with/without sensitivity maps
  - Adjoint consistency: <Ax, y> == <x, A^H y> (dot-product test)
  - Batch-IFFT helper (_scatter_ifft_crop_batch) vs per-row reference
  - Batch-FFT helper (_fft_pad_gather_batch) vs per-row reference
  - Batch helper self-adjointness

All tests that receive the ``device`` fixture run on both CPU and CUDA
(CUDA cases are skipped automatically when a GPU is not available).
"""

import numpy as np
import torch
import pytest

from pygrog.operator._sparse_fft import SparseFFT, scatter_add, gather
from pygrog._base._fftc import fft, ifft

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_op(grid_shape, image_shape, n_samples, *, smaps=None, seed=0):
    rng = np.random.default_rng(seed)
    grid_size = int(np.prod(grid_shape))
    indices = rng.integers(0, grid_size, size=n_samples)
    weights = np.ones(n_samples, dtype=np.float32)
    return SparseFFT(grid_shape, image_shape, indices, weights, smaps=smaps)


def _cdot(a, b):
    """Complex dot product <a, b> = sum_i a_i^* b_i  (via torch.vdot)."""
    return torch.vdot(a.flatten(), b.flatten())


# ---------------------------------------------------------------------------
# Primitives: scatter_add and gather
# ---------------------------------------------------------------------------


def test_primitives_scatter_accumulates():
    """Two samples hitting the same bin should accumulate."""
    grid = torch.zeros(8, dtype=torch.complex64)
    data = torch.ones(3, dtype=torch.complex64)
    idx = torch.tensor([0, 0, 4], dtype=torch.int64)
    w = torch.ones(3, dtype=torch.float32)
    scatter_add(grid, data, idx, w)
    assert grid[0].real.item() == pytest.approx(2.0)
    assert grid[4].real.item() == pytest.approx(1.0)
    assert grid[1].real.item() == pytest.approx(0.0)  # untouched


def test_primitives_scatter_weight():
    """Scattered value is weight * data."""
    grid = torch.zeros(4, dtype=torch.complex64)
    data = torch.ones(2, dtype=torch.complex64)
    idx = torch.tensor([0, 2], dtype=torch.int64)
    w = torch.tensor([2.0, 0.5], dtype=torch.float32)
    scatter_add(grid, data, idx, w)
    assert grid[0].real.item() == pytest.approx(2.0)
    assert grid[2].real.item() == pytest.approx(0.5)


def test_primitives_gather_selects():
    """gather returns weight * grid[idx]."""
    grid = torch.arange(8).to(torch.complex64)
    idx = torch.tensor([0, 3, 7], dtype=torch.int64)
    w = torch.ones(3, dtype=torch.float32)
    out = gather(grid, idx, w)
    expected = torch.tensor([0.0, 3.0, 7.0], dtype=torch.complex64)
    torch.testing.assert_close(out, expected)


def test_primitives_gather_applies_weights():
    grid = torch.ones(4, dtype=torch.complex64)
    idx = torch.tensor([0, 1, 2], dtype=torch.int64)
    w = torch.tensor([2.0, 0.5, 3.0], dtype=torch.float32)
    out = gather(grid, idx, w)
    expected = torch.tensor([2.0, 0.5, 3.0], dtype=torch.complex64)
    torch.testing.assert_close(out, expected)


def test_primitives_scatter_gather_roundtrip():
    """scatter then gather recovers data for non-colliding indices."""
    grid = torch.zeros(64, dtype=torch.complex64)
    data = torch.randn(8, dtype=torch.complex64)
    idx = torch.arange(0, 64, 8, dtype=torch.int64)  # [0, 8, 16, …]
    w = torch.ones(8, dtype=torch.float32)
    scatter_add(grid, data, idx, w)
    out = gather(grid, idx, w)
    torch.testing.assert_close(out, data, rtol=1e-5, atol=1e-6)


def test_primitives_complex_scatter():
    """Scatter works on complex data."""
    grid = torch.zeros(4, dtype=torch.complex64)
    data = torch.tensor([1 + 2j, 3 + 4j], dtype=torch.complex64)
    idx = torch.tensor([0, 2], dtype=torch.int64)
    w = torch.ones(2, dtype=torch.float32)
    scatter_add(grid, data, idx, w)
    assert grid[0] == pytest.approx(complex(1, 2))
    assert grid[2] == pytest.approx(complex(3, 4))


# ---------------------------------------------------------------------------
# SparseFFT output shapes
# ---------------------------------------------------------------------------


def test_sparse_fft_forward_no_smaps(device):
    op = _make_op((32, 32), (24, 24), 256)
    ksp = torch.randn(4, 256, dtype=torch.complex64, device=device)
    img = op.adjoint(ksp)
    assert img.shape == (4, 24, 24)
    assert img.device.type == device.type


def test_sparse_fft_forward_with_smaps(device):
    smaps = torch.randn(4, 24, 24, dtype=torch.complex64)
    op = _make_op((32, 32), (24, 24), 256, smaps=smaps)
    ksp = torch.randn(4, 256, dtype=torch.complex64, device=device)
    img = op.adjoint(ksp)
    assert img.shape == (24, 24)
    assert img.device.type == device.type


def test_sparse_fft_adjoint_no_smaps(device):
    op = _make_op((32, 32), (24, 24), 256)
    img = torch.randn(4, 24, 24, dtype=torch.complex64, device=device)
    ksp = op.forward(img)
    assert ksp.shape == (4, 256)
    assert ksp.device.type == device.type


def test_sparse_fft_adjoint_with_smaps(device):
    smaps = torch.randn(4, 24, 24, dtype=torch.complex64)
    op = _make_op((32, 32), (24, 24), 256, smaps=smaps)
    img = torch.randn(24, 24, dtype=torch.complex64, device=device)
    ksp = op.forward(img)
    assert ksp.shape == (4, 256)
    assert ksp.device.type == device.type


def test_sparse_fft_call_adjoint_flag(device):
    """__call__ with adjoint=True is identical to .adjoint()."""
    op = _make_op((32, 32), (24, 24), 256)
    ksp = torch.randn(4, 256, dtype=torch.complex64, device=device)
    img = torch.randn(4, 24, 24, dtype=torch.complex64, device=device)
    torch.testing.assert_close(op(img), op.forward(img))
    torch.testing.assert_close(op(ksp, adjoint=True), op.adjoint(ksp))


def test_sparse_fft_3d_shape(device):
    op = _make_op((16, 16, 16), (12, 12, 12), 128)
    ksp = torch.randn(2, 128, dtype=torch.complex64, device=device)
    img = op.adjoint(ksp)
    assert img.shape == (2, 12, 12, 12)


# ---------------------------------------------------------------------------
# Adjointness: <Ax, y> == <x, A^H y>
# ---------------------------------------------------------------------------


def test_adjointness_no_smaps(device):
    op = _make_op((32, 32), (24, 24), 256, seed=1)
    x = torch.randn(4, 256, dtype=torch.complex64, device=device)
    y = torch.randn(4, 24, 24, dtype=torch.complex64, device=device)
    torch.testing.assert_close(
        _cdot(op.adjoint(x), y), _cdot(x, op.forward(y)), rtol=1e-4, atol=1e-5
    )


def test_adjointness_with_smaps(device):
    torch.manual_seed(0)
    smaps = torch.randn(4, 24, 24, dtype=torch.complex64)
    op = _make_op((32, 32), (24, 24), 256, smaps=smaps, seed=2)
    x = torch.randn(4, 256, dtype=torch.complex64, device=device)
    y = torch.randn(24, 24, dtype=torch.complex64, device=device)
    torch.testing.assert_close(
        _cdot(op.adjoint(x), y), _cdot(x, op.forward(y)), rtol=1e-4, atol=1e-5
    )


def test_adjointness_3d(device):
    op = _make_op((16, 16, 16), (12, 12, 12), 128, seed=3)
    x = torch.randn(2, 128, dtype=torch.complex64, device=device)
    y = torch.randn(2, 12, 12, 12, dtype=torch.complex64, device=device)
    torch.testing.assert_close(
        _cdot(op.adjoint(x), y), _cdot(x, op.forward(y)), rtol=1e-4, atol=1e-5
    )


def test_adjointness_weighted(device):
    """Non-unit density-compensation weights preserve adjointness."""
    rng = np.random.default_rng(42)
    grid_size = 32 * 32
    idx = rng.integers(0, grid_size, size=200)
    w = rng.uniform(0.5, 2.0, size=200).astype(np.float32)
    op = SparseFFT((32, 32), (24, 24), idx, w)
    x = torch.randn(3, 200, dtype=torch.complex64, device=device)
    y = torch.randn(3, 24, 24, dtype=torch.complex64, device=device)
    torch.testing.assert_close(
        _cdot(op.adjoint(x), y), _cdot(x, op.forward(y)), rtol=1e-4, atol=1e-5
    )


# ---------------------------------------------------------------------------
# Batch FFT helpers: numerical consistency with per-row reference path
# ---------------------------------------------------------------------------


def test_batch_helpers_scatter_ifft_crop_vs_loop(device):
    """_scatter_ifft_crop_batch must match per-row reference."""
    op = _make_op((32, 32), (24, 24), 256, seed=4)
    B_batch = 6
    ksp = torch.randn(B_batch, 256, dtype=torch.complex64, device=device)

    batch_out = op._scatter_ifft_crop_batch(ksp)  # (B, *image_shape)

    dev = ksp.device
    sp = op.sort_perm.to(dev)
    idx = op.indices.to(dev)
    sw = op.sqrt_weights.to(dev)

    loop_imgs = []
    for b in range(B_batch):
        grid = torch.zeros(op.grid_size, dtype=ksp.dtype, device=dev)
        scatter_add(grid, ksp[b][sp], idx, sw)
        full_img = ifft(
            grid.reshape(op.grid_shape), axes=op.fft_axes
        )  # IFFT at grid size
        img = full_img[op._pad_slices]  # center-crop image (not k-space)
        loop_imgs.append(img)
    loop_out = torch.stack(loop_imgs)

    torch.testing.assert_close(batch_out, loop_out, rtol=1e-5, atol=1e-5)


def test_batch_helpers_fft_pad_gather_vs_loop(device):
    """_fft_pad_gather_batch must match per-row reference."""
    op = _make_op((32, 32), (24, 24), 256, seed=5)
    B_batch = 6
    imgs = torch.randn(B_batch, 24, 24, dtype=torch.complex64, device=device)

    batch_out = op._fft_pad_gather_batch(imgs)  # (B, n_samples)

    dev = imgs.device
    idx = op.indices.to(dev)
    sw = op.sqrt_weights.to(dev)
    inv = op.inv_perm.to(dev)

    loop_ksps = []
    for b in range(B_batch):
        padded = torch.zeros(*op.grid_shape, dtype=imgs.dtype, device=dev)
        padded[op._pad_slices] = imgs[b]  # zero-pad image in image space
        ksp_b = gather(fft(padded, axes=op.fft_axes).reshape(-1), idx, sw)[
            inv
        ]  # FFT at grid size
        loop_ksps.append(ksp_b)
    loop_out = torch.stack(loop_ksps)

    torch.testing.assert_close(batch_out, loop_out, rtol=1e-5, atol=1e-5)


def test_batch_helpers_are_adjoints(device):
    """_scatter_ifft_crop_batch and _fft_pad_gather_batch are exact adjoints."""
    op = _make_op((32, 32), (24, 24), 256, seed=6)
    B_batch = 5
    x = torch.randn(B_batch, 256, dtype=torch.complex64, device=device)
    y = torch.randn(B_batch, 24, 24, dtype=torch.complex64, device=device)
    lhs = _cdot(op._scatter_ifft_crop_batch(x), y)
    rhs = _cdot(x, op._fft_pad_gather_batch(y))
    torch.testing.assert_close(lhs, rhs, rtol=1e-4, atol=1e-5)
