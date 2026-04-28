"""Tests for ToeplitzOp (internal Toeplitz NUFFT accelerator used by NLINV).

Covers:
  - ToeplitzOp with a pre-computed PSF:
      * Output shape
      * Self-adjointness (<Tx, y> == <x, Ty> for real PSF)
      * Positive semi-definiteness (<Tx, x>.real >= 0 for non-negative PSF)

All device-parametrized tests run on CPU; CUDA skipped when not available.
"""

import torch

from pygrog._toep._toep_op import ToeplitzOp

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cdot(a, b):
    return torch.vdot(a.flatten(), b.flatten())


def _real_uniform_psf(ishape, os_factor=2):
    """PSF in frequency domain representing uniform sampling density.

    A real-valued, non-negative PSF makes ToeplitzOp self-adjoint and PSD.
    Using all-ones here is equivalent to a fully-sampled Cartesian grid
    (constant density in k-space).
    """
    os_shape = tuple(s * os_factor for s in ishape)
    return torch.ones(*os_shape, dtype=torch.complex64)


# ===========================================================================
# ToeplitzOp
# ===========================================================================


def test_toeplitz_op_output_shape(device):
    shape = (12, 16)
    psf = _real_uniform_psf(shape)
    op = ToeplitzOp(psf, ishape=shape, axes=(-2, -1))
    x = torch.randn(*shape, dtype=torch.complex64, device=device)
    assert op(x).shape == shape


def test_toeplitz_op_output_device(device):
    shape = (12, 12)
    psf = _real_uniform_psf(shape)
    op = ToeplitzOp(psf, ishape=shape, axes=(-2, -1))
    x = torch.randn(*shape, dtype=torch.complex64, device=device)
    assert op(x).device.type == device.type


def test_toeplitz_op_self_adjoint(device):
    """<Tx, y> == <x, Ty> for a real-valued PSF."""
    shape = (16, 16)
    psf = _real_uniform_psf(shape)
    op = ToeplitzOp(psf, ishape=shape, axes=(-2, -1))
    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=torch.complex64, device=device)
    y = torch.randn(*shape, dtype=torch.complex64, device=device)
    lhs = _cdot(op(x), y)
    rhs = _cdot(x, op(y))
    torch.testing.assert_close(lhs, rhs, rtol=1e-4, atol=1e-5)


def test_toeplitz_op_positive_semi_definite(device):
    """<Tx, x>.real >= 0 for a non-negative PSF."""
    shape = (16, 16)
    psf = _real_uniform_psf(shape)
    op = ToeplitzOp(psf, ishape=shape, axes=(-2, -1))
    torch.manual_seed(1)
    x = torch.randn(*shape, dtype=torch.complex64, device=device)
    inner = _cdot(op(x), x)
    assert inner.real.item() >= -1e-4, f"PSD violated: <Tx,x> = {inner}"


def test_toeplitz_op_3d_shape(device):
    shape = (8, 8, 8)
    psf = _real_uniform_psf(shape)
    op = ToeplitzOp(psf, ishape=shape, axes=(-3, -2, -1))
    x = torch.randn(*shape, dtype=torch.complex64, device=device)
    assert op(x).shape == shape


# ===========================================================================
# High-level Toeplitz `.normal()` for SparseFFT / ORC / Subspace
# ===========================================================================
import numpy as np
import pytest
from pygrog.operator import SparseFFT
from pygrog.gadgets import with_subspace
from pygrog.gadgets._off_resonance import OffResonanceSparseFFT
from pygrog.gadgets._subspace import SubspaceSparseFFT  # noqa: F401


def _make_sparse_fft(grid_shape, n_samples, *, n_coils=None, osf=1, seed=0,
                     toeplitz=True):
    image_shape = tuple(s // osf for s in grid_shape)
    rng = np.random.default_rng(seed)
    grid_size = int(np.prod(grid_shape))
    indices = torch.from_numpy(
        rng.integers(0, grid_size, size=n_samples).astype(np.int64))
    weights = torch.from_numpy(
        (rng.random(n_samples).astype(np.float32) + 0.1))
    smaps = None
    if n_coils is not None:
        smaps = torch.from_numpy(
            (rng.standard_normal((n_coils, *image_shape))
             + 1j * rng.standard_normal((n_coils, *image_shape))
             ).astype(np.complex64) * 0.5)
    return SparseFFT(grid_shape, image_shape, indices, weights,
                     smaps=smaps, toeplitz=toeplitz)


def _err(a, b):
    return (a - b).abs().max().item() / (b.abs().max().item() + 1e-30)


@pytest.mark.parametrize("osf", [1, 2])
@pytest.mark.parametrize("with_smaps", [False, True])
def test_sparse_fft_normal_matches_nested(osf, with_smaps):
    grid = (32 * osf, 32 * osf)
    op_t = _make_sparse_fft(grid, 600, n_coils=4 if with_smaps else None,
                            osf=osf, toeplitz=True)
    op_n = _make_sparse_fft(grid, 600, n_coils=4 if with_smaps else None,
                            osf=osf, toeplitz=False)
    image_shape = op_t.image_shape
    if with_smaps:
        x = torch.randn(*image_shape, dtype=torch.complex64)
    else:
        x = torch.randn(op_t.smaps.shape[0] if op_t.smaps is not None else 1,
                        *image_shape, dtype=torch.complex64)
    y_t = op_t.normal(x)
    y_n = op_n.forward(op_n.adjoint(op_n.forward(x))) if False else op_n.normal(x)
    # op_n.toeplitz is False so normal falls back to forward(adjoint(.))
    assert _err(y_t, y_n) < 1e-4


@pytest.mark.parametrize("osf", [1, 2])
def test_orc_normal_matches_nested(osf):
    grid = (32 * osf, 32 * osf)
    image_shape = (32, 32)
    n_samples = 600
    L = 4
    rng = np.random.default_rng(11)
    indices = torch.from_numpy(rng.integers(0, np.prod(grid), n_samples).astype(np.int64))
    weights = torch.from_numpy((rng.random(n_samples).astype(np.float32) + 0.1))
    smaps = torch.from_numpy((rng.standard_normal((4, *image_shape))
                              + 1j * rng.standard_normal((4, *image_shape))
                              ).astype(np.complex64) * 0.5)
    base = SparseFFT(grid, image_shape, indices, weights, smaps=smaps,
                     toeplitz=True)
    base_n = SparseFFT(grid, image_shape, indices, weights, smaps=smaps,
                       toeplitz=False)
    B = torch.from_numpy((rng.standard_normal((n_samples, L))
                          + 1j * rng.standard_normal((n_samples, L))
                          ).astype(np.complex64))
    C = torch.from_numpy((rng.standard_normal((L, *image_shape))
                          + 1j * rng.standard_normal((L, *image_shape))
                          ).astype(np.complex64))
    op_t = OffResonanceSparseFFT(base, B, C, toeplitz=True)
    op_n = OffResonanceSparseFFT(base_n, B, C, toeplitz=False)
    x = torch.randn(*image_shape, dtype=torch.complex64)
    y_t = op_t.normal(x)
    y_n = op_n.normal(x)
    assert _err(y_t, y_n) < 1e-4


def _make_subspace_base(grid, image_shape, T, n_pts, n_coils, *,
                        toeplitz, seed=13):
    """Build a SparseFFT with 2-D natural_shape ``(T, n_pts)``."""
    import types
    n_samples = T * n_pts
    rng = np.random.default_rng(seed)
    indices = torch.from_numpy(
        rng.integers(0, int(np.prod(grid)), n_samples).astype(np.int64))
    weights = torch.from_numpy(
        (rng.random(n_samples).astype(np.float32) + 0.1))
    smaps = torch.from_numpy((rng.standard_normal((n_coils, *image_shape))
                              + 1j * rng.standard_normal((n_coils, *image_shape))
                              ).astype(np.complex64) * 0.5)
    sort_perm = torch.argsort(indices)
    inv_perm = torch.empty_like(sort_perm)
    inv_perm[sort_perm] = torch.arange(n_samples)
    plan = types.SimpleNamespace(
        grid_shape=tuple(grid), image_shape=tuple(image_shape),
        grid_size=int(np.prod(grid)),
        indices=indices[sort_perm],
        sqrt_weights=torch.sqrt(weights)[sort_perm],
        sort_perm=sort_perm, inv_perm=inv_perm,
        natural_shape=(T, n_pts), n_samples=n_samples,
    )
    return SparseFFT(plan=plan, smaps=smaps, toeplitz=toeplitz)


@pytest.mark.parametrize("osf", [1, 2])
def test_subspace_normal_matches_nested(osf):
    grid = (32 * osf, 32 * osf)
    image_shape = (32, 32)
    T = 12
    K = 3
    n_pts = 50
    base = _make_subspace_base(grid, image_shape, T, n_pts, n_coils=4,
                               toeplitz=True)
    base_n = _make_subspace_base(grid, image_shape, T, n_pts, n_coils=4,
                                 toeplitz=False)
    rng = np.random.default_rng(17)
    basis = torch.from_numpy((rng.standard_normal((K, T))
                              + 1j * rng.standard_normal((K, T))
                              ).astype(np.complex64))
    op_t = with_subspace(base, basis, encoding_axis=-2, toeplitz=True)
    op_n = with_subspace(base_n, basis, encoding_axis=-2, toeplitz=False)
    x = torch.randn(K, *image_shape, dtype=torch.complex64)
    y_t = op_t.normal(x)
    y_n = op_n.normal(x)
    assert _err(y_t, y_n) < 1e-4


# ---------------------------------------------------------------------------
# Auto-toggle: CPU defaults to Toeplitz on
# ---------------------------------------------------------------------------
def test_sparse_fft_toeplitz_auto_cpu():
    op = _make_sparse_fft((32, 32), 100, n_coils=2, toeplitz=None)
    assert op.toeplitz is True
