"""Tests for Toeplitz normal operators.

Covers:
  - ToeplitzOp with a pre-computed PSF:
      * Output shape
      * Self-adjointness (<Tx, y> == <x, Ty> for real PSF)
      * Positive semi-definiteness (<Tx, x>.real >= 0 for non-negative PSF)
  - toeplitz_normal(SparseFFT):
      * Result matches direct A^H A x  (uses grid_shape == image_shape so
        the approximation is exact)
      * Self-adjointness

All device-parametrized tests run on CPU; CUDA skipped when not available.
"""

import numpy as np
import torch
import pytest

from pygrog._toep._toep_op import ToeplitzOp
from pygrog.operator._sparse_fft import SparseFFT
from pygrog.operator._toeplitz import toeplitz_normal


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

class TestToeplitzOp:
    """ToeplitzOp with a pre-computed PSF tensor."""

    def test_output_shape(self, device):
        shape = (12, 16)
        psf = _real_uniform_psf(shape)
        op = ToeplitzOp(psf, ishape=shape, axes=(-2, -1))
        x = torch.randn(*shape, dtype=torch.complex64, device=device)
        assert op(x).shape == shape

    def test_output_device(self, device):
        shape = (12, 12)
        psf = _real_uniform_psf(shape)
        op = ToeplitzOp(psf, ishape=shape, axes=(-2, -1))
        x = torch.randn(*shape, dtype=torch.complex64, device=device)
        assert op(x).device.type == device.type

    def test_self_adjoint(self, device):
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

    def test_positive_semi_definite(self, device):
        """<Tx, x>.real >= 0 for a non-negative PSF."""
        shape = (16, 16)
        psf = _real_uniform_psf(shape)
        op = ToeplitzOp(psf, ishape=shape, axes=(-2, -1))
        torch.manual_seed(1)
        x = torch.randn(*shape, dtype=torch.complex64, device=device)
        inner = _cdot(op(x), x)
        assert inner.real.item() >= -1e-4, f"PSD violated: <Tx,x> = {inner}"

    def test_3d_shape(self, device):
        shape = (8, 8, 8)
        psf = _real_uniform_psf(shape)
        op = ToeplitzOp(psf, ishape=shape, axes=(-3, -2, -1))
        x = torch.randn(*shape, dtype=torch.complex64, device=device)
        assert op(x).shape == shape


# ===========================================================================
# toeplitz_normal
# ===========================================================================

class TestToeplitzNormal:
    """toeplitz_normal(op)(x) must equal op.forward(op.adjoint(x)).

    We set grid_shape == image_shape (no oversampling) so that the Toeplitz
    density approximation is exact and the two paths agree to floating-point
    precision.
    """

    # Square grid with no oversampling so all k-space samples land in the
    # center crop → Toeplitz approximation is exact.
    _SHAPE = (16, 16)
    _N_SAMPLES = 80
    _N_COILS = 3

    def _build(self, seed=0):
        torch.manual_seed(seed)
        smaps = torch.randn(self._N_COILS, *self._SHAPE, dtype=torch.complex64)
        return _make_op(self._SHAPE, self._SHAPE, self._N_SAMPLES, smaps=smaps, seed=seed)

    def test_matches_direct_aha(self, device):
        """toeplitz_normal(op)(x) == op.forward(op.adjoint(x))."""
        op = self._build(seed=30)
        normal = toeplitz_normal(op)
        x = torch.randn(*self._SHAPE, dtype=torch.complex64, device=device)
        direct = op.forward(op.adjoint(x))
        result = normal(x)
        torch.testing.assert_close(result, direct, rtol=1e-4, atol=1e-4)

    def test_self_adjoint(self, device):
        """toeplitz_normal(op) is self-adjoint: <Nx, y> == <x, Ny>."""
        op = self._build(seed=31)
        normal = toeplitz_normal(op)
        x = torch.randn(*self._SHAPE, dtype=torch.complex64, device=device)
        y = torch.randn(*self._SHAPE, dtype=torch.complex64, device=device)
        torch.testing.assert_close(
            _cdot(normal(x), y), _cdot(x, normal(y)), rtol=1e-4, atol=1e-4
        )

    def test_positive_semi_definite(self, device):
        """<Nx, x>.real >= 0 (A^H A is always PSD)."""
        op = self._build(seed=32)
        normal = toeplitz_normal(op)
        x = torch.randn(*self._SHAPE, dtype=torch.complex64, device=device)
        inner = _cdot(normal(x), x)
        assert inner.real.item() >= -1e-4, f"PSD violated: <Nx,x> = {inner}"

    def test_requires_smaps(self):
        """toeplitz_normal raises ValueError when no smaps are given."""
        op = _make_op(self._SHAPE, self._SHAPE, self._N_SAMPLES)  # no smaps
        with pytest.raises(ValueError, match="smaps"):
            toeplitz_normal(op)

