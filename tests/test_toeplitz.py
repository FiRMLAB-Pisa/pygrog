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
