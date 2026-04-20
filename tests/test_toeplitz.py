"""Tests for Toeplitz operator."""

import torch
import pytest

from pygrog._toep._toep_op import ToeplitzOp


class TestToeplitzOp:
    def test_call_shape(self):
        shape = (16, 16)
        # Create a dummy PSF kernel (2x oversampled)
        oshape = tuple(2 * s for s in shape)
        psf = torch.randn(*oshape, dtype=torch.complex64)
        op = ToeplitzOp(psf, ishape=shape, axes=(-2, -1))
        x = torch.randn(*shape, dtype=torch.complex64)
        y = op(x)
        assert y.shape == shape
