"""Tests for gadgets."""

import torch
import pytest

from pygrog.gadgets._svd_extension import SubspaceProjection


class TestSubspaceProjection:
    def test_fit_and_forward(self):
        n_frames, n_spatial = 20, 100
        data = torch.randn(n_frames, n_spatial, dtype=torch.complex64)
        proj = SubspaceProjection(n_components=5)
        proj.fit(data)
        assert proj.basis.shape == (5, n_frames)

        coeff = proj.forward(data)
        assert coeff.shape == (5, n_spatial)

    def test_roundtrip(self):
        n_frames, nx, ny = 20, 8, 8
        data = torch.randn(n_frames, nx, ny, dtype=torch.complex64)
        proj = SubspaceProjection(n_components=n_frames)
        proj.fit(data.reshape(n_frames, -1))

        coeff = proj.forward(data)
        recon = proj.adjoint(coeff)
        assert torch.allclose(recon, data, atol=1e-4)
