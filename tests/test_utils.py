"""Tests for _utils module."""

import numpy as np
import torch
import pytest

from pygrog._utils import resize, normalize_axes, rescale_coords, estimate_shape


class TestResize:
    def test_center_crop_shape(self):
        x = torch.randn(8, 8)
        assert resize(x, (4, 4)).shape == (4, 4)

    def test_center_crop_content(self):
        """Cropped output contains the center values of the original."""
        x = torch.arange(64, dtype=torch.float32).reshape(8, 8)
        y = resize(x, (4, 4))
        # Center of (8,8) is rows [2:6], cols [2:6]
        torch.testing.assert_close(y, x[2:6, 2:6])

    def test_center_pad_shape(self):
        x = torch.randn(4, 4)
        assert resize(x, (8, 8)).shape == (8, 8)

    def test_center_pad_content(self):
        """Padded output has original in center; rest is zero."""
        x = torch.ones(4, 4, dtype=torch.float32)
        y = resize(x, (8, 8))
        torch.testing.assert_close(y[2:6, 2:6], x)
        assert y[0, 0].item() == pytest.approx(0.0)
        assert y[7, 7].item() == pytest.approx(0.0)

    def test_no_change(self):
        x = torch.randn(6, 6)
        torch.testing.assert_close(resize(x, (6, 6)), x)

    def test_mixed_axes(self):
        """Crop one axis, pad the other."""
        x = torch.randn(4, 8)
        y = resize(x, (8, 4))
        assert y.shape == (8, 4)

    def test_1d(self):
        x = torch.arange(8, dtype=torch.float32)
        y = resize(x, (4,))
        # Center of length-8 is indices [2:6]
        torch.testing.assert_close(y, x[2:6])

    def test_3d(self):
        x = torch.randn(8, 8, 8)
        assert resize(x, (4, 6, 10)).shape == (4, 6, 10)


class TestNormalizeAxes:
    def test_positive(self):
        assert normalize_axes(3, [0, 2]) == [0, 2]

    def test_negative(self):
        assert normalize_axes(3, [-1, -2]) == [2, 1]

    def test_mixed(self):
        assert normalize_axes(4, [0, -1]) == [0, 3]


class TestRescaleCoords:
    def test_shape(self, rng):
        coords = rng.standard_normal((2, 100))
        shape = (32, 32)
        out = rescale_coords(coords, shape)
        assert out.shape == coords.shape

    def test_range(self, rng):
        """Rescaled coords stay within [-shape/2, shape/2]."""
        coords = rng.standard_normal((2, 100))
        shape = (32, 32)
        out = rescale_coords(coords, shape)
        for d in range(2):
            assert out[d].max() <= shape[d] / 2
            assert out[d].min() >= -shape[d] / 2

    def test_symmetric_input(self):
        """Symmetric coords map to the full [-shape/2, shape/2] range."""
        coords = np.array([[-1.0, 1.0], [-1.0, 1.0]], dtype=np.float32)
        shape = (32, 32)
        out = rescale_coords(coords, shape)
        assert abs(out[0].min() - (-16.0)) < 1.0
        assert abs(out[0].max() - 16.0) < 1.0


class TestEstimateShape:
    def test_2d(self, rng):
        coords = rng.standard_normal((2, 100))
        shape = estimate_shape(coords)
        assert len(shape) == 2
        assert all(s > 0 for s in shape)

    def test_3d(self, rng):
        coords = rng.standard_normal((3, 200))
        shape = estimate_shape(coords)
        assert len(shape) == 3
        assert all(s > 0 for s in shape)

