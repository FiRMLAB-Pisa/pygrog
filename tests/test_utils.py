"""Tests for _utils module."""

import numpy as np
import torch
import pytest

from pygrog._utils import resize, normalize_axes, rescale_coords, estimate_shape


class TestResize:
    def test_center_crop(self):
        x = torch.randn(8, 8)
        y = resize(x, (4, 4))
        assert y.shape == (4, 4)

    def test_center_pad(self):
        x = torch.randn(4, 4)
        y = resize(x, (8, 8))
        assert y.shape == (8, 8)

    def test_no_change(self):
        x = torch.randn(6, 6)
        y = resize(x, (6, 6))
        assert torch.allclose(x, y)

    def test_mixed(self):
        x = torch.randn(4, 8)
        y = resize(x, (8, 4))
        assert y.shape == (8, 4)


class TestNormalizeAxes:
    def test_positive(self):
        assert normalize_axes(3, [0, 2]) == [0, 2]

    def test_negative(self):
        assert normalize_axes(3, [-1, -2]) == [2, 1]


class TestRescaleCoords:
    def test_shape(self, rng):
        coords = rng.standard_normal((2, 100))
        shape = (32, 32)
        out = rescale_coords(coords, shape)
        assert out.shape == coords.shape

    def test_range(self, rng):
        coords = rng.standard_normal((2, 100))
        shape = (32, 32)
        out = rescale_coords(coords, shape)
        for d in range(2):
            assert out[d].max() <= shape[d] / 2
            assert out[d].min() >= -shape[d] / 2


class TestEstimateShape:
    def test_2d(self, rng):
        coords = rng.standard_normal((2, 100))
        shape = estimate_shape(coords)
        assert len(shape) == 2
        assert all(s > 0 for s in shape)
