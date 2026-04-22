"""Regression tests for GrogInterpolator semantics and reconstruction."""

import numpy as np
import torch
from pygrog.operator import SparseFFT
from pygrog.calib import GrogInterpolator


def test_ret_image_matches_sparsefft_forward_rss():
    """ret_image should equal SparseFFT.forward(interpolate(...)) with RSS."""
    rng = np.random.default_rng(123)

    shape = (12, 12)
    image_shape = (10, 10)
    n_coils = 3
    n_views = 5
    n_readout = 7

    coords = rng.standard_normal((n_views, n_readout, 2)).astype(np.float32)

    grog = GrogInterpolator(
        shape=shape,
        coords=coords,
        kernel_width=2,
        image_shape=image_shape,
    )

    calib = (
        rng.standard_normal((n_coils, *shape))
        + 1j * rng.standard_normal((n_coils, *shape))
    ).astype(np.complex64)
    grog.calc_interp_table(calib, lamda=0.01, precision=1)

    data = (
        rng.standard_normal((n_coils, n_views, n_readout))
        + 1j * rng.standard_normal((n_coils, n_views, n_readout))
    ).astype(np.complex64)

    sparse = grog.interpolate(data, ret_image=False)
    image = grog.interpolate(data, ret_image=True)

    op = SparseFFT(plan=grog.plan)
    expected = op.forward(torch.as_tensor(np.asarray(sparse)))
    expected = expected.abs().square().sum(0).sqrt().cpu().numpy()

    np.testing.assert_allclose(image, expected, rtol=1e-5, atol=1e-5)
    assert sparse.shape[0] == n_coils
    assert sparse.shape[1] == int(grog.plan.n_samples)
    assert image.shape == image_shape


def test_interpolate_returns_sparse_neighbor_expansion_not_dense_grid():
    """interpolate(ret_image=False) must expose sparse neighbor-expanded samples."""
    rng = np.random.default_rng(321)

    shape = (14, 14)
    n_coils = 2
    n_views = 6
    n_readout = 8

    # Width-2 square kernel creates up to 4 replicas per source sample.
    coords = rng.standard_normal((n_views, n_readout, 2)).astype(np.float32)

    grog = GrogInterpolator(
        shape=shape,
        coords=coords,
        kernel_width=2,
        kernel_shape="square",
        image_shape=shape,
    )

    calib = (
        rng.standard_normal((n_coils, *shape))
        + 1j * rng.standard_normal((n_coils, *shape))
    ).astype(np.complex64)
    grog.calc_interp_table(calib, lamda=0.01, precision=1)

    data = (
        rng.standard_normal((n_coils, n_views, n_readout))
        + 1j * rng.standard_normal((n_coils, n_views, n_readout))
    ).astype(np.complex64)

    sparse = grog.interpolate(data, ret_image=False)

    # Sparse API contract: shape is (n_coils, n_samples) and uses plan.n_samples.
    assert sparse.shape == (n_coils, int(grog.plan.n_samples))

    # Dense-grid output would have exactly prod(grid_shape) samples; sparse should differ
    # for neighbor-expanded interpolation in this random setup.
    dense_grid_size = int(np.prod(grog.plan.grid_shape))
    assert sparse.shape[1] != dense_grid_size

    # Neighbor expansion should increase sample count beyond original trajectory points.
    original_points = n_views * n_readout
    assert sparse.shape[1] > original_points
