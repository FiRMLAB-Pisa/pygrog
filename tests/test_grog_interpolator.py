"""Regression tests for GrogInterpolator semantics and reconstruction."""

import numpy as np
import pytest
import torch
from pygrog.operator import SparseFFT
from pygrog.calib import GrogInterpolator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_grog(rng, shape, n_coils, n_views, n_readout, image_shape=None, **kw):
    """Create and fully initialise a GrogInterpolator."""
    if image_shape is None:
        image_shape = shape
    coords = rng.standard_normal((n_views, n_readout, 2)).astype(np.float32)
    grog = GrogInterpolator(
        shape=shape,
        coords=coords,
        kernel_width=2,
        oversamp=2.0,
        image_shape=image_shape,
        **kw,
    )
    calib = (
        rng.standard_normal((n_coils, *shape))
        + 1j * rng.standard_normal((n_coils, *shape))
    ).astype(np.complex64)
    grog.calc_interp_table(calib, lamda=0.01, precision=1)
    return grog, coords


def _random_kspace(rng, n_coils, n_views, n_readout):
    return (
        rng.standard_normal((n_coils, n_views, n_readout))
        + 1j * rng.standard_normal((n_coils, n_views, n_readout))
    ).astype(np.complex64)


def test_ret_image_matches_sparsefft_forward_rss():
    """ret_image should equal SparseFFT.forward(sqrt_w * interpolate(...)) with RSS."""
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
        oversamp=2.0,
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

    # Caller must pre-multiply by plan.pre_weights before passing to
    # SparseFFT.adjoint — this is what ret_image=True does internally.
    op = SparseFFT(plan=grog.plan)
    sqrt_w = grog.plan.pre_weights
    sparse_t = torch.as_tensor(np.asarray(sparse))
    sparse_weighted = sparse_t * sqrt_w.to(sparse_t.dtype).unsqueeze(0)
    expected = op.adjoint(sparse_weighted)
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
        oversamp=2.0,
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


# ---------------------------------------------------------------------------
# Error handling: interpolate before calc_interp_table
# ---------------------------------------------------------------------------


def test_interpolate_raises_before_calc_interp_table():
    """interpolate() must raise RuntimeError if calc_interp_table() was never called."""
    rng = np.random.default_rng(7)
    shape = (10, 10)
    n_coils = 2
    n_views = 3
    n_readout = 5

    coords = rng.standard_normal((n_views, n_readout, 2)).astype(np.float32)
    grog = GrogInterpolator(shape=shape, coords=coords, image_shape=shape)

    data = _random_kspace(rng, n_coils, n_views, n_readout)

    with pytest.raises(RuntimeError, match="calc_interp_table"):
        grog.interpolate(data)


# ---------------------------------------------------------------------------
# Shot-by-shot gridding
# ---------------------------------------------------------------------------


def test_shot_by_shot_returns_correct_grid_shape():
    """collect_shots() must return a tensor of shape (n_coils, *grid_shape)."""
    rng = np.random.default_rng(99)
    shape = (12, 12)
    n_coils = 3
    n_views = 4
    n_readout = 6

    grog, _ = _make_grog(rng, shape, n_coils, n_views, n_readout)
    data = _random_kspace(rng, n_coils, n_views, n_readout)

    for v in range(n_views):
        result = grog(data[:, v, :], shot_index=v)
        assert result is None  # accumulating shots returns None

    gridded = grog.collect_shots()
    assert gridded.shape == (n_coils, *grog.plan.grid_shape)


def test_collect_shots_raises_before_any_shots():
    """collect_shots() must raise RuntimeError when no shots have been accumulated."""
    rng = np.random.default_rng(55)
    shape = (10, 10)
    n_coils = 2
    n_views = 3
    n_readout = 5

    grog, _ = _make_grog(rng, shape, n_coils, n_views, n_readout)

    with pytest.raises(RuntimeError, match="No shots accumulated"):
        grog.collect_shots()
