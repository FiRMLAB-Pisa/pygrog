"""Test GROG interpolator."""

import pytest
import numpy as np

from numpy.testing import assert_allclose

from pygrog.grog import _GrogInterpolator


def test_initialization(cartesian_2d_data):
    """Test that the interpolator initializes correctly."""
    _, coords, shape = cartesian_2d_data

    # Basic initialization
    interp = _GrogInterpolator(coords, shape)
    assert interp.shape == (16, 16)
    assert interp.oversamp == 1.0
    assert interp.precision == 1
    assert interp.weighting_mode == "count"
    assert interp.nsteps == 11  # For precision=1

    # With explicit parameters
    interp = _GrogInterpolator(
        coords, shape, oversamp=1.5, precision=2, weighting_mode="distance"
    )
    assert interp.shape == (16, 16)
    assert interp.oversamp == 1.5
    assert interp.precision == 2
    assert interp.weighting_mode == "distance"
    assert interp.nsteps == 101  # For precision=2


def test_invalid_weighting_mode(cartesian_2d_data):
    """Test that invalid weighting mode raises an error."""
    _, coords, shape = cartesian_2d_data

    with pytest.raises(ValueError):
        _GrogInterpolator(coords, shape, weighting_mode="invalid")


def test_kernels_setting_2d(cartesian_2d_data, identity_interpolator):
    """Test setting 2D GRAPPA kernels."""
    _, coords, shape = cartesian_2d_data

    interp = _GrogInterpolator(coords, shape)
    interp.set_kernels(identity_interpolator)

    assert interp._kernels_set
    assert interp._n_coils == 4
    assert interp._grog_table.shape == (11 * 11, 4, 4)  # nsteps^2 x nc x nc


def test_kernels_setting_3d(cartesian_3d_data, identity_interpolator_3d):
    """Test setting 3D GRAPPA kernels."""
    _, coords, shape = cartesian_3d_data

    interp = _GrogInterpolator(coords, shape)
    interp.set_kernels(identity_interpolator_3d)

    assert interp._kernels_set
    assert interp._n_coils == 4
    assert interp._grog_table.shape == (11 * 11 * 11, 4, 4)  # nsteps^3 x nc x nc


def test_missing_kernels(cartesian_2d_data):
    """Test error handling when GRAPPA kernels are missing."""
    _, coords, shape = cartesian_2d_data

    interp = _GrogInterpolator(coords, shape)

    # Missing x kernel
    with pytest.raises(ValueError):
        interp.set_kernels({"y": np.eye(4)})

    # Missing y kernel
    with pytest.raises(ValueError):
        interp.set_kernels({"x": np.eye(4)})


def test_missing_z_kernel_for_3d(cartesian_3d_data):
    """Test error when missing z kernel for 3D data."""
    _, coords, shape = cartesian_3d_data

    interp_3d = _GrogInterpolator(coords, shape)
    with pytest.raises(ValueError):
        interp_3d.set_kernels({"x": np.eye(4), "y": np.eye(4)})


def test_call_without_kernels(cartesian_2d_data):
    """Test error when calling without setting kernels first."""
    data, coords, shape = cartesian_2d_data
    interp = _GrogInterpolator(coords, shape)
    with pytest.raises(RuntimeError):
        interp(data)


def test_interpolation_2d(cartesian_2d_data, identity_interpolator):
    """Test interpolation with 2D Cartesian grid data."""
    data, coords, shape = cartesian_2d_data
    interp = _GrogInterpolator(coords, shape)
    interp.set_kernels(identity_interpolator)

    output, indexes, weights = interp(data)

    # Check shapes
    assert output.shape == data.shape
    assert indexes.shape == coords.shape
    assert weights.shape == (*coords.shape[:-1], 1)

    # Since points are on grid and kernels are identity matrices,
    # output should be very close to input
    assert_allclose(output, data, rtol=1e-5)

    # Indexes should correspond to coordinates plus grid offset
    expected_indexes = np.asarray(shape) * coords + np.asarray(shape) // 2
    assert_allclose(indexes, expected_indexes, rtol=1e-10)

    # Each point should have weight 1.0 since they're on grid positions
    assert_allclose(weights, np.ones(weights.shape), rtol=1e-5)


def test_interpolation_3d(cartesian_3d_data, identity_interpolator_3d):
    """Test interpolation with 3D Cartesian grid data."""
    data, coords, shape = cartesian_3d_data
    interp = _GrogInterpolator(coords, shape)
    interp.set_kernels(identity_interpolator_3d)

    output, indexes, weights = interp(data)

    # Check shapes
    assert output.shape == data.shape
    assert indexes.shape == coords.shape
    assert weights.shape == (*coords.shape[:-1], 1)

    # Since points are on grid and kernels are identity matrices,
    # output should be very close to input
    assert_allclose(output, data, rtol=1e-5)

    # Indexes should correspond to coordinates plus grid offset
    expected_indexes = np.asarray(shape) * coords + np.asarray(shape) // 2
    assert_allclose(indexes, expected_indexes, rtol=1e-10)

    # Each point should have weight 1.0 since they're on grid positions
    assert_allclose(weights, np.ones(weights.shape), rtol=1e-5)


def test_shot_by_shot_processing(identity_interpolator):
    """Test shot-by-shot processing functionality."""
    n_coils = 4
    n_readout = 16
    n_views = 2

    # Create Cartesian coordinates for a simple trajectory
    coords = np.zeros((n_views, n_readout, 2), dtype=np.float32)
    for v in range(n_views):
        for r in range(n_readout):
            coords[v, r, 0] = -0.5 + r / n_readout  # x coordinate
            coords[v, r, 1] = -0.5 + v / n_views  # y coordinate

    # Create test data with coils as rightmost dimension
    data = np.zeros((n_views, n_readout, n_coils), dtype=np.complex64)

    # Set a signal point in each view
    data[0, 8, :] = 1.0 + 0j
    data[1, 4, :] = 0.5 + 0j

    # Create interpolator and set kernels
    grog = _GrogInterpolator(coords=coords, shape=(16, 16))
    grog.set_kernels(identity_interpolator)

    # Process whole dataset
    whole_output, whole_indices, whole_weights = grog(data)

    # Process shot-by-shot
    shot_outputs = []
    shot_indices = []
    shot_weights = []
    for v in range(n_views):
        # Extract single shot data: (readout, coils)
        shot_data = data[v]
        # Process with shot index
        output, indices, weights = grog(shot_data, shot_index=(v,))
        shot_outputs.append(output)
        shot_indices.append(indices)
        shot_weights.append(weights)

    # Verify shot-by-shot results are consistent
    assert_allclose(whole_output, np.stack(shot_outputs, axis=0))
    assert_allclose(whole_indices, np.stack(shot_indices, axis=0))
    assert_allclose(whole_weights, np.stack(shot_weights, axis=0))


def test_count_vs_distance_weighting(cartesian_2d_data, identity_interpolator):
    """Test that both weighting methods give same results for grid-aligned data."""
    data, coords, shape = cartesian_2d_data

    # With count weighting
    interp_count = _GrogInterpolator(coords, shape, weighting_mode="count")
    interp_count.set_kernels(identity_interpolator)
    _, _, weights_count = interp_count(data)

    # With distance weighting
    interp_distance = _GrogInterpolator(coords, shape, weighting_mode="distance")
    interp_distance.set_kernels(identity_interpolator)
    _, _, weights_distance = interp_distance(data)

    # For grid-aligned points, both weightings should give similar results
    assert_allclose(weights_count, weights_distance, rtol=1e-5)
    assert_allclose(weights_count, np.ones_like(weights_count), rtol=1e-5)


def test_oversamp_scaling(cartesian_2d_data, identity_interpolator):
    """Test that oversampling correctly scales coordinates."""
    data, coords, shape = cartesian_2d_data

    # Without oversampling
    interp1 = _GrogInterpolator(coords, shape, oversamp=1.0)
    interp1.set_kernels(identity_interpolator)
    _, indexes1, _ = interp1(data)

    # With oversampling
    interp2 = _GrogInterpolator(coords, shape, oversamp=2.0)
    interp2.set_kernels(identity_interpolator)
    _, indexes2, _ = interp2(data)

    # Indexes should differ by the scaling factor
    # For oversamp=2.0, coordinates are doubled before rounding
    expected_indexes2 = np.zeros_like(indexes1)
    for i in range(2):
        expected_indexes2[..., i] = 2 * shape[i] * coords[..., i] + shape[i]

    assert_allclose(indexes2, expected_indexes2, rtol=1e-10)
