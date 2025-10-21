"""Test GROG interpolator."""

import pytest
import numpy as np

from numpy.testing import assert_allclose

from pygrog.grog import _GrogInterpolator


def test_initialization(cartesian_2d_coords):
    """Test that the interpolator initializes correctly."""
    # Basic initialization
    interp = _GrogInterpolator(cartesian_2d_coords, (10, 10))
    assert interp.shape == (10, 10)
    assert interp.oversamp == 1.0
    assert interp.precision == 1
    assert interp.weighting_mode == "count"
    assert interp.nsteps == 11  # For precision=1
    
    # With explicit parameters
    interp = _GrogInterpolator(
        cartesian_2d_coords, (10, 10), 
        oversamp=1.5, 
        precision=2, 
        weighting_mode="distance"
    )
    assert interp.shape == (10, 10)
    assert interp.oversamp == 1.5
    assert interp.precision == 2
    assert interp.weighting_mode == "distance"
    assert interp.nsteps == 101  # For precision=2


def test_invalid_weighting_mode(cartesian_2d_coords):
    """Test that invalid weighting mode raises an error."""
    with pytest.raises(ValueError):
        _GrogInterpolator(cartesian_2d_coords, (10, 10), weighting_mode="invalid")


def test_kernels_setting_2d(cartesian_2d_coords, identity_interpolator):
    """Test setting 2D GRAPPA kernels."""
    interp = _GrogInterpolator(cartesian_2d_coords, (10, 10))
    interp.set_kernels(identity_interpolator)
    
    assert interp._kernels_set
    assert interp._n_coils == 4
    assert interp._grog_table.shape == (11*11, 4, 4)  # nsteps^2 x nc x nc


def test_kernels_setting_3d(cartesian_3d_coords, identity_interpolator_3d):
    """Test setting 3D GRAPPA kernels."""
    interp = _GrogInterpolator(cartesian_3d_coords, (10, 10, 10))
    interp.set_kernels(identity_interpolator_3d)
    
    assert interp._kernels_set
    assert interp._n_coils == 4
    assert interp._grog_table.shape == (11*11*11, 4, 4)  # nsteps^3 x nc x nc


def test_missing_kernels(cartesian_2d_coords):
    """Test error handling when GRAPPA kernels are missing."""
    interp = _GrogInterpolator(cartesian_2d_coords, (10, 10))
    
    # Missing x kernel
    with pytest.raises(ValueError):
        interp.set_kernels({"y": np.eye(4)})
        
    # Missing y kernel
    with pytest.raises(ValueError):
        interp.set_kernels({"x": np.eye(4)})


def test_missing_z_kernel_for_3d(cartesian_3d_coords):
    """Test error when missing z kernel for 3D data."""
    interp_3d = _GrogInterpolator(cartesian_3d_coords, (10, 10, 10))
    with pytest.raises(ValueError):
        interp_3d.set_kernels({"x": np.eye(4), "y": np.eye(4)})


def test_call_without_kernels(cartesian_2d_coords, uniform_2d_data):
    """Test error when calling without setting kernels first."""
    interp = _GrogInterpolator(cartesian_2d_coords, (10, 10))
    with pytest.raises(RuntimeError):
        interp(uniform_2d_data)


def test_interpolation_2d(cartesian_2d_coords, uniform_2d_data, identity_interpolator):
    """Test interpolation with 2D Cartesian grid data."""
    interp = _GrogInterpolator(cartesian_2d_coords, (10, 10))
    interp.set_kernels(identity_interpolator)
    
    output, indexes, weights = interp(uniform_2d_data)
    
    # Check shapes
    assert output.shape == uniform_2d_data.shape
    assert indexes.shape == cartesian_2d_coords.shape
    assert weights.shape == (*cartesian_2d_coords.shape[:-1], 1)
    
    # Since points are on grid and kernels are identity matrices,
    # output should be very close to input
    assert_allclose(output, uniform_2d_data, rtol=1e-5)
    
    # Indexes should correspond to coordinates plus grid offset
    expected_indexes = cartesian_2d_coords + np.array([5, 5])
    assert_allclose(indexes, expected_indexes, rtol=1e-10)
    
    # Each point should have weight 1.0 since they're on grid positions
    assert_allclose(weights, np.ones(weights.shape), rtol=1e-5)


def test_interpolation_3d(cartesian_3d_coords, uniform_3d_data, identity_interpolator_3d):
    """Test interpolation with 3D Cartesian grid data."""
    interp = _GrogInterpolator(cartesian_3d_coords, (10, 10, 10))
    interp.set_kernels(identity_interpolator_3d)
    
    output, indexes, weights = interp(uniform_3d_data)
    
    # Check shapes
    assert output.shape == uniform_3d_data.shape
    assert indexes.shape == cartesian_3d_coords.shape
    assert weights.shape == (*cartesian_3d_coords.shape[:-1], 1)
    
    # Since points are on grid and kernels are identity matrices,
    # output should be very close to input
    assert_allclose(output, uniform_3d_data, rtol=1e-5)
    
    # Indexes should correspond to coordinates plus grid offset
    expected_indexes = cartesian_3d_coords + np.array([5, 5, 5])
    assert_allclose(indexes, expected_indexes, rtol=1e-10)
    
    # Each point should have weight 1.0 since they're on grid positions
    assert_allclose(weights, np.ones(weights.shape), rtol=1e-5)


def test_single_shot_2d(cartesian_2d_coords, uniform_2d_data, identity_interpolator):
    """Test single shot processing in 2D with Cartesian grid data."""
    interp = _GrogInterpolator(cartesian_2d_coords, (10, 10))
    interp.set_kernels(identity_interpolator)
    
    # Process single shot - second view
    shot_index = (1,)
    shot_data = uniform_2d_data[1]  # Shape: (4, 4) - readouts, coils
    
    output, indexes, weights = interp(shot_data, shot_index)
    
    # Check shapes
    assert output.shape == shot_data.shape
    assert indexes.shape == (4, 2)  # readouts, ndim
    assert weights.shape == (4, 1)  # readouts, 1
    
    # Output should match input for identity kernels
    assert_allclose(output, shot_data, rtol=1e-5)
    
    # Indexes should match original coordinates plus grid offset
    expected_indexes = cartesian_2d_coords[1] + np.array([5, 5])
    assert_allclose(indexes, expected_indexes, rtol=1e-10)
    
    # Weights should be 1.0
    assert_allclose(weights, np.ones(weights.shape), rtol=1e-5)


def test_single_shot_3d(cartesian_3d_coords, uniform_3d_data, identity_interpolator_3d):
    """Test single shot processing in 3D with Cartesian grid data."""
    interp = _GrogInterpolator(cartesian_3d_coords, (10, 10, 10))
    interp.set_kernels(identity_interpolator_3d)
    
    # Process single shot - first slice, second view
    shot_index = (0, 1)
    shot_data = uniform_3d_data[0, 1]  # Shape: (4, 4) - readouts, coils
    
    output, indexes, weights = interp(shot_data, shot_index)
    
    # Check shapes
    assert output.shape == shot_data.shape
    assert indexes.shape == (4, 3)  # readouts, ndim
    assert weights.shape == (4, 1)  # readouts, 1
    
    # Output should match input for identity kernels
    assert_allclose(output, shot_data, rtol=1e-5)
    
    # Indexes should match original coordinates plus grid offset
    expected_indexes = cartesian_3d_coords[0, 1] + np.array([5, 5, 5])
    assert_allclose(indexes, expected_indexes, rtol=1e-10)
    
    # Weights should be 1.0
    assert_allclose(weights, np.ones(weights.shape), rtol=1e-5)


def test_count_vs_distance_weighting(cartesian_2d_coords, uniform_2d_data, identity_interpolator):
    """Test that both weighting methods give same results for grid-aligned data."""
    # With count weighting
    interp_count = _GrogInterpolator(cartesian_2d_coords, (10, 10), weighting_mode="count")
    interp_count.set_kernels(identity_interpolator)
    _, _, weights_count = interp_count(uniform_2d_data)
    
    # With distance weighting
    interp_distance = _GrogInterpolator(cartesian_2d_coords, (10, 10), weighting_mode="distance")
    interp_distance.set_kernels(identity_interpolator)
    _, _, weights_distance = interp_distance(uniform_2d_data)
    
    # For grid-aligned points, both weightings should give similar results
    assert_allclose(weights_count, weights_distance, rtol=1e-5)
    assert_allclose(weights_count, np.ones_like(weights_count), rtol=1e-5)


def test_oversamp_scaling(cartesian_2d_coords, uniform_2d_data, identity_interpolator):
    """Test that oversampling correctly scales coordinates."""
    # Without oversampling
    interp1 = _GrogInterpolator(cartesian_2d_coords, (10, 10), oversamp=1.0)
    interp1.set_kernels(identity_interpolator)
    _, indexes1, _ = interp1(uniform_2d_data)
    
    # With oversampling
    interp2 = _GrogInterpolator(cartesian_2d_coords, (10, 10), oversamp=2.0)
    interp2.set_kernels(identity_interpolator)
    _, indexes2, _ = interp2(uniform_2d_data)
    
    # Indexes should differ by the scaling factor
    # For oversamp=2.0, coordinates are doubled before rounding
    expected_indexes2 = np.zeros_like(indexes1)
    for i in range(2):
        expected_indexes2[..., i] = cartesian_2d_coords[..., i] * 2 + 5
        
    assert_allclose(indexes2, expected_indexes2, rtol=1e-10)