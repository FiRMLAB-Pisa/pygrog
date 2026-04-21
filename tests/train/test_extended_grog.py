"""Test Extended GROG interpolator."""

import os
import tempfile

import numpy as np
import pytest

from numpy.testing import assert_allclose

from pygrog.grog import _ExtendedGrogInterpolator


def test_create_grog_interpolator(identity_interpolator, cartesian_2d_data):
    """Test creating a GROG interpolator instance."""
    _, coords, shape = cartesian_2d_data

    # Create interpolator without kernels
    grog = _ExtendedGrogInterpolator(
        shape=shape, coords=coords, shot_coords=np.zeros_like(coords[..., 0])
    )

    # Check that the plan was created but doesn't have grog_table
    assert grog.plan is not None
    assert grog._interp_kernel is None  # No table yet
    assert grog._precision is None

    # Set the kernels
    grog.set_kernels(identity_interpolator)

    # Now there should be a interp_kernel in memory
    assert grog._interp_kernel is not None
    assert grog._precision is not None


def test_apply_interpolation(identity_interpolator, cartesian_2d_data):
    """Test applying interpolation to entire dataset."""
    data, coords, shape = cartesian_2d_data

    # Create interpolator and set kernels
    grog = _ExtendedGrogInterpolator(
        shape=shape, coords=coords, shot_coords=np.zeros_like(coords[..., 0])
    )
    grog.set_kernels(identity_interpolator)

    # Apply interpolation
    output = grog(data)

    # With identity matrices and Cartesian coordinates, output should preserve signal
    assert_allclose(np.sum(np.abs(output)), np.sum(np.abs(data)), rtol=1e-1)

    # Check that indices were returned
    indices = grog.metadata.indexes
    assert indices.shape[0] == output.shape[1]


def test_save_load_interpolator_without_kernels(cartesian_2d_data):
    """Test saving and loading interpolator without kernels."""
    _, coords, shape = cartesian_2d_data

    # Create interpolator without setting kernels
    grog = _ExtendedGrogInterpolator(
        shape=shape, coords=coords, shot_coords=np.zeros_like(coords[..., 0])
    )

    # Save to temporary file
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Test pickle format
        pkl_path = os.path.join(tmp_dir, "grog_plan.npy")
        grog.to_file(pkl_path)

        # Load from pickle file
        loaded_grog = _ExtendedGrogInterpolator.from_file(pkl_path)

        # Check that core parts of the plan are equal
        assert_allclose(grog.plan.coords, loaded_grog.plan.coords)

        # Verify no grog_table is loaded
        assert loaded_grog._interp_kernel is None


def test_save_load_set_kernels_workflow(identity_interpolator, cartesian_2d_data):
    """Test saving trajectory plan, loading it, and then setting kernels."""
    data, coords, shape = cartesian_2d_data

    # Create interpolator without kernels
    grog = _ExtendedGrogInterpolator(
        shape=shape, coords=coords, shot_coords=np.zeros_like(coords[..., 0])
    )

    # Save to temporary file
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Test numpy format
        npy_path = os.path.join(tmp_dir, "grog_plan.npy")
        grog.to_file(npy_path)

        # Load from numpy file
        loaded_grog = _ExtendedGrogInterpolator.from_file(npy_path)

        # Set kernels on loaded interpolator
        loaded_grog.set_kernels(identity_interpolator)

        # Now we should be able to use it
        assert loaded_grog._interp_kernel is not None

        # Test interpolation
        output = loaded_grog(data)
        assert output.shape[0] > 0
        assert_allclose(np.sum(np.abs(output)), np.sum(np.abs(data)), rtol=1e-1)


def test_missing_kernels(cartesian_2d_data):
    """Test error when trying to use interpolator without setting kernels."""
    data, coords, shape = cartesian_2d_data

    # Create interpolator without kernels
    grog = _ExtendedGrogInterpolator(
        shape=shape, coords=coords, shot_coords=np.zeros_like(coords[..., 0])
    )

    # Try to use it without setting kernels
    with pytest.raises(RuntimeError, match="GRAPPA kernels have not been set"):
        grog(data)


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
    shot_coords = np.arange(n_views)[..., None]
    shot_coords, _ = np.broadcast_arrays(shot_coords, coords[..., 0])

    # Create test data with coils as rightmost dimension
    data = np.zeros((n_views, n_readout, n_coils), dtype=np.complex64)

    # Set a signal point in each view
    data[0, 8, :] = 1.0 + 0j
    data[1, 4, :] = 0.5 + 0j

    # Create interpolator and set kernels
    grog = _ExtendedGrogInterpolator(
        shape=(16, 16), coords=coords, shot_coords=shot_coords
    )
    grog.set_kernels(identity_interpolator)

    # Process whole dataset
    whole_output = grog(data)

    # Process shot-by-shot
    shot_output = None
    for v in range(n_views):
        # Extract single shot data: (readout, coils)
        shot_data = data[v]

        # Process with shot index
        output = grog(shot_data, shot_index=v)

        # Aggregate
        if shot_output is None:
            shot_output = output
        else:
            shot_output = np.stack((shot_output, output), axis=0).mean(axis=0)

    # Check that total signal is preserved
    assert_allclose(whole_output, shot_output, rtol=1e-2)


def test_3d_interpolation(identity_interpolator_3d, cartesian_3d_data):
    """Test 3D interpolation functionality."""
    data, coords, shape = cartesian_3d_data

    # Create interpolator and set 3D kernels
    grog = _ExtendedGrogInterpolator(
        shape=shape,
        coords=coords,
        shot_coords=np.zeros_like(coords[..., 0]),
        oversamp=[1.0, 1.0, 1.0],
    )
    grog.set_kernels(identity_interpolator_3d)

    # Apply interpolation
    output = grog(data)

    # With identity matrices and Cartesian coordinates, output should preserve signal
    assert_allclose(np.sum(np.abs(output)), np.sum(np.abs(data)), rtol=1e-1)

    # Check that we have 3D indices
    indices = grog.metadata.indexes
    assert indices.shape[-1] == 1  # Just the grid index since no stack axes


def test_with_stack_axes(identity_interpolator):
    """Test interpolation with stack axes."""
    n_coils = 4
    n_readout = 16
    n_stacks = 2  # e.g., 2 slices

    # Create Cartesian coordinates with stack axis
    coords = np.zeros((n_stacks, n_readout, 2), dtype=np.float32)
    for s in range(n_stacks):
        for r in range(n_readout):
            coords[s, r, 0] = -0.5 + r / n_readout  # x coordinate
            coords[s, r, 1] = -0.5  # y coordinate fixed
    shot_coords = np.arange(n_stacks)[..., None]
    shot_coords, _ = np.broadcast_arrays(shot_coords, coords[..., 0])
    stack_coords = np.arange(n_stacks)[..., None]
    stack_coords, _ = np.broadcast_arrays(stack_coords, coords[..., 0])
    stack_coords = stack_coords[..., None]

    # Create test data with coils as rightmost dimension
    data = np.zeros((n_stacks, n_readout, n_coils), dtype=np.complex64)

    # Set a signal point in each stack
    data[0, 8, :] = 1.0 + 0j
    data[1, 4, :] = 0.5 + 0j

    # Create interpolator with stack_axes=[0] and set kernels
    grog = _ExtendedGrogInterpolator(
        shape=(16, 16),
        coords=coords,
        shot_coords=shot_coords,
        stack_coords=stack_coords,
    )
    grog.set_kernels(identity_interpolator)

    # Process dataset
    output = grog(data)

    # Verify output has signals from both stacks
    assert output.shape[0] > 0
    assert np.sum(np.abs(output)) > 0

    # Verify indices include stack dimension
    indices = grog.metadata.indexes
    assert indices.shape[-1] >= 2  # Should have (stack_idx, point_idx)
