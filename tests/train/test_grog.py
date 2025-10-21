"""Test GROG interpolator."""

import pytest
import numpy as np
import tempfile
import os

from pygrog.grog import GROGInterpolator


@pytest.fixture
def identity_interpolator():
    """Create identity GROG interpolator matrices for 2D."""
    n_coils = 4
    return {
        "x": np.eye(n_coils, dtype=np.complex64),
        "y": np.eye(n_coils, dtype=np.complex64)
    }


@pytest.fixture
def identity_interpolator_3d():
    """Create identity GROG interpolator matrices for 3D."""
    n_coils = 4
    return {
        "x": np.eye(n_coils, dtype=np.complex64),
        "y": np.eye(n_coils, dtype=np.complex64),
        "z": np.eye(n_coils, dtype=np.complex64)
    }


@pytest.fixture
def cartesian_2d_data():
    """Create 2D Cartesian k-space data and coordinates."""
    n_coils = 4
    matrix_size = 16
    
    # Create Cartesian coordinates
    x, y = np.meshgrid(
        np.linspace(-0.5, 0.5, matrix_size),
        np.linspace(-0.5, 0.5, matrix_size),
        indexing='ij'
    )
    coords = np.stack([y.flatten(), x.flatten()], axis=-1)
    
    # Create test data with coils as the rightmost dimension
    data = np.ones((matrix_size**2, n_coils), dtype=np.complex64)
        
    return data, coords, (matrix_size, matrix_size)


@pytest.fixture
def cartesian_3d_data():
    """Create 3D Cartesian k-space data and coordinates."""
    n_coils = 4
    matrix_size = 8  # Smaller for 3D to keep test fast
    
    # Create Cartesian coordinates
    x, y, z = np.meshgrid(
        np.linspace(-0.5, 0.5, matrix_size),
        np.linspace(-0.5, 0.5, matrix_size),
        np.linspace(-0.5, 0.5, matrix_size),
        indexing='ij'
    )
    coords = np.stack([z.flatten(), y.flatten(), x.flatten()], axis=-1)
    
    # Create test data with coils as the rightmost dimension
    data = np.ones((matrix_size**3, n_coils), dtype=np.complex64)
    
    return data, coords, (matrix_size, matrix_size, matrix_size)


def test_create_grog_interpolator(identity_interpolator, cartesian_2d_data):
    """Test creating a GROG interpolator instance."""
    _, coords, shape = cartesian_2d_data
    
    # Create interpolator without kernels
    grog = GROGInterpolator(
        coords=coords,
        shape=shape
    )
    
    # Check that the plan was created but doesn't have grog_table
    assert grog.plan is not None
    assert grog._grog_table is None  # No table yet
    assert "output_shape" in grog.plan
    
    # Set the kernels
    grog.set_kernels(identity_interpolator)
    
    # Now there should be a grog_table in memory but not in the plan
    assert grog._grog_table is not None
    assert "grog_table" not in grog.plan
    
    # Check that output shape matches input shape
    assert grog.output_shape == shape


def test_apply_interpolation(identity_interpolator, cartesian_2d_data):
    """Test applying interpolation to entire dataset."""
    data, coords, shape = cartesian_2d_data
    
    # Create interpolator and set kernels
    grog = GROGInterpolator(
        coords=coords,
        shape=shape,
    )
    grog.set_kernels(identity_interpolator)
    
    # Apply interpolation
    output, indices = grog(data)
    
    # With identity matrices and Cartesian coordinates, output should preserve signal
    assert np.allclose(np.sum(np.abs(output)), np.sum(np.abs(data)), rtol=1e-1)
    
    # Check that indices were returned
    assert indices.shape[0] == output.shape[0]


def test_save_load_interpolator_without_kernels(cartesian_2d_data):
    """Test saving and loading interpolator without kernels."""
    _, coords, shape = cartesian_2d_data
    
    # Create interpolator without setting kernels
    grog = GROGInterpolator(
        coords=coords,
        shape=shape
    )
    
    # Save to temporary file
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Test pickle format
        pkl_path = os.path.join(tmp_dir, "grog_plan.pkl")
        grog.to_file(pkl_path)
        
        # Load from pickle file
        loaded_grog = GROGInterpolator.from_file(pkl_path)
        
        # Check that core parts of the plan are equal
        assert grog.output_shape == loaded_grog.output_shape
        
        # Verify no grog_table is loaded
        assert loaded_grog._grog_table is None


def test_save_load_set_kernels_workflow(identity_interpolator, cartesian_2d_data):
    """Test saving trajectory plan, loading it, and then setting kernels."""
    data, coords, shape = cartesian_2d_data
    
    # Create interpolator without kernels
    grog = GROGInterpolator(
        coords=coords,
        shape=shape
    )
    
    # Save to temporary file
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Test numpy format
        npy_path = os.path.join(tmp_dir, "grog_plan.npy")
        grog.to_file(npy_path)
        
        # Load from numpy file
        loaded_grog = GROGInterpolator.from_file(npy_path)
        
        # Set kernels on loaded interpolator
        loaded_grog.set_kernels(identity_interpolator)
        
        # Now we should be able to use it
        assert loaded_grog._grog_table is not None
        
        # Test interpolation
        output, indices = loaded_grog(data)
        assert output.shape[0] > 0
        assert np.allclose(np.sum(np.abs(output)), np.sum(np.abs(data)), rtol=1e-1)


def test_missing_kernels(cartesian_2d_data):
    """Test error when trying to use interpolator without setting kernels."""
    data, coords, shape = cartesian_2d_data
    
    # Create interpolator without kernels
    grog = GROGInterpolator(
        coords=coords,
        shape=shape
    )
    
    # Try to use it without setting kernels
    with pytest.raises(RuntimeError, match="GRAPPA kernels have not been set"):
        grog(data)


def test_invalid_radius(cartesian_2d_data):
    """Test error handling for invalid radius."""
    _, coords, shape = cartesian_2d_data
    
    # Create interpolator with invalid radius
    with pytest.raises(ValueError, match="Maximum GRAPPA shift is 1.0"):
        GROGInterpolator(
            coords=coords,
            shape=shape,
            radius=1.5
        )


def test_invalid_weighting_mode(cartesian_2d_data):
    """Test error handling for invalid weighting mode."""
    _, coords, shape = cartesian_2d_data
    
    # Create interpolator with invalid weighting mode
    with pytest.raises(ValueError, match="Weighting mode can be either"):
        GROGInterpolator(
            coords=coords,
            shape=shape,
            weighting_mode="invalid"
        )


def test_coil_count_mismatch(identity_interpolator, cartesian_2d_data):
    """Test error handling for coil count mismatch."""
    data, coords, shape = cartesian_2d_data
    
    # Create interpolator
    grog = GROGInterpolator(
        coords=coords,
        shape=shape
    )
    grog.set_kernels(identity_interpolator)
    
    # Create data with wrong coil count
    wrong_data = np.zeros((data.shape[0], 5), dtype=np.complex64)
    
    # Apply interpolation with wrong data
    with pytest.raises(ValueError, match="Input data has .* coils but kernels expect"):
        grog(wrong_data)


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
            coords[v, r, 1] = -0.5 + v / n_views    # y coordinate
            
    # Create test data with coils as rightmost dimension
    data = np.zeros((n_views, n_readout, n_coils), dtype=np.complex64)
    
    # Set a signal point in each view
    data[0, 8, :] = 1.0 + 0j
    data[1, 4, :] = 0.5 + 0j
    
    # Create interpolator and set kernels
    grog = GROGInterpolator(
        coords=coords,
        shape=(16, 16)
    )
    grog.set_kernels(identity_interpolator)
    
    # Process whole dataset
    whole_output, whole_indices = grog(data)
    
    # Process shot-by-shot
    shot_outputs = []
    shot_indices = []
    for v in range(n_views):
        # Extract single shot data: (readout, coils)
        shot_data = data[v]  
        # Process with shot index
        output, indices = grog(shot_data, shot_index=(v,))
        shot_outputs.append(output)
        shot_indices.append(indices)
    
    # Verify shot-by-shot results are consistent
    whole_sum = np.sum(np.abs(whole_output))
    shot_sum = sum(np.sum(np.abs(output)) for output in shot_outputs)
    
    # Check that total signal is preserved
    assert np.isclose(whole_sum, shot_sum, rtol=1e-2)


def test_3d_interpolation(identity_interpolator_3d, cartesian_3d_data):
    """Test 3D interpolation functionality."""
    data, coords, shape = cartesian_3d_data
    
    # Create interpolator and set 3D kernels
    grog = GROGInterpolator(
        coords=coords,
        shape=shape,
        oversamp=[1.0, 1.0, 1.0]
    )
    grog.set_kernels(identity_interpolator_3d)
    
    # Apply interpolation
    output, indices = grog(data)
    
    # With identity matrices and Cartesian coordinates, output should preserve signal
    assert np.allclose(np.sum(np.abs(output)), np.sum(np.abs(data)), rtol=1e-1)
    
    # Check that we have 3D indices
    assert indices.shape[1] == 1  # Just the grid index since no stack axes


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
            
    # Create test data with coils as rightmost dimension
    data = np.zeros((n_stacks, n_readout, n_coils), dtype=np.complex64)
    
    # Set a signal point in each stack
    data[0, 8, :] = 1.0 + 0j
    data[1, 4, :] = 0.5 + 0j
    
    # Create interpolator with stack_axes=[0] and set kernels
    grog = GROGInterpolator(
        coords=coords,
        shape=(16, 16),
        stack_axes=[0]
    )
    grog.set_kernels(identity_interpolator)
    
    # Process dataset
    output, indices = grog(data)
    
    # Verify output has signals from both stacks
    assert output.shape[0] > 0
    assert np.sum(np.abs(output)) > 0
    
    # Verify indices include stack dimension
    assert indices.shape[1] >= 2  # Should have (stack_idx, point_idx)


def test_complex_dataset_structure(identity_interpolator):
    """Test with more complex dataset structure (batch, stack, view, readout, coil)."""
    n_coils = 4
    n_readout = 8
    n_views = 2
    n_stacks = 2  # e.g., slices
    n_batches = 2  # e.g., time points
    
    # Create coordinates with stack axis
    coords = np.zeros((n_stacks, n_views, n_readout, 2), dtype=np.float32)
    for s in range(n_stacks):
        for v in range(n_views):
            for r in range(n_readout):
                coords[s, v, r, 0] = -0.5 + r / n_readout  # x varies by readout
                coords[s, v, r, 1] = -0.5 + v / n_views    # y varies by view
    
    # Create test data
    data = np.zeros((n_batches, n_stacks, n_views, n_readout, n_coils), dtype=np.complex64)
    
    # Set a signal point in different locations
    data[0, 0, 0, 4, :] = 1.0 + 0j  # batch 0, stack 0, view 0
    data[0, 1, 1, 2, :] = 0.8 + 0j  # batch 0, stack 1, view 1
    data[1, 0, 1, 6, :] = 0.6 + 0j  # batch 1, stack 0, view 1
    data[1, 1, 0, 3, :] = 0.4 + 0j  # batch 1, stack 1, view 0
    
    # Create interpolator and set kernels
    grog = GROGInterpolator(
        coords=coords,
        shape=(16, 16),
        stack_axes=[0]  # First dimension (after batches) is stack
    )
    grog.set_kernels(identity_interpolator)
    
    # Process dataset
    output, indices = grog(data)
    
    # Verify output shape has correct batch dimensions
    assert output.shape[0] == n_batches
    
    # Test processing a single shot from the dataset
    b, s, v = 0, 1, 1  # batch 0, stack 1, view 1
    shot_data = data[b, s, v]  # shape: (n_readout, n_coils)
    shot_output, shot_indices = grog(shot_data, shot_index=(b, s, v))
    
    # Verify shot output has data
    assert np.sum(np.abs(shot_output)) > 0


def test_different_oversamp(identity_interpolator, cartesian_2d_data):
    """Test with different oversampling factors."""
    data, coords, shape = cartesian_2d_data
    
    # Create interpolator with higher oversampling and set kernels
    grog = GROGInterpolator(
        coords=coords,
        shape=shape,
        oversamp=2.0
    )
    grog.set_kernels(identity_interpolator)
    
    # Apply interpolation
    output, indices = grog(data)
    
    # Check output shape is larger
    assert grog.output_shape[0] >= shape[0] * 1.9  # Allow for rounding
    assert grog.output_shape[1] >= shape[1] * 1.9
    
    # Check that signal is preserved
    assert np.allclose(np.sum(np.abs(output)) / 2.0**2, np.sum(np.abs(data)), rtol=1e-1)
    
    # Test with different oversampling per dimension
    grog2 = GROGInterpolator(
        coords=coords,
        shape=shape,
        oversamp=[1.5, 2.0]
    )
    grog2.set_kernels(identity_interpolator)
    
    # Check asymmetric output shape
    assert grog2.output_shape[0] >= shape[0] * 1.4  # Allow for rounding
    assert grog2.output_shape[1] >= shape[1] * 1.9


def test_different_weighting(identity_interpolator, cartesian_2d_data):
    """Test with different weighting modes."""
    data, coords, shape = cartesian_2d_data
    
    # Create interpolator with distance weighting and set kernels
    grog_distance = GROGInterpolator(
        coords=coords,
        shape=shape,
        weighting_mode="distance"
    )
    grog_distance.set_kernels(identity_interpolator)
    
    # Create interpolator with average weighting and set kernels
    grog_average = GROGInterpolator(
        coords=coords,
        shape=shape,
        weighting_mode="average"
    )
    grog_average.set_kernels(identity_interpolator)
    
    # Apply interpolation
    output_distance, _ = grog_distance(data)
    output_average, _ = grog_average(data)
    
    # Both should preserve signal sum, though individual values might differ
    assert np.allclose(np.sum(np.abs(output_distance)), np.sum(np.abs(data)), rtol=1e-1)
    assert np.allclose(np.sum(np.abs(output_average)), np.sum(np.abs(data)), rtol=1e-1)


def test_kernel_setting_workflow():
    """Test typical GROG workflow with precomputation and runtime usage."""
    # Setup data
    n_coils = 4
    matrix_size = 16
    x, y = np.meshgrid(
        np.linspace(-0.5, 0.5, matrix_size),
        np.linspace(-0.5, 0.5, matrix_size),
        indexing='ij'
    )
    coords = np.stack([y.flatten(), x.flatten()], axis=-1)
    data = np.ones((matrix_size**2, n_coils), dtype=np.complex64)
    shape = (matrix_size, matrix_size)
    
    # Create identity kernels
    kernels = {
        "x": np.eye(n_coils, dtype=np.complex64),
        "y": np.eye(n_coils, dtype=np.complex64)
    }
    
    # Workflow: Precompute trajectory info, save, load, set kernels at runtime
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Step 1: Precomputation
        precomp_grog = GROGInterpolator(
            coords=coords,
            shape=shape
        )
        
        # Save precomputed plan
        plan_path = os.path.join(tmp_dir, "precomputed_plan.pkl")
        precomp_grog.to_file(plan_path)
        
        # Step 2: Runtime
        runtime_grog = GROGInterpolator.from_file(plan_path)
        runtime_grog.set_kernels(kernels)
        
        # Use the interpolator
        output, indices = runtime_grog(data)
        
        # Check output
        assert output.shape[0] > 0
        assert np.allclose(np.sum(np.abs(output)), np.sum(np.abs(data)), rtol=1e-1)