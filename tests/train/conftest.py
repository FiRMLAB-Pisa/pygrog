"""Common fixtures for GROG interpolator tests."""

import pytest
import numpy as np


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


@pytest.fixture
def uniform_2d_data():
    """Create uniform 2D data filled with ones."""
    # 3 views, 4 readouts, 4 coils
    return np.ones((3, 4, 4), dtype=np.complex64)


@pytest.fixture
def uniform_3d_data():
    """Create uniform 3D data filled with ones."""
    # 2 slices, 3 views, 4 readouts, 4 coils
    return np.ones((2, 3, 4, 4), dtype=np.complex64)


@pytest.fixture
def cartesian_2d_coords():
    """Create 2D coordinates exactly on a Cartesian grid."""
    matrix_size = 16

    # Create Cartesian coordinates
    x, y = np.meshgrid(
        np.linspace(-0.5, 0.5, matrix_size),
        np.linspace(-0.5, 0.5, matrix_size),
        indexing='ij'
    )
    coords = np.stack([y.flatten(), x.flatten()], axis=-1)
    return coords


@pytest.fixture
def cartesian_3d_coords():
    """Create 3D coordinates exactly on a Cartesian grid."""
    matrix_size = 8  # Smaller for 3D to keep test fast
    
    # Create Cartesian coordinates
    x, y, z = np.meshgrid(
        np.linspace(-0.5, 0.5, matrix_size),
        np.linspace(-0.5, 0.5, matrix_size),
        np.linspace(-0.5, 0.5, matrix_size),
        indexing='ij'
    )
    coords = np.stack([z.flatten(), y.flatten(), x.flatten()], axis=-1)
    
    return coords