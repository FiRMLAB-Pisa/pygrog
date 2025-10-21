"""Python implementation of the GRAPPA operator formalism."""

__all__ = ["_GrogInterpolator"]

import numpy as np
import numba as nb

from scipy.linalg import fractional_matrix_power as fmp

class _GrogInterpolator:
    """
    GRAPPA Operator Gridding (GROG) interpolator class.
    
    This class handles the creation of GROG interpolation plans and
    their application to non-Cartesian k-space data.
    """
    
    def __init__(
        self,
        coords,
        shape,
        oversamp=1.0,
        precision=1,
        weighting_mode="count",
    ):
        """
        Create a new GROG interpolator with trajectory information.
        
        Parameters
        ----------
        coords : np.ndarray
            Fourier domain coordinates array of shape ``(stack1...stackN, view, readouts, ndims)``.
        shape : list[int] | tuple[int, ...]
            Cartesian grid size of shape ``(ndim,)``.
        oversamp : float | list[float] | tuple[float, ...], optional
            Cartesian grid oversampling factor. The default is ``1.0``
        precision : int, optional
            Number of decimal digits in GROG kernel power. The default is ``1``.
            This determines the number of steps (nsteps = 2*10^precision + 1)
        weighting_mode : str, optional
            Method for weighting samples. Options are:
            - 'count': inverse of counts per grid point (default)
            - 'distance': weighting based on distance from grid point
        """
        # Store configuration parameters
        self.coords = coords
        self.original_coord_shape = coords.shape
        self.shape = shape
        self.oversamp = oversamp
        self.precision = precision
        self.weighting_mode = weighting_mode
        
        if weighting_mode not in ["count", "distance"]:
            raise ValueError("weighting_mode must be either 'count' or 'distance'")
        
        # calculate interpolation stepsize based on precision
        pfac = 10.0**precision
        radius = 0.5  # maximum shift is 0.5 in either direction
        self.radius = np.ceil(pfac * radius) / pfac
        self.nsteps = int(2 * self.radius / 10**(-precision) + 1)  # ensure odd number
        
        # Create the trajectory-based part of the plan
        self.plan = self._create_trajectory_plan()
        
        # Initialize runtime attributes
        self._grog_table = None
        self._kernels_set = False
        
    def set_kernels(self, grappa_kernels):
        """
        Set the GRAPPA kernels and compute the GROG table for interpolation.
        
        Parameters
        ----------
        grappa_kernels : dict
            Dictionary of GRAPPA kernels with keys 'x', 'y', and optionally 'z'
            for 3D interpolation.
        """
        # Check required keys in kernels
        ndim = self.plan["ndim"]
        if "x" not in grappa_kernels or "y" not in grappa_kernels:
            raise ValueError("GRAPPA kernels must include 'x' and 'y' operators")
        if ndim == 3 and "z" not in grappa_kernels:
            raise ValueError("3D interpolation requires 'z' operator in GRAPPA kernels")
            
        # Get number of coils from kernels
        n_coils = grappa_kernels["x"].shape[0]
            
        # compute exponentials
        nsteps = self.nsteps
        deltas = (np.arange(nsteps) - (nsteps - 1) // 2) / (nsteps - 1)
        
        # pre-compute partial operators
        Dx = _grog_power(grappa_kernels["x"], deltas)  # (nsteps, nc, nc)
        Dy = _grog_power(grappa_kernels["y"], deltas)  # (nsteps, nc, nc)
        if "z" in grappa_kernels and grappa_kernels["z"] is not None:
            Dz = _grog_power(grappa_kernels["z"], deltas)  # (nsteps, nc, nc), 3D only
        else:
            Dz = None
            
        # Compute grog table
        self._grog_table = _prepare_grog_table(Dx, Dy, Dz, nsteps, ndim)
        self._n_coils = n_coils
        self._kernels_set = True
    
    def __call__(self, input_data, shot_index=None):
        """
        Apply the GROG interpolation to input data.
        
        Parameters
        ----------
        input_data : np.ndarray
            Input Non-Cartesian kspace data. When shot_index is None,
            shape should be (..., ncoils). When shot_index is provided,
            shape should be (readouts, ncoils) for a single shot.
            
        shot_index : tuple or int, optional
            Index of the shot to process. If provided, only interpolates
            data for this specific shot. Default is None (process all data).
        
        Returns
        -------
        output : np.ndarray
            Output sparse Cartesian kspace with same shape as input.
        indexes : np.ndarray
            Sampled k-space points indexes with shape (..., ndim).
        weights : np.ndarray
            Sample weights with shape (..., 1).
        """
        if not self._kernels_set:
            raise RuntimeError("GRAPPA kernels have not been set. Call set_kernels() first.")
        
        # Extract plan components
        coord = self.plan["coord"]
        shape = self.plan["shape"]
        ndim = self.plan["ndim"]
        nsteps = self.plan["nsteps"]
        
        if shot_index is not None:
            # Process single shot
            return self._process_single_shot(input_data, shot_index, coord, shape, ndim, nsteps)
        else:
            # Process entire dataset
            return self._process_full_dataset(input_data, coord, shape, ndim, nsteps)
    
    def _process_single_shot(self, shot_data, shot_index, coord, shape, ndim, nsteps):
        """
        Process a single shot for GROG interpolation.
        
        Parameters
        ----------
        shot_data : np.ndarray
            Data for a single shot with shape (readouts, ncoils).
        shot_index : tuple or int
            Index of the shot in the trajectory.
        coord : np.ndarray
            Coordinates array.
        shape : tuple
            Output shape.
        ndim : int
            Number of dimensions.
        nsteps : int
            Number of interpolation steps.
            
        Returns
        -------
        output : np.ndarray
            Interpolated data for the shot with same shape as input.
        indexes : np.ndarray
            Grid indexes for the shot with shape (readouts, ndim).
        weights : np.ndarray
            Weights for the shot with shape (readouts, 1).
        """
        # Convert shot_index to tuple if it's an integer
        if isinstance(shot_index, int):
            shot_index = (shot_index,)
        
        # Calculate the flat index for this shot
        coord_shape = self.original_coord_shape[:-1]  # Exclude coordinate dimension
        
        # For readout-only case, adjust shot_index handling
        if len(shot_index) < len(coord_shape) - 1:  # -1 for readouts dimension
            # Incomplete index provided, assume it's for views/batches
            remaining_dims = len(coord_shape) - len(shot_index) - 1  # -1 for readouts
            shot_index = shot_index + (0,) * remaining_dims
        
        # Calculate multidimensional index without readouts
        multi_index = shot_index + (slice(None),)
        
        # Get coordinates for this shot
        shot_coord = self.coords[multi_index]
        
        # Build indexes by rounding coordinates
        indexes = np.round(shot_coord).astype(int)
        
        # Calculate displacements from grid points
        displacements = indexes - shot_coord
        
        # Convert displacements to table indices
        lut = np.floor(10 * displacements).astype(int) + int(nsteps // 2)
        lut_flat = _flatten_lut(lut, nsteps)
        
        # Reshape input to match interpolation format
        input_reshaped = shot_data.reshape(-1, 1, shot_data.shape[-1])  # (nsamples, 1, ncoils)
        
        # Create output array
        output = np.zeros_like(input_reshaped)
        
        # Perform interpolation
        _interp(output, input_reshaped, self._grog_table, lut_flat)
        
        # Remove batch dimension
        output = output[:, 0, :]  # (nsamples, ncoils)
        
        # Adjust indexes for the output grid
        if np.isscalar(shape):
            shape = [shape] * ndim
        indexes = indexes + np.array(shape[:ndim][::-1]) // 2
        indexes = indexes.astype(int)
        
        # Ensure bounds are within grid
        for n in range(ndim):
            outside = indexes[..., n] < 0
            output[outside] = 0.0
            indexes[..., n][outside] = 0
            outside = indexes[..., n] >= shape[::-1][n]
            indexes[..., n][outside] = shape[::-1][n] - 1
            output[outside] = 0.0
        
        # Calculate weights
        if self.weighting_mode == "count":
            # Create flat index for unique counting
            unfolding = [1] + list(np.cumprod(list(shape)[::-1]))[: ndim - 1]
            flattened_idx = np.sum(indexes * np.array(unfolding, dtype=int)[:, np.newaxis], axis=1)
            
            # Count-based weights - inverse of occurrence count
            unique_idx, inverse, counts = np.unique(flattened_idx, return_inverse=True, return_counts=True)
            weights = 1 / counts[inverse]
        else:  # distance-based
            # Distance-based weights - closer samples get higher weights
            distances = np.sqrt(np.sum(displacements**2, axis=-1))
            
            # Invert distances to get weights (closer = higher weight)
            epsilon = 1e-10
            distance_weights = 1.0 / (distances + epsilon)
            
            # Create flat index for unique grouping
            unfolding = [1] + list(np.cumprod(list(shape)[::-1]))[: ndim - 1]
            flattened_idx = np.sum(indexes * np.array(unfolding, dtype=int)[:, np.newaxis], axis=1)
            
            # Group by target grid point
            unique_idx, inverse = np.unique(flattened_idx, return_inverse=True)
            
            # Initialize weights array
            weights = np.zeros_like(distances)
            
            # For each unique grid point
            for i, idx in enumerate(unique_idx):
                # Get the indices of samples that map to this grid point
                mask = (flattened_idx == idx)
                
                # Calculate normalized weights for this group
                group_weights = distance_weights[mask]
                norm_factor = np.sum(group_weights)
                if norm_factor > 0:
                    group_weights = group_weights / norm_factor
                
                # Assign the normalized weights
                weights[mask] = group_weights
        
        # Reshape weights to have shape (..., 1)
        weights = weights[..., np.newaxis]
        
        return output, indexes, weights
    
    def _process_full_dataset(self, input_data, coord, shape, ndim, nsteps):
        """
        Process the entire dataset for GROG interpolation.
        
        Parameters
        ----------
        input_data : np.ndarray
            Full dataset with shape (..., ncoils).
        coord : np.ndarray
            Coordinates array.
        shape : tuple
            Output shape.
        ndim : int
            Number of dimensions.
        nsteps : int
            Number of interpolation steps.
            
        Returns
        -------
        output : np.ndarray
            Interpolated data with same shape as input.
        indexes : np.ndarray
            Grid indexes with shape (..., ndim).
        weights : np.ndarray
            Weights with shape (..., 1).
        """
        # Store original data shape for reshaping output
        original_shape = input_data.shape
        
        # Reshape input for processing
        ncoils = original_shape[-1]
        input_flattened = input_data.reshape(-1, ncoils)
        
        # Add a dummy batch dimension
        input_reshaped = input_flattened[:, np.newaxis, :]  # (nsamples, 1, ncoils)
        
        # Build indexes by rounding coordinates
        coord_flat = coord.reshape(-1, ndim)
        indexes_flat = np.round(coord_flat).astype(int)
        
        # Calculate displacements from grid points
        displacements_flat = indexes_flat - coord_flat
        
        # Convert displacements to table indices
        lut = np.floor(10 * displacements_flat).astype(int) + int(nsteps // 2)
        lut_flat = _flatten_lut(lut, nsteps)
        
        # Create output array
        output_flat = np.zeros_like(input_reshaped)
        
        # Perform interpolation
        _interp(output_flat, input_reshaped, self._grog_table, lut_flat)
        
        # Remove batch dimension
        output_flat = output_flat[:, 0, :]  # (nsamples, ncoils)
        
        # Adjust indexes for the output grid
        if np.isscalar(shape):
            shape = [shape] * ndim
        indexes_flat = indexes_flat + np.array(shape[:ndim][::-1]) // 2
        indexes_flat = indexes_flat.astype(int)
        
        # Calculate weights
        if self.weighting_mode == "count":
            # Create flat index for unique counting
            unfolding = [1] + list(np.cumprod(list(shape)[::-1]))[: ndim - 1]
            flattened_idx = np.sum(indexes_flat * np.array(unfolding, dtype=int), axis=1)
            
            # Count-based weights - inverse of occurrence count
            unique_idx, inverse, counts = np.unique(flattened_idx, return_inverse=True, return_counts=True)
            weights_flat = 1 / counts[inverse]
        else:  # distance-based
            # Distance-based weights - closer samples get higher weight
            distances = np.sqrt(np.sum(displacements_flat**2, axis=1))
            
            # Invert distances to get weights (closer = higher weight)
            epsilon = 1e-10
            distance_weights = 1.0 / (distances + epsilon)
            
            # Create flat index for unique grouping
            unfolding = [1] + list(np.cumprod(list(shape)[::-1]))[: ndim - 1]
            flattened_idx = np.sum(indexes_flat * np.array(unfolding, dtype=int), axis=1)
            
            # Group by target grid point
            unique_idx, inverse = np.unique(flattened_idx, return_inverse=True)
            
            # Initialize weights array
            weights_flat = np.zeros_like(distances)
            
            # For each unique grid point
            for i, idx in enumerate(unique_idx):
                # Get the indices of samples that map to this grid point
                mask = (flattened_idx == idx)
                
                # Calculate normalized weights for this group
                group_weights = distance_weights[mask]
                norm_factor = np.sum(group_weights)
                if norm_factor > 0:
                    group_weights = group_weights / norm_factor
                
                # Assign the normalized weights
                weights_flat[mask] = group_weights
        
        # Ensure bounds are within grid
        for n in range(ndim):
            outside = indexes_flat[:, n] < 0
            output_flat[outside] = 0.0
            indexes_flat[outside, n] = 0
            outside = indexes_flat[:, n] >= shape[::-1][n]
            indexes_flat[outside, n] = shape[::-1][n] - 1
            output_flat[outside] = 0.0
        
        # Reshape outputs back to original shapes
        output = output_flat.reshape(original_shape)
        indexes = indexes_flat.reshape(self.original_coord_shape)
        
        # Add extra dimension to weights
        weights_flat = weights_flat[..., np.newaxis]
        weights = weights_flat.reshape(self.original_coord_shape[:-1] + (1,))
        
        return output, indexes, weights
    
    def _create_trajectory_plan(self):
        """
        Create the trajectory-dependent part of a GROG interpolation plan.
            
        Returns
        -------
        plan : dict
            A dictionary containing trajectory-dependent information for interpolation.
        """
        coords = self.coords
        shape = self.shape
        oversamp = self.oversamp
        nsteps = self.nsteps
        
        # Get dimensions from coords
        ndim = coords.shape[-1]
        
        # expand oversamp
        if np.isscalar(oversamp):
            oversamp = tuple(ndim * [oversamp])
        elif len(oversamp) == 1:
            oversamp = tuple(ndim * oversamp[0])
        else:
            oversamp = tuple(oversamp)
            
        # Rescale coordinates for oversampling
        rescaled_coords = np.copy(coords)
        for i in range(ndim):
            rescaled_coords[..., i] *= oversamp[i]
        
        # Create plan dictionary
        plan = {
            "coord": rescaled_coords,
            "shape": shape,
            "oversamp": oversamp,
            "nsteps": nsteps,
            "ndim": ndim,
        }
        
        return plan


# Utility functions
def _flatten_lut(lut, nsteps):
    """Flatten LUT indices based on dimensionality."""
    ndim = lut.shape[-1]
    if ndim == 2:
        return lut[..., 0] + lut[..., 1] * nsteps
    else:  # ndim == 3
        return lut[..., 0] + lut[..., 1] * nsteps + lut[..., 2] * nsteps**2


def _prepare_grog_table(Dx, Dy, Dz, nsteps, ndim):
    """Prepare the GROG operator table."""
    # Convert to numpy arrays
    Dx = np.asarray(Dx)
    Dy = np.asarray(Dy)
    
    if ndim == 2:
        # 2D case
        Dx = Dx[None, :, ...]  # (1, nsteps, nc, nc)
        Dy = Dy[:, None, ...]  # (nsteps, 1, nc, nc)
        Dx = np.repeat(Dx, nsteps, axis=0)  # (nsteps, nsteps, nc, nc)
        Dy = np.repeat(Dy, nsteps, axis=1)  # (nsteps, nsteps, nc, nc)
        Dx = Dx.reshape(-1, *Dx.shape[-2:])  # (nsteps**2, nc, nc)
        Dy = Dy.reshape(-1, *Dy.shape[-2:])  # (nsteps**2, nc, nc)
        grog_table = Dx @ Dy  # (nsteps**2, nc, nc)
        
    elif ndim == 3:
        # 3D case
        if Dz is None:
            raise ValueError("3D interpolation requires Z operator")
        
        Dz = np.asarray(Dz)
        Dx = Dx[None, None, :, ...]  # (1, 1, nsteps, nc, nc)
        Dy = Dy[None, :, None, ...]  # (1, nsteps, 1, nc, nc)
        Dz = Dz[:, None, None, ...]  # (nsteps, 1, 1, nc, nc)
        
        # Repeat to create a grid of all combinations
        Dx = np.repeat(Dx, nsteps, axis=0)  # (nsteps, 1, nsteps, nc, nc)
        Dx = np.repeat(Dx, nsteps, axis=1)  # (nsteps, nsteps, nsteps, nc, nc)
        Dy = np.repeat(Dy, nsteps, axis=0)  # (nsteps, nsteps, 1, nc, nc)
        Dy = np.repeat(Dy, nsteps, axis=2)  # (nsteps, nsteps, nsteps, nc, nc)
        Dz = np.repeat(Dz, nsteps, axis=1)  # (nsteps, nsteps, 1, nc, nc)
        Dz = np.repeat(Dz, nsteps, axis=2)  # (nsteps, nsteps, nsteps, nc, nc)
        
        # Reshape to flat combinations
        Dx = Dx.reshape(-1, *Dx.shape[-2:])  # (nsteps**3, nc, nc)
        Dy = Dy.reshape(-1, *Dy.shape[-2:])  # (nsteps**3, nc, nc)
        Dz = Dz.reshape(-1, *Dz.shape[-2:])  # (nsteps**3, nc, nc)
        
        # Combine all operators
        grog_table = Dx @ Dy @ Dz  # (nsteps**3, nc, nc)
    
    else:
        raise ValueError(f"GROG interpolation only supports 2D or 3D data, got {ndim}D")
        
    return grog_table


def _grog_power(G, exponents):
    """Compute matrix powers of GROG operators."""
    D, idx = [], 0
    for exp in exponents:
        if np.isclose(exp, 0.0):
            _D = np.eye(G.shape[0], dtype=G.dtype)
        else:
            _D = fmp(G, np.abs(exp)).astype(G.dtype)
            if np.sign(exp) < 0:
                _D = np.linalg.pinv(_D).astype(G.dtype)
        D.append(_D)
        idx += 1

    return np.stack(D, axis=0)

@nb.njit(fastmath=True, cache=True)  # pragma: no cover
def _dot_product(out, in_a, in_b):
    """Matrix-vector multiplication helper."""
    row, col = in_b.shape
    for i in range(row):
        for j in range(col):
            out[j] += in_b[i][j] * in_a[j]
    return out


@nb.njit(fastmath=True, parallel=True)  # pragma: no cover
def _interp(data_out, data_in, interp, lut):
    """Numba-optimized interpolation kernel."""
    nsamples, batch_size, _ = data_in.shape
    for i in nb.prange(nsamples * batch_size):
        sample = i // batch_size
        batch = i % batch_size
        idx = lut[sample]
        _dot_product(
            data_out[sample][batch], data_in[sample][batch], interp[idx]
        )