"""Python implementation of the GRAPPA operator formalism."""

__all__ = ["_GrogInterpolator"]

import numpy as np
import numba as nb

from ._utils import rescale_coords, prepare_grog_table, grog_power

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
        Dx = grog_power(grappa_kernels["x"], deltas)  # (nsteps, nc, nc)
        Dy = grog_power(grappa_kernels["y"], deltas)  # (nsteps, nc, nc)
        if "z" in grappa_kernels and grappa_kernels["z"] is not None:
            Dz = grog_power(grappa_kernels["z"], deltas)  # (nsteps, nc, nc), 3D only
        else:
            Dz = None
            
        # Compute grog table
        self._grog_table = prepare_grog_table(Dx, Dy, Dz, nsteps, ndim)
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
                
        if shot_index is not None:
            # Process single shot
            return self._process_single_shot(input_data, shot_index)
        else:
            # Process entire dataset
            return self._process_full_dataset(input_data)
    
    def _process_single_shot(self, shot_data, shot_index):
        """
        Process a single shot for GROG interpolation.
        
        Parameters
        ----------
        shot_data : np.ndarray
            Data for a single shot with shape (readouts, ncoils).
        shot_index : tuple or int
            Index of the shot in the trajectory.
            
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
            
        # Add a dummy batch dimension
        input_reshaped = shot_data[:, np.newaxis, :]  # (nsamples, 1, ncoils)
        
        # Convert displacements to table indices
        lut = self.plan["lut"][shot_index]
        lut_flat = _flatten_lut(lut, self.nsteps)
        
        # Create output array
        output = np.zeros_like(input_reshaped)
        
        # Perform interpolation
        _interp(output, input_reshaped, self._grog_table, lut_flat)
        
        # Remove batch dimension
        output = output[:, 0, :]  # (nsamples, ncoils)
                
        return output, self.plan["indexes"][shot_index], self.plan["weights"][shot_index][..., None]
        
    def _process_full_dataset(self, input_data):
        """
        Process the entire dataset for GROG interpolation.
        
        Parameters
        ----------
        input_data : np.ndarray
            Full dataset with shape (..., ncoils).
            
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
                
        # Convert displacements to table indices
        lut = self.plan["lut"]
        lut_flat = _flatten_lut(lut, self.nsteps).ravel()
        
        # Create output array
        output_flat = np.zeros_like(input_reshaped)
        
        # Perform interpolation
        _interp(output_flat, input_reshaped, self._grog_table, lut_flat)
        
        # Remove batch dimension
        output_flat = output_flat[:, 0, :]  # (nsamples, ncoils)
        
        # Reshape outputs back to original shapes
        output = output_flat.reshape(original_shape)
        
        return output, self.plan["indexes"], self.plan["weights"][..., None]
    
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
        
        # Expand oversamp
        if np.isscalar(oversamp):
            oversamp = tuple(ndim * [oversamp])
        elif len(oversamp) == 1:
            oversamp = tuple(ndim * oversamp[0])
        else:
            oversamp = tuple(oversamp)
            
        # Apply oversampling
        shape = [int(np.ceil(oversamp[n] * shape[n])) for n in range(len(shape))]
            
        # Rescale coordinates for oversampling
        coords = rescale_coords(coords, shape)
                
        # Create indexes
        indexes = np.round(coords)
        
        # Get displacement
        displacement = indexes - coords
        
        # Convert displacements to table indices
        lut = np.floor(10.0 * displacement).astype(int) + int(nsteps // 2)
        
        # Adjust indexes for the output grid
        if np.isscalar(shape):
            shape = [shape] * ndim
        indexes = indexes + np.array(shape[:ndim][::-1]) // 2
        
        # Enforce integer indexes
        indexes = indexes.astype(int) 
        
        # Create flat index for unique counting
        unfolding = [1] + list(np.cumprod(list(shape)[::-1]))[:ndim-1]
        flattened_indexes = np.sum(indexes * np.array(unfolding, dtype=int), axis=-1)
        
        # Count-based weights - inverse of occurrence count
        _shape = flattened_indexes.shape
        unique_idx, inverse, counts = np.unique(flattened_indexes.ravel(), return_inverse=True, return_counts=True)
        
        # Calculate weights
        if self.weighting_mode == "distance":
            # Distance-based weights - closer samples get higher weight
            distances = np.sqrt(np.sum(displacement**2, axis=-1))
            
            # Invert distances to get weights (closer = higher weight)
            epsilon = 1e-10
            weights = 1.0 / (distances + epsilon)
            
            # Ravel weights
            weights = weights.ravel()

            # Compute total weight per grid location
            total_weight = np.zeros(flattened_indexes.max() + 1, dtype=weights.dtype)
            np.add.at(total_weight, flattened_indexes, weights)  # accumulate weights per unique index
            
            # Normalize
            weights = weights / total_weight[flattened_indexes]
        else:
            weights = 1 / counts[inverse]
            
        # Reshape back to original
        weights = weights.reshape(*_shape)

        # Create plan dictionary
        plan = {
            "lut": lut,
            "indexes": indexes,
            "weights": weights,
            "ndim": coords.shape[-1],
        }
        
        return plan


# %% Utility functions
def _flatten_lut(lut, nsteps):
    """Flatten LUT indices based on dimensionality."""
    ndim = lut.shape[-1]
    if ndim == 2:
        return lut[..., 0] + lut[..., 1] * nsteps
    else:  # ndim == 3
        return lut[..., 0] + lut[..., 1] * nsteps + lut[..., 2] * nsteps**2

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