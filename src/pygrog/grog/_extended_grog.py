"""Extended GROG with improved caching and performance."""

__all__ = ["_ExtendedGrogInterpolator"]

import pickle
import os
import pathlib

from typing import Any

import numpy as np
import numba as nb

from numpy.typing import NDArray
from scipy.spatial import KDTree

from ._utils import rescale_coords, prepare_grog_table, grog_power


class _ExtendedGrogInterpolator:
    """
    GRAPPA Operator Gridding (GROG) interpolator class.
    
    This class handles the creation of GROG interpolation plans and
    their application to non-Cartesian k-space data.
    
    Attributes
    ----------
    plan : dict[str, Any]
        The GROG interpolation plan.
    """
    
    def __init__(
        self,
        coords: NDArray,
        shape: list[int] | tuple[int, ...],
        stack_axes: list[int] | tuple[int, ...] | None = None,
        oversamp: float | list[float] | tuple[float, ...] | None = None,
        radius: float = 0.75,
        precision: int = 1,
        weighting_mode: str = "distance",
    ):
        """
        Create a new GROG interpolator with trajectory information.
        
        Parameters
        ----------
        coords : NDArray
            Fourier domain coordinates array of shape ``(..., ndims)``.
        shape : list[int] | tuple[int, ...]
            Cartesian grid size of shape ``(ndim,)``.
        stack_axes: list[int] | tuple[int, ...] | None
            Indices marking stack axes. The default is None.
        oversamp: float | list[float] | tuple[float, ...] | None
            Cartesian grid oversampling factor. The default is ``None``
        radius: float
            Spreading radius. The default is ``0.75``.
        precision: int
            Number of decimal digits in GROG kernel power. The default is ``1``.
        weighting_mode: str
            Non Cartesian samples accumulation mode.
            The default is ``"distance"``.
        """
        # Store configuration parameters
        self.coords = coords
        self.shape = shape
        self.stack_axes = stack_axes
        self.oversamp = oversamp
        self.radius = radius
        self.precision = precision
        self.weighting_mode = weighting_mode
        
        # Create the trajectory-based part of the plan
        self.plan = self._create_trajectory_plan(
            coords=coords,
            shape=shape,
            stack_axes=stack_axes,
            oversamp=oversamp,
            radius=radius,
            precision=precision,
            weighting_mode=weighting_mode
        )
        
        # Initialize runtime attributes - these are NOT stored in the plan
        self._grog_table = None
        self._kernels_set = False
    
    def set_kernels(self, grappa_kernels: dict[str, NDArray]) -> None:
        """
        Set the GRAPPA kernels and compute the GROG table for interpolation.
        
        Parameters
        ----------
        grappa_kernels: dict[str, NDArray]
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
            
        # Compute exponends
        radius = self.radius
        nsteps = self.plan["nsteps"]
        deltas = 2 * radius * (np.linspace(0, 1, nsteps) - 0.5)
        
        # Pre-compute partial operators
        Dx = grog_power(grappa_kernels["x"], deltas)  # (nsteps, nc, nc)
        Dy = grog_power(grappa_kernels["y"], deltas)  # (nsteps, nc, nc)
        if "z" in grappa_kernels and grappa_kernels["z"] is not None:
            Dz = grog_power(grappa_kernels["z"], deltas)  # (nsteps, nc, nc), 3D only
        else:
            Dz = None
            
        # Compute grog table (not stored in plan)
        self._grog_table = prepare_grog_table(Dx, Dy, Dz, nsteps, ndim)
        self._n_coils = n_coils
        self._kernels_set = True
    
    @classmethod
    def from_file(cls, filepath: str | pathlib.Path) -> "_ExtendedGrogInterpolator":
        """
        Load a GROG interpolator from a saved file.
        
        Parameters
        ----------
        filepath : str | pathlib.Path
            Path to the saved interpolator file.
            
        Returns
        -------
        GROGInterpolator
            Loaded interpolator instance.
        """
        filepath = pathlib.Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Plan file {filepath} not found")
            
        interpolator = cls.__new__(cls)  # Create instance without calling __init__
        
        # Choose deserialization method based on file extension
        if filepath.suffix == '.npy':
            interpolator.plan = np.load(filepath, allow_pickle=True).item()
        elif filepath.suffix == '.pkl':
            with open(filepath, 'rb') as f:
                interpolator.plan = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file extension: {filepath.suffix}")
        
        # Initialize runtime attributes
        interpolator._grog_table = None
        interpolator._kernels_set = False
        
        # Set attributes from plan
        interpolator.radius = interpolator.plan.get("radius", 0.75)
        interpolator.precision = interpolator.plan.get("precision", 1)
            
        return interpolator
    
    def to_file(self, filepath: str | pathlib.Path) -> None:
        """
        Save the GROG interpolator plan to a file.
        
        Parameters
        ----------
        filepath : str | pathlib.Path
            Path where to save the interpolator.
        """
        filepath = pathlib.Path(filepath)
        os.makedirs(filepath.parent, exist_ok=True)
        
        # Choose serialization method based on file extension
        if filepath.suffix == '.npy':
            np.save(filepath, self.plan)
        elif filepath.suffix == '.pkl':
            with open(filepath, 'wb') as f:
                pickle.dump(self.plan, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            raise ValueError(f"Unsupported file extension: {filepath.suffix}. Use .npy or .pkl")
    
    def __call__(
        self, 
        input_data: NDArray, 
        shot_index: tuple[int, ...] | None = None
    ) -> tuple[NDArray, NDArray]:
        """
        Apply the GROG interpolation to input data.
        
        Parameters
        ----------
        input_data : NDArray
            Input Non-Cartesian kspace with coils as the rightmost dimension:
            - When shot_index=None: shape ``(batch1,...,batchN,stack1,...,stackN,view,readout,coils)``
            - When shot_index is provided: shape ``(readout,coils)``
        shot_index : tuple[int, ...] | None, optional
            Index tuple for processing a single shot, representing ``(batch_idx1,...,batch_idxN,
            stack_idx1,...,stack_idxN,view_idx)``. 
            
            If None, processes the entire dataset at once.
        
        Returns
        -------
        output : NDArray
            Output sparse Cartesian kspace.
        indexes : NDArray
            Sampled k-space points indexes.
        """
        if not self._kernels_set:
            raise RuntimeError("GRAPPA kernels have not been set. Call set_kernels() first.")
            
        if shot_index is not None:
            # Process single shot with the given index
            return self._apply_shot(input_data, shot_index)
        else:
            # Process whole dataset
            return self._apply_whole_dataset(input_data)
    
    def _apply_shot(self, shot_data: NDArray, shot_index: tuple[int, ...]) -> tuple[NDArray, NDArray]:
        """
        Apply GROG interpolation to a single shot.
        
        Parameters
        ----------
        shot_data : NDArray
            Shot data with shape ``(readout_points, coils)``
        shot_index : tuple[int, ...]
            Index tuple ``(batch_idx1,...,batch_idxN,stack_idx1,...,stack_idxN,view_idx)``
            
        Returns
        -------
        output : NDArray
            Interpolated shot data
        indexes : NDArray
            K-space point indexes
        """
        # Extract the number of dimensions from the plan
        n_batch_dims = len(shot_index) - len(self.plan["stack_shape"]) - 1
        n_stack_dims = len(self.plan["stack_shape"])
        
        # Extract batch, stack and view indices
        stack_indices = shot_index[n_batch_dims:n_batch_dims+n_stack_dims] if n_stack_dims > 0 else ()
        view_idx = shot_index[-1]
        
        # Calculate the readout offset in the flattened coordinates
        signal_shape = self.plan["signal_shape"]
        readout_size = signal_shape[-1]
        view_offset = view_idx * readout_size
        
        # Convert stack indices to flat stack index if needed
        if n_stack_dims > 0:
            stack_idx = np.ravel_multi_index(stack_indices, self.plan["stack_shape"])
        else:
            stack_idx = 0
            
        # Filter plan components for this shot
        shot_mask = (self.plan["source_stack_indices"] == stack_idx) & \
                   (self.plan["source_readout_indices"] >= view_offset) & \
                   (self.plan["source_readout_indices"] < view_offset + readout_size)
        
        # Extract the relevant components from the plan
        shot_source_indices = self.plan["source_readout_indices"][shot_mask] - view_offset
        shot_target_indices = self.plan["target_indices"][shot_mask]
        shot_weights = self.plan["sample_weights"][shot_mask]
        shot_grog_indices = self.plan["grog_indices"][shot_mask]
        
        # Get unique target indices and create bin information
        unique_targets, inverse = np.unique(shot_target_indices, return_inverse=True)
        sort_order = np.argsort(inverse)
        
        # Apply sorting
        sorted_source_indices = shot_source_indices[sort_order]
        sorted_target_indices = shot_target_indices[sort_order]
        sorted_weights = shot_weights[sort_order]
        sorted_grog_indices = shot_grog_indices[sort_order]
        
        # Create bins for unique targets
        unique_targets, bin_starts, bin_counts = np.unique(
            sorted_target_indices, return_index=True, return_counts=True
        )
        
        # Perform interpolation
        n_coils = shot_data.shape[-1]
        
        # Check coil compatibility
        if n_coils != self._n_coils:
            raise ValueError(f"Input data has {n_coils} coils but kernels expect {self._n_coils}")
        
        # Prepare data for interpolation: (readout, 1, coils)
        shot_data_reshaped = shot_data.reshape(shot_data.shape[0], 1, n_coils)
        
        # Create output array
        output = np.zeros((len(unique_targets), 1, n_coils), dtype=np.complex64)
        
        # Apply interpolation
        _interpolation(
            output,
            shot_data_reshaped,
            sorted_source_indices,
            sorted_weights,
            unique_targets,
            bin_starts,
            bin_counts,
            sorted_grog_indices,
            self._grog_table,
        )
        
        # Reshape output: (targets, coils)
        output = output[:, 0, :]
        
        # Create appropriate index array for output points
        if n_stack_dims == 0:
            indexes = unique_targets
        else:
            # Unravel stack index back to coordinates
            stack_coords = np.array(np.unravel_index(stack_idx, self.plan["stack_shape"]))
            # Expand to match output shape
            stack_coords = np.tile(stack_coords[:, np.newaxis], (1, len(unique_targets)))
            # Combine with target indices
            indexes = np.vstack((stack_coords, unique_targets[np.newaxis, :])).T
            
        # Make sure indexes is at least 2D
        indexes = np.atleast_2d(indexes.T).T
            
        return output, indexes
        
    def _apply_whole_dataset(self, dataset: NDArray) -> tuple[NDArray, NDArray]:
        """
        Apply GROG interpolation to the entire dataset at once.
        
        Parameters
        ----------
        dataset : NDArray
            Input dataset with shape ``(batch1,...,batchN,stack1,...,stackN,view,readout,coils)``
            
        Returns
        -------
        output : NDArray
            Interpolated output data
        indexes : NDArray
            K-space point indexes
        """
        # Extract plan components
        source_indices = self.plan["source_indices"]
        sample_weights = self.plan["sample_weights"]
        bin_starts = self.plan["bin_starts"]
        bin_counts = self.plan["bin_counts"] 
        grog_indices = self.plan["grog_indices"]
        stack_shape = self.plan["stack_shape"]
        unique_targets = self.plan["unique_targets"]
        
        # Determine batch and signal shapes
        signal_shape = self.plan["signal_shape"]
        batch_shape = dataset.shape[:-len(signal_shape)-len(stack_shape)-1]
        
        # Get number of batches and coils
        n_batches = int(np.prod(batch_shape)) if batch_shape else 1
        n_coils = dataset.shape[-1]
        
        # Check compatibility
        if n_coils != self._n_coils:
            raise ValueError(f"Input data has {n_coils} coils but kernels expect {self._n_coils}")
        
        # reshape data to (nsamples, nbatches, ncoils)
        reshaped_data = dataset.reshape(-1, n_coils)  # Flatten all dimensions except coils
        reshaped_data = reshaped_data.reshape(n_batches, -1, n_coils)  # Separate batches
        reshaped_data = reshaped_data.transpose(1, 0, 2)  # (samples, batches, coils)
        reshaped_data = np.ascontiguousarray(reshaped_data)
        
        # Perform interpolation
        output = do_interpolation(
            reshaped_data,
            source_indices,
            sample_weights, 
            unique_targets,
            bin_starts,
            bin_counts,
            grog_indices,
            self._grog_table,
        )
        
        # Reshape output: (batches, coils, unique_targets) -> (batches, unique_targets, coils)
        output = output.transpose(1, 0, 2)
        
        # Final reshape to match expected output format
        if n_batches == 1:
            output = output.reshape(-1, n_coils)  # (unique_targets, coils)
        else:
            output = output.reshape(*batch_shape, -1, n_coils)  # (batch1,...,batchN, unique_targets, coils)
        
        # Create appropriate index array for output points
        if len(stack_shape) == 0:
            indexes = unique_targets
        elif len(stack_shape) == 1:
            indexes = np.stack((self.plan["stack_indices"], unique_targets), axis=-1)
        else:
            # Unravel stack indices to original coordinates
            stack_coords = np.array(np.unravel_index(self.plan["stack_indices"], stack_shape))
            # Combine with target indices
            indexes = np.vstack((stack_coords, unique_targets[np.newaxis, :])).T
            
        # Make sure indexes is at least 2D
        indexes = np.atleast_2d(indexes.T).T
        
        return output, indexes
    
    @property
    def output_shape(self) -> tuple[int, ...]:
        """Get the output shape of the interpolated data."""
        return self.plan["output_shape"]
        
    def _create_trajectory_plan(
        self,
        coords: NDArray,
        shape: list[int] | tuple[int, ...],
        stack_axes: list[int] | tuple[int, ...] | None = None,
        oversamp: float | list[float] | tuple[float, ...] | None = None,
        radius: float = 0.75,
        precision: int = 1,
        weighting_mode: str = "distance",
    ) -> dict[str, Any]:
        """
        Create the trajectory-dependent part of a GROG interpolation plan.
        
        Parameters
        ----------
        coords : NDArray
            Fourier domain coordinates array of shape ``(..., ndims)``.
        shape : list[int] | tuple[int, ...]
            Cartesian grid size of shape ``(ndim,)``.
        stack_axes: list[int] | tuple[int, ...] | None
            Indices marking stack axes. The default is None.
        oversamp: float | list[float] | tuple[float, ...] | None
            Cartesian grid oversampling factor. The default is ``None`` 
        radius: float
            Spreading radius. The default is ``0.75``.
        precision: int
            Number of decimal digits in GROG kernel power. The default is ``1``.
        weighting_mode: str
            Non Cartesian samples accumulation mode.
            The default is ``"distance"``.
            
        Returns
        -------
        plan : dict[str, Any]
            A dictionary containing trajectory-dependent information for interpolation.
        """
        if radius > 1.0:
            raise ValueError(f"Maximum GRAPPA shift is 1.0, requested {radius}")
        if weighting_mode.lower() not in ["average", "distance"]:
            raise ValueError(
                f"Weighting mode can be either 'average' or 'distance', requested {weighting_mode}"
            )
        weighting_mode = weighting_mode.lower()
        stack_axes = check_stack(stack_axes)

        # calculate interpolation stepsize
        pfac = 10.0**precision
        radius = np.ceil(pfac * radius) / pfac
        radius = radius.item()
        nsteps = 2 * radius / 10 ** (-precision) + 1
        nsteps = int(nsteps)

        # Get dimensions from coords
        if stack_axes is None:
            stack_shape = ()
        else:
            stack_shape = coords.shape[: len(stack_axes)]
        signal_shape = coords.shape[len(stack_shape) : -1]
        ndim = coords.shape[-1]

        # Expand oversamp
        if oversamp is None:
            if ndim == 2:
                oversamp = 1.0
            else:
                oversamp = [1.0, 1.0, 1.2]
        if np.isscalar(oversamp):
            oversamp = tuple(ndim * [oversamp])
        elif len(oversamp) == 1:
            oversamp = tuple(ndim * oversamp[0])
        else:
            oversamp = tuple(oversamp)

        # Determine output shape
        output_shape = np.ceil(np.asarray(oversamp) * np.asarray(shape)).astype(int)
        output_shape = tuple([ax.item() for ax in output_shape])

        # get number of stacks and samples
        n_stacks = int(np.prod(stack_shape))
        n_samples = int(np.prod(signal_shape))

        # reshape coordinates
        coords = coords.reshape(-1, ndim)
        coords = rescale_coords(coords, shape)

        # generate stack coordinate
        if len(stack_shape) > 0:
            stack_coords = np.meshgrid(
                *[np.arange(0, stack_size) for stack_size in stack_shape],
                indexing="ij",
            )
            stack_coords = np.stack([ax.ravel() for ax in stack_coords], axis=-1)
            stack_coords = np.repeat(stack_coords, n_samples, axis=0)
            stack_coords = stack_coords.astype(int)
            stack_coords = stack_coords * np.asarray(stack_shape[1:] + (1,), dtype=int)
            stack_coords_flat = stack_coords.sum(axis=-1)
        else:
            stack_coords = np.zeros(n_samples, dtype=int)
            stack_coords_flat = stack_coords

        # build target grid
        grid = np.meshgrid(
            *[
                np.linspace(
                    -shape[n] // 2, shape[n] // 2 - 1, int(np.ceil(oversamp[n] * shape[n]))
                )
                for n in range(ndim)
            ],
            indexing="ij",
        )
        grid = np.stack([ax.ravel() for ax in grid], axis=-1).astype(np.float32)

        # perform kd search and organize results
        (
            source_indices, 
            sample_weights, 
            grog_indices, 
            target_indices, 
            unique_targets,
            stack_indices,
            bin_starts, 
            bin_counts
        ) = prepare_interpolation_data(
            n_stacks,
            grid,
            coords,
            stack_coords_flat,
            radius,
            precision,
            weighting_mode,
        )

        # Calculate readout (view, point) indices for shot-by-shot processing
        source_readout_indices = source_indices % n_samples
        source_stack_indices = stack_coords_flat[source_indices]

        # Create trajectory-dependent part of the plan
        plan = {
            # Configuration parameters
            "radius": radius,
            "precision": precision,
            
            # Source and target indices
            "source_indices": source_indices.astype(np.int32),  
            "target_indices": target_indices.astype(np.int32),
            "unique_targets": unique_targets.astype(np.int32),
            "stack_indices": stack_indices.astype(np.int32),
            
            # For shot-by-shot processing
            "source_readout_indices": source_readout_indices.astype(np.int32),
            "source_stack_indices": source_stack_indices.astype(np.int32),
            
            # Weights and grog indices
            "sample_weights": sample_weights.astype(np.float32),
            "grog_indices": grog_indices.astype(np.int32),
            
            # Binning information for accumulation
            "bin_starts": bin_starts.astype(np.int32),
            "bin_counts": bin_counts.astype(np.int32),
            
            # Dimensions
            "n_stacks": n_stacks,
            "stack_shape": stack_shape,
            "signal_shape": signal_shape,
            "output_shape": output_shape,
            "ndim": ndim,
            "nsteps": nsteps,
        }
        
        return plan

# %% subroutines
def prepare_interpolation_data(
    n_stacks, grid, coords, stack_coords, radius, precision, weighting_mode
):
    """Prepare interpolation data structures using KD-tree search."""
    pfac = 10.0**precision
    stepsize = 10 ** (-precision)
    nsteps = 2 * radius / 10 ** (-precision) + 1
    nsteps = int(nsteps)

    # perform kd search
    unsorted_indices = _kdtree(grid, coords, radius)

    # flatten object array
    unsorted_indices_val, unsorted_indices_idx = flatten_indices(
        n_stacks, unsorted_indices, stack_coords
    )

    # Get the unique bins and the inverse mapping:
    unique_bins, inverse = _unique(unsorted_indices_idx, return_inverse=True)

    # Use the inverse mapping to get a sort order that groups identical bins together:
    sort_order = np.argsort(inverse)

    # Apply the sort order to both arrays:
    bin_idx = unsorted_indices_idx[sort_order]
    bin_val = unsorted_indices_val[sort_order]

    # Now, using _unique on the sorted bin_idx, get the start indices and counts:
    unique_bins, bin_starts, bin_counts = _unique(
        bin_idx, return_index=True, return_counts=True
    )
    
    # Extract stack indices and target indices
    stack_indices = unique_bins[:, 0]
    target_indices = unique_bins[:, 1]

    # Compute distances
    target_coords = grid[np.repeat(target_indices, bin_counts, axis=0), :]
    source_coords = coords[bin_val, :]
    distances = target_coords - source_coords

    # Compute weights
    ndim = coords.shape[-1]
    if weighting_mode == "distance":
        weight_scale = ndim**0.5 * radius * 1.00001
        weights = weight_scale - (distances**2).sum(axis=-1) ** 0.5
    elif weighting_mode == "average":
        weights = np.ones(distances.shape[0], dtype=np.float32)

    # Compute table index
    tab_idx = (radius + np.round(distances * pfac) / pfac) / stepsize
    tab_idx = np.round(tab_idx).astype(np.float32)
    tab_flattening = np.asarray([1.0, nsteps, nsteps**2], dtype=np.float32)
    tab_idx = tab_idx * tab_flattening[:ndim]
    tab_idx = np.round(tab_idx).astype(int).sum(axis=-1)

    return bin_val, weights, tab_idx, target_indices, target_indices, stack_indices, bin_starts, bin_counts


def check_stack(stack_axes):
    """Validate and normalize stack axes specification."""
    if stack_axes is not None:
        _stack_axes = np.sort(np.atleast_1d(stack_axes))
        if _stack_axes.size == 1 and _stack_axes.item() != 0:
            raise ValueError("if we have a single stack axis, it must be the leftmost")
        elif _stack_axes.size > 1:
            if _stack_axes[0] != 0:
                raise ValueError("If provided, stack axis must start from 0")
            _stack_stride = np.unique(np.diff(_stack_axes))
            if _stack_stride.size > 1 or _stack_stride.item() != 1:
                raise ValueError("If provided, stack axes must be contiguous")
        return _stack_axes.tolist()
    return None


def _kdtree(grid, coords, radius):
    """Build KD-tree and query points within radius."""
    kdtree = KDTree(coords)
    return kdtree.query_ball_point(grid, r=radius, workers=-1)


def flatten_indices(n_stacks, indices, stack_coords):
    """Flatten indices array from KD-tree results."""
    counts = np.asarray([len(index) for index in indices])

    # Find nonzeros
    nonzeros = np.where(counts)[0]
    counts = counts[nonzeros]

    # Build
    flattened_indices_val = np.concatenate(indices[nonzeros])
    flattened_indices_idx = np.repeat(nonzeros, counts)
    flattened_indices_idx = np.stack(
        (stack_coords[flattened_indices_val], flattened_indices_idx), axis=-1
    )

    return flattened_indices_val, flattened_indices_idx


def _unique(arr, return_index=False, return_inverse=False, return_counts=False):
    """Optimized unique function for lexicographic arrays."""
    sorted_idx = np.lexsort(arr.T)
    sorted_arr = arr[sorted_idx]

    unique_mask = np.empty(arr.shape[0], dtype=bool)
    unique_mask[0] = True
    unique_mask[1:] = np.any(sorted_arr[1:] != sorted_arr[:-1], axis=1)
    unique_mask_idx = np.where(unique_mask)[0]

    unique_vals = sorted_arr[unique_mask_idx]

    results = [unique_vals]

    if return_index:
        index = sorted_idx[unique_mask_idx]
        results.append(index)

    if return_inverse:
        inverse = np.empty(arr.shape[0], dtype=int)
        inverse[sorted_idx] = np.cumsum(unique_mask) - 1
        results.append(inverse)

    if return_counts:
        counts = np.diff(np.append(unique_mask_idx, arr.shape[0]))
        results.append(counts)

    return tuple(results) if len(results) > 1 else results[0]


def do_interpolation(
    input_data,
    source_indices,
    sample_weights,
    target_indices,
    bin_starts,
    bin_counts,
    grog_indices,
    grog_table,
):
    """Perform GROG interpolation using precomputed plan."""
    nbatches = input_data.shape[1]
    ncoils = input_data.shape[2]

    # Enforce datatype
    input_data = input_data.astype(np.complex64)
    source_indices = source_indices.astype(np.int32)
    sample_weights = sample_weights.astype(np.float32)
    target_indices = target_indices.astype(np.int32)
    bin_starts = bin_starts.astype(np.int32)
    bin_counts = bin_counts.astype(np.int32)
    grog_indices = grog_indices.astype(np.int32)
    grog_table = grog_table.astype(np.complex64)

    # Preallocate output
    output = np.zeros((len(target_indices), nbatches, ncoils), dtype=np.complex64)

    # Perform interpolation
    _interpolation(
        output,
        input_data,
        source_indices,
        sample_weights,
        target_indices,
        bin_starts,
        bin_counts,
        grog_indices,
        grog_table,
    )

    return output


@nb.njit(fastmath=True, cache=True, inline="always")  # pragma: no cover
def _matvec(y, A, x):
    """Matrix-vector multiplication helper."""
    ni, nj = A.shape
    for i in range(ni):
        for j in range(nj):
            y[i] += A[i][j] * x[j]


@nb.njit(fastmath=True, cache=True, parallel=True)  # pragma: no cover
def _interpolation(
    output,
    input_data,
    source_indices,
    weights,
    target_indices,
    bin_starts,
    bin_counts,
    grog_indices,
    grog_table,
):
    """Numba-optimized interpolation kernel."""
    nsamples = output.shape[0]
    nbatches = input_data.shape[1]

    for n in nb.prange(nsamples):
        bin_start = bin_starts[n]
        bin_count = bin_counts[n]
        total_weight = 0.0

        for b in range(bin_count):
            idx = bin_start + b
            source_index = source_indices[idx]

            # Get weight
            total_weight += weights[idx]

            # Perform interpolation for each element in batch
            for batch in range(nbatches):
                _matvec(
                    output[n, batch],
                    grog_table[grog_indices[idx]],
                    weights[idx] * input_data[source_index, batch],
                )

        # Normalize
        if total_weight > 0:
            output[n] = output[n] / total_weight
