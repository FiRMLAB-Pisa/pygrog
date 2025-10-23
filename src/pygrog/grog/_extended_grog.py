"""Regridder subroutines for ExtendedGrogInterpolator class."""

__all__ = ["_ExtendedGrogInterpolator"]

import os
import pathlib

from types import SimpleNamespace

from scipy.spatial import KDTree
from numpy.typing import NDArray

import numpy as np
import numba as nb

from _utils import rescale_coords, prepare_grog_table, grog_power

class _ExtendedGrogInterpolator:
    """
    Create a new GROG interpolator with trajectory information.
    
    Parameters
    ----------
    shape : list[int] | tuple[int, ...]
        Cartesian grid size of shape ``(ndims,)``.
        Here, ``ndims`` represent the encoding direction with following axes
        order:

            * ``2D imaging``: ``(x, y)``
            * ``3D imaging``: ``(x, y, z)`

    coords : NDArray
        Non Cartesian coordinates of shape ``(nsamples, ndims)``.
        Here, ``ndims`` represent the encoding direction with following axes
        order:

            * ``2D imaging``: ``(x, y)``
            * ``3D imaging``: ``(x, y, z)``

    shot_coords : NDArray
        For multi-shot acquisition, the shot index for each k-space sample.
        Shape is ``(nsamples,)``.
    stack_coords : NDArray | None
        For multi-volumes acquisitions (i.e., stack-of-trajectories, dynamic
        or multi-contrast scans), this array contain the stack index for each
        sample. Shape is ``(nsamples, nstack_axes)``. The default is ``None``.
    time_coords : NDArray | None
        Flattened sampling time map of each grid point of shape ``(nlocations,)``.
        Units are ``[s]``. The default is ``None``.
    oversamp: float | list[float] | tuple[float, ...] | None
        Cartesian grid oversampling factor. The default is ``1.0``.
    radius : float
        Interpolation radius in kspace displacement units, i.e., for non-oversampled
        grid, distance along a given axis between two Cartesian locations is ``1.0``.

    """
    
    def __init__(
            self,
            shape: int | list[int] | tuple[int, ...],
            coords: NDArray,
            shot_coords: NDArray,
            stack_coords: NDArray = None,
            time_coords: NDArray = None,
            oversamp: float | list[float] | tuple[float, ...] | None = None,
            radius: float = 0.75,
            ):
        
        # Rescale coordinates
        coords = rescale_coords(coords, shape)
        
        # Flatten trajectories
        data_n_axes = len(coords.shape)
        coords = coords.reshape(-1, coords.shape[-1])
        shot_coords = shot_coords.ravel()
        stack_coords = coords.reshape(-1, stack_coords.shape[-1]) if stack_coords is not None else None
        time_coords = time_coords.ravel() if time_coords is not None else None
        
        # Build plan
        self.plan = _ExtendedGrogPlan(
            shape, 
            coords, 
            shot_coords,
            stack_coords,
            time_coords,
            oversamp,
            radius)
        
        # Store number of axes in data inside plan
        self.plan.data_n_axes = data_n_axes

        # Initialize runtime attributes - these are NOT stored in the plan
        self._precision = None
        self._interp_kernel = None
        self._kernels_set = False
        
    def set_kernels(self, grappa_kernels: dict, precision: int = 1):
        """
        Set the GRAPPA kernels and compute the GROG table for interpolation.
        
        Parameters
        ----------
        grappa_kernels : dict
            Dictionary of GRAPPA kernels with keys 'x', 'y', and optionally 'z'
            for 3D interpolation.
        """
        # Check required keys in kernels
        ndim = self.plan.coords.shape[-1]
        if "x" not in grappa_kernels or "y" not in grappa_kernels:
            raise ValueError("GRAPPA kernels must include 'x' and 'y' operators")
        if ndim == 3 and "z" not in grappa_kernels:
            raise ValueError("3D interpolation requires 'z' operator in GRAPPA kernels")
            
        # compute exponentials
        nsteps = 2 * self.plan.radius / 10 ** (-precision) + 1
        nsteps = int(nsteps)
        deltas = (np.arange(nsteps) - (nsteps - 1) // 2) / (nsteps - 1)
        
        # pre-compute partial operators
        Dx = grog_power(grappa_kernels["x"], deltas)  # (nsteps, nc, nc)
        Dy = grog_power(grappa_kernels["y"], deltas)  # (nsteps, nc, nc)
        if "z" in grappa_kernels and grappa_kernels["z"] is not None:
            Dz = grog_power(grappa_kernels["z"], deltas)  # (nsteps, nc, nc), 3D only
        else:
            Dz = None
            
        # Compute grog table
        self._precision = precision
        self._interp_kernel = prepare_grog_table(Dx, Dy, Dz, nsteps, ndim)
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
        
        # Deserialize
        interpolator.plan = np.load(filepath, allow_pickle=True).item()
        
        # Initialize runtime attributes
        interpolator._interp_kernel = None
        interpolator._kernels_set = False
            
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
        
        # Serialize
        np.save(filepath, self.plan)
        
    def __call__(self, input: NDArray, shot_index: int | None = None) -> NDArray:
        if not self._kernels_set:
            raise RuntimeError("GRAPPA kernels have not been set. Call set_kernels() first.")
            
        # Reshape data from (..., ncoils) to (nbatches, nsamples, ncoils)
        data = input.reshape(*input.shape[:-self.plan.data_n_axes], -1, input.shape[-1])
        if len(data.shape) < 3:
            data = data[None, ...]
        data = data.reshape(-1, *data.shape[-2:])
        
        return _ExtendedGrogRegridder(data, self.plan, self._precision, self._interp_kernel, shot_index)
    
    @property
    def metadata(self):        
        indexes = self.plan.samples_map.cart_output[self.plan.sampled_locations]
        time_coords = self.plan.time_coords
        
        return SimpleNamespace(shape=self.plan.shape, indexes=indexes, time=time_coords)    
    
# %% Methods
def _ExtendedGrogPlan(
    shape: int | list[int] | tuple[int, ...],
    coords: NDArray,
    shot_coords: NDArray,
    stack_coords: NDArray = None,
    time_coords: NDArray = None,
    oversamp: float | list[float] | tuple[float, ...] | None = None,
    radius: float = 0.75,
) -> SimpleNamespace:
    """
    Create a plan for GROG regridding.
    
    This subroutine is based solely on ahead-of-time info.
    Therefore, it can be precomputed offline for a given trajectory.
    
    Parameters
    ----------
    shape : list[int] | tuple[int, ...]
        Cartesian grid size of shape ``(ndims,)``.
        Here, ``ndims`` represent the encoding direction with following axes
        order:

            * ``2D imaging``: ``(x, y)``
            * ``3D imaging``: ``(x, y, z)`

    coords : NDArray
        Non Cartesian coordinates of shape ``(nsamples, ndims)``.
        Here, ``ndims`` represent the encoding direction with following axes
        order:

            * ``2D imaging``: ``(x, y)``
            * ``3D imaging``: ``(x, y, z)``

    shot_coords : NDArray
        For multi-shot acquisition, the shot index for each k-space sample.
        Shape is ``(nsamples,)``.
    stack_coords : NDArray | None
        For multi-volumes acquisitions (i.e., stack-of-trajectories, dynamic
        or multi-contrast scans), this array contain the stack index for each
        sample. Shape is ``(nsamples, nstack_axes)``. The default is ``None``.
    time_coords : NDArray | None
        Flattened sampling time map of each grid point of shape ``(nlocations,)``.
        Units are ``[s]``. The default is ``None``.
    oversamp: float | list[float] | tuple[float, ...] | None
        Cartesian grid oversampling factor. The default is ``1.0``.
    radius : float
        Interpolation radius in kspace displacement units, i.e., for non-oversampled
        grid, distance along a given axis between two Cartesian locations is ``1.0``.

    Returns
    -------
    SimpleNamespace
        Structure with the following attributes:
            
            * shape ``(tuple[int])``: Cartesian grid size of shape ``(nx, ny)`` or ``(nx, ny, nz)``
            * oversamp ``(tuple[float])``: Cartesian grid oversampling factor along ``x``, ``y`` and ``z``.
            * radius ``(float)``: Interpolation radius in kspace displacement units.
            * sample_map ``(SimpleNamespace)``: map from Non Cartesian to Cartesian locations.
            * coords ``(NDArray)``: Non Cartesian coordinates of shape ``(nsamples, ndims)``.
            * grid ``(NDArray)``: Cartesian coordinates of shape ``(nlocations, ndims)``.
            * time_coords ``(NDArray)``: Sampling time of each acquired sample, of shape ``(nlocations,)``.

    """    
    # Build KDtree
    kdtree = KDTree(coords)
    
    # Build grid
    grid, shape, oversamp = _build_cartesian_grid(coords.shape[-1], shape, oversamp)
    
    # Query Non-Cartesian points within given radius from each grid point
    # (i.e., grid-driven interpolation)
    samples_map = kdtree.query_ball_point(grid, r=radius, workers=-1)
    
    # Sample map is the flattened list of Cartesian grid points:
    #
    # grid.shape = (nx, ny) -> len(samples_map)  = nx * ny
    #
    # The element of the list contain the indexes of all k Non-Cartesian samples
    # falling withing the given radius:
    #
    # grid[0] = (coord_idx_1,...,coord_idx_k) # Non-Cartesian indexes in neighbourhood of grid[0, 0]
    # samples_in_grid_0_0 = [data[idx] for idx in grid[0]] # Here, assume that data.shape = coords.shape[:-1], i.e., no batches
    #
    # Since coords is sparse, most samples_map elements are empty. Let's now convert samples_map to a sparse array:
    #
    samples_map = _dense2sparse(samples_map)
    
    # Now, assign shot index
    samples_map.shot_index = shot_coords[samples_map.noncart_input]
    
    # Out samples_map will map each Non-Cartesian sample to the 2D or 3D Cartesian grid.
    # If we have multiple k-space volumes (i.e., dynamic or multicontrast MRI, or stack-of-readouts)
    # we do not want to mix samples belonging to different volumes. To avoid this, let's expand samples_map
    # to also contain stack position.
    samples_map = _append_stack_dimensions(samples_map, stack_coords)
    
    # Find sampled locations
    sampled_locations=np.where(np.r_[True, np.any(samples_map.cart_output[1:] != samples_map.cart_output[:-1], axis=1)])[0]

    # If provided, get the sampling time for each target grid point
    if time_coords is not None:
        time_coords = time_coords[samples_map.cart_output]
        time_coords = time_coords[sampled_locations]
    
    return SimpleNamespace(
        shape=shape, 
        oversamp=oversamp, 
        radius=radius, 
        samples_map=samples_map, 
        grid=grid,
        coords=coords,
        time_coords=time_coords,
        sampled_locations=sampled_locations,
        )

def _ExtendedGrogRegridder(
        data: NDArray,
        plan: SimpleNamespace,
        precision: int,
        interp_kernel: NDArray,
        index: int | None = None,
) -> tuple[NDArray, SimpleNamespace]: 
    """
    Perform extended GROG regridding.
    
    Specifically, this extended GROG interpolates all Non Cartesian
    samples within given radius to the target Cartesian location
    (grid-driven interpolation) rather than interpolating each Non Cartesian
    sample to the nearest Cartesian location (data-driven interpolation).
    
    Parameters
    ----------
    data : NDArray
        Input Non-Cartesian dataset of shape ``(nbatches, nsamples, ncoils)``.
    plan : SimpleNamespace
        Structure with the following attributes:

           * shape ``(tuple[int])``: Cartesian grid size of shape ``(nx, ny)`` or ``(nx, ny, nz)``
           * oversamp ``(tuple[float])``: Cartesian grid oversampling factor along ``x``, ``y`` and ``z``.
           * radius ``(float)``: Interpolation radius in kspace displacement units.
           * sample_map ``(SimpleNamespace)``: map from Non Cartesian to Cartesian locations.
           * time_coords ``(NDArray)``: Sampling time of each acquired sampling, of shape ``(nsamples, ndims+nstack_axes)``.

    precision : int
        Number of decimal digits for rounding distance when precomputing interpolation
        kernels.
    interp_kernel : NDArray
        Precomputed table of GROG interpolation kernels of shape ``(nsteps**ndims, ncoils, ncoils)``.
        Here, ``nsteps`` is determined by ``precision``. Each interpolation kernel represents a
        shift across k-space with distance ``(dx, dy, dz)``. Nsteps is the discretization step
        for ``dx``, ``dy`` and ``dz``.
    index : int
        Shot index corresponding to the input dataset. If not 

    Returns
    -------
    NDArray
        Interpolated Cartesian points of shape ``(nbatches, nsamples, ncoils)``.

    """
    # Unpack plan
    shape = plan.shape
    samples_map = plan.samples_map
    coords = plan.coords
    grid = plan.grid
    radius = plan.radius
    
    # If shot index is provided, get the samples map corresponding to current
    # shot
    if index is not None:
        act_samples_map = _select_shot(samples_map, index)
    else:
        act_samples_map = samples_map
     
    # We now need to build the interpolator
    interpolator = _compute_interpolator(act_samples_map, grid, coords, interp_kernel, radius, precision)
    
    # We now bin the map to enable fast grid-driven GROG interpolation
    # The interpolation subroutine will spawn one thread for each bin,
    # corresponding to a single Cartesian location. This way, interpolation
    # is embarassingly parallel (no race conditions in writing onto a given location).
    # 
    # Each thread will then read all the bin_size Non-cartesian samples from each bin_start location
    # in samples_map.noncart_input.
    output_grid_bin = _bin_output_cartesian_grid(act_samples_map)
    
    # Now we can apply the GROG interpolation to input data, obtaining
    # Cartesian samples and the corresponding indexes.
    return _interpolation(data, shape, output_grid_bin, act_samples_map, interpolator)
    

# %% Subroutines
def _build_cartesian_grid(ndim, shape, oversamp):
    # Make sure shape is a tuple
    if np.isscalar(shape):
        shape = tuple(ndim * [shape])
    elif len(shape) == 1:
        shape = tuple(ndim * shape[0])
    else:
        shape = tuple(shape)
    
    # Defaults for oversamp
    if oversamp is None:
        oversamp = 1.0 if ndim == 2 else [1.0, 1.0, 1.2]
        
    # Make sure oversamp is a tuple
    if np.isscalar(oversamp):
        oversamp = tuple(ndim * [oversamp])
    elif len(oversamp) == 1:
        oversamp = tuple(ndim * oversamp[0])
    else:
        oversamp = tuple(oversamp)
        
    # Build grid
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
        
    return grid, shape, oversamp


def _dense2sparse(dense_map):
    counts = np.asarray([len(el) for el in dense_map], dtype=np.int32)
    
    # Find non-empty elements of input dense map
    nonzeros = np.where(counts)[0]
    counts = counts[nonzeros]
    
    # Store number of non-zero Cartesian locations
    nlocations = nonzeros.size

    # Build sparse map
    noncart_input = np.concatenate(dense_map[nonzeros]).astype(np.int32)
    cart_output = np.repeat(nonzeros, counts).astype(np.int32)
    
    return SimpleNamespace(
        noncart_input=noncart_input, 
        cart_output=cart_output, 
        nlocations=nlocations, 
        n_stack_axes=0, 
        shot_index=None
        )

      
def _append_stack_dimensions(samples_map, stack_coords):
    # Make sure samples_map.cart_output has shape (nsamples, naxes), 
    # with samples_map.cart_output[..., -1] being the on-grid indexes
    samples_map.cart_output = samples_map.cart_output[..., None]
    
    # If provided, include stack in cart_output
    if stack_coords is not None:
        # Pick, for each output Cartesian location, the stack location for the corresponding
        # Non-Cartesian input sample
        stack_coords = stack_coords[samples_map.noncart_input, :]
        
        # Append this after Grid indexes
        samples_map.cart_output = np.concatenate((samples_map.cart_output, stack_coords), axis=-1)
        samples_map.n_stack_axes = stack_coords.shape[-1]
        
        # Sort accouting for new stack indexes to enable binning
        _, inverse = _unique(samples_map.cart_output, return_inverse=True)

        # Use the inverse mapping to get a sort order that groups identical bins together:
        sort_order = np.argsort(inverse)

        # Apply the sort order to both arrays
        samples_map.cart_output = samples_map.cart_output[sort_order]
        samples_map.noncart_input = samples_map.noncart_input[sort_order]
        samples_map.shot_index = samples_map.shot_index[sort_order]
        
        # Count new unique output locations
        samples_map.nlocations = _unique(samples_map.cart_output).shape[0]

    return samples_map


def _select_shot(samples_map, index):
    idx = samples_map.shot_index == index
    
    # Select
    noncart_input = samples_map.noncart_input[idx]
    cart_output = samples_map.cart_output[idx, :]
    n_stack_axes = samples_map.n_stack_axes
    nlocations = samples_map.nlocations
    
    return SimpleNamespace(noncart_input=noncart_input, cart_output=cart_output, nlocations=nlocations, n_stack_axes=n_stack_axes, shot_index=None)


def _compute_interpolator(samples_map, grid, coords, interp_kernel, radius, precision):
    ndims = coords.shape[-1]
    pfac = 10.0**precision
    stepsize = 10 ** (-precision)
    nsteps = 2 * radius / 10 ** (-precision) + 1
    nsteps = int(nsteps)
    
    # First, we need to compute the distances from each Non-Cartesian input to the corresponding Cartesian output
    # Get Non-Cartesian source and Cartesian target coordinates
    source_coords = coords[samples_map.noncart_input, :] # (..., ndims)
    target_coords = grid[samples_map.cart_output[..., 0], :] # (..., ndims)
    
    # Compute distance between source and target
    distances = target_coords - source_coords
    
    # We can now use distances to compute the kernel table index, i.e.,
    # the index in the precompute kernel table to select the appropriate precomputed
    # interpolation kernel to grid each Non Cartesian sample to the target Cartesian location.
    interp_idx = (radius + np.round(distances * pfac) / pfac) / stepsize
    interp_idx = np.round(interp_idx).astype(np.float32)
    
    interp_idx *= np.asarray([1.0, nsteps, nsteps**2], dtype=np.float32)[:ndims]
    interp_idx = np.round(interp_idx).astype(np.int32).sum(axis=-1)
        
    return SimpleNamespace(idx=interp_idx, kernel=interp_kernel)


def _bin_output_cartesian_grid(samples_map):
    bin_starts = np.where(np.r_[True, np.any(samples_map.cart_output[1:] != samples_map.cart_output[:-1], axis=1)])[0]
    bin_sizes = np.diff(np.r_[bin_starts, samples_map.cart_output.shape[0]])
    return SimpleNamespace(starts=bin_starts, sizes=bin_sizes)


def _interpolation(input, shape, output_grid_bin, samples_map, interpolator):
    nbatches = input.shape[0]
    nlocations = samples_map.nlocations
    ncoils = input.shape[-1]

    # Enforce datatype
    input = np.ascontiguousarray(input.swapaxes(0, 1)).astype(np.complex64)

    # Preallocate output
    output = np.zeros((nlocations, nbatches, ncoils), dtype=np.complex64)
    
    # Perform interpolation
    _numba_interpolation(
        output, input, 
        samples_map.cart_output, samples_map.noncart_input, 
        output_grid_bin.starts, output_grid_bin.sizes, 
        interpolator.kernel, interpolator.idx
        )
    
    return np.ascontiguousarray(output.swapaxes(0, 1))


# %% Numba helpers
# @nb.njit(fastmath=True, cache=True, inline="always")  # pragma: no cover
def _matvec(y, A, x):
    ni, nj = A.shape
    for i in range(ni):
        for j in range(nj):
            y[i] += A[i][j] * x[j]


# @nb.njit(fastmath=True, cache=True, parallel=True)  # pragma: no cover
def _numba_interpolation(output, input, # output and input data
    cart_output, noncart_input, # indexes of cartesian grid points and noncartesian samples
    bin_starts, bin_sizes, # output cartesian grid bins starts and sizes
    interp_kernel, interp_idx, # precomputed interpolator kernels and indexes to select the appropriate one
):
    nbins = bin_starts.shape[0]
    nbatches = input.shape[1]

    # Parallelize over bins
    for n in nb.prange(nbins):
        # Get current bin start and size
        bin_start = bin_starts[n]
        bin_size = bin_sizes[n]
        
        # Get target Cartesian location corresponding to current bin
        target_idx = cart_output[bin_start, 0]

        # Interpolate all Non-Cartesian samples assigned to current bin
        for b in range(bin_size):
            source_index = noncart_input[bin_start + b]
            
            # Get current interpolator
            _kernel_value = interp_kernel[interp_idx[source_index]]

            # Perform interpolation for each element in batch
            for batch in range(nbatches):
                _matvec(
                    output[target_idx, batch], # (ncoils,)
                    _kernel_value, # (ncoils, ncoils)
                    input[source_index, batch], # (ncoils,)
                )

        # Normalize current bin according to bin size
        output[n] = output[n] / bin_size
        

# %% Other helpers
def _unique(arr, return_index=False, return_inverse=False, return_counts=False):
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
