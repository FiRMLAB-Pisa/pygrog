"""GROG interpolator class."""

__all__ = ["GrogInterpolator"]

import gc
import os
import pathlib
import psutil

from types import SimpleNamespace
from itertools import chain
from numpy.typing import NDArray

import h5py
import numpy as np
import numba as nb

from scipy.spatial import KDTree

from ._grappa import KernelTable

class GrogInterpolator:
    """
    GrogInterpolator object.
    
    Initializers creates the interpolation plan, which is executed via `__call__`
    method. 
    
    Class contains serialization (`to_file()`) and deserialization (`from_file()`)
    methods to enable offline precomputation of the kernel.
    
    Interpolator can be either used to the dataset as whole (`obj(data)`) or
    shot-by-shot (`obj(data, shot_idx)`). In the latter case, full dataset shape
    must be specified beforehand via `set_dataset_shape()` method.
    
    Parameters
    ----------
    shape : int | list[int] | tuple[int]
        Spatial image size ``(x, y)`` or ``(x, y, z)``.
        If scalar, assumes isotropic grid.
    coords : NDArray[float], optional
        Fourier domain coordinate array of shape ``(*others, 1, k2, k1, k0, ndim)``.
    oversamp : float | list[float] | tuple[float], optional
        Grid oversampling. If scalar, assumes isotropic.
        The default is ``(1.0, 1.0)`` for 2D and ``(1.0, 1.0, 1.2)`` for 3D.
    radius : float, optional
        Interpolation radius. The default is ``0.75``.
    time_map : NDArray[float], optional
        Time map corresponding to each acquired sample of shape ``. 
        Shape is ``(*others, 1, k2, k1, k0, ndim)`` and units are ``[s]``. 
        The default is ``None``.
        
    Attributes
    ----------
    plan : SimpleNamespace
        GROG plan. Contains info solely based on k-space trajectory, hence
        can be precomputed offline, saved via `to_file()` method and retrieved
        via `from_file()` class method.
    _interpolator_set : bool
        Whether interpolator has been computed or not. Initialize to ``False``.
    _dataset_shape_set : bool
        Whether full data size has been provided or not. Initialize to ``False``.
        
    Examples
    --------
    Object can be created by passing image shape and coordinates:

    >>> nx, ny = 200, 200 # shape contains the spatial part of image shape
    >>> grog = GrogInterpolator(shape=(nx, ny), coords=coords, oversamp=1.0, radius=0.75) 
    
    Plan can be serialized to disk (either as ``.npy`` file, or embedded in a ``.h5`` file, such as MRD dataset):
        
    >>> grog.to_file('path-to-file.npy') # Save as NumPy file
    >>> grog.to_file('path-to-mrd-file.mrd') # Embed into HDF5 '.mrd'
    
    The cached interpolator can be deserialized directly from disk, hence skipping the
    computationaly expensive planning:
        
    >>> grog = GrogInterpolator.from_file('path-to-file.npy') # or '.mrd' / '.h5'
    
    Now, a low resolution k-space region - either fully sampled or synthesized via NLINV,
    must be passed to compute GRAPPA interpolators:
        
    >>> grog.calc_interp_table(training_data, lamda=0.01, precision=1) # round distances to 1 decimal digit (-0.2, -0.1, 0.0, 0.1, ...)
    
    Now, interpolator is ready to be called on dataset:
        
    >>> gridded_data = grog(data)
    
    Data can also be gridded shot-by-shot. To do so, we must first pass the expected full data shape,
    known beforehand. This is required because coords might rely on broadcast when multiple shots
    share the same k-space trajectory, thus leading to potential crash when indexing:
        
    >>> grog.set_dataset_shape(data.shape)
    >>> gridded_data = [grog(data[n], (n,)) for n in range(data.shape[0])]
    
    By default, output data have shape ``(*input_data.shape[:-1], num_grid_locs_per_sample * input_data.shape[-1])``.
    This is done to enable subsequent use in iterative algorithms. If this is not necessary, i.e., for non-iterative
    reconstruction, we can direcly obtain the reconstructed individual coil images as:
        
    >>> img = grog(data, ret_image=True)
    
    Again, this can be done shot by shot:
        
    >>> img = 0.0
    >>> for n in range(data.shape[0]):
    >>>     img += grog(data[n], shot_index=(n,), ret_image=True)
    
    Notes
    -----
    Implements the GROG algorithm as described in [1]_.

    References
    ----------
    .. [1] Seiberlich, Nicole, et al. "Self‐calibrating GRAPPA
           operator gridding for radial and spiral trajectories."
           Magnetic Resonance in Medicine: An Official Journal of the
           International Society for Magnetic Resonance in Medicine
           59.4 (2008): 930-935.

    """
    
    _interpolator_set = False
    _dataset_shape_set = False
    
    def __init__(
            self,
            shape: int | list[int] | tuple[int], 
            coords: NDArray[float], 
            oversamp : float | list[float] | tuple[float] | None = None, 
            radius: float = 0.75, 
            time_map: NDArray[float] | None = None,
    ):
        self.plan = _CreateGrogPlan(shape, coords, oversamp, radius, time_map)
        
    def metadata(self):
        """
        Return metadata associated with Cartesian dataset.
        
        These can be used to convert sparse into dense Cartesian data,
        in order to get an image.

        """
        oshape = np.asarray(self.plan.shape) * np.asarray(self.plan.oversamp)
        oshape = np.ceil(oshape).astype(int).tolist()
        return SimpleNamespace(
            shape=tuple(self.plan.shape), 
            oshape=tuple(oshape),
            indexes=self.plan.indexes, 
            weights=self.plan.weights, 
            time_map=self.plan.time_map,
            )
    
    @classmethod
    def from_file(cls, filepath: str | pathlib.Path) -> "GrogInterpolator":
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
        elif filepath.suffix == '.mrd' or filepath.suffix == '.h5':
            with h5py.File(filepath, 'r') as dset:
                interpolator.plan = _load_plan_from_mrd(dset)
        else:
            raise ValueError(f"Unsupported file extension: {filepath.suffix}")
            
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
        elif filepath.suffix == '.mrd' or filepath.suffix == '.h5':
            with h5py.File(filepath, 'r+') as dset:
                _store_plan_inside_mrd(dset, self.plan)
        else:
            raise ValueError(f"Unsupported file extension: {filepath.suffix}. Use .npy, .mrd or .h5")
        
    def calc_interp_table(self, train_data: NDArray[complex], lamda: float = 0.01, precision: int = 1):
        """
        Set the GRAPPA kernels and compute the GROG table for interpolation.
        
        Parameters
        ----------
        train_data : NDArray[complex]
            Calibration k-space region of shape ``(coils, z_cal, y_cal, x_cal)``.
        precision : int
            Number of decimal digits to round shifts.
        lamda : float
            L2 regularization for GRAPPA kernel estimation.
            
        """
        pfac = 10.0**precision
        stepsize = 10 ** (-precision)
        
        # calculate kernel table
        interp_kernel, nsteps, ndim = KernelTable(train_data, self.plan.radius, precision, lamda)
                        
        # Get distance between source and target
        distances = self.plan.distances
        
        # We can now use distances to compute the kernel table index, i.e.,
        # the index in the precompute kernel table to select the appropriate precomputed
        # interpolation kernel to grid each Non Cartesian sample to the target Cartesian location.
        interp_idx = (self.plan.radius + np.round(distances * pfac) / pfac) / stepsize
        interp_idx = np.round(interp_idx).astype(np.float32)
        
        interp_idx *= np.asarray([1.0, nsteps, nsteps**2], dtype=np.float32)[:ndim]
        interp_idx = np.round(interp_idx).astype(np.int32).sum(axis=-1)
        interp_idx[interp_idx < 0] = 0
        interp_idx[interp_idx > interp_kernel.shape[0] - 1] = interp_kernel.shape[0]  - 1
            
        self._interpolator = SimpleNamespace(idx=interp_idx, kernel=interp_kernel, width=self.plan.kernel_width)
        self._interpolator_set = True
        
    def set_dataset_shape(self, shape: tuple[int] | list[int]):
        """
        Set full dataset shape. Required for shot-by-shot interpolation.

        Parameters
        ----------
        shape : tuple[int] | list[int]
            Set full dataset shape ``(*other, coil, k2, k1, k0)``.

        """
        if self._dataset_shape_set:
            return
        dataset_shape = tuple(shape)
        dataset_shape[-4] = 1 # No need to replicate coil axis
        dummy_dataset = np.ones(dataset_shape)
        
        if self._interpolator_set: # Directly broadcast interpolator indexes
            self._interpolator.idx, _ = np.broadcast_arrays(self._interpolator.idx, dummy_dataset)
        else: # Broadcast Plan displacements
            self.plan.distances, _ = np.broadcast_arrays(self.plan.distances, dummy_dataset)
        
        self._dataset_shape_set = True
        
    def interpolate(
            self, 
            data: NDArray[complex],
            shot_index: int | tuple[int] | None = None,
            ret_image: bool = False,
        ) -> NDArray[complex]:
        """
        Apply interpolator.
        
        Parameters
        ----------
        data : NDArray[complex]
            Input k-space dataset of shape ``(*other, coils, k2, k1, k0)`` (full dataset interpolation).
            or ``(coils, k0)`` (shot-by-shot interpolation).
        shot_index : int | tuple[int] | None, optional
            Shot index for current input data. If not provided, assume this is the whole dataset.
            The default is ``None``.
        ret_image : bool, optional
            Return reconstructed image. If ``False``, return sparse Cartesian k-space data instead
            The default is ``False``.

        Returns
        -------
        NDArray[complex]
            If ``ret_image`` is ``True``, return ``(other*, coil, z, y, x)`` image.
            If it is ``False``, return ``(other*, coil, k2, k1, kernel.width * k0)`` sparse
            Cartesian k-space samples.

        """
        if self._interpolator_set is False:
            raise RuntimeError("GRAPPA kernels have not been set. Call calc_interp_table() first.")
        if shot_index is not None:
            if self._dataset_shape_set is False:
                raise RuntimeError("For shot-by-shot interpolation, provide full data shape via set_dataset_shape().")
            if np.isscalar(shot_index):
                shot_index = (shot_index,)
            shot_index = tuple(shot_index)
        
        # Apply regridding
        output = _GrogRegridder(data, self._interpolator, shot_index)
        
        # Force garbage collection
        gc.collect()
        
        # If required, reconstruct
        if ret_image:
            ...
                        
        return output
            
    def __call__(
            self, 
            data: NDArray[complex], 
            shot_index: int | tuple[int] | None = None, 
            ret_image: bool = False
        ) -> NDArray[complex]:
        return self.interpolate(data, shot_index, ret_image)
        
# %% Methods
def _CreateGrogPlan(
    shape, 
    coords, 
    oversamp=None, 
    radius=0.75, 
    time_map=None
):
    ndim = coords.shape[-1]
    
    # Preprocess args
    shape = _default_shape(ndim, shape)
    oversamp = _default_oversamp(ndim, oversamp)
    coords = _rescale_coords(coords, shape[-ndim:])
    coords_shape = coords.shape
    if time_map is not None:
        time_map, _ = np.broadcast_arrays(time_map, coords[..., 0])
    
    # Create grid
    grid = _create_grid(ndim, shape, oversamp)
    
    # Get weights to average each source point
    weights = _estimate_weights(grid, coords, radius)
    
    # Get flattened jagget array of target indexes
    indexes, jagged_converter = _estimate_indexes(grid, coords, radius)
    kernel_width = jagged_converter.shape[-1]
    
    # For each Non Cartesian source, find distance from each Cartesian target
    distances, cart_target_coords = _estimate_distances(grid, coords, indexes, jagged_converter)
    
    # Assign weight to the correct Non Cartesian sample
    weights = jagged_converter.to_standard_array(weights[indexes])
    
    # Assign to each sample its sampling time
    if time_map is not None:
        time_map = jagged_converter.to_jagged_array(time_map)

    # Convert jagged indexes array to standard (zero-filled) array
    indexes = jagged_converter.to_standard_array(indexes)
    
    # Flatten
    distances = distances.reshape(-1, ndim)
    cart_target_coords = cart_target_coords.reshape(-1, ndim)
    weights = weights.ravel()
    if time_map is not None:
        time_map = time_map.ravel()
    indexes = indexes.ravel()
    
    # Reformat for output
    distances = distances.reshape(*coords_shape[:-2], coords_shape[-2] * kernel_width, ndim)
    cart_target_coords = cart_target_coords.reshape(*coords_shape[:-2], coords_shape[-2] * kernel_width, ndim)
    
    # Weights, time_map and indexes can be flattened to (*others, nsamples_per_volume)
    weights = weights.reshape(*coords_shape[:-5], -1)
    if time_map is not None:
        time_map = time_map.reshape(*coords_shape[:-5], -1)
    indexes = indexes.reshape(*coords_shape[:-5], -1)
        
    # Weights, time_map and indexes can be flattened to (*others, nsamples_per_volume)
    weights = weights.reshape(*coords_shape[:-5], -1)
    if time_map is not None:
        time_map = time_map.reshape(*coords_shape[:-5], -1)
    indexes = indexes.reshape(*coords_shape[:-5], -1)
    
    # Exclude fake points from computation
    idx = np.argsort(weights);
    true_begin = np.argmax(weights[idx] > 0)
    
    # Radial balancing
    radial_coords = (cart_target_coords**2).sum(axis=-1)**0.5
    radial_coords = radial_coords.reshape(*coords_shape[:-5], -1)
    radialsort = np.argsort(radial_coords[true_begin:])
    radial_coords = radial_coords[true_begin:][radialsort]
    
    # Get num workers
    num_workers = psutil.cpu_count(logical=False)
    
    # Find center threadmask
    num_center_samples = np.argmax(radial_coords > 0.0)
    center_bin_starts, center_bin_counts  = _split_indices(num_center_samples, num_workers)
    
    # Find outer threadmask
    _, outer_bin_counts  = _split_into_uniform_bins(radial_coords[center_bin_starts[-1]+center_bin_counts[-1]:], num_workers)
    outer_bin_starts = np.r_[center_bin_starts[-1] + center_bin_counts[-1], center_bin_starts[-1] + center_bin_counts[-1] + outer_bin_counts]
    
    # gridsort = np.concatenate((idx[:true_begin], idx[true_begin:][radialsort]))
    # datasort = np.zeros_like(gridsort) 
    # np.put_along_axis(datasort, gridsort, np.arange(gridsort.shape[-1]), axis=-1)
    
    # # Sort according to weights
    # radial_coords = radial_coords[gridsort]
    # weights = weights[gridsort]
    # if time_map is not None:
    #     time_map = time_map[gridsort]
    # indexes = indexes[gridsort]
        
    return SimpleNamespace(
            shape=shape, 
            oversamp=oversamp, 
            coords=cart_target_coords,
            radius=radius, 
            kernel_width=kernel_width,
            distances=distances,
            indexes=indexes,
            weights=weights,
            time_map=time_map,
        )

def _GrogRegridder(data, interpolator, shot_idx):
    
    # Prepare data and kernel indexes for interpolation
    if shot_idx is None:
        data = data[..., None].swapaxes(-5, -1) # (*other, coil, k2, k1, k0) -> (*other, 1, k2, k1, k0, coil)
        indexes = interpolator.idx # (1, k2, k1, k0'=interpolator.width * k0)
    else:
        data = data.T # (coil, k0) -> (k0, coil)
        indexes = interpolator.idx[shot_idx] # (k0'=interpolator.width * k0,)
    indexes = np.ascontiguousarray(indexes.ravel()) # (samples,) or (k0',)
        
    # Expand along readout
    data = np.repeat(data, interpolator.width, axis=-2) # (..., k0, coil) -> (..., k0', coil)
    data_shape = data.shape # (..., k0', coil)
    
    # Flatten data for computation
    if shot_idx is None:
        nbatches = np.prod(data_shape[:-5]).astype(int).item()
        data = data.reshape(nbatches, -1, data.shape[-1]) # (*other, 1, k2, k1, k0', coil) -> (batches, samples, coil)
        data = data.swapaxes(0, 1) # (batches, samples, coil) -> (samples, batches, coil) 
    else:
        data = data[:, None, :] # (k0', coil) -> (k0', 1, coil)
    data = np.ascontiguousarray(data)
    
    # Actual interpolation
    _interpolate(data, indexes, interpolator.kernel)
    
    # Reshape back
    if shot_idx is None:
        data = data.swapaxes(0, 1) # (samples, batches, coil) -> (batches, samples, coil) 
        data = data.reshape(data_shape) # (batches, samples, coil) -> (*other, 1, k2, k1, k0', coil)
        data = data.swapaxes(-5, -1)[..., 0]  # (*other, 1, k2, k1, k0, coil) -> (*other, coil, k2, k1, k0)
    else:
        data = data[:, 0, :].T # (k0', 1, coil) -> (coil, k0')
        
    # Make sure output is contiguous
    return np.ascontiguousarray(data)
    
#%% Subroutines
def _store_plan_inside_mrd(dset, plan):
    dataset_grp = dset.require_group("dataset")
    grp = dataset_grp.require_group("grog_plan")
    
    for key, value in vars(plan).items():
        if isinstance(value, np.ndarray):
            # If dataset already exists, overwrite it
            if key in grp:
                del grp[key]
            grp.create_dataset(key, data=value)
            
        elif isinstance(value, (np.integer, np.floating)):
            grp.attrs[key] = value.item()  # convert NumPy scalar to native Python type
        
        elif isinstance(value, (int, float)):
            grp.attrs[key] = value
        
        elif isinstance(value, tuple):
            if key in grp:
                del grp[key]
            grp.create_dataset(key, data=np.array(value))
        
        elif value is None:
            grp.attrs[key] = "__NONE__"
        
        else:
            raise TypeError(f"Unsupported type for key '{key}': {type(value)}")
            
def _load_plan_from_mrd(dset):
    grp = dset["dataset/grog_plan"]

    loaded = {}
    # read datasets
    for key in grp.keys():
        loaded[key] = grp[key][()]
    # read attributes
    for key, val in grp.attrs.items():
        if val == "__NONE__":
            loaded[key] = None
        else:
            loaded[key] = val

    return SimpleNamespace(**loaded)

def _default_shape(ndim, shape):
    if np.isscalar(shape):
        shape = ndim * [shape]
        
    shape = tuple(shape)
        
    return shape
    
def _default_oversamp(ndim, oversamp):
    if oversamp is None:
        if ndim == 2:
            oversamp = (1.0, 1.0)
        else:
            oversamp = (1.0, 1.0, 1.2)
    
    if np.isscalar(oversamp):
        oversamp = ndim * [oversamp]
        
    oversamp = tuple(oversamp)
    if len(oversamp) != ndim:
        raise ValueError(f"Oversampling {oversamp }does not match number of dimensions {ndim}")
        
    return oversamp
    
def _rescale_coords(coords, amp):
    cmax = abs(coords).reshape(-1, coords.shape[-1]).max(axis=0)
    if np.isscalar(amp):
        amp = coords.shape[-1] * [amp]
    return 0.5 * np.asarray(amp, dtype=coords.dtype) * coords / cmax

def _create_grid(ndim, shape, oversamp):
    grid = np.meshgrid(
        *[
            np.linspace(
                -shape[n] // 2, shape[n] // 2 - 1, int(np.ceil(oversamp[n] * shape[n]))
            )
            for n in range(ndim)
        ],
        indexing="ij",
    )
    return np.stack([ax.ravel() for ax in grid], axis=-1).astype(np.float32)

def _estimate_weights(grid, coords, radius):
    kdtree_coords = KDTree(coords.reshape(-1, coords.shape[-1]))
    
    # Query Non Cartesian points within given radius from each grid point
    weights = kdtree_coords.query_ball_point(grid, r=radius, workers=-1)
    weights = [1 / len(w) if len(w) else 0.0 for w in weights]
    return np.asarray(weights, dtype=np.float32)

class _JaggedConverter:
    
    def __init__(self, num_sources, num_targets_per_source):
        # Build row indexes
        self.row_idx = np.repeat(np.arange(num_sources, dtype=int), num_targets_per_source)

        # Build col indexes        
        total = int(num_targets_per_source.sum())       
        starts = np.concatenate(([0], np.cumsum(num_targets_per_source)[:-1]))
        self.col_idx = np.arange(total, dtype=np.int32) - np.repeat(starts, num_targets_per_source).astype(np.int32)   
        
        self.num_sources = num_sources
        self.max_num_targets_per_source = np.max(num_targets_per_source)
        self.num_targets_per_source = num_targets_per_source
        
    def to_standard_array(self, input):
        output = np.zeros((self.num_sources, self.max_num_targets_per_source), dtype=input.dtype)
        output[self.row_idx, self.col_idx] = input
        return output
    
    def to_jagged_array(self, input):
        return np.repeat(input, self.num_targets_per_source)
    
    @property
    def shape(self):
        return (self.num_sources, self.max_num_targets_per_source)
    
def _estimate_indexes(grid, coords, radius):
    ndim = coords.shape[-1]
    kdtree_grid = KDTree(grid)
       
    # Query Cartesian points within given radius from each sample (jagged array)
    indexes = kdtree_grid.query_ball_point(coords.reshape(-1, ndim), r=radius, workers=-1)
 
    # Count number of Cartesian target for each sample
    num_targets_per_source = np.asarray([len(idx) for idx in indexes], dtype=np.int32)
        
    # Build converter from jagged to standard array
    jagged_onverter = _JaggedConverter(indexes.shape[0], num_targets_per_source)
            
    # Flatten indexes
    indexes = np.fromiter(chain.from_iterable(indexes.tolist()), dtype=np.int32)
    
    return indexes, jagged_onverter

def _estimate_distances(grid, coords, indexes, jagged_converter):
    ndim = coords.shape[-1]
    coords = np.ascontiguousarray(coords.reshape(-1, ndim).T)
    grid = np.ascontiguousarray(grid.T)
    distances = []
    cart_target_coords = []
    for ax in range(ndim):
        _cart_target_coords = grid[ax][indexes]
        _noncart_source_coords = jagged_converter.to_jagged_array(coords[ax])
        _distance = _cart_target_coords - _noncart_source_coords
        distances.append(jagged_converter.to_standard_array(_distance))
        cart_target_coords.append(jagged_converter.to_standard_array(_cart_target_coords))
    return np.stack(distances, axis=-1), np.stack(cart_target_coords, axis=-1)

def _split_indices(N, n):
    q, r = divmod(N, n)
    starts = []
    counts = []
    start = 0
    for i in range(n):
        end = start + q + (1 if i < r else 0)
        starts.append(start)
        counts.append(end-start)
        start = end
    return np.asarray(starts, dtype=np.uint32), np.asarray(counts, dtype=np.uint32)

def _split_into_uniform_bins(samples, n_bins):
    starts = np.r_[0, np.where(np.diff(samples) != 0)[0] + 1]
    counts = np.diff(np.r_[starts, len(samples)]).astype(np.uint32)
    
    total = counts.sum().item()
    target = total / n_bins

    bin_starts = [0]
    bin_counts = []
    acc = 0.0

    for i, c in enumerate(counts):
        acc += c
        if acc >= target and len(bin_counts) < n_bins - 1:
            bin_counts.append(acc)
            bin_starts.append(i + 1)
            acc = 0.0

    # Last bin takes remaining counts
    bin_counts.append(acc)
    return np.asarray(bin_starts, dtype=np.uint32), np.asarray(bin_counts, dtype=np.uint32)
    
def _compute_interpolator(samples_map, distances, radius, interp_kernel, precision):
    ndims = distances.shape[-1]
    pfac = 10.0**precision
    stepsize = 10 ** (-precision)
    nsteps = 2 * radius / 10 ** (-precision) + 1
    nsteps = int(nsteps)
        
    # We can now use distances to compute the kernel table index, i.e.,
    # the index in the precompute kernel table to select the appropriate precomputed
    # interpolation kernel to grid each Non Cartesian sample to the target Cartesian location.
    interp_idx = (radius + np.round(distances * pfac) / pfac) / stepsize
    interp_idx = np.round(interp_idx).astype(np.float32)
    
    interp_idx *= np.asarray([1.0, nsteps, nsteps**2], dtype=np.float32)[:ndims]
    interp_idx = np.round(interp_idx).astype(np.int32).sum(axis=-1)
        
    return SimpleNamespace(idx=interp_idx, kernel=interp_kernel)

# %% Numba helpers
@nb.njit(parallel=True, fastmath=True, cache=True, inline="always")
def _interpolate(data, indexes, kernel):
    """
    Numba-friendly interpolation: data is expected with shape (nsamples, nbatches, ncoils)
    kernel is an array of shape (n_kernels, ncoils, ncoils)
    indexes is an integer array of length nsamples selecting a kernel per-sample.

    For each sample n and batch b:
        data[n, b, :] := kernel[indexes[n]] @ data[n, b, :]
    Implemented with explicit loops to avoid .dot attribute lookups and shape-checks
    that prevent good extraction/hoisting in Numba.
    """
    nsamples, nbatches, ncoils = data.shape

    for n in nb.prange(nsamples):
        kv = kernel[indexes[n]]  # (ncoils, ncoils)
        for b in range(nbatches):
            # compute result in-place (overwrites data[n, b, :])
            # Use an explicit mat-vec multiply so Numba can optimize it better.
            # temp accumulator per output component i
            for i in range(ncoils):
                s = 0.0
                # kv[i, j] times data[n, b, j]
                for j in range(ncoils):
                    s += kv[i, j] * data[n, b, j]
                data[n, b, i] = s
