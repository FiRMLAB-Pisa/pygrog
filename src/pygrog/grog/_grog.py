"""GROG interpolator class."""

__all__ = ["GrogInterpolator"]

import gc
import os
import pathlib

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
    Implements the GROG algorithm as described in [1]_. In this implementation,
    we assume that coords matches the shape of data, except for coil axes - 
    broadcasting, for the moment, is not automatically handled.

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
    _dataset_shape = None
    _data = []
    
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
                bin_global_start=self.plan.bin_global_start,
                bin_starts=self.plan.bin_starts,
                bin_counts=self.plan.bin_counts,
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
        if self._dataset_shape_set is False:
            raise RuntimeError("Please provide full data shape via set_dataset_shape().")
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
            
        self.plan.distances = None
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
        
        # Store full dataset shape
        self.dataset_shape = list(shape)
        self.dataset_shape[-1] *= self.plan.kernel_width
        self.dataset_shape = tuple(self.dataset_shape)
        
        # Broadcast dataplan
        dataset_shape = list(shape)
        dataset_shape[-4] = 1 # No need to replicate coil axis
        dataset_shape[-1] = 1 # Let broadcast handle automatically the readout dimension
        dummy_dataset = np.ones(dataset_shape, dtype=np.uint8)
        
        # Broadcast Plan
        self.plan.distances, _ = np.broadcast_arrays(self.plan.distances, dummy_dataset[..., None])
        self.plan.coords, _ = np.broadcast_arrays(self.plan.coords, dummy_dataset[..., None])        
        self.plan.indexes, _ = np.broadcast_arrays(self.plan.indexes, dummy_dataset)
        self.plan.weights, _ = np.broadcast_arrays(self.plan.weights, dummy_dataset)
        if self.plan.time_map is not None:
            self.plan.time_map, _ = np.broadcast_arrays(self.plan.time_map, dummy_dataset)
        
        self._dataset_shape_set = True
        
    def interpolate(
            self, 
            data: NDArray[complex],
            shot_index: int | tuple[int] | None = None,
            ret_image: bool = False,
        ) -> None | NDArray[complex]:
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
            The default is ``False``. Ignored if ``shot_index`` is provided.

        Returns
        -------
        None | NDArray[complex]
            If ``shot_index`` is provided, return ``None``.
            Otherwise, if ``ret_image`` is ``True``, return ``(other*, coil, z, y, x)`` image.
            If it is ``False``, return ``(other*, coil, 1, 1, k2 * k1 * k0 * kernel.width)`` sparse
            Cartesian k-space samples.

        """
        if self._interpolator_set is False:
            raise RuntimeError("GRAPPA kernels have not been set. Call calc_interp_table() first.")
        if shot_index is not None:
            if np.isscalar(shot_index):
                shot_index = (shot_index,)
            shot_index = tuple(shot_index)
        
            # Apply regridding
            self._data.append(_GrogRegridder(data, self._interpolator, shot_index))
     
            # Force garbage collection
            gc.collect()
            
            return
            
        # Perform
        self._data = _GrogRegridder(data, self._interpolator, shot_index)
        data, coords = self.sort_data()
        
        # If required, reconstruct
        if ret_image:
            ...
                        
        # Force garbage collection
        gc.collect()
        
        return data, coords
    
    def sort_data(self):
        if isinstance(self._data, list):
            data = np.stack(self._data, axis=0)
            data = data.reshape(*self._dataset_shape[:-1], self._dataset_shape[-1] * self.plan.kernel_width)
        else:
            data = self._data
            
        # Infer data shape
        coords_shape = list(data.shape)
        coords_shape[-4] = 1 # Coil does not matter 
        ncoils = data.shape[-4]
            
        # Unpack
        time_map = self.plan.time_map
        weights = self.plan.weights
        indexes = self.plan.indexes
        coords = self.plan.coords
        ndim = coords.shape[-1]
                        
        # Flatten indexes, weights and time_map to (*other, nsamples_per_volume)
        if time_map is not None:
            time_map = time_map.reshape(*coords_shape[:-5], -1)
        else:
            time_map = None
        weights = weights.reshape(*coords_shape[:-5], -1)
        indexes = indexes.reshape(*coords_shape[:-5], -1)
        
        # Flatten batch axes
        if time_map is not None:
            time_map = time_map.reshape(-1, time_map.shape[-1])
        weights = weights.reshape(-1, weights.shape[-1])
        indexes = indexes.reshape(-1, indexes.shape[-1])
        coords = coords.reshape(*coords_shape, -1)
        
        # Sort based on weights
        gridsort = np.argsort(weights, axis=-1)
        sorted_weights = np.stack([weights[n, gridsort[n]] for n in range(gridsort.shape[0])], axis=0)
        sorted_indexes = np.stack([indexes[n, gridsort[n]] for n in range(gridsort.shape[0])], axis=0)
        
        # Compute bin for each batch axes
        bin_global_start = np.argmax(sorted_weights > 0, axis=-1)
        for n in range(bin_global_start.shape[0]):
            _tmp = sorted_indexes[n][bin_global_start[n]:]
            _order1 = np.argsort(_tmp)
            _tmp = _tmp[_order1]
            
            # Find number of repetitions for each element
            # _starts = np.r_[0, np.where(np.diff(_tmp) != 0)[0] + 1]
            # _counts = np.diff(np.r_[_starts, len(_tmp)]).astype(np.uint32)
            # _counts = np.repeat(_counts, _counts)
            
            # # Sort based on counts
            # _order2 = np.argsort(-_counts)
            
            # Sort nonzero weight part
            _nonzero_part = gridsort[n][bin_global_start[n]:]
            _nonzero_reordered = _nonzero_part[_order1]#[_order2]
            
            # Retrieve unsorted zero weight part
            _zero_part = gridsort[n][:bin_global_start[n]]
            
            # Concatenate back to output
            gridsort[n] = np.r_[_zero_part, _nonzero_reordered]
            
        # Free some memory
        del sorted_indexes
        del sorted_weights
                
        # Reorder adjoints
        if time_map is not None:
            time_map = np.stack([time_map[n, gridsort[n]] for n in range(gridsort.shape[0])], axis=0)
        weights = np.stack([weights[n, gridsort[n]] for n in range(gridsort.shape[0])], axis=0)
        indexes = np.stack([indexes[n, gridsort[n]] for n in range(gridsort.shape[0])], axis=0)
        
        # Perform binning
        bin_starts = []
        bin_counts = []
        for n in range(bin_global_start.shape[0]):
            _starts = np.r_[0, np.where(np.diff(indexes[n][bin_global_start[n]:]) != 0)[0] + 1]
            _counts = np.diff(np.r_[_starts, len(indexes[n][bin_global_start[n]:])]).astype(np.uint32)
            
            # Retain
            bin_starts.append(_starts)
            bin_counts.append(_counts)
                    
        # Reshape data to (nbatches, ncoils, nsamples)
        batch_axes = data.shape[:-4]
        
        # Sort data
        data = data.reshape(*batch_axes, ncoils, -1)
        data = data.reshape(-1, *data.shape[-2:])
        for b in range(len(gridsort)):
            for n in range(ncoils):
                data[b, n, :] = data[b, n, gridsort[b]]
        data = data.reshape(*batch_axes, ncoils, 1, 1, -1)
        
        # Sort coordinates (e.g., for compatibility with other frameworks)
        coords = coords.reshape(-1, *coords.shape[-4:])
        coords = coords.reshape(coords.shape[0], -1, ndim)
        for b in range(len(gridsort)):
            for ax in range(ndim):
                coords[b, :, ax] = coords[b, gridsort[b], ax]
        coords = coords.reshape(*batch_axes, 1, 1, 1, -1, ndim)
        
        # Remove batch axis if singleton
        if len(gridsort) == 1:
            if time_map is not None:
                time_map = time_map[0]
            weights = weights[0]
            indexes = indexes[0]
            bin_starts = bin_starts[0]
            bin_counts = bin_counts[0]
            bin_global_start = bin_global_start[0]
            
        # Remove unused stuff
        self._data = None
            
        # Update plan
        self.plan.time_map = time_map
        self.plan.weights = weights
        self.plan.indexes = indexes
        self.plan.bin_starts = bin_starts
        self.plan.bin_counts = bin_counts
        self.plan.bin_global_start = bin_global_start
        
        return data, coords
        
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
        
    # 'xyz'->'zyx'
    shape = shape[::-1]
    oversamp = oversamp[::-1]
    coords = coords[..., ::-1]
    
    # Create grid
    grid = _create_grid(ndim, shape, oversamp)
    
    # Get weights to average each source point
    weights = _estimate_weights(grid, coords, radius)
    
    # Get flattened jagget array of target indexes
    indexes, jagged_converter = _estimate_indexes(grid, coords, radius)
    kernel_width = jagged_converter.max_num_targets_per_source
        
    # For each Non Cartesian source, find distance from each Cartesian target
    distances, cart_target_coords = _estimate_distances(grid, coords, indexes, jagged_converter)

    # Assign to each sample its sampling time
    if time_map is not None:
        time_map = jagged_converter.to_jagged_array(time_map)
        
    # Assign weight to the correct Non Cartesian sample
    weights = jagged_converter.to_standard_array(weights[indexes])

    # Convert jagged indexes array to standard (zero-filled) array
    indexes = jagged_converter.to_standard_array(indexes)
    
    # Reshape for output
    if time_map is not None:
        time_map = time_map.reshape(*coords_shape[:-2], -1)
    weights = weights.reshape(*coords_shape[:-2], -1)
    indexes = indexes.reshape(*coords_shape[:-2], -1)    
    distances = distances.reshape(*coords_shape[:-2], coords_shape[-2] * kernel_width, ndim)
    cart_target_coords = cart_target_coords.reshape(*coords_shape[:-2], coords_shape[-2] * kernel_width, ndim)
        
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
            bin_global_start=None,
            bin_starts=None,
            bin_counts=None,
        )

def _GrogRegridder(data, interpolator, shot_idx):    
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
            # Regular NumPy array
            if key in grp:
                del grp[key]
            grp.create_dataset(key, data=value)
        
        elif isinstance(value, (np.integer, np.floating)):
            grp.attrs[key] = value.item()
        
        elif isinstance(value, (int, float)):
            grp.attrs[key] = value
        
        elif isinstance(value, tuple):
            if key in grp:
                del grp[key]
            grp.create_dataset(key, data=np.array(value))
        
        elif isinstance(value, list):
            # list of numpy arrays → VLEN dataset
            dtype = h5py.vlen_dtype(value[0].dtype)
            if key in grp:
                del grp[key]
            ds = grp.create_dataset(key, (len(value),), dtype=dtype)
            for n in range(len(value)):
                ds[n] = value[n].tolist()
        
        elif value is None:
            grp.attrs[key] = "__NONE__"
        
        else:
            raise TypeError(f"Unsupported type for key '{key}': {type(value)}")
            
def _load_plan_from_mrd(dset):
    grp = dset["dataset/grog_plan"]
    loaded = {}

    # read datasets
    for key in grp.keys():
        data = grp[key][()]
        # check for VLEN arrays (object arrays) → convert to list of NumPy arrays
        if isinstance(data, np.ndarray) and data.dtype.kind == 'O':
            loaded[key] = [np.array(x) for x in data]
        else:
            loaded[key] = data

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
    jagged_converter = _JaggedConverter(indexes.shape[0], num_targets_per_source)
            
    # Flatten indexes
    indexes = np.fromiter(chain.from_iterable(indexes.tolist()), dtype=np.int32)
    
    return indexes, jagged_converter

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
