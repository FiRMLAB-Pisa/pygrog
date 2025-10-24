"""GROG interpolator class."""

__all__ = ["GrogInterpolator"]

import gc
import os
import pathlib

from types import SimpleNamespace
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
    nsamples = coords.shape[-2]
    
    # Preprocess args
    shape = _default_shape(ndim, shape)
    oversamp = _default_oversamp(ndim, oversamp)
    coords = _rescale_coords(coords, shape[-ndim:])
    
    # Create grid
    grid = _create_grid(ndim, shape, oversamp)
    
    # Create KDTree
    kdtree = KDTree(grid)
    
    # Query Cartesian points within given radius from each sample
    _samples_map = kdtree.query_ball_point(coords.reshape(-1, ndim), r=radius, workers=-1)
    
    # Count number of Cartesian target for each sample
    num_targets_per_source = [len(el) for el in _samples_map]
    
    # Find max number of targets per source
    max_num_targets_per_source = np.max(num_targets_per_source)
    
    # Pad to have equal number of targets per each source 
    pad = [max_num_targets_per_source - count for count in num_targets_per_source]
    
    # Build weights
    weights = [1 / count if count else 0.0 for count in num_targets_per_source]
    weights = np.asarray(weights, dtype=np.float32)[..., None]
    weights = np.repeat(weights, max_num_targets_per_source, -1)
    
    cols = np.arange(max_num_targets_per_source)
    mask = cols >= (max_num_targets_per_source - np.asarray(pad)[:, None])
    weights[mask] = 0.0
    
    # Pad samples_map
    # Compute lengths
    lens = np.asarray(num_targets_per_source)
    samples_map = np.zeros((_samples_map.shape[0], max_num_targets_per_source), dtype=np.int32)
    
    # Flatten data
    _samples_map = np.concatenate(_samples_map)
    
    # Build fancy indices
    row_idx = np.repeat(np.arange(len(samples_map)), lens)
    col_idx = np.concatenate([np.arange(l) for l in lens])
    
    # Fancy assignment
    samples_map[row_idx, col_idx] = _samples_map
    
    # Free some memory
    del _samples_map
    del row_idx
    del col_idx
    del mask
    del pad
    gc.collect()
    
    # Reshape indexes map
    samples_map = samples_map.reshape(*coords.shape[:-2], nsamples * max_num_targets_per_source)
    
    # Reshape weighs
    weights = weights.reshape(*coords.shape[:-2], nsamples * max_num_targets_per_source)

    # Get Cartesian coordinates
    cart_output_coords = np.stack([grid[samples_map, ax] for ax in range(ndim)], axis=-1)
    
    # Get distances
    distances = cart_output_coords - np.repeat(coords, max_num_targets_per_source, axis=-2)
    
    # Time map
    if time_map is not None:
        time_map, _ = np.broadcast_arrays(time_map, coords[..., 0])
        time_map = np.repeat(time_map, max_num_targets_per_source, -1)
    
    return SimpleNamespace(
        shape=shape, 
        oversamp=oversamp, 
        radius=radius, 
        kernel_width=max_num_targets_per_source,
        distances=distances,
        indexes=samples_map,
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

    # Allocate output
    output = np.zeros_like(data) # (samples, batches, coil) or (k0', 1, coil)
    
    # Actual interpolation
    _interpolate(output, data, indexes, interpolator.kernel)
    
    # Reshape back
    if shot_idx is None:
        output = output.swapaxes(0, 1) # (samples, batches, coil) -> (batches, samples, coil) 
        output = output.reshape(data_shape) # (batches, samples, coil) -> (*other, 1, k2, k1, k0', coil)
        output = output.swapaxes(-5, -1)[..., 0]  # (*other, 1, k2, k1, k0, coil) -> (*other, coil, k2, k1, k0)
    else:
        output = output[:, 0, :].T # (k0', 1, coil) -> (coil, k0')
        
    return np.ascontiguousarray(output)
    
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

def _rescale_coords(coords, amp):
    cmax = abs(coords).reshape(-1, coords.shape[-1]).max(axis=0)
    if np.isscalar(amp):
        amp = coords.shape[-1] * [amp]
    return 0.5 * np.asarray(amp, dtype=coords.dtype) * coords / cmax

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
@nb.njit(fastmath=True, cache=True, inline="always")  # pragma: no cover
def _matvec(y, A, x):
    ni, nj = A.shape
    for i in range(ni):
        for j in range(nj):
            y[i] += A[i][j] * x[j]
            
@nb.njit(fastmath=True, cache=True, inline="always")  # pragma: no cover
def _interpolate(output, data, indexes, kernel):
    nbatches, nsamples, ncoils = output.shape
    
    for n in nb.prange(nsamples):
        kernel_value = kernel[indexes[n]] 
        for b in range(nbatches):
            for batch in range(nbatches):
                _matvec(
                    output[n, batch], # (ncoils,)
                    kernel_value, # (ncoils, ncoils)
                    data[n, b], # (ncoils,)
                )
