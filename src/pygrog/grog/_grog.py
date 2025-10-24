"""
"""

__all__ = []

import gc

from types import SimpleNamespace

import numpy as np

from scipy.spatial import KDTree


class GrogInterpolator:
    
    def __init__(self):
        pass
    
    @classmethod
    def from_file(self):
        ...
        
    def to_file(self):
        ...
        
    def calc_grappa_kernel(self):
        ...
        
    def interpolate(self):
        ...
        
    def __call__(self):
        return self.interpolate()
        
    
    
# %%
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

def _GrogRegridder():
    ...

#%% Subroutines
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
    