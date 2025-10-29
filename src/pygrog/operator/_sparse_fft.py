"""Sparse FF routines with batching support."""

__all__ = ["SparseFFT"]

from types import SimpleNamespace
from numpy.typing import NDArray

import numpy as np
try:
    import cupy as cp
    __CUPY_AVAILABLE__ = True
except ImportWarning:
    __CUPY_AVAILABLE__ = False

from .._base import fft, ifft
from .._sigpy import resize, get_array_module, to_device

class SparseFFT:
    """
    Sparse Fast Fourier transform class.
    
    Implements indexing and its adjoint (grid), i.e., transforms sparse k-space
    data into dense k-space data, followed by FFT to retrieve the image.
    
    Parameters
    ----------
    metadata: SimpleNamespace
        Structure with the following fields:
            
            * shape: tuple[int]
                Image shape ``(z, y, x)``
            * grid_shape: tuple[int]
                Oversampled grid shape ``(osf_z * z, osf_y * y, osf_x * x)``.
            * indexes: NDArray[int]
                Indexes to perform sparse-to-dense and dense-to-sparse operations,
                shaped ``(prod(other), k2*k1*k0*kernel_width)``.
            * weights: NDArray[float]
                Weights assigned to each grid location based on number of source points,
                shaped ``(prod(other), k2*k1*k0*kernel_width)``.
            * start_index: list[int]
                First index for each batch corresponding to non-zero weight.

    smaps: NDarray[complex], optional
        Coil sensitivity maps of shape ``(coils, z, y, x)``. If not provided,
        only ``adj_op()`` is allowed, and Sum-of-Squares of the channels will be performed.
        
    """
    
    def __init__(self, metadata: SimpleNamespace, smaps: NDArray[complex] | None = None):
        self.shape = metadata.shape
        self.grid_shape = metadata.oshape
        self.indexes = metadata.indexes
        self.weights = metadata.weights
        self.start_index = metadata.bin_global_start
        self.smaps = smaps
        
    def to_device(self, device_id: int):
        """
        Move internal parameters to GPU.

        Parameters
        ----------
        device_id : int
            ID of the target device. CPU has ``device_id == -1``,
            n-th GPU has ``device_id == n`` (``n >= 0``)

        """
        self.indexes = to_device(self.indexes, device_id)
        self.weights = to_device(self.weights, device_id)
        if self.smaps is not None:
            self.smaps = to_device(self.smaps, device_id)
        
    def op(self, image: NDArray[complex]) -> NDArray[complex]:
        """
        Perform sparse forward FFT (image -> k-space).

        Parameters
        ----------
        image : NDArray[complex]
            Input image of shape ``(*other, 1, z, y, x)``.

        Raises
        ------
        RuntimeError
            If ``smaps`` is not provided..

        Returns
        -------
        NDArray[complex]
            Sparse k-space of shape ``(*other, coil, 1, 1, k2*k1*k0*kernel_width)``.

        """
        if self.smaps is None:
            raise RuntimeError('Please provide smaps to run forward operation')
        return sparse_fft(
            image,
            self.start_index,
            self.indexes, 
            self.weights, 
            self.shape, 
            self.grid_shape, 
            self.smaps
        )
    
    def adj_op(self, data: NDArray[complex]) -> NDArray[complex]:
        """
        Perform sparse adjoint FFT (k-space -> image).

        Parameters
        ----------
        data : NDArray[complex]
            Input sparse k-space of shape ``(*other, coil, 1, 1, k2*k1*k0*kernel_width)``.

        Returns
        -------
        NDArray[complex]
            Coil-combined image of shape ``(*other, 1, z, y, x)``.

        """
        return sparse_ifft(
            data,
            self.start_index,
            self.indexes, 
            self.weights, 
            self.shape, 
            self.grid_shape, 
            self.smaps
        )
        

# %% Subroutines
def sparse_ifft(data, start_index, indexes, weights, shape, grid_shape, smaps=None):
    xp = get_array_module(data)
    
    # Flatten batch axes
    batch_axes = data.shape[:-4]
    nbatches = np.prod(batch_axes).astype(int).item()
    ncoils = data.shape[-4]
    data = data.reshape(nbatches, ncoils, -1)
    
    # Allocate output
    image = xp.zeros((nbatches, *shape), data.dtype)
        
    # Loop over batches
    if nbatches > 1:
        for b in range(nbatches):
            _sparse_ifft(
                image[b], 
                smaps,
                data[b, :, start_index[b]:] * (weights[b, start_index[b]:]**0.5), 
                indexes[b, start_index[b]:], 
                grid_shape
            )
    else:
        _sparse_ifft(
            image[0], 
            smaps, 
            data[0, :, start_index:] * (weights[start_index:]**0.5), 
            indexes[start_index:], 
            grid_shape
        )
        
    return image.reshape(*batch_axes, 1, *shape)
    

def sparse_fft(image, start_index, indexes, weights, shape, grid_shape, smaps):
    xp = get_array_module(image)
    
    # Flatten batch axes
    batch_axes = image.shape[:-4]
    nbatches = np.prod(batch_axes).astype(int).item()
    ncoils = smaps.shape[0]
    image = image.reshape(nbatches, *image.shape[-4:])
    
    # Allocate output
    data = xp.zeros((nbatches, ncoils, indexes.shape[-1]), dtype=image.dtype)
    
    # Loop over batches
    if nbatches > 1:
        for b in range(nbatches):
            _sparse_fft(
                data[b], 
                smaps, 
                image[b], 
                indexes[b, start_index[b]:], 
                grid_shape
            )
            data[b] *= (weights[b]**0.5)
    else:
        _sparse_fft(
            data[0], 
            smaps, 
            image[0], 
            indexes, 
            grid_shape
        )
        data *= (weights**0.5)
        
    return data.reshape(*batch_axes, ncoils, 1, 1, -1)

    
def _sparse_ifft(img, smaps, ksp, indexes, gshape):
    xp = get_array_module(ksp)
    
    # Allocate temporary output
    for n in range(ksp.shape[0]):
        _ksp = xp.zeros(gshape, img.dtype) # gridded kspace
        if xp.__name__ == 'numpy':
            np.add.at(_ksp.ravel(), indexes, ksp[n])
        else:
            cp.add.at(_ksp.ravel().real, indexes, ksp[n].real)
            cp.add.at(_ksp.ravel().imag, indexes, ksp[n].imag)
            
        # Transform
        _img = ifft(_ksp)
        
        # Remove oversampling
        _img = resize(_img, img.shape)
        
        # Accumulate
        if smaps is not None:
            img += (smaps[n].conj() * _img)
        else:
            img += (_img.conj() * _img)
            
    # If SoS, take square root
    if smaps is None:
        img = img**0.5
        
def _sparse_fft(ksp, smaps, img, indexes, gshape):    
    for n in range(ksp.shape[0]):
        # Apply coil sensitivity
        _img = smaps[n] * img[0]
        
        # Pad
        _img = resize(_img, gshape)
        
        # Transform
        _ksp = fft(_img)
        
        # Keep
        ksp[n] = _ksp.ravel()[indexes]
        