"""Sparse FFT operator for GROG-gridded data."""

__all__ = ["SparseFFT"]

import numpy as np
from numpy.typing import NDArray

import torch

from .._utils import resize
from .._base._fftc import fft, ifft


class SparseFFT:
    """
    Sparse FFT / IFFT operator.

    Maps between non-uniform k-space samples and Cartesian image space
    using a GROG interpolation plan (pre-computed indices and weights).

    Parameters
    ----------
    grid_shape : tuple[int, ...]
        Full Cartesian k-space grid shape, e.g. ``(nz, ny, nx)``.
    out_shape : tuple[int, ...]
        Output image shape (center-crop), e.g. ``(nz, ny, nx)``.
    indices : NDArray[np.intp]
        Flat grid indices for each non-uniform sample, shape ``(n_samples,)``.
    weights : NDArray[np.floating]
        Per-sample density weights, shape ``(n_samples,)``.
    dual_stream : bool, optional
        Use dual-stream CUDA pipelining.  The default is ``False``.

    """

    def __init__(
        self,
        grid_shape: tuple[int, ...],
        out_shape: tuple[int, ...],
        indices: NDArray[np.intp],
        weights: NDArray[np.floating],
        dual_stream: bool = False,
    ):
        self.grid_shape = tuple(grid_shape)
        self.out_shape = tuple(out_shape)
        self.grid_size = int(np.prod(grid_shape))
        self.ndim = len(grid_shape)
        self.fft_axes = tuple(range(-self.ndim, 0))
        self.dual_stream = dual_stream

        # Store plan as torch tensors
        self.indices = torch.as_tensor(indices.ravel().astype(np.int64))
        self.weights = torch.as_tensor(weights.ravel().astype(np.float32))

    def forward(self, sparse_kspace: torch.Tensor) -> torch.Tensor:
        """
        Sparse k-space -> image (adjoint NUFFT direction).

        Scatter-add sparse data onto Cartesian grid, then IFFT + crop.

        Parameters
        ----------
        sparse_kspace : torch.Tensor
            Input data of shape ``(n_coils, n_samples)`` (complex).

        Returns
        -------
        torch.Tensor
            Image of shape ``(n_coils, *out_shape)`` (complex).

        """
        n_coils = sparse_kspace.shape[0]
        device = sparse_kspace.device
        dtype = sparse_kspace.dtype
        indices = self.indices.to(device)
        weights = self.weights.to(device, dtype=torch.float32)

        if self.dual_stream and device.type == "cuda":
            return self._forward_dual_stream(sparse_kspace, indices, weights)

        # Per-coil gridding
        out_shape = (n_coils,) + self.out_shape
        image = torch.zeros(out_shape, dtype=dtype, device=device)

        for c in range(n_coils):
            grid = torch.zeros(self.grid_size, dtype=dtype, device=device)
            grid.index_add_(0, indices, weights.to(dtype) * sparse_kspace[c])
            grid = grid.reshape(self.grid_shape)
            img_c = ifft(grid, oshape=self.out_shape, axes=self.fft_axes)
            image[c] = img_c

        return image

    def adjoint(self, image: torch.Tensor) -> torch.Tensor:
        """
        Image -> sparse k-space (forward NUFFT direction).

        FFT image to Cartesian grid, then gather at sparse locations.

        Parameters
        ----------
        image : torch.Tensor
            Input image of shape ``(n_coils, *out_shape)`` (complex).

        Returns
        -------
        torch.Tensor
            Sparse k-space of shape ``(n_coils, n_samples)`` (complex).

        """
        n_coils = image.shape[0]
        device = image.device
        dtype = image.dtype
        indices = self.indices.to(device)
        weights = self.weights.to(device, dtype=torch.float32)
        n_samples = indices.shape[0]

        if self.dual_stream and device.type == "cuda":
            return self._adjoint_dual_stream(image, indices, weights)

        output = torch.zeros(n_coils, n_samples, dtype=dtype, device=device)

        for c in range(n_coils):
            grid = fft(image[c], oshape=self.grid_shape, axes=self.fft_axes)
            grid_flat = grid.reshape(-1)
            output[c] = weights.to(dtype) * grid_flat[indices]

        return output

    def _forward_dual_stream(self, sparse_kspace, indices, weights):
        """Dual-stream CUDA forward: overlap FFT and data transfer."""
        n_coils = sparse_kspace.shape[0]
        device = sparse_kspace.device
        dtype = sparse_kspace.dtype
        out_shape = (n_coils,) + self.out_shape
        image = torch.zeros(out_shape, dtype=dtype, device=device)

        s1 = torch.cuda.Stream()
        s2 = torch.cuda.Stream()

        for c in range(n_coils):
            stream = s1 if c % 2 == 0 else s2
            with torch.cuda.stream(stream):
                grid = torch.zeros(self.grid_size, dtype=dtype, device=device)
                grid.index_add_(0, indices, weights.to(dtype) * sparse_kspace[c])
                grid = grid.reshape(self.grid_shape)
                image[c] = ifft(grid, oshape=self.out_shape, axes=self.fft_axes)

        torch.cuda.synchronize()
        return image

    def _adjoint_dual_stream(self, image, indices, weights):
        """Dual-stream CUDA adjoint."""
        n_coils = image.shape[0]
        device = image.device
        dtype = image.dtype
        n_samples = indices.shape[0]
        output = torch.zeros(n_coils, n_samples, dtype=dtype, device=device)

        s1 = torch.cuda.Stream()
        s2 = torch.cuda.Stream()

        for c in range(n_coils):
            stream = s1 if c % 2 == 0 else s2
            with torch.cuda.stream(stream):
                grid = fft(image[c], oshape=self.grid_shape, axes=self.fft_axes)
                output[c] = weights.to(dtype) * grid.reshape(-1)[indices]

        torch.cuda.synchronize()
        return output
    if sens_maps is not None:
        sens_maps = np.asarray(sens_maps)
        if sens_maps.shape[0] != ncoils:
            raise ValueError("sens_maps must have first dim == number of coils (samples.shape[0])")
        if sens_maps.shape[1:] != tuple(out_shape):
            raise ValueError("sens_maps spatial shape must match out_shape")

    # accumulators
    if sens_maps is not None:
        combined = np.zeros(out_shape, dtype=complex_dtype)
    else:
        sos = np.zeros(out_shape, dtype=real_dtype)
        
    # Apply weighting
    samples = weights * samples

    # Process each coil
    for v in range(ncoils):
        # weighted k-space samples for coil v, flattened
        vals = samples[v].ravel()

        # accumulate into grid_flat for this coil
        if use_bincount_for_speed:
            if np.iscomplexobj(vals):
                real_part = np.bincount(indexes_flat, weights=vals.real, minlength=prod_grid)
                imag_part = np.bincount(indexes_flat, weights=vals.imag, minlength=prod_grid)
                grid_flat = real_part + 1j * imag_part
            else:
                grid_flat = np.bincount(indexes_flat, weights=vals, minlength=prod_grid)
                if not np.iscomplexobj(grid_flat):
                    grid_flat = grid_flat.astype(complex_dtype, copy=False)
        else:
            grid_flat = np.zeros(prod_grid, dtype=complex_dtype)
            np.add.at(grid_flat, indexes_flat, vals.astype(complex_dtype))

        grid3 = grid_flat.reshape(grid_shape)
        grid3 = np.fft.ifftshift(grid3)
        img_k = np.fft.ifftn(grid3)
        img_k = np.fft.ifftshift(img_k)
        img_crop = _center_crop(img_k, out_shape)

        if sens_maps is not None:
            s_map_v = sens_maps[v]
            combined += img_crop * np.conj(s_map_v)
        else:
            sos += np.abs(img_crop) ** 2

    # Finalize
    if sens_maps is not None:
        return combined

    return np.sqrt(sos)
