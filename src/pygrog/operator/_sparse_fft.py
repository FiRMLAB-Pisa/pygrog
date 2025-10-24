"""
Gridding + 3D IFFT + coil combination pipeline.

Assumptions:
- samples.shape == (ncoils, 500, 48, 6752)
- weights.shape == (1, 500, 48, 6752)  (broadcast over coils)
- indexes.shape == (1, 500, 48, 6752)  (flat indices into the target grid)
- grid_shape: full k-space grid, e.g. (220, 220, 264)
- out_shape: final image shape after center-crop, e.g. (220, 220, 220)

Behavior:
- Accumulate weighted samples into the k-space grid for each coil (handles duplicate indices).
- Perform optional ifftshift then inverse 3D FFT.
- Center-crop to out_shape.
- If sens_maps is provided (shape (ncoils, *out_shape)), combine coils by multiplying each coil image
  by the complex conjugate of the corresponding sensitivity map and summing across coils. Optionally
  normalize by sum(|s|^2) before taking magnitude.
- If sens_maps is not provided, perform sum-of-squares (sqrt of sum |img_coil|^2) across coils.
- Final return is a real-valued magnitude image of shape out_shape by default. If you need the complex
  combined image when sens_maps are provided, set return_complex_combined=True.

"""

__all__ = ["reconstruct_and_combine"]

import numpy as np

def _center_crop(volume, out_shape):
    """Center-crop a 3D 'volume' to out_shape."""
    in_shape = volume.shape
    assert len(in_shape) == len(out_shape) == 3
    slices = []
    for i in range(3):
        start = (in_shape[i] - out_shape[i]) // 2
        if start < 0:
            raise ValueError(f"Output shape {out_shape} is larger than input {in_shape} along axis {i}")
        end = start + out_shape[i]
        slices.append(slice(start, end))
    return volume[slices[0], slices[1], slices[2]]


def reconstruct_and_combine(
        samples, 
        weights, 
        indexes,
        grid_shape,
        out_shape,
        sens_maps=None,
        use_bincount_for_speed=True,
):
    """
    Reconstruct and combine coil images.

    Parameters
    ----------
    samples : NDArray 
        Shape (ncoils, 500, 48, 6752)  (dtype can be float or complex)
    weights : NDArray
        Shape (1, 500, 48, 6752)  (will be broadcast over coils)
    indexes: NDArray
        Shape (1, 500, 48, 6752)  (flat indices into grid_shape)
    grid_shape : tuple[ints]
        Full k-space grid (kx, ky, kz)
    out_shape : tuple[ints] 
        Desired output image shape after cropping
    sens_maps : NDArray, optional
        (ncoils, out_x, out_y, out_z) with complex coil sensitivity maps
    use_bincount_for_speed : bool, optional
        Use np.bincount for accumulation (fast). 
        For complex, real/imag split is used.

    Returns
    -------
    NDArray 
        Shape out_shape. By default real-valued magnitude. If return_complex_combined is True
        and sens_maps provided, returns complex combined image (out_shape).

    """
    samples = np.asarray(samples)
    weights = np.asarray(weights)
    indexes = np.asarray(indexes)

    if samples.ndim < 4:
        raise ValueError("samples must have shape (ncoils, ..., ... , ...)")
    if indexes.size != weights.size:
        raise ValueError("weights and indexes must have the same number of elements")
    if samples.shape[1:] != weights.shape[1:] or samples.shape[1:] != indexes.shape[1:]:
        raise ValueError("samples[1:], weights[1:], and indexes[1:] must match in shape")
    ncoils = samples.shape[0]
    prod_grid = int(np.prod(grid_shape))

    # weights_flat = weights[0].ravel()
    indexes_flat = indexes[0].ravel().astype(np.intp)
    if np.any(indexes_flat < 0) or np.any(indexes_flat >= prod_grid):
        raise IndexError("Some indexes are out of bounds for grid_shape")

    complex_dtype = np.complex64
    real_dtype = np.float32

    # Validate sens_maps if provided
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
