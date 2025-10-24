"""Utilities for SVD-based subspace estimation (e.g., Bloch or Coil Compression)."""

from __future__ import annotations

__all__ = [
    "SVDCompression",
    "compress_coil",
    "estimate_coil_subspace",
]

import torch

from numpy.typing import NDArray

from mrinufft._array_compat import with_torch

from ._acr import extract_acr


def estimate_coil_subspace(
    data: NDArray[complex],
    num_coils: int = None,
    explained_variance_ratio: float | None = None,
    cal_width: int = 24,
    coords: NDArray[float] | None = None,
    shape: int = None,
    ndim: int | None = None,
) -> SVDCompression:
    """
    Estimate Bloch subspace basis.

    Parameters
    ----------
    training_data : NDArray[complex]
        Training dataset of shape ``(*other, coil, k2, k1, k0)``.
    num_coils : int, optional
        Size of subspace basis (i.g., subspace size).
        User can either specify this or the target explained variance ratio.
    explained_variance_ratio : float, optional
        Explained variance ratio for the given basis size. User can either specify this
        or the desired subspace size.
    coords : NDArray, optional
        Fourier domain coordinate array of shape ``(*others, coils, k2, k1, k0, ndim)``.
        Required for Non Cartesian datasets. 
        The default is ``None``.
    shape : int, optional
        Matrix size of shape ``(ndim,)``.
        Required for Non Cartesian datasets. The default is ``None``.
    ndim : int, optional
        Number of spatial dimensions. Required for Cartesian datasets.
        The default is ``None``.

    Returns
    -------
    basis : SVDCompression
        SVD compression object.

    """
    coil_axis = -4

    # if axis is not 0, this is batched
    if len(data.shape[:coil_axis]) != sum(data.shape[:coil_axis]):
        batched = True
    else:
        batched = False

    # extract calibration region
    if coords is None:
        training_data = extract_acr(
            data,
            cal_width,
            ndim=ndim,
        )
        # reshape training data to (num_samples, num_coils)
        training_data = training_data[..., None].swapaxes(coil_axis - 1, -1)
    else:
        ndim = coords.shape[-1]
        training_data, _, _ = extract_acr(data, cal_width, coords=coords, shape=shape)
        training_data = training_data[..., None].swapaxes(coil_axis - 1, -1)

    if batched:
        training_data = training_data.reshape(
            training_data.shape[0], -1, training_data.shape[-1]
        )
    else:
        training_data = training_data.reshape(-1, training_data.shape[-1])

    return SVDCompression(training_data, num_coils, explained_variance_ratio)


def compress_coil(
    data: NDArray[complex],
    num_coils: int = None,
    explained_variance_ratio: float | None = None,
    cal_width: int = 24,
    coords: NDArray[float] | None = None,
    shape: int | None = None,
    ndim: int | None = None,
) -> NDArray[complex]:
    """
    Compress data along ``coil`` axis.

    Parameters
    ----------
    data : NDArray[complex]
        Input dataset of shape ``(*other, coil, k2, k1, k0)``.
    num_coils : int, optional
        Size of subspace basis (i.g., subspace size).
        User can either specify this or the target explained variance ratio.
    explained_variance_ratio : float, optional
        Explained variance ratio for the given basis size. User can either specify this
        or the desired subspace size.
    coords : NDArray, optional
        Fourier domain coordinate array of shape ``(*others, coils, k2, k1, k0, ndim)``.
        Required for Non Cartesian datasets. 
        The default is ``None``.
    shape : int, optional
        Matrix size of shape ``(ndim,)``.
        Required for Non Cartesian datasets. The default is ``None``.
    ndim : int, optional
        Number of spatial dimensions. Required for Cartesian datasets.
        The default is ``None``.

    Returns
    -------
    NDArray[complex]
        Compressed dataset of shape ``(*other, coil_reduced, k2, k1, k0)``.

    """
    coil_axis = -4
    basis = estimate_coil_subspace(
        data, num_coils, explained_variance_ratio, cal_width, ndim, coords, shape
    )

    return basis(data, axis=coil_axis)


class SVDCompression:
    """
    Subspace basis estimator via SVD.

    Parameters
    ----------
    training_data : NDArray
        Training data for subspace estimation of shape ``(num_samples, space_size)``.
    num_coeff : int, optional
        Size of subspace basis (i.g., subspace size). User can either specify this
        or the target explained variance ratio.
    explained_variance_ratio : float, optional
        Explained variance ratio for the given basis size. User can either specify this
        or the desired subspace size.

    Attributes
    ----------
    basis : NDArray
        Subspace basis of shape ``(space_size, subspace_size)``
    explained_variance_ratio : float
        Explained variance ratio for the given basis size.

    """

    @with_torch
    def __init__(
        self,
        training_data: NDArray[complex],
        num_coeff: int | None = None,
        explained_variance_ratio: float | None = None,
    ):
        if num_coeff is None and explained_variance_ratio is None:
            raise ValueError("Please specify 'num_coeff' or 'explained_variance_ratio'.")
        if num_coeff is not None and explained_variance_ratio is not None:
            raise ValueError(
                "Please either specify 'num_coeff' or 'explained_variance_ratio', not both."
            )

        # Perform SVD compression
        U, S, Vh = torch.linalg.svd(training_data, full_matrices=False)

        # Get variance
        num_samples = training_data.shape[-2]
        explained_variance = S**2 / (num_samples - 1)
        total_variance = explained_variance.sum(axis=-1)
        if explained_variance.ndim > 1:
            while total_variance.ndim < explained_variance.ndim:
                total_variance = total_variance[..., None]
        self._explained_variance_ratio = explained_variance / total_variance

        # Get output coefficients from variance ratio.
        if explained_variance_ratio is not None:
            cum_variance = torch.cumsum(self._explained_variance_ratio, -1)
            self._num_coeff = (cum_variance <= explained_variance_ratio).sum(axis=-1).max().item()
            self._num_coeff = max(1, self._num_coeff)
        else:
            self._num_coeff = num_coeff
            if self._num_coeff > training_data.shape[-1]:
                raise ValueError(
                    f"Requested number of coefficients {num_coeff} larger than space {training_data.shape[-1]}"
                )

        self._basis = Vh.swapaxes(-1, -2).conj()[..., : self._num_coeff]

    @with_torch
    def __call__(self, input, axis=0):  # noqa
        space_size = input.shape[axis]

        # apply compression
        if len(self._basis.shape) == 2:
            _output = input.swapaxes(axis, -1)
            _tmp_shape = _output.shape
            _output = _output.reshape(-1, space_size)
            _output = _output @ self._basis
        else:
            _output = input.swapaxes(axis, -1)
            _tmp_shape = _output.shape
            _output = _output.reshape(_tmp_shape[0], -1, space_size)
            _output = torch.einsum("nki,nij->nkj", _output, self._basis)

        # reshape
        _output = _output.reshape(*_tmp_shape[:-1], self._num_coeff).swapaxes(axis, -1)
        return _output.contiguous()

    @property
    def basis(self):  # noqa
        return self._basis

    @property
    def num_coeff(self):  # noqa
        return self._num_coeff

    @property
    def explained_variance_ratio(self):  # noqa
        return self._explained_variance_ratio

