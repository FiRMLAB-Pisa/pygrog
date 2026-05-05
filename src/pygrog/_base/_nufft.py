"""Non Uniform Fast Fourier Transform."""

__all__ = ["nufft", "nufft_adjoint"]

import contextlib
import math

import numpy as np
from numpy.typing import NDArray

import torch

import mrinufft
from mrinufft._array_compat import with_torch
from mrinufft._utils import proper_trajectory
from mrinufft.operators.base import FourierOperatorBase

from .._utils import estimate_shape

# ---------------------------------------------------------------------------
# pytorch-finufft backend for mri-nufft (auto-registers via __init_subclass__)
# ---------------------------------------------------------------------------
_PYTORCH_FINUFFT_AVAILABLE = True
try:
    from pytorch_finufft.functional import finufft_type1, finufft_type2
except ImportError:
    _PYTORCH_FINUFFT_AVAILABLE = False


class _MRIPytorchFinufft(FourierOperatorBase):
    """MRI NUFFT operator backed by pytorch-finufft.

    Device-agnostic: runs on CPU or CUDA depending on where the input lives.
    Inputs/outputs are transparently converted to/from any array library by
    the ``@with_torch`` decorator.

    Parameters
    ----------
    samples : array-like
        Sample locations of shape ``(n_samples, ndim)`` in ``[-pi, pi]``.
    shape : tuple[int, ...]
        Image-space shape.
    density : bool or array, optional
        Density compensation weights. Default is ``False``.
    n_coils : int, optional
        Number of coils. Default is ``1``.
    n_batchs : int, optional
        Number of batches. Default is ``1``.
    smaps : array, optional
        Sensitivity maps of shape ``(n_coils, *shape)``. Default is ``None``.
    squeeze_dims : bool, optional
        Squeeze singleton batch/coil dimensions on output. Default is ``True``.
    upsampfac : float, optional
        NUFFT oversampling factor. Default is ``2.0``.
    eps : float, optional
        Desired numerical precision. Default is ``1e-6``.
    """

    backend = "pytorch-finufft"
    available = _PYTORCH_FINUFFT_AVAILABLE
    autograd_available = True

    def __init__(
        self,
        samples,
        shape,
        density=False,
        n_coils=1,
        n_batchs=1,
        smaps=None,
        squeeze_dims=True,
        upsampfac=2.0,
        eps=1e-6,
        **_kwargs,
    ):
        super().__init__()
        self.shape = shape

        # Convert samples to contiguous numpy float32 in [-pi, pi]
        if isinstance(samples, torch.Tensor):
            samples = samples.detach().cpu().numpy()
        samples = proper_trajectory(
            np.asarray(samples).astype(np.float32, copy=False), normalize="pi"
        )
        self._samples = np.ascontiguousarray(
            samples
        )  # (n_samples, ndim) — required by base class n_samples property
        # Store transposed (ndim, n_samples) already contiguous so _get_points
        # doesn't create a non-contiguous view via .T every call (which causes
        # finufft "Argument x/y/z not C-contiguous" warnings at runtime).
        self._samples_T = np.ascontiguousarray(samples.T)  # (ndim, n_samples)
        self.dtype = np.float32

        self.n_coils = n_coils
        self.n_batchs = n_batchs
        self.squeeze_dims = squeeze_dims
        self.upsampfac = float(upsampfac)
        self.eps = float(eps)

        self.compute_density(density)
        self.compute_smaps(smaps)

    def _get_points(self, device):
        """Return trajectory as (ndim, n_samples) float32 tensor on *device*."""
        return torch.as_tensor(self._samples_T, device=device)

    def _safe_squeeze(self, arr):
        if self.squeeze_dims:
            with contextlib.suppress(ValueError, IndexError):
                arr = arr.squeeze(axis=1)
            with contextlib.suppress(ValueError, IndexError):
                arr = arr.squeeze(axis=0)
        return arr

    @with_torch
    def op(self, data, _out=None):
        """Forward NUFFT: image → non-uniform k-space (type 2)."""
        points = self._get_points(data.device)
        B, C = self.n_batchs, self.n_coils
        n_bc = B * (1 if self.uses_sense else C)

        data = data.reshape(n_bc, *self.shape).to(torch.complex64)
        ksp = torch.stack(
            [
                finufft_type2(
                    points,
                    data[i],
                    modeord=0,
                    isign=-1,
                    upsampfac=self.upsampfac,
                    eps=self.eps,
                )
                for i in range(n_bc)
            ]
        )
        ksp = ksp.reshape(B, C, self.n_samples) / float(self.norm_factor)
        return self._safe_squeeze(ksp)

    @with_torch
    def adj_op(self, coeffs, _out=None):
        """Adjoint NUFFT: non-uniform k-space → image (type 1)."""
        points = self._get_points(coeffs.device)
        B, C, K = self.n_batchs, self.n_coils, self.n_samples

        coeffs = coeffs.reshape(B * C, K).to(torch.complex64)
        img = torch.stack(
            [
                finufft_type1(
                    points,
                    coeffs[i],
                    output_shape=self.shape,
                    modeord=0,
                    isign=1,
                    upsampfac=self.upsampfac,
                    eps=self.eps,
                )
                for i in range(B * C)
            ]
        )
        img = img.reshape(B, C, *self.shape) / float(self.norm_factor)
        return self._safe_squeeze(img)


def nufft(
    input: NDArray[complex],  # noqa: A002
    coords: NDArray[float],
    oversamp: float = 1.25,
    eps: float = 1e-3,
    normalize_coords: bool = True,
) -> NDArray[complex]:
    """
    Non-uniform Fast Fourier Transform.

    Parameters
    ----------
    input : NDArray[complex]
        Input signal domain array of shape
        ``(..., n_{ndim - 1}, ..., n_1, n_0)``,
        where ``ndim`` is specified by ``coord.shape[-1]``. The nufft
        is applied on the last ``ndim axes``, and looped over
        the remaining axes.
    coords : NDArray[float]
        Fourier domain coordinate array of shape ``(..., ndim)``.
        ``ndim`` determines the number of dimensions to apply the NUFFT.
    oversamp : float, optional
        Oversampling factor. The default is ``1.25``.
    eps : float, optional
        Desired numerical precision. The default is ``1e-6``.
    normalize_coords : bool, optional
        Normalize coordinates between -pi and pi. If ``False``,
        assume they are correctly normalized already. The default
        is ``True``.

    Returns
    -------
    NDArray[complex]
        Fourier domain data of shape
        ``input.shape[:-ndim] + coords.shape[:-1]``.

    """
    ndim = coords.shape[-1]
    ishape = input.shape[-ndim:]
    plan = __nufft_init__(coords, ishape, oversamp, eps, normalize_coords)
    output = _apply(plan, input)
    return output.reshape(*output.shape[:-1], *coords.shape[:-1])


def _estimate_menon_dcf(
    coords: torch.Tensor,
    shape: tuple[int, ...],
    oversamp: float,
    eps: float,
) -> torch.Tensor:
    """Estimate non-Cartesian DCF using mrinufft PIPE (Menon) iterations."""
    coords_np = np.asarray(coords.detach().cpu(), dtype=np.float32)
    samples = coords_np.reshape(-1, coords_np.shape[-1])

    # PIPE (Menon) is implemented by finufft/cufinufft in mrinufft. The
    # private pytorch-finufft backend is used for forward/adjoint operators in
    # this project, but may not expose a `pipe` implementation.
    backends = (
        ("cufinufft", "finufft")
        if coords.device.type == "cuda"
        else (
            "finufft",
            "cufinufft",
        )
    )
    backend = None
    for candidate in backends:
        iface = FourierOperatorBase.interfaces.get(candidate)
        if iface is None or not iface[0]:
            continue
        try:
            op_cls = mrinufft.get_operator(candidate)
        except ValueError:
            continue
        if hasattr(op_cls, "pipe"):
            backend = candidate
            break

    if backend is None:
        raise RuntimeError(
            "Failed to estimate Menon DCF: no PIPE-capable mrinufft backend "
            "available (expected finufft or cufinufft)."
        )

    from mrinufft.density.nufft_based import pipe as pipe_density

    dcf = pipe_density(
        samples,
        tuple(int(s) for s in shape),
        backend=backend,
        osf=float(oversamp),
        eps=float(eps),
    )
    dcf = np.asarray(dcf)
    if np.iscomplexobj(dcf):
        # DCF is expected to be real-valued. Some backends return a complex
        # array with negligible imaginary residuals from numerical noise.
        dcf = np.real_if_close(dcf, tol=1000)
        if np.iscomplexobj(dcf):
            dcf = np.real(dcf)

    return torch.as_tensor(dcf, dtype=torch.float32, device=coords.device).reshape(
        coords.shape[:-1]
    )


def nufft_adjoint(
    input: NDArray[complex],  # noqa: A002
    coords: NDArray[float],
    oshape: list[int] | tuple[int] | None = None,
    oversamp: float = 1.25,
    eps: float = 1e-3,
    normalize_coords: bool = True,
) -> NDArray[complex]:
    """
    Adjoint non-uniform Fast Fourier Transform.

    Parameters
    ----------
    input : ArrayLike
        Input Fourier domain array of shape
        ``(..., n_{ndim - 1}, ..., n_1, n_0)``,
        where ``ndim`` is specified by ``coord.shape[-1]``. The nufft
        is applied on the last ``ndim axes``, and looped over
        the remaining axes.
    coord : NDArray[float]
        Fourier domain coordinate array of shape ``(..., ndim)``.
        ``ndim`` determines the number of dimensions to apply the NUFFT.
    oshape : list[int] | tuple[int] | None, optional
        Output shape of the form ``(..., n_{ndim - 1}, ..., n_1, n_0)``.
        The default is ``None`` (estimated from ``coord``).
    oversamp : float, optional
        Oversampling factor. The default is ``1.25``.
    eps : float, optional
        Desired numerical precision. The default is ``1e-6``.
    normalize_coords : bool, optional
        Normalize coordinates between -pi and pi. If ``False``,
        assume they are correctly normalized already. The default
        is ``True``.

    Returns
    -------
    NDArray[complex]
        Signal domain data of shape
        ``input.shape[:-ndim] + coords.shape[:-1]``.

    """
    fourier_ndim = len(coords) - 1
    input = input.reshape(*input.shape[:-fourier_ndim], -1)  # noqa: A001
    plan = __nufft_init__(coords, oshape, oversamp, eps, normalize_coords)
    return _apply_adj(plan, input)


# %% local subroutines
def __nufft_init__(
    coords: NDArray[float],
    shape: list[int] | tuple[int] | None = None,
    oversamp: float = 1.25,
    eps: float = 1e-6,
    normalize_coords: bool = True,
):
    # Convert coords to numpy float32 for operator initialization
    with contextlib.suppress(AttributeError):
        coords = coords.numpy(force=True)  # torch tensor
    with contextlib.suppress(AttributeError):
        coords = coords.get()  # cupy array
    coords = np.asarray(coords, dtype=np.float32)

    if shape is None:
        shape = estimate_shape(coords)

    # normalize to [-pi, pi]: scale each spatial dimension by its global max
    # coords is numpy (..., ndim) at this point — normalize per last-axis dimension
    if normalize_coords:
        cmax = np.abs(coords).reshape(-1, coords.shape[-1]).max(axis=0)  # (ndim,)
        cmax = np.where(cmax == 0, 1.0, cmax)  # avoid divide by zero
        coords = (math.pi * coords / cmax).astype(np.float32)

    return mrinufft.get_operator("pytorch-finufft")(
        samples=coords.reshape(-1, coords.shape[-1]),
        shape=shape,
        squeeze_dims=True,
        upsampfac=oversamp,
        eps=eps,
    )


@with_torch
def _apply(plan, input):  # noqa: A002
    # reshape from (..., *grid_shape) to (B, *grid_shape)
    ndim = plan.ndim
    broadcast_shape = input.shape[:-ndim]
    input = input.reshape(-1, *input.shape[-ndim:])  # noqa: A001

    # actual computation
    if input.ndim == ndim:
        output = plan.op(input)
    else:
        output = torch.stack([plan.op(batch) for batch in input])

    # reshape from (B, samples) to (..., samples)
    if output.ndim != 1:
        output = output.reshape(*broadcast_shape, *output.shape[1:])

    return output


@with_torch
def _apply_adj(plan, input):  # noqa: A002
    # reshape from (..., samples) to (B, samples)
    nsamples = plan.n_samples
    broadcast_shape = input.shape[:-1]
    input = input.reshape(-1, nsamples)  # noqa: A001

    # actual computation
    if input.ndim == 1:
        output = plan.adj_op(input)
    else:
        output = torch.stack([plan.adj_op(batch) for batch in input])

    # reshape from (B, *grid_shape) to (..., *grid_shape)
    if input.ndim != 1:
        output = output.reshape(*broadcast_shape, *output.shape[1:])

    return output
