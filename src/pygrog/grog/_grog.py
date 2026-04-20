"""GROG interpolator class — data-driven approach.

Each non-Cartesian source point is replicated to all target grid points inside
a local kernel neighbourhood (square/cube, circle/sphere, or cross), and the
appropriate precomputed GRAPPA operator is applied to each replica.

This is a *pure-torch* implementation (scipy is not needed at runtime).
"""

__all__ = ["GrogInterpolator"]

import gc
import os
import pathlib

from types import SimpleNamespace
from numpy.typing import NDArray

import h5py
import numpy as np
import torch

from ._grappa import KernelTable


# ---------------------------------------------------------------------------
# Torch C++ extension (lazy, cached)
# ---------------------------------------------------------------------------
_torch_ext = None
_torch_ext_checked = False


def _get_torch_ext():
    """Return the ``_pygrog_torch`` extension module.

    Load priority:
      1. Pre-compiled wheel extension (``pygrog._pygrog_torch``)
      2. JIT compilation via ``torch.utils.cpp_extension.load()``

    Raises
    ------
    RuntimeError
        If neither the pre-built extension nor JIT compilation succeeds.
    """
    global _torch_ext, _torch_ext_checked
    if _torch_ext_checked:
        return _torch_ext

    # 1. try pre-built wheel extension
    try:
        import pygrog._pygrog_torch as _ext
        _torch_ext = _ext
        _torch_ext_checked = True
        return _torch_ext
    except ImportError:
        pass

    # 2. JIT compilation
    jit_error = None
    try:
        import torch.utils.cpp_extension as _cpp_ext

        _HERE = pathlib.Path(__file__).parent.parent.parent.parent.parent
        _CSRC = _HERE / "csrc" / "torch"

        sources = [
            str(_CSRC / "module.cpp"),
            str(_CSRC / "grog_interp.cpp"),
            str(_CSRC / "sparse_ops.cpp"),
            str(_CSRC / "sparse_ops_avx2.cpp"),
            str(_CSRC / "sparse_ops_avx512.cpp"),
        ]
        define_macros = []
        if torch.cuda.is_available() and (_CSRC / "grog_interp_cuda.cu").exists():
            sources.append(str(_CSRC / "grog_interp_cuda.cu"))
            sources.append(str(_CSRC / "sparse_ops_cuda.cu"))
            define_macros.append(("COMPILE_WITH_CUDA", "1"))

        _torch_ext = _cpp_ext.load(
            name="_pygrog_torch_jit",
            sources=sources,
            extra_cflags=["-O3", "-std=c++17", "-fopenmp", "-march=native",
                          "-DPYGROG_MARCH_NATIVE"],
            extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"],
            extra_ldflags=["-fopenmp"],
            define_macros=define_macros,
            verbose=False,
        )
        _torch_ext_checked = True
        return _torch_ext
    except Exception as exc:
        jit_error = exc

    raise RuntimeError(
        "pygrog requires the _pygrog_torch C++ extension but it could not "
        "be loaded.  Install from a precompiled wheel (`pip install pygrog`) "
        "or build from source with a C++17 compiler "
        "(`pip install --no-build-isolation -e .`).\n"
        f"JIT compilation error: {jit_error}"
    )


# =====================================================================
# GrogInterpolator
# =====================================================================
class GrogInterpolator:
    """Data-driven GROG interpolator.

    For every non-Cartesian sample, a fixed-size neighbourhood of Cartesian
    grid points is identified.  Each source sample is replicated to all its
    neighbours and a pre-computed GRAPPA kernel is applied.  The results are
    accumulated onto the Cartesian grid using ``scatter_add``.

    Parameters
    ----------
    shape : int | list[int] | tuple[int]
        Spatial image size ``(ny, nx)`` or ``(nz, ny, nx)``.
    coords : NDArray[float]
        Non-Cartesian coordinates, shape ``(..., npts, ndim)`` scaled so that
        the range ``(-shape/2, shape/2)`` covers the full FOV.
    oversamp : float | list[float] | tuple[float] | None
        Grid oversampling factor.
    kernel_width : int
        Side-length of the interpolation kernel in grid units.  Default ``2``.
    kernel_shape : str
        ``'circle'`` (2D) / ``'sphere'`` (3D), ``'square'`` / ``'cube'``,
        or ``'cross'``.  Default ``'circle'`` / ``'sphere'``.
    time_map : NDArray[float] | None
        Sampling time per point, same leading shape as *coords*.
    """

    _interpolator_set = False
    _dataset_shape_set = False
    _data: list | None = None

    def __init__(
        self,
        shape: int | list[int] | tuple[int],
        coords: NDArray[float],
        oversamp: float | list[float] | tuple[float] | None = None,
        kernel_width: int = 2,
        kernel_shape: str = "circle",
        time_map: NDArray[float] | None = None,
        image_shape: int | list[int] | tuple[int] | None = None,
    ):
        self.plan = _create_plan(
            shape, coords, oversamp, kernel_width, kernel_shape, time_map
        )
        # Attach FFT plan fields to self.plan for SparseFFT consumption
        _attach_fft_plan(self.plan, image_shape)

    # ------------------------------------------------------------------
    # metadata / convenience
    # ------------------------------------------------------------------
    def metadata(self):
        """Return metadata associated with the Cartesian output grid."""
        oshape = np.asarray(self.plan.shape) * np.asarray(self.plan.oversamp)
        oshape = np.ceil(oshape).astype(int).tolist()
        return SimpleNamespace(
            shape=tuple(self.plan.shape),
            oshape=tuple(oshape),
            target_idx=self.plan.target_idx,
            weights=self.plan.weights,
            time_map=self.plan.time_map,
        )

    def fft_plan(self, image_shape=None):
        """Return the pre-built FFT plan (stored in ``self.plan``).

        If *image_shape* differs from the one used at construction time,
        a new plan is built on the fly (not cached).

        Parameters
        ----------
        image_shape : tuple[int, ...] | None
            Target image shape.  Defaults to the one used at construction.

        Returns
        -------
        types.SimpleNamespace
            The plan namespace — has ``grid_shape``, ``image_shape``,
            ``indices``, ``sqrt_weights``, ``sort_perm``, ``inv_perm``,
            ``grid_size``, ``n_samples``, plus all original GROG plan
            fields.
        """
        if image_shape is not None and tuple(image_shape) != tuple(self.plan.image_shape):
            # Build a fresh plan with a different crop size
            import copy
            plan = copy.copy(self.plan)
            _attach_fft_plan(plan, image_shape)
            return plan
        return self.plan

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    @classmethod
    def from_file(cls, filepath: str | pathlib.Path) -> "GrogInterpolator":
        """Load a saved GROG interpolator."""
        filepath = pathlib.Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Plan file {filepath} not found")

        obj = cls.__new__(cls)
        if filepath.suffix == ".npy":
            obj.plan = np.load(filepath, allow_pickle=True).item()
        elif filepath.suffix in (".mrd", ".h5"):
            with h5py.File(filepath, "r") as dset:
                obj.plan = _load_plan_from_mrd(dset)
        else:
            raise ValueError(f"Unsupported file extension: {filepath.suffix}")
        return obj

    def to_file(self, filepath: str | pathlib.Path) -> None:
        """Save the GROG plan to disk."""
        filepath = pathlib.Path(filepath)
        os.makedirs(filepath.parent, exist_ok=True)
        if filepath.suffix == ".npy":
            np.save(filepath, self.plan)
        elif filepath.suffix in (".mrd", ".h5"):
            with h5py.File(filepath, "r+") as dset:
                _store_plan_inside_mrd(dset, self.plan)
        else:
            raise ValueError(
                f"Unsupported extension: {filepath.suffix}. Use .npy, .mrd or .h5"
            )

    # ------------------------------------------------------------------
    # Interpolator setup
    # ------------------------------------------------------------------
    def calc_interp_table(
        self,
        train_data: NDArray[complex],
        lamda: float = 0.01,
        precision: int = 1,
    ):
        """Compute the GRAPPA kernel table and per-replica kernel index.

        Parameters
        ----------
        train_data : complex array, (coils, ...)
            Calibration k-space region.
        lamda : float
            Tikhonov regularisation.
        precision : int
            Decimal digits for distance rounding.
        """
        interp_kernel, nsteps, ndim = KernelTable(
            train_data, self.plan.radius, precision, lamda,
        )

        distances = self.plan.distances  # (..., npts, kw, ndim)
        pfac = 10.0 ** precision
        stepsize = 10 ** (-precision)

        # Quantise distances → kernel table index
        interp_idx = (self.plan.radius + np.round(distances * pfac) / pfac) / stepsize
        interp_idx = np.round(interp_idx).astype(np.float32)
        strides = np.array(
            [nsteps ** k for k in range(ndim)], dtype=np.float32
        )
        interp_idx = np.round(interp_idx * strides).astype(np.int32).sum(axis=-1)
        interp_idx = np.clip(interp_idx, 0, interp_kernel.shape[0] - 1)

        self.plan.distances = None  # free memory
        self._interp_kernel = torch.as_tensor(interp_kernel)  # (K, C, C)
        self._interp_idx = torch.as_tensor(interp_idx.astype(np.int64))  # (..., npts, kw)
        self._interpolator_set = True

    # ------------------------------------------------------------------
    # Dataset shape (for shot-by-shot)
    # ------------------------------------------------------------------
    def set_dataset_shape(self, shape: tuple[int] | list[int]):
        """Register the full dataset shape for shot-by-shot interpolation."""
        if self._dataset_shape_set:
            return
        self.dataset_shape = tuple(shape)
        self._dataset_shape_set = True

    # ------------------------------------------------------------------
    # Forward interpolation
    # ------------------------------------------------------------------
    def interpolate(
        self,
        data: NDArray[complex] | torch.Tensor,
        shot_index: int | tuple[int] | None = None,
        ret_image: bool = False,
    ) -> NDArray[complex] | torch.Tensor | None:
        """Apply GROG gridding.

        Parameters
        ----------
        data : complex array or Tensor
            ``(*batch, coils, ..., npts)`` for full-dataset, or
            ``(coils, npts)`` for shot-by-shot.
        shot_index : int | tuple[int] | None
            If given, accumulate a single shot into the internal buffer.
        ret_image : bool
            If *True*, return IFFT-reconstructed images.

        Returns
        -------
        Gridded k-space (or image), or *None* when accumulating shots.
        """
        if not self._interpolator_set:
            raise RuntimeError(
                "GRAPPA kernels not set. Call calc_interp_table() first."
            )

        is_numpy = isinstance(data, np.ndarray)
        data_t = torch.as_tensor(data)

        if shot_index is not None:
            if np.isscalar(shot_index):
                shot_index = (shot_index,)
            shot_index = tuple(shot_index)
            gridded = self._grid_shot(data_t, shot_index)
            if self._data is None:
                self._data = []
            self._data.append(gridded)
            gc.collect()
            return None

        gridded = self._grid_full(data_t)

        if ret_image:
            ...  # placeholder for IFFT reconstruction

        gc.collect()

        out = gridded.numpy() if is_numpy else gridded
        return out

    def __call__(self, data, shot_index=None, ret_image=False):
        return self.interpolate(data, shot_index, ret_image)

    # ------------------------------------------------------------------
    # Core gridding
    # ------------------------------------------------------------------
    def _grid_full(self, data: torch.Tensor) -> torch.Tensor:
        """Grid the full dataset at once.

        data : (coils, ..., npts) where ... matches coords leading dims
        plan.target_idx : (..., npts, kw) int64
        plan.weights    : (..., npts, kw) float32
        _interp_idx     : (..., npts, kw) int64
        _interp_kernel  : (K, C, C) complex
        """
        plan = self.plan
        target_idx = torch.as_tensor(plan.target_idx)       # (..., npts, kw)
        weights = torch.as_tensor(plan.weights)              # (..., npts, kw)
        interp_idx = self._interp_idx                        # (..., npts, kw)
        kernel = self._interp_kernel                         # (K, C, C)
        grid_size = int(np.prod(plan.grid_shape))

        kw = target_idx.shape[-1]
        ncoils = data.shape[0]

        # Flatten all source points: data → (C, N), plan → (N, kw)
        data_flat = data.reshape(ncoils, -1)                 # (C, N)
        N = data_flat.shape[1]
        target_flat = target_idx.reshape(-1, kw)             # (N, kw)
        weights_flat = weights.reshape(-1, kw)               # (N, kw)
        idx_flat = interp_idx.reshape(-1, kw)                # (N, kw)

        # Replicate each source → kw copies: (C, N*kw)
        data_rep = data_flat.unsqueeze(-1).expand(-1, -1, kw).reshape(ncoils, N * kw)

        # Flatten indexes
        idx_1d = idx_flat.reshape(-1)                        # (N*kw,)
        target_1d = target_flat.reshape(-1)                  # (N*kw,)
        weights_1d = weights_flat.reshape(-1)                # (N*kw,)

        # Apply GRAPPA kernels: (N*kw, 1, C)
        data_interp = data_rep.T.unsqueeze(1).contiguous()   # (N*kw, 1, C)
        _interpolate(data_interp, idx_1d, kernel)
        data_interp = data_interp[:, 0, :]                   # (N*kw, C)

        # Weight
        data_interp = data_interp * weights_1d[:, None].to(data_interp.dtype)

        # Scatter onto grid: (C, grid_size)
        output = torch.zeros(ncoils, grid_size, dtype=data.dtype, device=data.device)
        valid = target_1d >= 0
        if valid.any():
            t = target_1d[valid].unsqueeze(0).expand(ncoils, -1)
            output.scatter_add_(1, t, data_interp[valid].T)

        return output.reshape(ncoils, *plan.grid_shape)

    def _grid_shot(self, data: torch.Tensor, shot_index: tuple) -> torch.Tensor:
        """Grid a single shot and return its contribution."""
        # data: (C, npts)
        plan = self.plan
        target_idx = torch.as_tensor(plan.target_idx[shot_index])  # (npts, kw)
        weights = torch.as_tensor(plan.weights[shot_index])        # (npts, kw)
        interp_idx = self._interp_idx[shot_index]                  # (npts, kw)
        kernel = self._interp_kernel
        grid_size = int(np.prod(plan.grid_shape))
        ncoils = data.shape[0]
        npts = data.shape[-1]
        kw = target_idx.shape[-1]

        # Replicate: (C, npts) → (C, npts*kw)
        data_rep = data.unsqueeze(-1).expand(-1, -1, kw).reshape(ncoils, npts * kw)

        # Interpolate
        idx_1d = interp_idx.reshape(-1)
        data_interp = data_rep.T.unsqueeze(1).contiguous()  # (npts*kw, 1, C)
        _interpolate(data_interp, idx_1d, kernel)
        data_interp = data_interp[:, 0, :]  # (npts*kw, C)

        # Weight
        weights_1d = weights.reshape(-1)
        data_interp = data_interp * weights_1d[:, None].to(data_interp.dtype)

        # Scatter
        target_1d = target_idx.reshape(-1)
        valid = target_1d >= 0
        output = torch.zeros(ncoils, grid_size, dtype=data.dtype, device=data.device)
        if valid.any():
            t = target_1d[valid].unsqueeze(0).expand(ncoils, -1)
            output.scatter_add_(1, t, data_interp[valid].T)

        return output

    def collect_shots(self) -> torch.Tensor:
        """Sum accumulated shot contributions and return gridded data."""
        if self._data is None or len(self._data) == 0:
            raise RuntimeError("No shots accumulated.")
        output = torch.stack(self._data, dim=0).sum(dim=0)
        self._data = None
        # Reshape flat grid → spatial dims
        ncoils = output.shape[0]
        return output.reshape(ncoils, *self.plan.grid_shape)


# =====================================================================
# FFT plan attachment  (called at construction + on-demand)
# =====================================================================
def _attach_fft_plan(plan, image_shape=None):
    """Compute and attach FFT-related fields to an existing plan namespace.

    Adds: ``image_shape``, ``grid_size``, ``n_samples``, ``indices``,
    ``sqrt_weights``, ``sort_perm``, ``inv_perm``.
    """
    grid_shape = tuple(plan.grid_shape)
    if image_shape is None:
        image_shape = tuple(plan.shape)
    else:
        if np.isscalar(image_shape):
            image_shape = len(grid_shape) * (image_shape,)
        image_shape = tuple(image_shape)

    grid_size = int(np.prod(grid_shape))

    idx = torch.as_tensor(plan.target_idx).ravel().to(torch.int64)
    w = torch.as_tensor(plan.weights).ravel().to(torch.float32)
    sqrt_w = torch.sqrt(w)

    # Identify valid (non-sentinel) entries
    valid_mask = idx >= 0
    valid_idx = idx[valid_mask]
    valid_w = sqrt_w[valid_mask]
    n_valid = int(valid_mask.sum())

    # Sort valid entries by grid index for cache-friendly access
    sort_order = torch.argsort(valid_idx)
    sorted_idx = valid_idx[sort_order]
    sorted_w = valid_w[sort_order]

    # sort_perm / inv_perm are permutations of range(n_valid):
    # sort_perm[i] = which valid-order sample goes to sorted position i
    sort_perm = sort_order
    inv_perm = torch.empty(n_valid, dtype=torch.int64)
    inv_perm[sort_order] = torch.arange(n_valid)

    plan.image_shape = image_shape
    plan.grid_size = grid_size
    plan.n_samples = n_valid
    plan.valid_mask = valid_mask  # bool mask into target_idx.ravel()
    plan.indices = sorted_idx
    plan.sqrt_weights = sorted_w
    plan.sort_perm = sort_perm
    plan.inv_perm = inv_perm


# =====================================================================
# Plan creation  (pure torch / numpy — no scipy)
# =====================================================================
def _create_plan(shape, coords, oversamp, kernel_width, kernel_shape, time_map):
    """Build data-driven GROG plan.

    For each source point, enumerate all grid neighbours within the kernel
    and compute:

    - ``target_idx``: flat index into the oversampled grid  (…, npts, kw_eff)
    - ``distances``:  (target − source) in grid units        (…, npts, kw_eff, ndim)
    - ``weights``:    density-compensation (1/count)          (…, npts, kw_eff)
    - ``time_map``:   replicated time stamp                   (…, npts, kw_eff) or None

    All arrays preserve the leading shape of *coords* (except the last dim).
    """
    coords = np.asarray(coords, dtype=np.float32)
    ndim = coords.shape[-1]

    shape = _default_shape(ndim, shape)
    oversamp = _default_oversamp(ndim, oversamp)
    grid_shape = tuple(int(np.ceil(s * o)) for s, o in zip(shape, oversamp))

    # Rescale coords so that FOV spans [-shape/2, shape/2-1]
    coords_scaled = _rescale_coords(coords, shape[-ndim:])

    # --- enumerate kernel offsets ------------------------------------------
    offsets = _kernel_offsets(ndim, kernel_width, kernel_shape)  # (kw_eff, ndim)
    kw_eff = offsets.shape[0]
    radius = 0.5 * kernel_width  # used for KernelTable later

    # --- for each source, compute target grid indices ----------------------
    # Round source to nearest grid point, then add offsets
    # Grid spans [-shape[d]//2 .. shape[d]//2-1] with grid_shape[d] points
    # Convert coords_scaled to grid indices
    origins = np.array([-(s // 2) for s in shape], dtype=np.float32)  # (ndim,)
    grid_steps = np.array(
        [(s - 1) / (gs - 1) if gs > 1 else 1.0 for s, gs in zip(shape, grid_shape)],
        dtype=np.float32,
    )  # spacing between grid points in coord units

    # Source position → fractional grid index
    frac_idx = (coords_scaled - origins) / grid_steps  # (..., npts, ndim)

    # Nearest grid index
    nearest = np.round(frac_idx).astype(np.int32)  # (..., npts, ndim)

    # Target grid indices = nearest + offsets  → (..., npts, kw_eff, ndim)
    target_nd = nearest[..., np.newaxis, :] + offsets[np.newaxis, :]  # (..., npts, kw_eff, ndim)

    # Compute distances in coordinate space: target_coord - source_coord
    target_coords_space = origins + target_nd.astype(np.float32) * grid_steps
    distances = target_coords_space - coords_scaled[..., np.newaxis, :]  # (..., npts, kw_eff, ndim)

    # Clip out-of-bounds → mark as -1
    in_bounds = np.ones(target_nd.shape[:-1], dtype=bool)  # (..., npts, kw_eff)
    for d in range(ndim):
        in_bounds &= (target_nd[..., d] >= 0) & (target_nd[..., d] < grid_shape[d])

    # Compute flat index: sum_d( idx_d * stride_d )
    strides = np.array(
        [int(np.prod(grid_shape[d + 1:])) for d in range(ndim)], dtype=np.int64
    )
    target_flat = (target_nd.astype(np.int64) * strides).sum(axis=-1)  # (..., npts, kw_eff)
    target_flat[~in_bounds] = -1  # sentinel for out-of-bounds

    # --- density compensation (1/count) ------------------------------------
    # Count how many source replicas land on each grid point
    grid_size = int(np.prod(grid_shape))
    valid_targets = target_flat[target_flat >= 0]
    counts = np.bincount(valid_targets.ravel(), minlength=grid_size)

    # Weight per (source, neighbour) pair = 1 / count[target]
    weights = np.zeros_like(target_flat, dtype=np.float32)
    valid_mask = target_flat >= 0
    weights[valid_mask] = 1.0 / np.maximum(counts[target_flat[valid_mask]], 1).astype(np.float32)

    # --- time map replication ----------------------------------------------
    if time_map is not None:
        time_map = np.asarray(time_map, dtype=np.float32)
        time_map = np.broadcast_to(
            time_map[..., np.newaxis], target_flat.shape
        ).copy()

    return SimpleNamespace(
        shape=shape,
        oversamp=oversamp,
        grid_shape=grid_shape,
        kernel_width=kernel_width,
        kernel_shape=kernel_shape,
        radius=radius,
        target_idx=target_flat,
        distances=distances,
        weights=weights,
        time_map=time_map,
    )


# =====================================================================
# Kernel offset enumeration
# =====================================================================
def _kernel_offsets(ndim: int, width: int, shape: str) -> np.ndarray:
    """Return integer offsets for the interpolation neighbourhood.

    Parameters
    ----------
    ndim : int
        2 or 3.
    width : int
        Side-length of the kernel (e.g. 2 → offsets in {0, 1}).
    shape : str
        ``'circle'``/``'sphere'``, ``'square'``/``'cube'``, or ``'cross'``.

    Returns
    -------
    offsets : ndarray, shape (kw_eff, ndim), int32
    """
    half = (width - 1) / 2.0
    axes = [np.arange(width) - int(np.floor(half)) for _ in range(ndim)]
    grids = np.meshgrid(*axes, indexing="ij")
    all_offsets = np.stack([g.ravel() for g in grids], axis=-1).astype(np.int32)

    shape_lower = shape.lower()
    if shape_lower in ("square", "cube"):
        return all_offsets
    elif shape_lower in ("circle", "sphere"):
        r = np.sqrt((all_offsets.astype(np.float32) ** 2).sum(axis=-1))
        mask = r <= half + 0.5  # include points on the boundary
        return all_offsets[mask]
    elif shape_lower == "cross":
        # Keep only offsets where at most one axis is nonzero
        nonzero_axes = (all_offsets != 0).sum(axis=-1)
        mask = nonzero_axes <= 1
        return all_offsets[mask]
    else:
        raise ValueError(
            f"Unknown kernel_shape '{shape}'. Use 'circle'/'sphere', "
            f"'square'/'cube', or 'cross'."
        )


# =====================================================================
# Interpolation  (torch, with optional C++ extension)
# =====================================================================
def _interpolate(data: torch.Tensor, indexes: torch.Tensor, kernel: torch.Tensor):
    """Apply per-sample GRAPPA kernels.

    ``data[n, b, :] = kernel[indexes[n]] @ data[n, b, :]``

    Operates **in-place** on *data*  ``(N, B, C)``.
    """
    if indexes.dtype != torch.int64:
        indexes = indexes.to(torch.int64)

    ext = _get_torch_ext()
    result = ext.grog_interpolate(data, indexes, kernel)
    data.copy_(result)


# =====================================================================
# Helpers
# =====================================================================
def _default_shape(ndim, shape):
    if np.isscalar(shape):
        shape = ndim * [shape]
    return tuple(shape)


def _default_oversamp(ndim, oversamp):
    if oversamp is None:
        oversamp = (1.0, 1.0) if ndim == 2 else (1.0, 1.0, 1.2)
    if np.isscalar(oversamp):
        oversamp = ndim * [oversamp]
    oversamp = tuple(oversamp)
    if len(oversamp) != ndim:
        raise ValueError(
            f"Oversampling {oversamp} does not match ndim={ndim}"
        )
    return oversamp


def _rescale_coords(coords, amp):
    cmax = np.abs(coords).reshape(-1, coords.shape[-1]).max(axis=0)
    if np.isscalar(amp):
        amp = coords.shape[-1] * [amp]
    return 0.5 * np.asarray(amp, dtype=coords.dtype) * coords / cmax


# =====================================================================
# HDF5 serialisation (unchanged API)
# =====================================================================
def _store_plan_inside_mrd(dset, plan):
    dataset_grp = dset.require_group("dataset")
    grp = dataset_grp.require_group("grog_plan")
    for key, value in vars(plan).items():
        if isinstance(value, np.ndarray):
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
            dtype = h5py.vlen_dtype(value[0].dtype)
            if key in grp:
                del grp[key]
            ds = grp.create_dataset(key, (len(value),), dtype=dtype)
            for n in range(len(value)):
                ds[n] = value[n].tolist()
        elif value is None:
            grp.attrs[key] = "__NONE__"
        elif isinstance(value, str):
            grp.attrs[key] = value
        else:
            raise TypeError(f"Unsupported type for key '{key}': {type(value)}")


def _load_plan_from_mrd(dset):
    grp = dset["dataset/grog_plan"]
    loaded = {}
    for key in grp.keys():
        data = grp[key][()]
        if isinstance(data, np.ndarray) and data.dtype.kind == "O":
            loaded[key] = [np.array(x) for x in data]
        else:
            loaded[key] = data
    for key, val in grp.attrs.items():
        if val == "__NONE__":
            loaded[key] = None
        else:
            loaded[key] = val
    return SimpleNamespace(**loaded)
