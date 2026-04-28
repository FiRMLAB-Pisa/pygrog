"""GROG interpolator class — data-driven approach.

Each non-Cartesian source point is replicated to all target grid points inside
a local kernel neighbourhood (square/cube, circle/sphere, or cross), and the
appropriate precomputed GRAPPA operator is applied to each replica.

This is a *pure-torch* implementation (scipy is not needed at runtime).
"""

__all__ = ["GrogInterpolator", "GrogPlan"]

import gc
import pathlib

from types import SimpleNamespace
from numpy.typing import NDArray

import numpy as np
import torch

from ._grappa import KernelTable


# ---------------------------------------------------------------------------
# Plan container
# ---------------------------------------------------------------------------
class GrogPlan(SimpleNamespace):
    """Dynamic namespace for GROG/FFT plan fields.

    Behaves exactly like :class:`types.SimpleNamespace` (attributes can be set
    freely) but exposes :attr:`pre_weights` as a computed ``@property`` so it
    is always consistent with ``sqrt_weights`` and ``inv_perm``.
    """

    @property
    def pre_weights(self) -> "torch.Tensor":
        """``sqrt_weights`` reordered to match :meth:`~pygrog.calib.GrogInterpolator.interpolate` output.

        Returns a flat 1-D tensor of shape ``(n_samples,)``.

        Multiply sparse k-space by this factor **once** before an iterative
        reconstruction loop so that
        :class:`~pygrog.operator.SparseFFT` ``.forward`` / ``.adjoint`` satisfy
        the adjointness condition (i.e. cumulative weighting equals 1/count).
        """
        return self.sqrt_weights[self.inv_perm]

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
            extra_cflags=[
                "-O3",
                "-std=c++17",
                "-fopenmp",
                "-march=native",
                "-DPYGROG_MARCH_NATIVE",
            ],
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
    neighbours and a pre-computed GRAPPA kernel is applied.

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
        Side-length of the interpolation kernel in **oversampled**-grid units.
        Default ``2``.  Candidates whose GROG shift exceeds ±0.5 original-grid
        units in any dimension are automatically discarded, so this is a soft
        upper bound: effective neighbourhood shrinks for low *oversamp* values.
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
            ``pre_weights`` (computed property),
            ``grid_size``, ``n_samples``, plus all original GROG plan
            fields.
        """
        if image_shape is not None and tuple(image_shape) != tuple(
            self.plan.image_shape
        ):
            # Build a fresh plan with a different crop size
            import copy

            plan = copy.copy(self.plan)
            _attach_fft_plan(plan, image_shape)
            return plan
        return self.plan

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
            train_data,
            self.plan.radius,
            precision,
            lamda,
        )

        distances = self.plan.distances  # (..., npts, kw, ndim)
        pfac = 10.0**precision
        stepsize = 10 ** (-precision)

        # Quantise distances → kernel table index (all torch)
        interp_idx = (self.plan.radius + torch.round(distances * pfac) / pfac) / stepsize
        interp_idx = torch.round(interp_idx)
        strides = torch.tensor([nsteps**k for k in range(ndim)], dtype=torch.float32)
        interp_idx = torch.round(interp_idx * strides).to(torch.int32).sum(dim=-1)
        interp_idx = interp_idx.clamp(0, interp_kernel.shape[0] - 1)

        self.plan.distances = None  # free memory
        self._interp_kernel = torch.as_tensor(interp_kernel)  # (K, C, C)
        self._interp_idx = interp_idx.to(torch.int64)  # (..., npts, kw)
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
        """Apply GROG interpolation to sparse Cartesian neighbour samples.

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
        Sparse Cartesian samples ``(n_coils, n_samples)`` (or image), or
        *None* when accumulating shots.
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

        # Shape-preserving interpolation: route everything through
        # ``_sparse_full``.  It detects how many leading batch dims sit in
        # front of (coils, *spatial) using ``plan.target_idx.shape``.
        sparse = self._sparse_full(data_t)

        # Flatten from (*batch, C, *natural_shape) → (*batch, C, n_samples).
        # Public API contract: no-batch input → (n_coils, n_samples).
        nat_ndim = len(self.plan.natural_shape)
        n_coils = int(sparse.shape[-(nat_ndim + 1)])
        batch_prefix = sparse.shape[:-(nat_ndim + 1)]
        sparse_flat = sparse.reshape(*batch_prefix, n_coils, self.plan.n_samples)

        if ret_image:
            from ..operator._sparse_fft import SparseFFT

            op = SparseFFT(plan=self.plan)
            # Pre-multiply by plan.pre_weights (n_samples,) so that
            # SparseFFT.forward applies the second sqrt_w factor → full
            # density compensation w = 1/count.
            pre_w = self.plan.pre_weights.to(sparse_flat.dtype)  # (n_samples,)
            img_coils = op.forward(sparse_flat * pre_w.unsqueeze(0))
            image = img_coils.abs().square().sum(0).sqrt()  # RSS: (*image_shape,)
            gc.collect()
            out = image.numpy() if is_numpy else image
            return out

        gc.collect()

        out = sparse_flat.numpy() if is_numpy else sparse_flat
        return out

    def __call__(self, data, shot_index=None, ret_image=False):
        return self.interpolate(data, shot_index, ret_image)

    # ------------------------------------------------------------------
    # Core interpolation
    # ------------------------------------------------------------------
    def _sparse_full(self, data: torch.Tensor) -> torch.Tensor:
        """Shape-preserving GROG interpolation.

        Replicates each non-Cartesian source sample ``kw`` times (one per
        Cartesian neighbour in the kernel) and applies the pre-computed
        GRAPPA kernel in-place.  Invalid neighbours (outside the GROG
        validity window) keep whatever value the kernel produced; their
        weight is zero in :attr:`~pygrog.calib.GrogPlan.sqrt_weights`, so
        downstream scatter / gather calls treat them as no-ops.

        Parameters
        ----------
        data : torch.Tensor
            Shape ``(*batch, n_coils, *spatial)`` where ``*spatial`` matches
            ``plan.target_idx.shape[:-1]``.  Leading ``*batch`` dims are
            optional.

        Returns
        -------
        torch.Tensor
            Shape ``(*batch, n_coils, *spatial, kw)``.
        """
        plan = self.plan
        target_idx = torch.as_tensor(plan.target_idx)  # (*spatial, kw)
        interp_idx = self._interp_idx                  # (*spatial, kw)
        kernel = self._interp_kernel                   # (K, C, C)

        spatial = tuple(int(s) for s in target_idx.shape[:-1])
        kw = int(target_idx.shape[-1])
        ncoils_pos = data.ndim - len(spatial) - 1
        if ncoils_pos < 0:
            raise ValueError(
                f"data.ndim={data.ndim} too small for spatial={spatial} (+coils)"
            )
        batch_shape = tuple(int(s) for s in data.shape[:ncoils_pos])
        ncoils = int(data.shape[ncoils_pos])
        if tuple(int(s) for s in data.shape[ncoils_pos + 1:]) != spatial:
            raise ValueError(
                f"data spatial dims {data.shape[ncoils_pos + 1:]} != plan {spatial}"
            )

        B = int(np.prod(batch_shape)) if batch_shape else 1
        N = int(np.prod(spatial))

        # (B, C, N) → (B*C, N, kw) → (N*kw, B*C) for the in-place C++ kernel,
        # which expects (N, B, C) layout.
        data_bcn = data.reshape(B * ncoils, N)
        data_rep = (
            data_bcn.unsqueeze(-1).expand(-1, -1, kw).reshape(B * ncoils, N * kw)
        )
        # (N*kw, B*C) — interpret B*C as the (batch, coil) flattened axis
        data_interp = data_rep.T.reshape(N * kw, B, ncoils).contiguous()

        idx_1d = interp_idx.reshape(-1)  # (N*kw,)
        _interpolate(data_interp, idx_1d, kernel)

        # (N*kw, B, C) → (B, C, N, kw) → (*batch, C, *spatial, kw)
        out = data_interp.permute(1, 2, 0).contiguous().reshape(
            *batch_shape, ncoils, *spatial, kw
        )
        return out

    def _grid_shot(self, data: torch.Tensor, shot_index: tuple) -> torch.Tensor:
        """Grid a single shot and return its contribution."""
        # data: (C, npts)
        plan = self.plan
        target_idx = torch.as_tensor(plan.target_idx[shot_index])  # (npts, kw)
        weights = torch.as_tensor(plan.weights[shot_index])  # (npts, kw)
        interp_idx = self._interp_idx[shot_index]  # (npts, kw)
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

    Adds: ``image_shape``, ``grid_size``, ``n_samples``, ``natural_shape``,
    ``indices``, ``sqrt_weights``, ``sort_perm``, ``inv_perm``.

    Shape-preserving: ALL entries (including those whose ``target_idx == -1``
    fell outside the GROG validity window) are kept.  Invalid indices are
    clamped to ``0`` and their weight is set to ``0`` so that scatter writes
    a harmless zero into grid cell 0.  ``n_samples`` therefore equals
    ``prod(natural_shape) = prod(*spatial) * kw``.

    ``pre_weights`` is a :class:`GrogPlan` ``@property`` that returns
    ``sqrt_weights[inv_perm]`` (i.e. weights in natural / interpolate-output
    order).
    """
    grid_shape = tuple(plan.grid_shape)
    if image_shape is None:
        image_shape = tuple(plan.shape)
    else:
        if np.isscalar(image_shape):
            image_shape = len(grid_shape) * (image_shape,)
        image_shape = tuple(image_shape)

    grid_size = int(np.prod(grid_shape))

    target_idx = torch.as_tensor(plan.target_idx)
    weights = torch.as_tensor(plan.weights)
    natural_shape = tuple(int(s) for s in target_idx.shape)  # (*spatial, kw)

    idx = target_idx.ravel().to(torch.int64).clone()
    w = weights.ravel().to(torch.float32).clone()
    sqrt_w = torch.sqrt(w.clamp(min=0.0))

    # Out-of-window entries: clamp index to 0, force weight to 0.  After
    # this, the scatter/gather kernels can treat all positions uniformly.
    invalid = idx < 0
    if invalid.any():
        idx[invalid] = 0
        sqrt_w[invalid] = 0.0

    n_total = int(idx.numel())

    # Sort by grid index for cache-friendly scatter; invalid entries end up
    # contiguously at the front (their clamped index is 0) with zero weight.
    sort_perm = torch.argsort(idx)
    sorted_idx = idx[sort_perm]
    sorted_w = sqrt_w[sort_perm]

    inv_perm = torch.empty(n_total, dtype=torch.int64)
    inv_perm[sort_perm] = torch.arange(n_total)

    plan.image_shape = image_shape
    plan.grid_size = grid_size
    plan.n_samples = n_total
    plan.natural_shape = natural_shape  # (*spatial, kw)
    plan.indices = sorted_idx
    plan.sqrt_weights = sorted_w
    plan.sort_perm = sort_perm
    plan.inv_perm = inv_perm
    # pre_weights is a @property on GrogPlan — no eager assignment needed.


# =====================================================================
# Plan creation  (pure torch / numpy — no scipy)
# =====================================================================
def _create_plan(shape, coords, oversamp, kernel_width, kernel_shape, time_map):
    """Build data-driven GROG plan.

    For each source point, enumerate all grid neighbours within the kernel
    and compute:

    - ``target_idx``: flat index into the oversampled grid  (…, npts, kw_eff)
    - ``distances``:  (target - source) in grid units        (…, npts, kw_eff, ndim)
    - ``weights``:    density-compensation (1/count)          (…, npts, kw_eff)
    - ``time_map``:   replicated time stamp                   (…, npts, kw_eff) or None

    All arrays preserve the leading shape of *coords* (except the last dim).
    """
    coords = np.asarray(coords, dtype=np.float32)
    ndim = coords.shape[-1]

    shape = _default_shape(ndim, shape)
    oversamp = _default_oversamp(ndim, oversamp)
    grid_shape = tuple(
        int(np.ceil(s * o)) for s, o in zip(shape, oversamp, strict=False)
    )

    # Rescale coords so the maximum coordinate falls exactly on the last
    # valid grid point in every dimension, for any oversampling factor.
    #
    # The oversampled grid covers image-grid coordinates
    #   [origins_d, origins_d + (gs_d-1)*grid_steps_d]
    #   = [-(s_d//2),  s_d - 1 - s_d//2]
    # so the correct normalisation target is (s_d - 1 - s_d//2) per axis.
    # Using shape/2 instead (the old code) caused edge samples to round to
    # grid_shape (out of bounds) when oversamp > 1, silently discarding all
    # high-frequency edge samples and producing severely artefacted images
    # for any non-unity oversampling factor (e.g. osf=1.5, osf=2, …).
    amp = [2 * (s - 1 - s // 2) for s in shape]
    coords_scaled = _rescale_coords(coords, amp)

    # --- enumerate kernel offsets (numpy, simple integer table) -----------
    offsets = _kernel_offsets(ndim, kernel_width, kernel_shape)  # (kw_eff, ndim)

    # GROG validity constraint: G^d is physically meaningful only for
    # |d| <= 0.5 original-grid units (half the Cartesian spacing).
    # Neighbours exceeding this range are masked automatically below.
    # radius is ALWAYS 0.5 so the kernel table only pre-computes valid G^d.
    radius = 0.5

    # --- convert to torch for all tensor computations ---------------------
    coords_t = torch.from_numpy(coords_scaled)  # (..., npts, ndim) float32
    offsets_t = torch.from_numpy(offsets.astype(np.int32))  # (kw_eff, ndim)

    origins = torch.tensor(
        [-(s // 2) for s in shape], dtype=torch.float32
    )  # (ndim,)
    grid_steps = torch.tensor(
        [
            (s - 1) / (gs - 1) if gs > 1 else 1.0
            for s, gs in zip(shape, grid_shape, strict=False)
        ],
        dtype=torch.float32,
    )  # spacing between grid points in coord units, (ndim,)

    # Source position → fractional grid index
    frac_idx = (coords_t - origins) / grid_steps  # (..., npts, ndim)

    # Nearest grid index
    nearest = torch.round(frac_idx).to(torch.int32)  # (..., npts, ndim)

    # Target grid indices = nearest + offsets  → (..., npts, kw_eff, ndim)
    target_nd = nearest.unsqueeze(-2) + offsets_t  # (..., npts, kw_eff, ndim)

    # Compute distances in coordinate space: target_coord - source_coord
    target_coords_space = origins + target_nd.float() * grid_steps  # (..., npts, kw_eff, ndim)
    distances = target_coords_space - coords_t.unsqueeze(-2)  # (..., npts, kw_eff, ndim)

    # In-bounds mask (grid boundary)
    grid_shape_t = torch.tensor(list(grid_shape), dtype=torch.int32)
    in_bounds = ((target_nd >= 0) & (target_nd < grid_shape_t)).all(dim=-1)  # (..., npts, kw_eff)

    # GROG validity mask: each per-component shift must be within ±0.5
    # original-grid units so that G^d stays in the physically trained range.
    # This turns kw into a soft parameter — candidates outside the valid
    # radius are automatically discarded regardless of how large kw is.
    # Consequence: for low osf the effective neighbourhood shrinks (e.g.
    # osf=1.25 → only offset-0 survives; osf=2 → offsets ±1 survive ~50%).
    grog_valid = (distances.abs() <= 0.5).all(dim=-1)  # (..., npts, kw_eff)
    in_bounds = in_bounds & grog_valid

    # Flat index: sum_d( idx_d * stride_d )
    strides = torch.tensor(
        [int(np.prod(grid_shape[d + 1 :])) for d in range(ndim)], dtype=torch.int64
    )
    target_flat = (target_nd.to(torch.int64) * strides).sum(dim=-1)  # (..., npts, kw_eff)
    target_flat.masked_fill_(~in_bounds, -1)  # sentinel for out-of-bounds / GROG-invalid

    # --- EGROG distance-based density compensation -------------------------
    # For each (source i, target t) pair compute a Gaussian kernel value
    #   G(d) = exp(-||d_grid||^2 / 2),   d_grid in oversampled-grid units.
    # Then normalise per target so that sum_{i → t} w_it = 1:
    #   w_it = G(d_it) / sum_{j → t} G(d_jt)
    # For kernel_width=1 every source has exactly one target, so w=1 always
    # (identical to original GROG).  For kernel_width>1 this gives higher
    # weight to sources that are closer to the target, suppressing the radial
    # intensity modulation that arises from a uniform (box) average.
    grid_size = int(np.prod(grid_shape))
    valid_mask = target_flat >= 0

    # Euclidean distance in oversampled-grid units: (..., npts, kw_eff)
    dist_grid = distances / grid_steps  # grid_steps broadcasts over last dim
    dist_sq = (dist_grid**2).sum(dim=-1)  # (..., npts, kw_eff)
    gauss_vals = torch.exp(-0.5 * dist_sq)
    gauss_vals.masked_fill_(~valid_mask, 0.0)

    # Per-target Gaussian sum via scatter_add (torch equivalent of np.add.at)
    flat_targets = target_flat.reshape(-1)
    flat_gauss = gauss_vals.reshape(-1)
    flat_valid = flat_targets >= 0
    gauss_sum = torch.zeros(grid_size, dtype=torch.float64)
    gauss_sum.scatter_add_(
        0, flat_targets[flat_valid], flat_gauss[flat_valid].double()
    )

    # Normalised per-pair weights
    weights = torch.zeros_like(gauss_vals)
    weights[valid_mask] = (
        gauss_vals[valid_mask]
        / gauss_sum[target_flat[valid_mask]].clamp(min=1e-12).float()
    )

    # --- time map replication ----------------------------------------------
    if time_map is not None:
        time_map = (
            torch.as_tensor(np.asarray(time_map, dtype=np.float32))
            .unsqueeze(-1)
            .expand(target_flat.shape)
            .clone()
        )

    return GrogPlan(
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
        raise ValueError(f"Oversampling {oversamp} does not match ndim={ndim}")
    return oversamp


def _rescale_coords(coords, amp):
    cmax = np.abs(coords).reshape(-1, coords.shape[-1]).max(axis=0)
    if np.isscalar(amp):
        amp = coords.shape[-1] * [amp]
    return 0.5 * np.asarray(amp, dtype=coords.dtype) * coords / cmax
