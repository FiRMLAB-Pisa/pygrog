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

    Plan-level stacking
    -------------------
    A plan may carry a leading ``stack_shape = (*S,)`` prefix on every
    *trajectory-bound* field (``target_idx``, ``weights``, ``time_map``,
    ``indices``, ``sqrt_weights``, ``sort_perm``, ``inv_perm``).  Scalar
    fields (``grid_shape``, ``image_shape``, ``grid_size``, ``n_samples``,
    ``natural_shape``, …) remain global — they describe one stack element.

    Default ``stack_shape == ()`` means a single, unstacked trajectory and
    every field has exactly today's shape.
    """

    @property
    def stack_shape(self) -> tuple:
        """Tuple of leading stack axes; ``()`` if the plan is unstacked."""
        return tuple(getattr(self, "_stack_shape", ()))

    @stack_shape.setter
    def stack_shape(self, value):
        self._stack_shape = tuple(int(s) for s in (value or ()))

    @property
    def is_stacked(self) -> bool:
        return len(self.stack_shape) > 0

    @property
    def pre_weights(self) -> "torch.Tensor":
        """``sqrt_weights`` reordered to match :meth:`~pygrog.calib.GrogInterpolator.interpolate` output.

        Returns a tensor of shape ``(*stack_shape, n_samples)`` (or just
        ``(n_samples,)`` when unstacked).

        Multiply sparse k-space by this factor **once** before an iterative
        reconstruction loop so that
        :class:`~pygrog.operator.SparseFFT` ``.forward`` / ``.adjoint`` satisfy
        the adjointness condition (i.e. cumulative weighting equals 1/count).
        """
        sw = self.sqrt_weights
        ip = self.inv_perm
        if not self.is_stacked:
            return sw[ip]
        # Per-stack gather (vectorised).
        return torch.gather(sw, -1, ip)

    # ------------------------------------------------------------------
    @classmethod
    def _stack(cls, plans: list) -> "GrogPlan":
        """Assemble per-element plans into a single stacked plan.

        Internal helper.  Most users should pass an already-stacked
        ``coords`` array (with leading stack axes) to
        :class:`GrogInterpolator`; auto-detection turns those leading axes
        into a stacked plan with no need to call this method directly.

        Parameters
        ----------
        plans : list[GrogPlan]
            One plan per stack element; all must share ``shape``,
            ``oversamp``, ``grid_shape``, ``kernel_width``, ``kernel_shape``,
            ``radius``, ``image_shape``, ``grid_size``, ``n_samples``, and
            ``natural_shape``.

        Returns
        -------
        GrogPlan
            New plan with ``stack_shape = (len(plans),)`` and trajectory
            fields stacked along axis 0.

        Notes
        -----
        Every input plan must be unstacked (``stack_shape == ()``) — nested
        stacking is not supported.  Distances are dropped from the stacked
        plan since they are released after FFT-plan attachment.
        """
        if not plans:
            raise ValueError("Cannot stack an empty list of plans.")
        for p in plans:
            if getattr(p, "stack_shape", ()) != ():
                raise ValueError("GrogPlan.stack requires unstacked input plans.")
        ref = plans[0]
        global_fields = (
            "shape",
            "oversamp",
            "grid_shape",
            "kernel_width",
            "kernel_shape",
            "radius",
            "image_shape",
            "grid_size",
            "n_samples",
            "natural_shape",
        )
        for f in global_fields:
            ref_v = getattr(ref, f, None)
            for p in plans[1:]:
                if getattr(p, f, None) != ref_v:
                    raise ValueError(
                        f"GrogPlan.stack: field '{f}' differs across plans"
                    )

        out = cls(
            shape=ref.shape,
            oversamp=ref.oversamp,
            grid_shape=ref.grid_shape,
            kernel_width=ref.kernel_width,
            kernel_shape=ref.kernel_shape,
            radius=ref.radius,
            image_shape=ref.image_shape,
            grid_size=ref.grid_size,
            n_samples=ref.n_samples,
            natural_shape=ref.natural_shape,
        )
        out.stack_shape = (len(plans),)

        # Stacked trajectory-bound tensors.
        out.target_idx = torch.stack(
            [torch.as_tensor(p.target_idx) for p in plans],
            dim=0,
        )
        out.weights = torch.stack(
            [torch.as_tensor(p.weights) for p in plans],
            dim=0,
        )
        # distances is needed by calc_interp_table; preserve when present
        if all(getattr(p, "distances", None) is not None for p in plans):
            out.distances = torch.stack(
                [torch.as_tensor(p.distances) for p in plans],
                dim=0,
            )
        else:
            out.distances = None
        if any(p.time_map is not None for p in plans):
            if any(p.time_map is None for p in plans):
                raise ValueError(
                    "GrogPlan.stack: all plans must agree on time_map presence"
                )
            out.time_map = torch.stack(
                [torch.as_tensor(p.time_map) for p in plans],
                dim=0,
            )
        else:
            out.time_map = None
        # Stacked sorted/permutation fields produced by _attach_fft_plan.
        out.indices = torch.stack([p.indices for p in plans], dim=0)
        out.sqrt_weights = torch.stack([p.sqrt_weights for p in plans], dim=0)
        out.sort_perm = torch.stack([p.sort_perm for p in plans], dim=0)
        out.inv_perm = torch.stack([p.inv_perm for p in plans], dim=0)
        return out


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
        stack_shape: tuple[int, ...] = (),
    ):
        coords_arr = np.asarray(coords)
        stack_shape_eff = tuple(int(s) for s in (stack_shape or ()))
        if not stack_shape_eff and coords_arr.ndim == 4:
            # Conservative auto-stack detection for canonical 2D trajectories:
            # coords = (*stack, k1, k0, 2). We infer a single stack axis.
            # More complex layouts (ndim != 4) must pass stack_shape explicitly.
            stack_shape_eff = (int(coords_arr.shape[0]),)

        self.plan = _create_plan(
            shape,
            coords_arr,
            oversamp,
            kernel_width,
            kernel_shape,
            time_map,
            stack_shape=stack_shape_eff,
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
        interp_idx = (
            self.plan.radius + torch.round(distances * pfac) / pfac
        ) / stepsize
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
        grid: bool = False,
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
        grid : bool
            If *True*, scatter the interpolated sparse samples onto a dense
            oversampled Cartesian grid and return the tuple
            ``(gridded_kspace, mask, density)`` where:

            - ``gridded_kspace`` has shape ``(*batch, *stack, C, *grid_shape)``;
            - ``mask`` has shape ``(*stack, *grid_shape)``, real float (1 where
              at least one sample lands, 0 elsewhere);
            - ``density`` has shape ``(*stack, *grid_shape)``, real float (sum of
              squared GROG weights per grid cell — the Toeplitz PSF source).

            Mutually exclusive with *ret_image*.

        Returns
        -------
        Sparse Cartesian samples ``(n_coils, n_samples)`` (or image when
        *ret_image=True*), or ``(gridded_kspace, mask, density)`` tuple when
        *grid=True*, or *None* when accumulating shots.
        """
        if not self._interpolator_set:
            raise RuntimeError(
                "GRAPPA kernels not set. Call calc_interp_table() first."
            )

        if grid and ret_image:
            raise ValueError("'grid' and 'ret_image' are mutually exclusive.")

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
        # front of (*stack, coils, *spatial) using ``plan.stack_shape`` and
        # ``plan.target_idx.shape``.
        sparse = self._sparse_full(data_t)

        # Flatten from (*batch, *stack, C, *natural_shape) → (*batch, *stack,
        # C, n_samples).  Public API contract: no-batch / no-stack input →
        # (n_coils, n_samples).
        plan = self.plan
        s_shape = tuple(plan.stack_shape)
        nat_ndim = len(plan.natural_shape)
        n_coils = int(sparse.shape[-(nat_ndim + 1)])
        # batch_prefix = leading dims BEFORE *S; stack prefix is fixed = s_shape
        batch_prefix = sparse.shape[: sparse.ndim - (nat_ndim + 1) - len(s_shape)]
        sparse_flat = sparse.reshape(
            *batch_prefix,
            *s_shape,
            n_coils,
            plan.n_samples,
        )

        if ret_image:
            from ..operator._sparse_fft import SparseFFT

            op = SparseFFT(plan=self.plan)
            # Pre-multiply by plan.pre_weights so that SparseFFT.forward
            # applies the second sqrt_w factor → full density compensation
            # w = 1/count.  pre_weights shape: (*S, n_samples) or (n_samples,).
            pre_w = plan.pre_weights.to(sparse_flat.dtype)
            # broadcast over *batch, coil axes — pre_w needs a coil axis
            pw = pre_w.unsqueeze(-2) if pre_w.ndim > 1 else pre_w.unsqueeze(0)
            img_coils = op.adjoint(sparse_flat * pw)
            # RSS over coil axis (the one immediately before *image_shape).
            img_ax = img_coils.ndim - len(plan.image_shape) - 1
            image = img_coils.abs().square().sum(img_ax).sqrt()
            gc.collect()
            out = image.numpy() if is_numpy else image
            return out

        if grid:
            grid_kspace, masked_plan = self._grid_kspace(sparse_flat, plan)
            gc.collect()
            if is_numpy:
                import numpy as _np

                numpy_plan = masked_plan.__class__(
                    grid_shape=masked_plan.grid_shape,
                    image_shape=masked_plan.image_shape,
                    stack_shape=masked_plan.stack_shape,
                    mask=_np.asarray(masked_plan.mask),
                    density=_np.asarray(masked_plan.density),
                )
                return _np.asarray(grid_kspace), numpy_plan
            return grid_kspace, masked_plan

        gc.collect()

        out = sparse_flat.numpy() if is_numpy else sparse_flat
        return out

    def __call__(self, data, shot_index=None, ret_image=False, grid=False):
        return self.interpolate(data, shot_index, ret_image, grid)

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
            Shape ``(*batch, *stack, n_coils, *spatial)`` where ``*stack``
            matches ``plan.stack_shape`` (possibly empty) and ``*spatial``
            matches the per-stack-element grid (i.e. the trailing dims of
            ``plan.target_idx`` minus the last ``kw`` axis and minus the
            leading ``*stack`` prefix).  Leading ``*batch`` dims are
            optional.

        Returns
        -------
        torch.Tensor
            Shape ``(*batch, *stack, n_coils, *spatial, kw)``.
        """
        plan = self.plan
        target_idx = torch.as_tensor(plan.target_idx)
        interp_idx = self._interp_idx
        kernel = self._interp_kernel

        s_shape = tuple(int(s) for s in plan.stack_shape)
        s_ndim = len(s_shape)

        spatial = tuple(int(s) for s in target_idx.shape[s_ndim:-1])
        kw = int(target_idx.shape[-1])

        # ndim layout: data = (*batch, *stack, n_coils, *spatial)
        ncoils_pos = data.ndim - len(spatial) - 1
        if ncoils_pos < s_ndim:
            raise ValueError(
                f"data.ndim={data.ndim} too small for stack_shape={s_shape} + "
                f"spatial={spatial} (+coils)"
            )
        # Validate stack prefix.
        if s_ndim > 0:
            stack_prefix = tuple(
                int(s) for s in data.shape[ncoils_pos - s_ndim : ncoils_pos]
            )
            if stack_prefix != s_shape:
                raise ValueError(
                    f"data stack prefix {stack_prefix} != plan.stack_shape {s_shape}"
                )
        batch_shape = tuple(int(s) for s in data.shape[: ncoils_pos - s_ndim])
        ncoils = int(data.shape[ncoils_pos])
        if tuple(int(s) for s in data.shape[ncoils_pos + 1 :]) != spatial:
            raise ValueError(
                f"data spatial dims {data.shape[ncoils_pos + 1 :]} != plan {spatial}"
            )

        if s_ndim == 0:
            # Fast path — single trajectory (unchanged from prior behaviour).
            return self._sparse_full_single(
                data,
                batch_shape,
                ncoils,
                spatial,
                kw,
                target_idx,
                interp_idx,
                kernel,
            )

        # Stacked path — loop over *S (kernel call per stack element).
        # data: (*B, *S, C, *spatial)  →  view as (B, S, C, *spatial)
        B = int(np.prod(batch_shape)) if batch_shape else 1
        S = int(np.prod(s_shape))
        data_v = data.reshape(B, S, ncoils, *spatial)
        # interp_idx is (*S, *spatial, kw); flatten *S to S
        interp_idx_v = interp_idx.reshape(S, *spatial, kw)

        outs = []
        for s in range(S):
            outs.append(
                self._sparse_full_single(
                    data_v[:, s],
                    batch_shape,
                    ncoils,
                    spatial,
                    kw,
                    (
                        target_idx[s]
                        if s_ndim == 1
                        else target_idx.reshape(S, *spatial, kw)[s]
                    ),
                    interp_idx_v[s],
                    kernel,
                )
            )
        # Stack along *S, then reshape back to multi-axis stack prefix.
        # outs[i] has shape (*batch, C, *spatial, kw)
        stacked = torch.stack(
            outs, dim=len(batch_shape)
        )  # (*batch, S, C, *spatial, kw)
        return stacked.reshape(*batch_shape, *s_shape, ncoils, *spatial, kw)

    @staticmethod
    def _sparse_full_single(
        data,
        batch_shape,
        ncoils,
        spatial,
        kw,
        _target_idx,
        interp_idx,
        kernel,
    ):
        """Single-trajectory kernel call. ``data`` has shape ``(*batch, C, *spatial)``."""
        B = int(np.prod(batch_shape)) if batch_shape else 1
        N = int(np.prod(spatial))

        data_bcn = data.reshape(B * ncoils, N)
        data_rep = data_bcn.unsqueeze(-1).expand(-1, -1, kw).reshape(B * ncoils, N * kw)
        data_interp = data_rep.T.reshape(N * kw, B, ncoils).contiguous()

        idx_1d = interp_idx.reshape(-1)
        _interpolate(data_interp, idx_1d, kernel)

        out = (
            data_interp.permute(1, 2, 0)
            .contiguous()
            .reshape(*batch_shape, ncoils, *spatial, kw)
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

    # ------------------------------------------------------------------
    # Gridded k-space construction (for grid=True codepath)
    # ------------------------------------------------------------------
    def _grid_kspace(self, sparse_flat, plan):
        """Scatter density-compensated sparse samples onto the oversampled grid.

        Parameters
        ----------
        sparse_flat : torch.Tensor
            ``(*batch, *stack, C, n_samples)`` as returned by the
            main interpolation code path.
        plan : GrogPlan
            Attached FFT plan with ``indices``, ``sqrt_weights``,
            ``inv_perm``, ``grid_shape``, ``grid_size``, ``stack_shape``.

        Returns
        -------
        grid_kspace : torch.Tensor
            ``(*batch, *stack, C, *grid_shape)``
        mask : torch.Tensor
            ``(*stack, *grid_shape)`` real float32 — 1 where ≥1 sample lands.
        density : torch.Tensor
            ``(*stack, *grid_shape)`` real float32 — sum of squared weights.
        """
        from ..operator._sparse_fft import _scatter_add

        s_shape = tuple(plan.stack_shape)
        grid_shape = tuple(plan.grid_shape)
        grid_size = int(plan.grid_size)

        # Density-compensate the sparse data: multiply by pre_weights.
        # pre_weights shape: (*S, n_samples) or (n_samples,).
        pre_w = plan.pre_weights.to(sparse_flat.dtype)
        pw = pre_w.unsqueeze(-2) if pre_w.ndim > 1 else pre_w.unsqueeze(0)
        # sparse_dc: (*batch, *stack, C, n_samples)
        sparse_dc = sparse_flat * pw

        # Work out (*batch) leading dims.
        s_ndim = len(s_shape)
        # sparse_dc has trailing (n_samples,) after (*batch, *stack, C)
        n_coils = int(sparse_dc.shape[-(s_ndim + 2 if s_ndim else 2)])
        batch_prefix = tuple(
            int(s) for s in sparse_dc.shape[: sparse_dc.ndim - (s_ndim + 1 + 1)]
        )
        B_total = int(np.prod(batch_prefix)) if batch_prefix else 1
        S_total = int(np.prod(s_shape)) if s_shape else 1
        n_per = int(plan.n_samples)

        flat = sparse_dc.reshape(B_total, S_total, n_coils, n_per)

        # Use the packed-offset trick so that a single C++ call handles every
        # (batch, stack) combination at once.
        indices_full = plan.indices  # (*S, n_per) or (n_per,)
        sqrt_w_full = plan.sqrt_weights  # (*S, n_per) or (n_per,)
        inv_perm_full = plan.inv_perm  # (*S, n_per) or (n_per,)

        if S_total == 1:
            # Unstacked or single-stack-element: trivial case.
            idx = indices_full.to(sparse_dc.device).reshape(-1)
            sqw = sqrt_w_full.to(sparse_dc.device).reshape(-1)
            inv_perm_full.to(sparse_dc.device).reshape(-1)
            # The sparse_dc is already in pre_weight order, but flat is in
            # original order; reorder to match sorted indices for scatter_add.
            # sparse_dc is actually in natural (unsorted) order — we need to
            # reorder to sorted using the sort_perm before scattering.
            sort_p = plan.sort_perm.to(sparse_dc.device).reshape(-1)

            # Allocate output grids.
            grid_kspace = torch.zeros(
                B_total,
                n_coils,
                grid_size,
                dtype=sparse_dc.dtype,
                device=sparse_dc.device,
            )
            # density (real): sum of squared weights per grid cell
            density_flat = torch.zeros(
                grid_size,
                dtype=torch.float32,
                device=sparse_dc.device,
            )
            w_sq = (sqw * sqw).contiguous()
            density_flat.index_add_(0, idx.long(), w_sq)

            for b in range(B_total):
                for c in range(n_coils):
                    coil = flat[b, 0, c]  # (n_per,)
                    sorted_coil = coil[sort_p]  # reorder for scatter
                    _scatter_add(grid_kspace[b, c], sorted_coil, idx, sqw)

            density = density_flat.reshape(*grid_shape)
            mask = (density > 0).float()
            grid_kspace = grid_kspace.reshape(B_total, n_coils, *grid_shape)

        else:
            # Stacked: pack-offset all stack elements into a super-grid.
            idx_v = indices_full.reshape(S_total, n_per).to(sparse_dc.device)
            sqw_v = sqrt_w_full.reshape(S_total, n_per).to(sparse_dc.device)
            sp_v = plan.sort_perm.reshape(S_total, n_per).to(sparse_dc.device)

            # Offsets for super-grid: s-th stack element starts at s*grid_size
            grid_off = (
                torch.arange(S_total, device=sparse_dc.device, dtype=idx_v.dtype)
                * grid_size
            ).unsqueeze(
                -1
            )  # (S, 1)
            data_off = (
                torch.arange(S_total, device=sparse_dc.device, dtype=sp_v.dtype) * n_per
            ).unsqueeze(-1)

            idx_packed = (idx_v + grid_off).reshape(-1).contiguous()  # (S*n_per,)
            sqw_packed = sqw_v.reshape(-1).contiguous()
            sp_packed = (sp_v + data_off).reshape(-1).contiguous()

            # density across the super-grid
            density_super = torch.zeros(
                S_total * grid_size,
                dtype=torch.float32,
                device=sparse_dc.device,
            )
            w_sq_packed = (sqw_packed * sqw_packed).contiguous()
            density_super.index_add_(0, idx_packed.long(), w_sq_packed)

            grid_kspace = torch.zeros(
                B_total,
                S_total,
                n_coils,
                S_total * grid_size,
                dtype=sparse_dc.dtype,
                device=sparse_dc.device,
            )
            for b in range(B_total):
                # flat[b]: (S, C, n_per)
                ksp_b = flat[b].permute(1, 0, 2).reshape(n_coils, -1)  # (C, S*n_per)
                sorted_all = ksp_b[:, sp_packed]  # (C, S*n_per)
                g_b = torch.zeros(
                    n_coils,
                    S_total * grid_size,
                    dtype=sparse_dc.dtype,
                    device=sparse_dc.device,
                )
                for c in range(n_coils):
                    _scatter_add(g_b[c], sorted_all[c], idx_packed, sqw_packed)
                # Split super-grid back into (S, *grid_shape) slabs
                grid_kspace[b] = g_b.reshape(n_coils, S_total, grid_size).permute(
                    1, 0, 2
                )

            density = density_super.reshape(*s_shape, *grid_shape)
            mask = (density > 0).float()
            grid_kspace = grid_kspace.reshape(B_total, S_total, n_coils, grid_size)
            grid_kspace = grid_kspace.reshape(
                *batch_prefix if batch_prefix else (B_total,),
                *s_shape,
                n_coils,
                *grid_shape,
            )

        if not batch_prefix:
            grid_kspace = grid_kspace.reshape(*s_shape, n_coils, *grid_shape)

        from ..operator._masked_fft import MaskedFFTPlan

        masked_plan = MaskedFFTPlan(
            grid_shape=grid_shape,
            image_shape=tuple(plan.image_shape),
            stack_shape=s_shape,
            mask=mask,
            density=density,
        )
        return grid_kspace, masked_plan

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

    Plan stacking
    -------------
    If ``plan.stack_shape`` is a non-empty tuple ``(*S,)``, ``target_idx``
    is expected to carry a leading ``*S`` prefix (i.e. shape
    ``(*S, *natural_shape)``).  ``natural_shape`` and ``n_samples`` describe
    *one* stack element; ``indices``, ``sqrt_weights``, ``sort_perm``,
    ``inv_perm`` are produced with shape ``(*S, n_samples)``.
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

    stack_shape = tuple(getattr(plan, "stack_shape", ()) or ())
    s_ndim = len(stack_shape)
    natural_shape = tuple(
        int(s) for s in target_idx.shape[s_ndim:]
    )  # (*spatial, kw) per stack element
    n_per = int(np.prod(natural_shape)) if natural_shape else 0

    # Reshape to (*S, n_per) for vectorised per-stack operations.
    if s_ndim == 0:
        idx = target_idx.reshape(-1).to(torch.int64).clone()
        w = weights.reshape(-1).to(torch.float32).clone()
    else:
        idx = target_idx.reshape(*stack_shape, n_per).to(torch.int64).clone()
        w = weights.reshape(*stack_shape, n_per).to(torch.float32).clone()
    sqrt_w = torch.sqrt(w.clamp(min=0.0))

    # Out-of-window entries: clamp index to 0, force weight to 0.
    invalid = idx < 0
    if invalid.any():
        idx[invalid] = 0
        sqrt_w[invalid] = 0.0

    # Sort by grid index for cache-friendly scatter; per-stack along last axis.
    sort_perm = torch.argsort(idx, dim=-1)
    sorted_idx = torch.gather(idx, -1, sort_perm)
    sorted_w = torch.gather(sqrt_w, -1, sort_perm)

    if s_ndim == 0:
        inv_perm = torch.empty(n_per, dtype=torch.int64)
        inv_perm[sort_perm] = torch.arange(n_per)
    else:
        inv_perm = torch.empty_like(sort_perm)
        ar = torch.arange(n_per, dtype=torch.int64).expand(sort_perm.shape)
        inv_perm.scatter_(-1, sort_perm, ar)

    plan.image_shape = image_shape
    plan.grid_size = grid_size
    plan.n_samples = n_per
    plan.natural_shape = natural_shape  # (*spatial, kw) per stack element
    plan.indices = sorted_idx
    plan.sqrt_weights = sorted_w
    plan.sort_perm = sort_perm
    plan.inv_perm = inv_perm
    # pre_weights is a @property on GrogPlan — no eager assignment needed.


# =====================================================================
# Plan creation  (pure torch / numpy — no scipy)
# =====================================================================
def _create_plan(
    shape, coords, oversamp, kernel_width, kernel_shape, time_map, stack_shape=()
):
    """Build data-driven GROG plan.

    When ``stack_shape`` is non-empty, the leading ``*S`` axes of ``coords``
    are treated as independent stack elements: the underlying single-element
    builder is invoked once per element (so per-element normalisation,
    density compensation and rounding are bit-exact equivalent to building
    a list of separate unstacked plans) and the results are then assembled
    via :meth:`GrogPlan.stack`.
    """
    s_shape = tuple(int(s) for s in (stack_shape or ()))
    if s_shape:
        coords = np.asarray(coords)
        if tuple(coords.shape[: len(s_shape)]) != s_shape:
            raise ValueError(
                f"coords leading shape {coords.shape[: len(s_shape)]} does "
                f"not match stack_shape {s_shape}"
            )
        s_total = int(np.prod(s_shape))
        rest_shape = coords.shape[len(s_shape) :]
        coords_flat = coords.reshape(s_total, *rest_shape)
        if time_map is not None:
            tm = np.asarray(time_map)
            if tm.shape[: len(s_shape)] != s_shape:
                raise ValueError(
                    f"time_map leading shape {tm.shape[: len(s_shape)]} "
                    f"does not match stack_shape {s_shape}"
                )
            tm_flat = tm.reshape(s_total, *tm.shape[len(s_shape) :])
        else:
            tm_flat = [None] * s_total
        plans = [
            _create_plan_single(
                shape,
                coords_flat[i],
                oversamp,
                kernel_width,
                kernel_shape,
                tm_flat[i],
            )
            for i in range(s_total)
        ]
        # _attach_fft_plan must run before GrogPlan._stack since the latter
        # stacks indices/sort_perm/inv_perm.  Build them per element first.
        for p in plans:
            _attach_fft_plan(p, image_shape=None)
        out = GrogPlan._stack(plans)
        if len(s_shape) > 1:
            # Re-shape stacked tensors from (s_total, ...) to (*S, ...).
            for f in (
                "target_idx",
                "weights",
                "indices",
                "sqrt_weights",
                "sort_perm",
                "inv_perm",
            ):
                t = getattr(out, f)
                out_shape = s_shape + tuple(t.shape[1:])
                setattr(out, f, t.reshape(out_shape))
            if out.time_map is not None:
                t = out.time_map
                out.time_map = t.reshape(s_shape + tuple(t.shape[1:]))
            out.stack_shape = s_shape
        return out
    return _create_plan_single(
        shape,
        coords,
        oversamp,
        kernel_width,
        kernel_shape,
        time_map,
    )


def _create_plan_single(shape, coords, oversamp, kernel_width, kernel_shape, time_map):
    """Build data-driven GROG plan for a single (unstacked) trajectory.

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

    origins = torch.tensor([-(s // 2) for s in shape], dtype=torch.float32)  # (ndim,)
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
    target_coords_space = (
        origins + target_nd.float() * grid_steps
    )  # (..., npts, kw_eff, ndim)
    distances = target_coords_space - coords_t.unsqueeze(
        -2
    )  # (..., npts, kw_eff, ndim)

    # In-bounds mask (grid boundary)
    grid_shape_t = torch.tensor(list(grid_shape), dtype=torch.int32)
    in_bounds = ((target_nd >= 0) & (target_nd < grid_shape_t)).all(
        dim=-1
    )  # (..., npts, kw_eff)

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
    target_flat = (target_nd.to(torch.int64) * strides).sum(
        dim=-1
    )  # (..., npts, kw_eff)
    target_flat.masked_fill_(
        ~in_bounds, -1
    )  # sentinel for out-of-bounds / GROG-invalid

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
    gauss_sum.scatter_add_(0, flat_targets[flat_valid], flat_gauss[flat_valid].double())

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
