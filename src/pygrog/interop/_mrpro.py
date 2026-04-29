"""mrpro LinearOperator adapter for SparseFFT.

Wraps :class:`~pygrog.operator.SparseFFT` (or any pygrog operator with
``forward`` / ``adjoint`` methods) as an ``mrpro.operators.LinearOperator``
so it plugs into mrpro reconstruction pipelines and algorithms (PGD, CG,
PDHG, …) without modification.

Shape conventions (matching mrpro's k-space layout
``(..., coils, k2, k1, k0)``)
==========================================================================

The adapter speaks mrpro's native shapes at its boundary and handles the
GROG kernel-width axis (``kw``) internally so callers never need to
introduce a manual rearrangement.  For an operator whose underlying GROG
plan has ``natural_shape == (*trajectory_shape, kw)``:

* **k-space**: ``(*other, n_coils, *trajectory_shape[:-1], trajectory_shape[-1] * kw)``
  i.e. the kernel-width axis is fused into the last trajectory axis (k0).
* **image**:   ``(*other, *image_shape)`` — no coil axis when the wrapped
  operator carries coil sensitivities (smaps), since coils are already
  combined.

Gradients are computed via :mod:`pygrog.interop._torch` — explicit
``torch.autograd.Function`` subclasses that are also compatible with
``torch.func.grad`` / ``vmap`` (used internally by ``mrpro.algorithms``).
``adjoint_as_backward`` is intentionally *not* used.

mrpro ``LinearOperator`` contract (this adapter's convention):
  - ``forward(kspace) -> (image,)``  — backprojection (A^H)
  - ``adjoint(image)  -> (kspace,)`` — measurement   (A)
"""

__all__ = ["GrogInterpolator", "GrogLinearOp", "coil_compress", "nlinv_calib"]

import numpy as np
import torch

from ..calib import GrogInterpolator as _GrogInterpolatorBase
from ..utils import nlinv_calib as _nlinv_calib
from ._torch import grog_backproject, grog_measure


# ---------------------------------------------------------------------------
# KData ↔ pygrog field extraction helpers
# ---------------------------------------------------------------------------
def _kdata_extract(kdata):
    """Pull coords / k-space / shapes out of a mrpro ``KData``.

    Returns
    -------
    coords : np.ndarray, shape ``(*spatial, ndim)``
        Trajectory coordinates in pygrog scale (``[-shape/2, shape/2]``,
        where ``shape == encoding_matrix``).  ``*spatial`` matches the
        trajectory's ``(k2, k1, k0)`` axes (broadcast to dense if needed).
    data : torch.Tensor, shape ``(*other, n_coils, *spatial)``
        K-space data, complex.  ``*spatial`` matches *coords*.
    enc_shape : tuple[int, ...]
        Encoding-matrix shape in pygrog ordering (``(y, x)`` for 2D,
        ``(z, y, x)`` for 3D).
    recon_shape : tuple[int, ...]
        Recon-matrix shape (image FFT crop target).
    """
    traj = kdata.traj
    # kx/ky/kz: (*other, 1, k2, k1, k0).  Drop the coil-singleton axis.
    kx = traj.kx.squeeze(-4)
    ky = traj.ky.squeeze(-4)
    kz = traj.kz.squeeze(-4)

    # 2D vs 3D detection: if kz is identically zero and k2==1, treat as 2D.
    k2 = kdata.data.shape[-3]
    is_2d = (k2 == 1) and bool(torch.all(kz == 0).item())

    # Broadcast traj components to the data's spatial shape (k2, k1, k0).
    spatial = tuple(int(s) for s in kdata.data.shape[-3:])

    # Each component may have leading "other"/singleton dims; collapse to spatial.
    def _to_spatial(t):
        # Broadcast against an all-ones tensor of shape (*spatial,) so any
        # leading repetition or singleton broadcasts cleanly.
        if t.ndim < 3:
            t = t.reshape((1,) * (3 - t.ndim) + tuple(t.shape))
        # Take the trailing 3 dims and broadcast to spatial.
        target = torch.broadcast_to(t, t.shape[:-3] + spatial)
        # If there are leading "other" dims, take the first slice — the
        # trajectory is assumed identical across "other".
        while target.ndim > 3:
            target = target[0]
        return target

    kx_s = _to_spatial(kx)
    ky_s = _to_spatial(ky)
    kz_s = _to_spatial(kz)

    if is_2d:
        # 2D: (k1, k0, 2) in (y, x) order; drop the singleton k2 axis.
        coords = torch.stack([ky_s[0], kx_s[0]], dim=-1)
        enc_shape = (
            int(kdata.header.encoding_matrix.y),
            int(kdata.header.encoding_matrix.x),
        )
        recon_shape = (
            int(kdata.header.recon_matrix.y),
            int(kdata.header.recon_matrix.x),
        )
    else:
        coords = torch.stack([kz_s, ky_s, kx_s], dim=-1)
        enc_shape = (
            int(kdata.header.encoding_matrix.z),
            int(kdata.header.encoding_matrix.y),
            int(kdata.header.encoding_matrix.x),
        )
        recon_shape = (
            int(kdata.header.recon_matrix.z),
            int(kdata.header.recon_matrix.y),
            int(kdata.header.recon_matrix.x),
        )

    return (
        coords.detach().cpu().numpy().astype(np.float32),
        kdata.data,
        enc_shape,
        recon_shape,
    )


def _data_to_spatial(data, ndim):
    """Reshape KData ``(..., coils, k2, k1, k0)`` for pygrog interpolate.

    For 2D (``ndim==2``), drops the singleton k2 axis →
    ``(..., coils, k1, k0)``.  For 3D, leaves shape unchanged.
    """
    if ndim == 2:
        # k2 (axis -3) must be 1.
        assert (
            data.shape[-3] == 1
        ), f"2D interpolation expects k2 == 1; got {tuple(data.shape)}"
        new_shape = data.shape[:-3] + data.shape[-2:]
        return data.reshape(new_shape)
    return data


def _spatial_to_kdata(arr, ndim):
    """Inverse of :func:`_data_to_spatial`: re-insert the singleton k2 axis."""
    if ndim == 2:
        new_shape = (*arr.shape[:-2], 1, *arr.shape[-2:])
        return arr.reshape(new_shape)
    return arr


class GrogLinearOp:
    """Wrap a pygrog operator as an ``mrpro.operators.LinearOperator``.

    Because mrpro is an optional dependency, the class is built lazily on
    first instantiation so importing this module does not fail when mrpro
    is absent.

    Parameters
    ----------
    op : SparseFFT-like
        Any operator with ``forward(kspace) -> image`` and
        ``adjoint(image) -> kspace`` methods, plus a ``natural_shape``
        attribute whose last axis is the GROG kernel width.

    Raises
    ------
    ImportError
        If ``mrpro`` is not installed.

    Examples
    --------
    ::

        from pygrog.operator import SparseFFT
        from pygrog.interop import GrogLinearOp
        from mrpro.algorithms.optimizers import pgd

        base = SparseFFT(plan=grog.plan, smaps=smaps)
        mrpro_op = GrogLinearOp(base)
        # mrpro_op now consumes/produces standard
        # ``(*other, n_coils, k2, k1, k0)`` k-space and ``(*other, z, y, x)``
        # images, ready for any mrpro algorithm.
    """

    _mrpro_class = None  # cached at class level

    def __new__(cls, op):
        if cls._mrpro_class is None:
            cls._mrpro_class = cls._build_mrpro_class()
        return cls._mrpro_class(op)

    @staticmethod
    def _build_mrpro_class():
        try:
            from mrpro.operators import LinearOperator
        except ImportError as exc:
            raise ImportError(
                "mrpro is required for GrogLinearOp.  "
                "Install it with: pip install mrpro"
            ) from exc

        class _GrogLinearOpImpl(LinearOperator):
            """mrpro LinearOperator wrapping a pygrog SparseFFT-like operator.

            Hides the GROG kernel-width axis by fusing it into the last
            trajectory axis (k0), so the public shape contract matches
            mrpro's standard ``(..., coils, k2, k1, k0)`` k-space layout.
            """

            def __init__(self, op):
                super().__init__()
                self._op = op
                # natural_shape == (*trajectory_shape, kw).
                self._nat_shape = tuple(int(s) for s in op.natural_shape)
                if len(self._nat_shape) >= 2:
                    *traj, kw = self._nat_shape
                    self._mrpro_kshape = (*traj[:-1], traj[-1] * kw)
                else:
                    # 1-D natural shape: kw is the only axis and is left as-is.
                    self._mrpro_kshape = self._nat_shape

            def _to_natural(self, x: torch.Tensor) -> torch.Tensor:
                """Public mrpro k-shape → internal natural shape."""
                lead = x.shape[: -len(self._mrpro_kshape)]
                return x.reshape(*lead, *self._nat_shape)

            def _to_mrpro(self, x: torch.Tensor) -> torch.Tensor:
                """Internal natural shape → public mrpro k-shape."""
                lead = x.shape[: -len(self._nat_shape)]
                return x.reshape(*lead, *self._mrpro_kshape)

            def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
                """Backprojection: k-space → image (A^H).

                Parameters
                ----------
                x : torch.Tensor
                    K-space tensor with shape
                    ``(*other, n_coils, *trajectory_shape[:-1],
                    trajectory_shape[-1] * kw)``.

                Returns
                -------
                tuple of torch.Tensor
                    Image tensor with shape ``(*other, *image_shape)``.
                """
                x = self._to_natural(x)
                return (grog_backproject(x, self._op),)

            def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
                """Measurement: image → k-space (A).

                Parameters
                ----------
                x : torch.Tensor
                    Image tensor with shape ``(*other, *image_shape)``.

                Returns
                -------
                tuple of torch.Tensor
                    K-space tensor in mrpro layout
                    ``(*other, n_coils, *trajectory_shape[:-1],
                    trajectory_shape[-1] * kw)``.
                """
                ksp = grog_measure(x, self._op)
                return (self._to_mrpro(ksp),)

            @property
            def H(self):
                """Adjoint LinearOperator with Toeplitz-aware ``.gram``.

                In mrpro convention, ``self.forward`` is the
                backprojection (k→image) so ``self.H`` represents the
                acquisition (image→k).  We override the adjoint's
                ``gram`` (which mrpro defines as ``self.H @ self``) to
                short-circuit ``self.H @ self.H.H = adjoint @ forward
                = pygrog op.normal`` (image → image), enabling Toeplitz
                acceleration when the underlying op has ``toeplitz=True``.
                """
                outer = self
                return _make_acq_op(outer)

        return _GrogLinearOpImpl


def _make_acq_op(outer):
    """Build a Toeplitz-aware adjoint LinearOperator wrapping ``outer.H``."""
    from mrpro.operators import LinearOperator

    class _GrogAcqOp(LinearOperator):
        """Acquisition op (image→k) with Toeplitz-accelerated ``gram``."""

        def __init__(self):
            super().__init__()
            self._outer = outer

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
            # forward of acquisition = adjoint of original = pygrog adjoint
            return self._outer.adjoint(x)

        def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
            return self._outer.forward(x)

        @property
        def gram(self):
            """``self.H @ self`` short-circuit using ``op.normal``."""
            outer_ref = self._outer
            return _make_image_normal_op(outer_ref)

    return _GrogAcqOp()


def _make_image_normal_op(outer):
    """Self-adjoint LinearOperator computing ``op.normal`` on images."""
    from mrpro.operators import LinearOperator

    class _GrogImageNormalOp(LinearOperator):
        """Image-domain ``A^H A`` via ``pygrog op.normal``."""

        def __init__(self):
            super().__init__()
            self._outer = outer

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
            return (self._outer._op.normal(x),)

        def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
            # Self-adjoint.
            return (self._outer._op.normal(x),)

        @property
        def gram(self):
            return self  # already a normal operator

        @property
        def H(self):
            return self  # self-adjoint

    return _GrogImageNormalOp()


# ===========================================================================
# GrogInterpolator adapter (KData I/O)
# ===========================================================================
class GrogInterpolator(_GrogInterpolatorBase):
    """GROG interpolator with native :class:`mrpro.data.KData` I/O.

    Extracts the k-space trajectory and encoding/recon matrix sizes from
    ``kdata`` and configures the underlying pygrog
    :class:`pygrog.calib.GrogInterpolator` accordingly.  Calibration may
    be supplied as a calibration ``KData`` (for instance, a Cartesian
    centre block) or as a raw numpy/tensor patch.

    Parameters
    ----------
    kdata : mrpro.data.KData
        Source k-space whose trajectory drives the GROG plan.  Header
        ``encoding_matrix`` defines the Cartesian grid the trajectory
        spans; ``recon_matrix`` is the post-FFT crop.
    kernel_width, oversamp, kernel_shape, time_map
        Forwarded to :class:`pygrog.calib.GrogInterpolator`.

    Notes
    -----
    The result of :meth:`interpolate` is a fresh ``KData`` whose
    ``data`` carries the sparse Cartesian k-space (the GROG kernel-width
    axis is fused into ``k0``, matching :class:`GrogLinearOp`'s
    convention) and whose ``traj`` is updated to the gridded sample
    locations.  The companion :class:`pygrog.calib.GrogPlan` is returned
    alongside so callers can build a downstream
    :class:`~pygrog.operator.SparseFFT` / :class:`GrogLinearOp` directly.
    """

    def __init__(
        self,
        kdata,
        *,
        kernel_width: int = 2,
        oversamp: float | list | tuple | None = None,
        kernel_shape: str = "circle",
        time_map=None,
    ):
        coords_np, _, enc_shape, recon_shape = _kdata_extract(kdata)
        super().__init__(
            shape=enc_shape,
            coords=coords_np,
            oversamp=oversamp,
            kernel_width=kernel_width,
            kernel_shape=kernel_shape,
            time_map=time_map,
            image_shape=recon_shape,
        )
        self._enc_shape = enc_shape
        self._recon_shape = recon_shape
        self._ndim = len(enc_shape)

    # ------------------------------------------------------------------
    def calc_interp_table(self, calib, *, lamda: float = 0.01, precision: int = 1):
        """Fit the GRAPPA kernel table.

        Parameters
        ----------
        calib : KData | np.ndarray | torch.Tensor
            Fully-sampled calibration region.  When a ``KData`` is given,
            its ``data`` tensor is used; the leading ``other`` dims are
            squeezed and any singleton k2 axis is dropped for 2D plans.
        """
        try:
            from mrpro.data import KData as _KData
        except ImportError:
            _KData = None

        if _KData is not None and isinstance(calib, _KData):
            arr = calib.data
            # Squeeze leading "other" dims (assumed all size 1 for calibration).
            while arr.ndim > 3 + 1:  # coils + spatial(3)
                if arr.shape[0] != 1:
                    raise ValueError(
                        f"Calibration KData has non-singleton 'other' dims; "
                        f"got data shape {tuple(arr.shape)}."
                    )
                arr = arr[0]
            if self._ndim == 2:
                # (coils, 1, k1, k0) → (coils, k1, k0)
                arr = arr.squeeze(-3)
            calib_arr = arr.detach().cpu().numpy()
        else:
            calib_arr = calib

        super().calc_interp_table(calib_arr, lamda=lamda, precision=precision)

    # ------------------------------------------------------------------
    def interpolate(self, kdata, *, return_plan: bool = True, grid: bool = False):
        """GROG-interpolate ``kdata`` onto the sparse Cartesian grid.

        Parameters
        ----------
        kdata : mrpro.data.KData
            K-space data sharing the trajectory/encoding used at
            construction.
        return_plan : bool, optional
            If ``True`` (default) return the plan alongside the output.
        grid : bool, optional
            If ``True``, scatter the interpolated samples onto a dense
            oversampled Cartesian grid and return
            ``(KData, mask, density[, plan])``.  The returned ``KData``
            carries a regular Cartesian ``KTrajectory`` and has shape
            ``(*other, n_coils, k2, k1, k0)`` matching ``grid_shape``
            (``k2=1`` for 2D plans).  ``mask`` and ``density`` have shape
            ``(*stack, *grid_shape)`` and can be passed directly to
            :class:`~pygrog.operator.MaskedFFT`.

        Returns
        -------
        KData (or (KData, GrogPlan))
            When ``grid=False`` (default): ``data`` shape
            ``(*other, n_coils, k2', k1', k0')`` with the GROG kernel width
            fused into ``k0'``; ``traj`` updated to the gridded sample
            positions; ``header`` carried over.
        (KData, mask, density[, plan])
            When ``grid=True``: dense Cartesian KData with Cartesian
            trajectory, plus real-valued ``mask`` and ``density`` tensors.
        """
        from mrpro.data import KData, KTrajectory

        _coords_np, data_t, enc_shape, _ = _kdata_extract(kdata)
        if enc_shape != self._enc_shape:
            raise ValueError(
                f"KData encoding_matrix {enc_shape} does not match "
                f"interpolator {self._enc_shape}."
            )

        # Drop singleton k2 for 2D plans before calling the base interpolate.
        data_for_interp = _data_to_spatial(data_t, self._ndim)
        if grid:
            out = super().interpolate(data_for_interp, grid=True)
            grid_kspace, mask, density = (torch.as_tensor(t) for t in out)
            # grid_kspace: (*batch, *stack, C, *grid_shape)
            # Re-insert singleton k2 for 2D → (*batch, *stack, C, 1, gy, gx)
            grid_data = _spatial_to_kdata(grid_kspace, self._ndim)

            # Build a regular Cartesian KTrajectory covering the oversampled grid.
            plan = self.plan
            grid_shape = tuple(int(s) for s in plan.grid_shape)
            traj_1d = []
            for d, gs in enumerate(grid_shape):
                origin = float(-(enc_shape[d] // 2))
                step = (enc_shape[d] - 1) / (gs - 1) if gs > 1 else 1.0
                traj_1d.append(origin + torch.arange(gs, dtype=torch.float32) * step)
            # meshgrid → each tensor has shape (*grid_shape)
            grids = torch.meshgrid(*traj_1d, indexing="ij")

            if self._ndim == 2:
                # 2D: grids[0]=ky, grids[1]=kx, each (gy, gx)
                # mrpro KTrajectory layout: (*other, coil_singleton=1, k2, k1, k0)
                kz_new = torch.zeros(1, 1, 1, *grid_shape, dtype=torch.float32)
                ky_new = (
                    grids[0].unsqueeze(0).unsqueeze(0).unsqueeze(0)
                )  # (1,1,1,gy,gx)
                kx_new = grids[1].unsqueeze(0).unsqueeze(0).unsqueeze(0)
            else:
                # 3D: grids[0]=kz, grids[1]=ky, grids[2]=kx, each (gz,gy,gx)
                kz_new = grids[0].unsqueeze(0).unsqueeze(0)  # (1,1,gz,gy,gx)
                ky_new = grids[1].unsqueeze(0).unsqueeze(0)
                kx_new = grids[2].unsqueeze(0).unsqueeze(0)

            new_traj = KTrajectory(kz=kz_new, ky=ky_new, kx=kx_new)
            new_kdata = KData(header=kdata.header, data=grid_data, traj=new_traj)
            if return_plan:
                return new_kdata, mask, density, plan
            return new_kdata, mask, density
        sparse = super().interpolate(data_for_interp)
        # Reshape from flat (*other, coils, n_samples) → (*other, coils, *spatial, kw)
        sparse = torch.as_tensor(sparse)
        sparse = sparse.reshape(*sparse.shape[:-1], *self.plan.natural_shape)

        # Fuse kw into the last spatial axis to match mrpro layout.
        *_lead, n_coils = (*sparse.shape[:-(self._ndim + 1)], sparse.shape[-(self._ndim + 1)])
        spatial_kw = sparse.shape[-(self._ndim + 1) + 1 :]  # (*spatial, kw)
        spatial = spatial_kw[:-1]
        kw = int(spatial_kw[-1])
        # Flatten kw into the last spatial axis (k0 in mrpro layout).
        new_spatial = (*spatial[:-1], spatial[-1] * kw)
        new_data_shape = (*sparse.shape[: -(self._ndim + 1)], n_coils, *new_spatial)
        new_data = sparse.reshape(new_data_shape)
        # Re-insert singleton k2 for 2D so layout becomes (*other, coils, 1, k1, k0*kw).
        new_data = _spatial_to_kdata(new_data, self._ndim)

        # Build a new trajectory whose kx/ky/kz reflect the (oversampled)
        # Cartesian grid points each replicated GROG sample lands on.
        plan = self.plan
        target_idx = torch.as_tensor(plan.target_idx).to(torch.int64)  # (*spatial, kw)
        grid_shape = tuple(int(s) for s in plan.grid_shape)
        # Decode flat indices into per-axis grid coordinates.
        idx = target_idx.clamp_min(0)
        coords_grid = []
        rem = idx
        for d, gs in enumerate(grid_shape):
            stride = int(np.prod(grid_shape[d + 1 :])) if d + 1 < len(grid_shape) else 1
            coords_grid.append((rem // stride) % gs)
            rem = rem - (rem // stride) * stride if stride > 1 else rem
        # Convert grid index → physical coords in pygrog scale [-shape/2, shape/2).
        # grid_steps_d = (shape_d - 1) / (grid_shape_d - 1) for each axis.
        phys = []
        for d, gs in enumerate(grid_shape):
            origin = -(enc_shape[d] // 2)
            step = (enc_shape[d] - 1) / (gs - 1) if gs > 1 else 1.0
            phys.append(origin + coords_grid[d].float() * step)
        # phys is list of tensors shape (*spatial, kw); stack & flatten kw into last axis.
        phys_stack = [
            _spatial_to_kdata(
                p.reshape((*p.shape[:-2], p.shape[-2] * p.shape[-1])), self._ndim
            )
            for p in phys
        ]

        # Each phys_stack[d] now has shape matching mrpro layout (k2', k1', k0').
        # Reshape into KTrajectory's expected (*other=1, coil=1, k2', k1', k0').
        def _to_traj(t):
            t = t.unsqueeze(0).unsqueeze(0)  # (1, 1, k2', k1', k0')
            return t

        if self._ndim == 2:
            ky_new = _to_traj(phys_stack[0])
            kx_new = _to_traj(phys_stack[1])
            kz_new = torch.zeros_like(kx_new)
        else:
            kz_new = _to_traj(phys_stack[0])
            ky_new = _to_traj(phys_stack[1])
            kx_new = _to_traj(phys_stack[2])
        new_traj = KTrajectory(kz=kz_new, ky=ky_new, kx=kx_new)

        new_kdata = KData(header=kdata.header, data=new_data, traj=new_traj)
        if return_plan:
            return new_kdata, plan
        return new_kdata


# ===========================================================================
# NLINV calibration (always pygrog backend — preferred over framework NLINV)
# ===========================================================================
def nlinv_calib(
    kdata,
    *,
    cal_width: int = 24,
    ret_cal: bool = False,
    ret_image: bool = False,
    **kwargs,
):
    """Estimate coil sensitivities for a ``KData`` via pygrog NLINV.

    Uses :func:`pygrog.utils.nlinv_calib` (preferred over any framework's
    own NLINV) and reshapes the result into mrpro's smap layout
    ``(n_coils, z, y, x)`` (broadcasting trivially against the
    ``(*other, n_coils, z, y, x)`` :class:`mrpro.data.IData`/
    :class:`mrpro.operators.SensitivityOp` convention).

    Parameters
    ----------
    kdata : mrpro.data.KData
        Source k-space.
    cal_width : int
        Calibration patch width.
    ret_cal, ret_image
        Forwarded to :func:`pygrog.utils.nlinv_calib`.
    **kwargs
        Additional keyword arguments forwarded to
        :func:`pygrog.utils.nlinv_calib`.

    Returns
    -------
    smaps : torch.Tensor
        Coil sensitivities, shape ``(n_coils, z, y, x)`` (z=1 for 2D).
    *extras : optional
        Additional outputs from ``nlinv_calib`` (calibration k-space and/or
        reconstructed image), if requested.
    """
    coords_np, data_t, enc_shape, recon_shape = _kdata_extract(kdata)
    ndim = len(enc_shape)

    # Allow leading *other dims (batch).  Squeeze leading singleton other
    # axes so single-frame KData (the historic case) returns a result with
    # no leading batch dim.  Trajectory is assumed shared across *other.
    arr = data_t  # (*other, n_coils, k2, k1, k0)
    while arr.ndim > 4 and arr.shape[0] == 1:
        arr = arr[0]
    other_shape = tuple(int(s) for s in arr.shape[:-4])
    if ndim == 2:
        # Drop singleton k2 → (*other, n_coils, k1, k0).
        arr_sf = arr.squeeze(-3)
        coords_np = coords_np.reshape(-1, 2)
        y_in = arr_sf.reshape(*other_shape, arr_sf.shape[-3], -1)
    else:
        coords_np = coords_np.reshape(-1, 3)
        y_in = arr.reshape(*other_shape, arr.shape[-4], -1)

    out = _nlinv_calib(
        y_in,
        cal_width=cal_width,
        shape=recon_shape,
        coords=coords_np,
        ret_cal=ret_cal,
        ret_image=ret_image,
        **kwargs,
    )
    if not isinstance(out, tuple):
        out = (out,)
    smaps = out[0]
    smaps_t = torch.as_tensor(smaps)
    if ndim == 2:
        # (..., n_coils, y, x) → (..., n_coils, 1, y, x)
        smaps_t = smaps_t.unsqueeze(-3)
    extras = out[1:]
    if extras:
        return (smaps_t, *extras)
    return smaps_t


# ===========================================================================
# Coil compression — dispatch to mrpro's native KData.compress_coils
# ===========================================================================
def coil_compress(kdata, n_coils: int, *, batch_dims=None, joint_dims=...):
    """Coil-compress a ``KData`` via mrpro's native PCA compression.

    Thin dispatch to :meth:`mrpro.data.KData.compress_coils`; provided so
    callers using pygrog adapters across all three frameworks have a
    uniform API surface.

    Parameters
    ----------
    kdata : mrpro.data.KData
        Source k-space.
    n_coils : int
        Number of virtual coils to retain.
    batch_dims, joint_dims
        Forwarded to :meth:`KData.compress_coils`.

    Returns
    -------
    KData
        K-space with reduced coil dimension.
    """
    return kdata.compress_coils(n_coils, batch_dims=batch_dims, joint_dims=joint_dims)
