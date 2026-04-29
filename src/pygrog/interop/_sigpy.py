"""sigpy Linop adapter for SparseFFT.

Wraps a pygrog operator (with ``forward`` / ``adjoint`` methods) as a
``sigpy.linop.Linop`` so it plugs into sigpy reconstruction algorithms
(conjugate gradient, primal-dual hybrid gradient, etc.) without modification.

sigpy ``Linop`` contract:
  - Subclass ``sigpy.linop.Linop``
  - Implement ``_apply(self, input)`` → applies the forward direction
  - Implement ``_adjoint_linop(self)`` → returns a ``Linop`` for the adjoint
  - Inputs/outputs are numpy (CPU) or cupy (GPU) arrays

Array conversion (numpy/cupy ↔ torch) is handled transparently by
:class:`~pygrog.operator.SparseFFT`, which is decorated with
:func:`mrinufft._array_compat.with_torch` on its ``forward`` and ``adjoint``
methods.  The sigpy ``_apply`` method therefore passes arrays directly to
the operator and receives numpy/cupy back.
"""

__all__ = [
    "GrogLinop",
    "GrogNormalLinop",
    "GrogInterpolator",
    "nlinv_calib",
    "coil_compress",
]

import numpy as np

from ..calib import GrogInterpolator as _GrogInterpolatorBase
from ..utils import coil_compress as _coil_compress
from ..utils import nlinv_calib as _nlinv_calib


class GrogLinop:
    """Wrap a pygrog SparseFFT-like operator as a ``sigpy.linop.Linop``.

    The returned object is a real ``sigpy.linop.Linop`` with a working
    ``.H`` (adjoint) property, and therefore participates in all sigpy
    operator algebra (composition via ``*``, addition, scaling, etc.).

    Parameters
    ----------
    op : SparseFFT-like
        Any pygrog operator with ``forward(kspace) -> image`` and
        ``adjoint(image) -> kspace`` methods.

    Raises
    ------
    ImportError
        If ``sigpy`` is not installed.

    Examples
    --------
    ::

        from pygrog.operator import SparseFFT
        from pygrog.interop import GrogLinop

        base = SparseFFT(plan=grog.plan, smaps=smaps)
        A = GrogLinop(base)

        # Use inside sigpy CG reconstruction:
        import sigpy.alg as alg
        AHA = A.H * A
        # ... set up CG solver using AHA

    """

    # --- class factory (lazy) ---------------------------------------------
    _sigpy_class = None

    def __new__(cls, op):
        if cls._sigpy_class is None:
            cls._sigpy_class = cls._build_sigpy_class()
        return cls._sigpy_class(op)

    @staticmethod
    def _build_sigpy_class():
        try:
            from sigpy.linop import Linop
        except ImportError as exc:
            raise ImportError(
                "sigpy is required for GrogLinop.  "
                "Install it with: pip install sigpy"
            ) from exc

        class _GrogLinopImpl(Linop):
            """sigpy Linop wrapping a pygrog SparseFFT-like operator."""

            def __init__(self, op, *, _adjoint=False):
                self._op = op
                self._is_adjoint = _adjoint

                # Shapes for sigpy: oshape, ishape
                # Convention: forward = kspace → image (adjoint NUFFT direction)
                #   ishape = (n_coils, n_samples)
                #   oshape = image_shape (or (n_coils, *image_shape) without smaps)
                n_samples = op.indices.shape[0]
                n_coils = op.smaps.shape[0] if op.smaps is not None else 1

                if op.smaps is not None:
                    # forward: (n_coils, n_samples) → (*image_shape)
                    ksp_shape = [n_coils, n_samples]
                    img_shape = list(op.image_shape)
                else:
                    ksp_shape = [n_coils, n_samples]
                    img_shape = [n_coils, *list(op.image_shape)]

                if not _adjoint:
                    super().__init__(img_shape, ksp_shape)
                else:
                    super().__init__(ksp_shape, img_shape)

            def _apply(self, input):  # noqa: A002
                # SparseFFT.forward / .adjoint are decorated with with_torch,
                # so they accept numpy/cupy arrays and return the same type.
                if self._is_adjoint:
                    # adjoint: image → k-space.  Return flat (n_coils, n_samples)
                    # to match the declared ishape/oshape so sigpy doesn't complain.
                    op = self._op
                    n_coils = op.smaps.shape[0] if op.smaps is not None else 1
                    n_samples = op.indices.shape[0]
                    return op._adjoint_flat(input).reshape(n_coils, n_samples)
                return self._op.forward(input)

            def _adjoint_linop(self):
                return _GrogLinopImpl(self._op, _adjoint=not self._is_adjoint)

        return _GrogLinopImpl


# ===========================================================================
# Normal-operator Linop (Toeplitz short-circuit)
# ===========================================================================
class GrogNormalLinop:
    """sigpy ``Linop`` wrapping ``op.normal`` (i.e. ``A^H A``).

    Use this in CG / least-squares loops in place of ``A.H * A`` when the
    underlying pygrog operator's ``toeplitz`` flag is enabled (``A.H * A``
    in sigpy is a generic composed Linop and does NOT short-circuit to
    ``op.normal``).

    Parameters
    ----------
    op : SparseFFT-like
        Any pygrog operator with a ``.normal(image)`` method.

    Examples
    --------
    ::

        from pygrog.interop import GrogLinop, GrogNormalLinop
        A = GrogLinop(base)
        AHA = GrogNormalLinop(base)   # uses Toeplitz when base.toeplitz=True
        # Use AHA wherever you would use ``A.H * A``.
    """

    _sigpy_class = None

    def __new__(cls, op):
        if cls._sigpy_class is None:
            cls._sigpy_class = cls._build_sigpy_class()
        return cls._sigpy_class(op)

    @staticmethod
    def _build_sigpy_class():
        try:
            from sigpy.linop import Linop
        except ImportError as exc:
            raise ImportError(
                "sigpy is required for GrogNormalLinop.  "
                "Install it with: pip install sigpy"
            ) from exc

        class _GrogNormalLinopImpl(Linop):
            """Self-adjoint Linop that applies ``op.normal``."""

            def __init__(self, op):
                self._op = op
                if op.smaps is not None:
                    img_shape = list(op.image_shape)
                else:
                    n_coils = (
                        op.smaps.shape[0] if op.smaps is not None else 1
                    )
                    img_shape = [n_coils, *list(op.image_shape)]
                super().__init__(img_shape, img_shape)

            def _apply(self, input):  # noqa: A002
                return self._op.normal(input)

            def _adjoint_linop(self):
                return self  # self-adjoint

        return _GrogNormalLinopImpl


# ===========================================================================
# GrogInterpolator adapter (numpy ndarray I/O)
# ===========================================================================
class GrogInterpolator(_GrogInterpolatorBase):
    """GROG interpolator with numpy-array I/O for sigpy users.

    sigpy has no native non-Cartesian data container — calibration,
    coordinates and k-space are passed as plain :class:`numpy.ndarray`.
    This adapter is a thin sigpy-flavoured wrapper that mirrors the API
    of :class:`pygrog.interop.mrpro.GrogInterpolator` but accepts and
    returns sigpy's idiomatic types.

    Parameters
    ----------
    coords : np.ndarray, shape ``(*spatial, ndim)``
        Trajectory coordinates in pygrog scale (``[-shape/2, shape/2]``).
    shape : int | tuple[int, ...]
        Cartesian image shape spanned by the trajectory.
    image_shape : tuple[int, ...] | None, optional
        FFT crop target.  Defaults to *shape*.
    kernel_width, oversamp, kernel_shape, time_map
        Forwarded to :class:`pygrog.calib.GrogInterpolator`.

    Returns from :meth:`interpolate`
    --------------------------------
    Tuple ``(sparse_kspace_ndarray, GrogPlan)``.  The sparse k-space has
    shape ``(n_coils, *natural_shape)`` (i.e. ``(n_coils, *spatial, kw)``)
    matching the layout :class:`~pygrog.operator.SparseFFT` consumes.
    """

    def __init__(
        self,
        coords,
        shape,
        *,
        image_shape=None,
        kernel_width: int = 2,
        oversamp: float | list | tuple | None = None,
        kernel_shape: str = "circle",
        time_map=None,
    ):
        super().__init__(
            shape=shape,
            coords=np.asarray(coords),
            oversamp=oversamp,
            kernel_width=kernel_width,
            kernel_shape=kernel_shape,
            time_map=time_map,
            image_shape=image_shape,
        )

    def interpolate(self, kspace, *, return_plan: bool = True, **kwargs):
        """Interpolate ``kspace`` and (optionally) return the plan.

        Parameters
        ----------
        kspace : np.ndarray
            ``(*batch, n_coils, *spatial)`` complex k-space.
        return_plan : bool, optional
            If ``True`` (default) return ``(ndarray, GrogPlan)``;
            if ``False`` return the ndarray only.
        grid : bool, optional
            If ``True``, return ``(gridded_kspace, mask, density[, plan])``
            numpy arrays instead of the flat sparse output.
        """
        grid = kwargs.get("grid", False)
        out = super().interpolate(np.asarray(kspace), **kwargs)
        if grid:
            grid_kspace, mask, density = (np.asarray(t) for t in out)
            if return_plan:
                return grid_kspace, mask, density, self.plan
            return grid_kspace, mask, density
        out = np.asarray(out)
        # Reshape from flat (*batch, C, n_samples) → (*batch, C, *natural_shape)
        out = out.reshape(*out.shape[:-1], *self.plan.natural_shape)
        if return_plan:
            return out, self.plan
        return out


# ===========================================================================
# NLINV calibration — passthrough to pygrog.utils.nlinv_calib
# ===========================================================================
def nlinv_calib(
    kspace,
    coords,
    shape,
    *,
    cal_width: int = 24,
    ret_cal: bool = False,
    ret_image: bool = False,
    **kwargs,
):
    """Estimate coil sensitivities from non-Cartesian sigpy-style inputs.

    Thin numpy-friendly wrapper around :func:`pygrog.utils.nlinv_calib`
    (preferred over sigpy's :class:`sigpy.mri.app.JsenseRecon`).

    Parameters
    ----------
    kspace : np.ndarray
        Multi-coil non-Cartesian k-space, shape ``(n_coils, n_samples)``.
    coords : np.ndarray
        Trajectory coordinates, shape ``(n_samples, ndim)`` in pygrog
        scale.
    shape : tuple[int, ...]
        Image shape ``(y, x)`` or ``(z, y, x)``.
    cal_width : int
        Calibration patch width.
    **kwargs
        Forwarded to :func:`pygrog.utils.nlinv_calib`.

    Returns
    -------
    smaps : np.ndarray
        Coil sensitivities of shape ``(n_coils, *shape)``.
    *extras
        Optional ``(grappa_train, image)`` if requested via
        ``ret_cal=True`` / ``ret_image=True``.
    """
    out = _nlinv_calib(
        np.asarray(kspace),
        cal_width=cal_width,
        shape=tuple(shape),
        coords=np.asarray(coords),
        ret_cal=ret_cal,
        ret_image=ret_image,
        **kwargs,
    )
    if isinstance(out, tuple):
        return tuple(np.asarray(o) for o in out)
    return np.asarray(out)


# ===========================================================================
# Coil compression — passthrough to pygrog.utils.coil_compress
# ===========================================================================
def coil_compress(
    kspace,
    n_coils,
    *,
    traj=None,
    krad_thresh: float | None = None,
):
    """Coil-compress non-Cartesian k-space (numpy I/O).

    Thin wrapper around :func:`pygrog.utils.coil_compress`.

    Parameters
    ----------
    kspace : np.ndarray
        Multi-coil k-space, shape ``(n_coils, n_samples)``.
    n_coils : int | float
        Number of virtual coils (int) or energy threshold (float in
        ``(0, 1]``).
    traj : np.ndarray, optional
        Sampling trajectory ``(n_samples, ndim)`` for radius-based
        calibration extraction.
    krad_thresh : float, optional
        Relative k-space radius threshold for calibration selection.

    Returns
    -------
    compressed : np.ndarray
        Compressed k-space ``(n_virtual, n_samples)``.
    matrix : np.ndarray
        Compression matrix ``(n_virtual, n_coils)``.
    """
    return _coil_compress(
        np.asarray(kspace),
        n_coils,
        traj=None if traj is None else np.asarray(traj),
        krad_thresh=krad_thresh,
    )
