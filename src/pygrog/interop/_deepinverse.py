"""deepinv LinearPhysics adapter for SparseFFT.

Wraps :class:`~pygrog.operator.SparseFFT` as a
``deepinv.physics.LinearPhysics`` so it plugs into deepinv reconstruction
algorithms (FISTA, ADMM, PnP, RED, unrolled networks, …) without
modification.

Shape conventions (matching deepinv's ``(B, C, *spatial)`` layout)
==================================================================

The adapter speaks deepinv's native batched shapes at its boundary so it
can be dropped directly into ``optim_builder``, ``Trainer`` and the rest of
the deepinv ecosystem:

* **image**:   ``(B, 1, *image_shape)`` complex.  ``C == 1`` because the
  wrapped operator already coil-combines via ``smaps``.
* **k-space**: ``(B, n_coils, n_samples)`` complex (flattened sparse layout
  – deepinv treats measurements as opaque tensors, so the flat layout is
  the most permissive choice and avoids forcing callers to reshape on
  every call).

Gradients are computed via :mod:`pygrog.interop._torch` — explicit
``torch.autograd.Function`` subclasses whose backward is the adjoint of
the measurement operator — rather than relying on automatic
differentiation through the GROG kernels.

deepinv ``LinearPhysics`` contract:
  - Subclass ``deepinv.physics.LinearPhysics``
  - Implement ``A(x)``         : image  → k-space  (measurement)
  - Implement ``A_adjoint(y)`` : k-space → image   (backprojection)
  - ``A_dagger`` (pseudoinverse) is then provided by the base class.
"""

__all__ = ["GrogLinearPhysics", "GrogInterpolator", "nlinv_calib", "coil_compress"]

import numpy as np
import torch

from ..calib import GrogInterpolator as _GrogInterpolatorBase
from ..utils import coil_compress as _coil_compress
from ..utils import nlinv_calib as _nlinv_calib
from ._torch import grog_backproject, grog_measure


class GrogLinearPhysics:
    """Wrap a pygrog operator as a ``deepinv.physics.LinearPhysics``.

    Because deepinv is an optional dependency, the concrete subclass is
    built lazily on first instantiation.

    Parameters
    ----------
    op : SparseFFT-like
        Any operator with ``forward(kspace) -> image`` and
        ``adjoint(image) -> kspace`` methods.
    noise_model : deepinv.physics.NoiseModel or None, optional
        Noise model to attach.  Defaults to ``deepinv.physics.ZeroNoise()``.

    Raises
    ------
    ImportError
        If ``deepinv`` is not installed.

    Examples
    --------
    ::

        from pygrog.operator import SparseFFT
        from pygrog.interop import GrogLinearPhysics

        op = SparseFFT(plan=grog.plan, smaps=smaps)
        physics = GrogLinearPhysics(op)

        # x: (B, 1, H, W) complex,  y: (B, n_coils, n_samples) complex
        y = physics(x)
        x_hat = physics.A_dagger(y)
    """

    _deepinv_class = None  # cached at class level

    def __new__(cls, op, noise_model=None):
        if cls._deepinv_class is None:
            cls._deepinv_class = cls._build_class()
        return cls._deepinv_class(op, noise_model=noise_model)

    @staticmethod
    def _build_class():
        try:
            from deepinv.physics import LinearPhysics, ZeroNoise
        except ImportError as exc:
            raise ImportError(
                "deepinv is required for GrogLinearPhysics.  "
                "Install it with: pip install deepinv"
            ) from exc

        class _GrogLinearPhysicsImpl(LinearPhysics):
            """deepinv LinearPhysics wrapping a pygrog SparseFFT-like operator.

            Hides the (n_coils, *natural_shape) sparse-k-space layout behind
            deepinv's ``(B, C, *spatial)`` convention.  Coils live in the
            ``C`` axis on the measurement side; the image side has ``C=1``
            because the wrapped operator coil-combines via ``smaps``.
            """

            def __init__(self, op, noise_model=None):
                super().__init__(noise_model=noise_model or ZeroNoise())
                self._op = op
                self._n_coils = (
                    int(op.smaps.shape[0]) if getattr(op, "smaps", None) is not None
                    else None
                )
                self._n_samples = int(op.indices.shape[0])

            # ---- shape adapters ------------------------------------------
            def _strip_image(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
                """``(B, 1, *image)`` → ``(B, *image)`` for the wrapped op."""
                if x.ndim < 2 or x.shape[1] != 1:
                    raise ValueError(
                        "GrogLinearPhysics expects images of shape "
                        f"(B, 1, *image_shape); got {tuple(x.shape)}."
                    )
                return x[:, 0], x.shape[0]

            def _wrap_image(self, x: torch.Tensor) -> torch.Tensor:
                return x.unsqueeze(1)

            def _strip_kspace(self, y: torch.Tensor) -> torch.Tensor:
                """``(B, n_coils, n_samples)`` → ``(B, n_coils, n_samples)``.

                A no-op shape check; provided for symmetry with image side.
                """
                if y.ndim != 3 or y.shape[2] != self._n_samples:
                    raise ValueError(
                        "GrogLinearPhysics expects k-space of shape "
                        f"(B, n_coils, {self._n_samples}); got {tuple(y.shape)}."
                    )
                return y

            # ---- LinearPhysics API ---------------------------------------
            def A(self, x: torch.Tensor, **kwargs) -> torch.Tensor:  # noqa: ARG002
                """Forward measurement: image → k-space.

                Parameters
                ----------
                x : torch.Tensor
                    Image of shape ``(B, 1, *image_shape)``.

                Returns
                -------
                torch.Tensor
                    K-space of shape ``(B, n_coils, n_samples)``.
                """
                img, _ = self._strip_image(x)
                ksp = grog_measure(img, self._op)
                return ksp.reshape(ksp.shape[0], ksp.shape[1], -1)

            def A_adjoint(
                self, y: torch.Tensor, **kwargs  # noqa: ARG002
            ) -> torch.Tensor:
                """Backprojection: k-space → image.

                Parameters
                ----------
                y : torch.Tensor
                    K-space of shape ``(B, n_coils, n_samples)``.

                Returns
                -------
                torch.Tensor
                    Image of shape ``(B, 1, *image_shape)``.
                """
                y = self._strip_kspace(y)
                img = grog_backproject(y, self._op)
                return self._wrap_image(img)

            def A_adjoint_A(
                self, x: torch.Tensor, **kwargs  # noqa: ARG002
            ) -> torch.Tensor:
                """Self-adjoint operator ``A^H A``.

                Routes through the wrapped operator's ``.normal()``
                method, which uses the Toeplitz embedding when the op
                was constructed with ``toeplitz=True`` (default on CPU).

                Parameters
                ----------
                x : torch.Tensor
                    Image of shape ``(B, 1, *image_shape)``.

                Returns
                -------
                torch.Tensor
                    Same shape as input.
                """
                img, _ = self._strip_image(x)
                # Apply over the batch dim.
                out = torch.stack(
                    [self._op.normal(img[b]) for b in range(img.shape[0])],
                    dim=0,
                )
                return self._wrap_image(out)

        return _GrogLinearPhysicsImpl


# ===========================================================================
# GrogInterpolator adapter (deepinv batched-tensor I/O)
# ===========================================================================
class GrogInterpolator(_GrogInterpolatorBase):
    """GROG interpolator with deepinv-style batched torch I/O.

    deepinv has no native non-Cartesian data container; users typically
    pass batched tensors through their custom ``Physics`` object.  This
    adapter accepts and returns ``(B, ...)`` tensors so it composes
    naturally with deepinv pipelines.

    Parameters
    ----------
    coords : np.ndarray | torch.Tensor, shape ``(*spatial, ndim)``
        Trajectory coordinates in pygrog scale.
    shape : int | tuple[int, ...]
        Cartesian image shape spanned by the trajectory.
    image_shape, kernel_width, oversamp, kernel_shape, time_map
        Forwarded to :class:`pygrog.calib.GrogInterpolator`.
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
        coords_np = (
            coords.detach().cpu().numpy()
            if isinstance(coords, torch.Tensor)
            else np.asarray(coords)
        )
        super().__init__(
            shape=shape,
            coords=coords_np,
            oversamp=oversamp,
            kernel_width=kernel_width,
            kernel_shape=kernel_shape,
            time_map=time_map,
            image_shape=image_shape,
        )

    def interpolate(self, kspace, *, return_plan: bool = True, **kwargs):
        """Interpolate batched non-Cartesian k-space.

        Parameters
        ----------
        kspace : torch.Tensor
            ``(B, n_coils, *spatial)`` complex k-space.

        Returns
        -------
        sparse : torch.Tensor
            ``(B, n_coils, *spatial, kw)`` interpolated k-space (in
            pygrog "natural" layout — the kw axis is exposed because
            deepinv has no notion of ``k0``).
        plan : pygrog.calib.GrogPlan, optional
            Returned when ``return_plan=True`` (default).
        """
        if not isinstance(kspace, torch.Tensor):
            raise TypeError("GrogInterpolator (deepinv) expects a torch.Tensor.")
        if kspace.ndim < 3:
            raise ValueError(
                "Expected (B, n_coils, *spatial) tensor; got shape "
                f"{tuple(kspace.shape)}."
            )
        out = super().interpolate(kspace, **kwargs)
        out_t = torch.as_tensor(out)
        # Reshape from flat (*batch, C, n_samples) → (*batch, C, *natural_shape)
        out_t = out_t.reshape(*out_t.shape[:-1], *self.plan.natural_shape)
        if return_plan:
            return out_t, self.plan
        return out_t


# ===========================================================================
# NLINV calibration — pygrog backend (preferred over any framework NLINV)
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
    """Estimate coil sensitivities, returned in deepinv batched layout.

    Parameters
    ----------
    kspace : torch.Tensor
        ``(B, n_coils, n_samples)`` or ``(n_coils, n_samples)`` k-space.
    coords : torch.Tensor | np.ndarray
        Trajectory ``(n_samples, ndim)`` in pygrog scale.
    shape : tuple[int, ...]
        Image shape.
    cal_width : int
        Calibration patch width.
    **kwargs
        Forwarded to :func:`pygrog.utils.nlinv_calib`.

    Returns
    -------
    smaps : torch.Tensor
        ``(1, n_coils, *shape)`` coil sensitivities (deepinv "B=1, C=coils"
        layout).  The leading batch axis is included so callers can scale
        with the rest of a deepinv pipeline.
    *extras : optional
        Additional outputs from ``nlinv_calib``.
    """
    if isinstance(kspace, torch.Tensor):
        ks = kspace
        if ks.ndim == 3:
            if ks.shape[0] != 1:
                raise ValueError(
                    "nlinv_calib expects batch size 1; got "
                    f"shape {tuple(ks.shape)}."
                )
            ks = ks[0]
        ks_in = ks
    else:
        ks_in = np.asarray(kspace)

    coords_np = (
        coords.detach().cpu().numpy()
        if isinstance(coords, torch.Tensor)
        else np.asarray(coords)
    )

    out = _nlinv_calib(
        ks_in,
        cal_width=cal_width,
        shape=tuple(shape),
        coords=coords_np,
        ret_cal=ret_cal,
        ret_image=ret_image,
        **kwargs,
    )
    if not isinstance(out, tuple):
        out = (out,)
    smaps = torch.as_tensor(out[0]).unsqueeze(0)  # (1, n_coils, *shape)
    extras = tuple(torch.as_tensor(e) for e in out[1:])
    if extras:
        return (smaps, *extras)
    return smaps


# ===========================================================================
# Coil compression — pygrog backend
# ===========================================================================
def coil_compress(
    kspace,
    n_coils,
    *,
    traj=None,
    krad_thresh: float | None = None,
):
    """Coil-compress batched non-Cartesian k-space.

    Parameters
    ----------
    kspace : torch.Tensor
        ``(B, n_coils, n_samples)`` or ``(n_coils, n_samples)`` k-space.
    n_coils : int | float
        Target virtual-coil count or energy threshold.
    traj : torch.Tensor | np.ndarray, optional
        Trajectory for radius-based calibration extraction.
    krad_thresh : float, optional
        Relative k-space radius threshold.

    Returns
    -------
    compressed : torch.Tensor
        Same leading shape as input, with reduced coil dimension.
    matrix : torch.Tensor
        Compression matrix ``(n_virtual, n_coils)``.
    """
    is_torch = isinstance(kspace, torch.Tensor)
    has_batch = is_torch and kspace.ndim == 3 and kspace.shape[0] != 0

    if is_torch and has_batch:
        if kspace.shape[0] != 1:
            raise ValueError(
                "coil_compress expects batch size 1; got shape "
                f"{tuple(kspace.shape)}."
            )
        ks_in = kspace[0]
    else:
        ks_in = kspace

    traj_in = (
        traj.detach().cpu().numpy()
        if isinstance(traj, torch.Tensor)
        else (None if traj is None else np.asarray(traj))
    )

    compressed, matrix = _coil_compress(
        ks_in,
        n_coils,
        traj=traj_in,
        krad_thresh=krad_thresh,
    )
    compressed_t = torch.as_tensor(compressed)
    matrix_t = torch.as_tensor(matrix)
    if is_torch and has_batch:
        compressed_t = compressed_t.unsqueeze(0)
    return compressed_t, matrix_t
