"""deepinv LinearPhysics adapter for SparseFFT.

Wraps :class:`~pygrog.operator.SparseFFT` as a
``deepinv.physics.LinearPhysics`` so it plugs into deepinv reconstruction
algorithms (unrolled networks, PnP, RED, …) without modification.

Gradients are computed via :mod:`pygrog.interop._torch` — explicit
``torch.autograd.Function`` subclasses whose backward is the adjoint of the
measurement operator — rather than relying on automatic differentiation
through the GROG kernels.

deepinv ``LinearPhysics`` contract:
  - Subclass ``deepinv.physics.LinearPhysics``
  - Implement ``A(x)``         : image  → k-space  (measurement)
  - Implement ``A_adjoint(y)`` : k-space → image   (backprojection)
  - ``A_dagger`` (pseudoinverse) is then provided by the base class.
"""

__all__ = ["GrogLinearPhysics"]

import torch

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

        y = physics(x)               # noisy measurement
        x_hat = physics.A_dagger(y)  # pseudoinverse
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
            """deepinv LinearPhysics wrapping a pygrog SparseFFT-like operator."""

            def __init__(self, op, noise_model=None):
                super().__init__(noise_model=noise_model or ZeroNoise())
                self._op = op

            def A(self, x: torch.Tensor, **kwargs) -> torch.Tensor:  # noqa: ARG002
                """Forward measurement: image → k-space."""
                return grog_measure(x, self._op)

            def A_adjoint(self, y: torch.Tensor, **kwargs) -> torch.Tensor:  # noqa: ARG002
                """Backprojection: k-space → image."""
                return grog_backproject(y, self._op)

        return _GrogLinearPhysicsImpl
