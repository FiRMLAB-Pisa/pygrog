"""Interoperability with external frameworks.

The flat names (``GrogLinearOp``, ``GrogLinop``, ``GrogLinearPhysics``)
remain available at the package level for convenience.  Per-framework
preprocessing helpers (``GrogInterpolator``, ``nlinv_calib``,
``coil_compress``) collide across submodules — they are therefore
exposed via sub-namespaces::

    from pygrog.interop import mrpro as pg_mrpro
    from pygrog.interop import sigpy as pg_sigpy
    from pygrog.interop import deepinv as pg_deepinv

    grog = pg_mrpro.GrogInterpolator(kdata, kernel_width=2, oversamp=1.25)
    grog.calc_interp_table(calib_kdata)
    new_kdata, plan = grog.interpolate(kdata)
"""

__all__ = [
    "GrogLinearOp",
    "GrogLinop",
    "GrogNormalLinop",
    "GrogLinearPhysics",
    "grog_backproject",
    "grog_measure",
    "mrpro",
    "sigpy",
    "deepinv",
]

from ._torch import grog_backproject, grog_measure
from ._mrpro import GrogLinearOp
from ._sigpy import GrogLinop, GrogNormalLinop
from ._deepinverse import GrogLinearPhysics

# Sub-namespaces (so per-framework helpers don't collide).
from . import _mrpro as mrpro  # noqa: E402
from . import _sigpy as sigpy  # noqa: E402
from . import _deepinverse as deepinv  # noqa: E402
