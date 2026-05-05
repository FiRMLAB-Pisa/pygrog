"""Iterative solvers (CG, LSMR) and polynomial preconditioning for PyGROG.

These solvers are framework-agnostic: they operate on any object that exposes
``.forward(x)``, ``.adjoint(y)``, ``.normal(x)``, ``.image_shape``, and
``.device`` attributes (i.e. :class:`~pygrog.operator.SparseFFT`,
:class:`~pygrog.operator.MaskedFFT`, and the
:mod:`~pygrog.gadgets`-decorated variants).
"""

__all__ = []

from ._solvers import *  # noqa
from ._polynomial import *  # noqa
from ._mixin import *  # noqa

from . import _solvers  # noqa
from . import _polynomial  # noqa
from . import _mixin  # noqa

__all__.extend(_solvers.__all__)
__all__.extend(_polynomial.__all__)
__all__.extend(_mixin.__all__)
