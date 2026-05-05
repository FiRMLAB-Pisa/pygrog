"""Reconstruction gadgets: off-resonance correction and subspace projection."""

__all__ = []

from . import _off_resonance
from . import _subspace

from ._off_resonance import *  # noqa
from ._subspace import *  # noqa

from ._off_resonance import OffResonanceCorrection, with_off_resonance
from ._subspace import SubspaceSparseFFT


class SubspaceGadget(SubspaceSparseFFT):
    """Low-rank temporal/contrast subspace wrapper for sparse/gridded operators."""


class OffResonanceGadget(OffResonanceCorrection):
    """Low-rank B0/R2* off-resonance correction wrapper for sparse/gridded operators."""


def with_offresonance(*args, **kwargs):
    """Decorator-style constructor for off-resonance-wrapped operators."""
    return with_off_resonance(*args, **kwargs)


__all__.extend(_off_resonance.__all__)
__all__.extend(_subspace.__all__)
__all__.extend(
    [
        "OffResonanceGadget",
        "SubspaceGadget",
        "with_offresonance",
    ]
)
