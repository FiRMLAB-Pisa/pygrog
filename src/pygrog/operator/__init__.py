"""
Optimized operators for PyGROG
"""

__all__ = []

from ._sparse_fft import *  # noqa

# Import fast binning functions with fallback handling
try:
    from ._fast_binning import fast_binning_add_at
    _fast_binning_available = True
    __all__.append("fast_binning_add_at")
except ImportError:
    _fast_binning_available = False

try:
    from ._fast_binning import detect_simd_level
    __all__.append("detect_simd_level")
except ImportError:
    pass

from . import _sparse_fft # noqa

__all__.extend(_sparse_fft.__all__)

