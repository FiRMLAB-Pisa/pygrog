"""PyGROG package"""

# Version information
try:
    from ._version import __version__
except ImportError:
    # Fallback for development installs
    try:
        from setuptools_scm import get_version

        __version__ = get_version(root="../..")
    except (ImportError, LookupError):
        __version__ = "0.1.0+dev"

from . import calib  # noqa
from . import operator  # noqa
from . import gadgets  # noqa
from . import utils  # noqa
from . import interop  # noqa

# Iterative solvers + polynomial preconditioner.
from ._solve import cg, lsmr, solve, PolynomialPreconditioner  # noqa
