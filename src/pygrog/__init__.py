"""PyGROG package"""

import os
import sys

# On macOS, PyTorch and finufft each bundle their own libomp.dylib.
# Set these environment variables *before* any submodule imports so that
# OpenMP is initialised correctly:
#   KMP_DUPLICATE_LIB_OK=TRUE  – suppress abort() when two OMP runtimes load
#   OMP_NUM_THREADS=1          – prevent thread-pool deadlocks from two runtimes
# Using os.environ.setdefault so that explicit user overrides are respected.
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

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
