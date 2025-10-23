"""Sigpy module."""

import warnings

# Avoid strange bugs when importing sigpy before torch
try:
    import torch # noqa 
except ImportError:
    pass

# Suppress annoying Sigpy warning (TODO: fix)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from sigpy import * # noqa