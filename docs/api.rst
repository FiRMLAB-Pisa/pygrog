API Reference
=============

Public API grouped by subpackage.

Calibration Helpers
-------------------

.. autosummary::

   pygrog.utils.coil_compression
   pygrog.utils.nlinv

GROG
----

.. autosummary::

   pygrog.calib.GrogInterpolator

Gadgets
-------

.. autosummary::

   pygrog.gadgets.SubspaceGadget
   pygrog.gadgets.OffResonanceGadget
   pygrog.gadgets.with_subspace
   pygrog.gadgets.with_offresonance

Operators
---------

.. autosummary::

   pygrog.operator.SparseFFT
   pygrog.operator.MaskedFFT

Interoperability
----------------

.. autosummary::

   pygrog.interop.GrogLinop
   pygrog.interop.GrogLinearOp
   pygrog.interop.GrogLinearPhysics

.. toctree::
   :maxdepth: 1

   api/calib
   api/grog
   api/gadgets
   api/operators
   api/interop
