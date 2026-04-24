API Reference
=============

Calibration
-----------

Routines for GRAPPA kernel estimation and GROG interpolation.

.. autosummary::
   :toctree: generated
   :nosignatures:

   pygrog.calib.KernelTable
   pygrog.calib.GrogInterpolator

Operators
---------

Sparse FFT operator for non-Cartesian MRI.

.. autosummary::
   :toctree: generated
   :nosignatures:

   pygrog.operator.SparseFFT

Gadgets
-------

Reconstruction gadgets that wrap a base :class:`~pygrog.operator.SparseFFT`.

.. autosummary::
   :toctree: generated
   :nosignatures:

   pygrog.gadgets.SubspaceProjection
   pygrog.gadgets.SubspaceSparseFFT
   pygrog.gadgets.with_subspace
   pygrog.gadgets.OffResonanceCorrection
   pygrog.gadgets.OffResonanceSparseFFT

Utils
-----

Pre-processing utilities and coil calibration algorithms.

.. autosummary::
   :toctree: generated
   :nosignatures:

   pygrog.utils.coil_compress
   pygrog.utils.nlinv_calib

Interoperability
----------------

Adapters for third-party reconstruction frameworks.

.. autosummary::
   :toctree: generated
   :nosignatures:

   pygrog.interop.GrogLinop
   pygrog.interop.GrogLinearOp
