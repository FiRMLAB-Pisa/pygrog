===============
Getting Started
===============

.. toctree::
   :maxdepth: 2
   :hidden:
   :titlesonly:

   self
   install


Welcome to PyGROG! This library provides efficient, GPU-accelerated
implementations of the GROG (GRAPPA Operator Gridding) algorithm for
non-Cartesian MRI reconstruction, together with a family of Cartesian
sparse-FFT operators, calibration utilities, and reconstruction gadgets.

Installation
------------

Follow the :doc:`install` guide to install PyGROG from PyPI or build from
source (required for CUDA wheels).

Using PyGROG
------------

The shortest path from raw non-Cartesian k-space to an image is:

.. code-block:: python

    import numpy as np
    from pygrog.calib import GrogInterpolator
    from pygrog.operator import SparseFFT

    # 1. Build the GROG plan from the trajectory (geometry only)
    grog = GrogInterpolator(shape=(256, 256), coords=coords)

    # 2. Fit GRAPPA kernels from the auto-calibration region
    grog.calc_interp_table(acr_data)

    # 3. Grid + reconstruct in one call
    image = grog.interpolate(kspace_nc, ret_image=True)

For more detail see the :ref:`general_examples` gallery.

What's Next?
------------

- Explore the :ref:`general_examples` section for practical, runnable examples.
- Read the :doc:`api` for a complete reference of all classes and functions.
- Visit :doc:`explanations/index` to learn about the GRAPPA/GROG theory and
  model extensions (parallel imaging, off-resonance, subspace).
