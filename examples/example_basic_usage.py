"""
=======================================
Basic Usage: SparseFFT and GROG Gridding
=======================================

This example walks through the fundamental PyGROG building blocks:

1. **SparseFFT** — Cartesian-undersampled forward (image → k-space) and
   adjoint (k-space → image) operators for use in compressed-sensing MRI.
2. **GrogInterpolator** — non-Cartesian to Cartesian gridding via the
   GROG (GRAPPA Operator Gridding) algorithm.

For reproducibility all data are **synthetic**: a small Shepp-Logan-like
phantom is used as the reference image throughout.

.. note::

   PyGROG and `mri-nufft <https://mind-inria.github.io/mri-nufft/>`_ are
   complementary tools for non-Cartesian MRI.  mri-nufft provides general
   NUFFT backends (FINUFFT, cuFINUFFT, …) through a unified interface and
   should be preferred when a standard gridding/degridding NUFFT is needed.
   PyGROG provides a *data-driven* alternative: the GROG algorithm estimates
   the gridding operator from the k-space data themselves (via GRAPPA kernels),
   which avoids the need for a density-compensation function and can improve
   reconstruction accuracy in highly accelerated settings.  See
   :ref:`sphx_glr_generated_autoexamples_example_interop.py` for a side-by-side
   comparison using the mri-nufft interoperability layer.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

# %%
# Reference phantom
# =================
#
# We construct a tiny 2-D Shepp-Logan phantom by superimposing three
# ellipses.  All values are in ``[0, 1]``.


def _phantom(shape):
    """Simple 2-D phantom from overlapping ellipses."""
    ny, nx = shape
    yy, xx = np.mgrid[-1 : 1 : ny * 1j, -1 : 1 : nx * 1j]
    img = np.zeros(shape, dtype=np.float32)
    img += 1.0 * ((xx / 0.9) ** 2 + (yy / 0.9) ** 2 < 1)
    img += 0.4 * ((xx / 0.6) ** 2 + ((yy - 0.1) / 0.7) ** 2 < 1)
    img -= 0.6 * ((xx / 0.15) ** 2 + ((yy + 0.2) / 0.2) ** 2 < 1)
    return img.clip(0, 1)


image_shape = (64, 64)
image_ref = _phantom(image_shape)

fig, ax = plt.subplots()
ax.imshow(image_ref, cmap="gray", origin="lower")
ax.set_title("Reference phantom")
ax.axis("off")
plt.tight_layout()
plt.show()

# %%
# SparseFFT: forward and adjoint
# ==============================
#
# :class:`~pygrog.operator.SparseFFT` models a Cartesian k-space acquisition
# where only a *sparse* subset of grid positions is sampled — as in
# compressed-sensing MRI.  It is parameterised by:
#
# * ``grid_shape`` — oversampled Cartesian FFT grid (e.g. 1.5x the image).
# * ``image_shape`` — the reconstruction target (centre-cropped from grid).
# * ``indices`` — flat linear indices of the *sampled* Cartesian k-space
#   positions (``int64``).
# * ``weights`` — density-compensation weights (``float32``); ``sqrt(w)`` is
#   applied symmetrically in both directions.
#
# **Forward direction** (``SparseFFT.forward``):
#   sparse k-space ``(n_coils, n_samples)`` → image ``(*image_shape)``
#
# **Adjoint direction** (``SparseFFT.adjoint``):
#   image ``(*image_shape)`` → sparse k-space ``(n_coils, n_samples)``

from pygrog.operator import SparseFFT

# 1.5x oversampled Cartesian grid
grid_shape = (96, 96)

# Random undersampled Cartesian mask (20% of k-space)
rng = np.random.default_rng(42)
grid_size = grid_shape[0] * grid_shape[1]
n_samples = grid_size // 5  # 20 % undersampling
indices = rng.integers(0, grid_size, size=n_samples).astype(np.int64)

# Uniform weights (no density compensation for now)
weights = np.ones(n_samples, dtype=np.float32)

op = SparseFFT(grid_shape, image_shape, indices, weights)

print(f"Grid shape  : {op.grid_shape}")
print(f"Image shape : {op.image_shape}")
print(f"Samples     : {op.indices.shape[0]}")

# %%
# Adjoint: image → sparse k-space
# --------------------------------
#
# :meth:`~pygrog.operator.SparseFFT.adjoint` zero-pads the image to
# ``grid_shape``, applies the centred FFT, and then *gathers* the values at
# the sampled positions.

image_t = torch.as_tensor(image_ref).to(torch.complex64)
kspace = op.adjoint(image_t)  # (1, n_samples)  — 1 coil (no smaps)

print(f"\nAdjoint output shape: {kspace.shape}")

# %%
# Forward: sparse k-space → image
# --------------------------------
#
# :meth:`~pygrog.operator.SparseFFT.forward` *scatters* the k-space values
# into the oversampled Cartesian grid, applies the centred IFFT, and
# centre-crops the result to ``image_shape``.

image_recon = op.forward(kspace)  # (*image_shape)

print(f"Forward output shape: {image_recon.shape}")

# %%

fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))

axes[0].imshow(image_ref, cmap="gray", origin="lower")
axes[0].set_title("Reference")
axes[0].axis("off")

axes[1].imshow(image_recon.abs().numpy(), cmap="gray", origin="lower")
axes[1].set_title("Adjoint ∘ Adjoint (20% k-space)")
axes[1].axis("off")

error = image_recon.abs().numpy() - image_ref
axes[2].imshow(error, cmap="bwr", origin="lower", vmin=-0.3, vmax=0.3)
axes[2].set_title("Error")
axes[2].axis("off")

plt.suptitle("SparseFFT round-trip (adjoint then forward)")
plt.tight_layout()
plt.show()

# %%
# .. note::
#    ``SparseFFT.forward(SparseFFT.adjoint(x))`` is **not** the identity.
#    It is the *normal operator* ``A^H A x``.  Without density compensation
#    the reconstruction suffers from k-space weighting artefacts.  An
#    iterative CG-SENSE solver (e.g. via the sigpy interop) removes them.

# %%
# Adjointness verification
# ------------------------
#
# A dot-product test confirms ``<A x, y> = <x, A^H y>`` to numerical
# precision, which is required for the operator to be a valid linear map.

x = torch.randn(*image_shape, dtype=torch.complex64)
y = torch.randn(1, op.indices.shape[0], dtype=torch.complex64)

Ax = op.adjoint(x)
AHy = op.forward(y)

lhs = torch.vdot(Ax.flatten(), y.flatten()).real
rhs = torch.vdot(x.flatten(), AHy.flatten()).real

print("\nAdjointness check:")
print(f"  <A x, y>  = {lhs:.6f}")
print(f"  <x, A^H y>= {rhs:.6f}")
print(f"  Rel. error: {abs((lhs - rhs) / (abs(lhs) + 1e-12)):.2e}")

# %%
# Multi-coil reconstruction (with sensitivity maps)
# -------------------------------------------------
#
# When ``smaps`` is provided the forward operator applies coil-combination
# (conjugate smaps multiply-accumulate) and the adjoint applies coil
# expansion.  Here we use synthetic sensitivity maps.

n_coils = 4
smaps = torch.randn(n_coils, *image_shape, dtype=torch.complex64)
# Normalise so sum-of-squares across coils is 1
smaps = smaps / smaps.abs().square().sum(0, keepdim=True).sqrt()

op_mc = SparseFFT(grid_shape, image_shape, indices, weights, smaps=smaps)

kspace_mc = op_mc.adjoint(image_t)  # (n_coils, n_samples)
image_mc = op_mc.forward(kspace_mc)  # (*image_shape)  — coil-combined

print(f"\nMulti-coil adjoint shape: {kspace_mc.shape}")
print(f"Multi-coil forward shape: {image_mc.shape}")

# %%

fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
axes[0].imshow(image_ref, cmap="gray", origin="lower")
axes[0].set_title("Reference")
axes[0].axis("off")
axes[1].imshow(image_mc.abs().numpy(), cmap="gray", origin="lower")
axes[1].set_title(f"Multi-coil ({n_coils} coils), 20% k-space")
axes[1].axis("off")
plt.tight_layout()
plt.show()

# %%
# GROG interpolator (non-Cartesian → Cartesian)
# =============================================
#
# :class:`~pygrog.calib.GrogInterpolator` maps *non-Cartesian* k-space
# samples onto the Cartesian grid using GRAPPA kernels trained from the
# auto-calibration region (ACR).
#
# Workflow:
#
# 1. Build the plan from the trajectory (geometry only, no data).
# 2. Call :meth:`~pygrog.calib.GrogInterpolator.calc_interp_table` with ACR
#    data to fit the GRAPPA kernels.
# 3. Call :meth:`~pygrog.calib.GrogInterpolator.interpolate` on each
#    non-Cartesian dataset to obtain a Cartesian-gridded result, or pass
#    ``ret_image=True`` to get the coil-combined image directly.

from pygrog.calib import GrogInterpolator

# Synthetic 2-D golden-angle radial trajectory
n_spokes = 32
n_pts = 96
golden_angle = 111.246

angles = np.deg2rad(golden_angle * np.arange(n_spokes))
readout = np.linspace(-0.5, 0.5 - 1 / n_pts, n_pts)

kx = np.outer(np.cos(angles), readout) * image_shape[1]
ky = np.outer(np.sin(angles), readout) * image_shape[0]
coords = np.stack([kx, ky], axis=-1).astype(np.float32)  # (n_spokes, n_pts, 2)

# Build the plan (geometry only)
grog = GrogInterpolator(shape=image_shape, coords=coords, kernel_width=2)

print(f"\nGROG grid shape : {grog.plan.grid_shape}")
print(f"GROG image shape: {grog.plan.image_shape}")

# Simulate single-coil Cartesian ACR data (usually from scanner)
cal_kspace = np.fft.fftshift(np.fft.fft2(image_ref))
cal_data = cal_kspace[np.newaxis].astype(np.complex64)  # (1, ny, nx)

grog.calc_interp_table(cal_data, lamda=0.01, precision=1)
print("GROG kernels fitted.")

# Sample the phantom at trajectory positions (bilinear lookup)
ny, nx = image_shape
kx_idx = (coords[..., 0] + nx // 2).clip(0, nx - 1).astype(int)
ky_idx = (coords[..., 1] + ny // 2).clip(0, ny - 1).astype(int)
kspace_nc = cal_kspace[ky_idx, kx_idx].astype(np.complex64)[np.newaxis]
# add noise
kspace_nc = kspace_nc + 0.01 * (
    rng.standard_normal(kspace_nc.shape).astype(np.float32)
    + 1j * rng.standard_normal(kspace_nc.shape).astype(np.float32)
)

# GROG gridding and immediate image reconstruction
image_grog = grog.interpolate(kspace_nc, ret_image=True)  # (*image_shape) float32

print(f"GROG image shape: {image_grog.shape}")

# %%

fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
axes[0].imshow(image_ref, cmap="gray", origin="lower")
axes[0].set_title("Reference")
axes[0].axis("off")
axes[1].imshow(image_grog, cmap="gray", origin="lower")
axes[1].set_title("GROG gridded + IFFT (radial, 1 coil)")
axes[1].axis("off")
plt.tight_layout()
plt.show()

# %%
# .. note::
#    GROG performs a *non-iterative* gridding reconstruction.  For high
#    undersampling factors or noisy acquisitions, combine it with iterative
#    compressed-sensing reconstruction on top of :class:`~pygrog.operator.SparseFFT`.

plt.show()
