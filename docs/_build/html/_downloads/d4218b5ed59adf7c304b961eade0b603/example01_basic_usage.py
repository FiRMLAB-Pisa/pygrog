"""
============================================
Basic Usage: mri-nufft Baseline vs PyGROG
============================================

This example follows a realistic non-Cartesian pipeline:

1. Use ``brainweb-dl`` to load a brain phantom.
2. Use ``mri-nufft`` to generate the trajectory, sensitivity maps,
   non-Cartesian k-space data, and a reference adjoint reconstruction.
3. Use :class:`~pygrog.calib.GrogInterpolator` to grid and reconstruct,
   then compare PyGROG against the mri-nufft reference.
"""

import matplotlib.pyplot as plt
import numpy as np

from brainweb_dl import get_mri

from mrinufft import get_operator, initialize_2D_spiral
from mrinufft.density import voronoi

from pygrog.calib import GrogInterpolator
from pygrog.operator import MaskedFFT, SparseFFT


# sphinx_gallery_start_ignore
def _synthetic_smaps(shape, n_coils=4):
    """Create smooth multi-coil maps used to simulate acquisition."""
    ny, nx = shape
    yy, xx = np.mgrid[-1 : 1 : ny * 1j, -1 : 1 : nx * 1j]
    smaps = []
    for angle in np.linspace(0.0, 2.0 * np.pi, n_coils, endpoint=False):
        cx, cy = np.cos(angle), np.sin(angle)
        gauss = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * 0.45**2))
        phase = np.exp(1j * (cx * xx + cy * yy))
        smaps.append(gauss * phase)
    smaps = np.asarray(smaps, dtype=np.complex64)
    smaps /= np.sqrt((np.abs(smaps) ** 2).sum(0, keepdims=True)) + 1e-12
    return smaps


def _normalize_unit(image):
    return image / (image.max() + 1e-12)


def _format_panel(ax, image, title, *, cmap="gray", vmin=None, vmax=None):
    ax.imshow(image, cmap=cmap, origin="upper", vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
# sphinx_gallery_end_ignore


# %%
image = get_mri(0, "T1")
# sphinx_gallery_start_ignore
image = np.flip(image, axis=(0, 1, 2))[90].astype(np.float32)
image /= image.max() + 1e-8
shape = image.shape
n_coils = 16  # coils for simulation

samples = initialize_2D_spiral(Nc=48, Ns=600, nb_revolutions=10).astype(np.float32)
density = voronoi(samples)
# sphinx_gallery_end_ignore

# %%
smaps = _synthetic_smaps(shape, n_coils=n_coils)
# sphinx_gallery_start_ignore

nufft_sim = get_operator("finufft")(
    samples=samples,
    shape=shape,
    n_coils=n_coils,
    smaps=smaps,
    density=density,
    squeeze_dims=True,
)
kspace_nc = nufft_sim.op(image.astype(np.complex64))  # (n_coils, n_samples)
nufft_ref = get_operator("finufft")(
    samples=samples,
    shape=shape,
    n_coils=n_coils,
    smaps=smaps,
    density=density,
    squeeze_dims=True,
)
image_ref = nufft_ref.adj_op(kspace_nc)

print(f"k-space shape   : {kspace_nc.shape}")
print(f"image shape     : {shape}")
# sphinx_gallery_end_ignore

# %%
# PyGROG calibration setup
# ========================
#
# First, prepare the calibration data and initialize the GROG interpolator.
# This is the setup phase — the actual API calls are shown in the next cells.

# Calibrate GROG from the 24x24 k-space centre of compressed coil images.
# Using the full k-space degrades GRAPPA conditioning; the low-frequency
# centre is all that is needed to estimate the GRAPPA operators.
coil_calib = smaps * image[None, ...]
calib_cart_full = np.fft.fftshift(
    np.fft.fftn(np.fft.ifftshift(coil_calib, axes=(-2, -1)), axes=(-2, -1)),
    axes=(-2, -1),
).astype(np.complex64)
calib_size = 24
cy, cx = shape[0] // 2, shape[1] // 2
calib_cart = calib_cart_full[
    :,
    cy - calib_size // 2 : cy + calib_size // 2,
    cx - calib_size // 2 : cx + calib_size // 2,
]

# mri-nufft coordinates are in [-0.5, 0.5): scale to PyGROG grid units.
coords = (samples * np.asarray(shape, dtype=np.float32)).astype(np.float32)

# %%
# GROG Calibration
# ================
#
# Initialize :class:`~pygrog.calib.GrogInterpolator` with the calibration
# region and compute the GRAPPA operators via FFT-based kernel estimation.

grog = GrogInterpolator(
    shape=shape, coords=coords, kernel_width=2, oversamp=1.25, image_shape=shape
)
grog.calc_interp_table(calib_cart, lamda=0.01, precision=1)

# GrogInterpolator expects (n_coils, n_shots, n_readout).
kspace_nc_shaped = kspace_nc.astype(np.complex64).reshape(n_coils, *samples.shape[:2])

print(f"GROG initialized with {kspace_nc_shaped.shape[0]} coils")

# %%
# Path 1: Shortcut reconstruction (``ret_image=True``)
# ====================================================
#
# The simplest path: :meth:`~pygrog.calib.GrogInterpolator.interpolate`
# with ``ret_image=True`` handles density compensation internally and
# returns an RSS-combined image directly.

image_grog = grog.interpolate(kspace_nc_shaped, ret_image=True)
print(f"PyGROG shortcut image shape : {image_grog.shape}")

# %%
# Path 2: Explicit sparse IFFT path
# ==================================
#
# For iterative reconstruction or when you need the raw sparse k-space,
# call :meth:`~pygrog.calib.GrogInterpolator.interpolate` with
# ``ret_image=False`` to get the raw sparse samples, then pre-multiply
# by ``sqrt_weights`` before using :class:`~pygrog.operator.SparseFFT`.
#
# This ensures :class:`~pygrog.operator.SparseFFT` ``.forward`` and
# ``.adjoint`` satisfy the adjointness condition throughout your
# iterative reconstruction.

# Step 1: GROG-interpolate to raw sparse Cartesian samples (no weights applied).
kspace_sparse = grog.interpolate(kspace_nc_shaped, ret_image=False)
print(f"PyGROG sparse shape : {kspace_sparse.shape}")

# Step 2: Pre-multiply by plan.pre_weights once (caller's responsibility).
#   plan.pre_weights gives sqrt(density_compensation) in the same sample order
#   as interpolate() returns — no index arithmetic required.
sqrt_w = np.asarray(grog.plan.pre_weights)
sparse_weighted = kspace_sparse * sqrt_w[np.newaxis]

# Step 3: SparseFFT.adjoint applies sqrt_weights again → full density compensation.
op = SparseFFT(plan=grog.plan, smaps=smaps)
image_grog_explicit = np.abs(op.adjoint(sparse_weighted))

# %%
# Compare both PyGROG paths
# =========================
#
# Both paths should give nearly identical results. Here we measure
# NMSE against the mri-nufft reference adjoint reconstruction.

ref_abs = np.abs(image_ref)
grog_abs = np.abs(image_grog)
grog_exp_abs = image_grog_explicit

nmse_shortcut = ((grog_abs - ref_abs) ** 2).mean() / (ref_abs**2).mean()
nmse_explicit = ((grog_exp_abs - ref_abs) ** 2).mean() / (ref_abs**2).mean()
print(f"NMSE shortcut  (ret_image=True)    : {nmse_shortcut:.3e}")
print(f"NMSE explicit  (sparse IFFT path)  : {nmse_explicit:.3e}")
# sphinx_gallery_start_ignore
truth_abs = np.abs(image)
truth_norm = _normalize_unit(truth_abs)
ref_norm = _normalize_unit(ref_abs)
grog_norm = _normalize_unit(grog_exp_abs)

fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

_format_panel(axes[0], truth_norm, "Ground truth", vmin=0.0, vmax=1.0)
_format_panel(axes[1], ref_norm, "NUFFT", vmin=0.0, vmax=1.0)
_format_panel(axes[2], grog_norm, "PyGROG", vmin=0.0, vmax=1.0)

plt.show()
# sphinx_gallery_end_ignore
# %%
# Path 3: Dense-grid path with MaskedFFT
# =======================================
#
# Use :meth:`~pygrog.calib.GrogInterpolator.interpolate` with ``grid=True``
# to scatter density-compensated sparse samples onto the full oversampled
# Cartesian grid, then use :class:`~pygrog.operator.MaskedFFT` for standard
# FFT-based reconstruction. The ``mask`` and ``density`` tensors are passed
# directly to :class:`~pygrog.operator.MaskedFFT`, which performs
# density-compensated reconstruction using masked FFTs instead of NUFFT.

grid_kspace, masked_plan = grog.interpolate(kspace_nc_shaped, grid=True)

print(f"gridded k-space shape : {grid_kspace.shape}")
print(f"masked_plan           : {masked_plan}")
print(f"fraction of mask set  : {masked_plan.mask.float().mean().item():.1%}")

# Build the MaskedFFT operator from the plan — same one-liner as SparseFFT:
#   sparse = grog.interpolate(kspace)
#   op = SparseFFT(plan=grog.plan, smaps=smaps)
#
#   kgrid, plan = grog.interpolate(kspace, grid=True)
#   op = MaskedFFT(plan=plan, smaps=smaps)
masked_fft = MaskedFFT(plan=masked_plan, smaps=smaps)

# Adjoint pass: gridded k-space → SENSE-combined image.
image_masked = np.abs(masked_fft.adjoint(grid_kspace))

nmse_masked = ((image_masked - ref_abs) ** 2).mean() / (ref_abs**2).mean()
print(f"NMSE MaskedFFT path   : {nmse_masked:.3e}")

# sphinx_gallery_start_ignore
truth_masked_norm = _normalize_unit(truth_abs)
ref_masked_norm = _normalize_unit(ref_abs)
masked_norm = _normalize_unit(image_masked)

fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
_format_panel(axes[0], truth_masked_norm, "Ground truth", vmin=0.0, vmax=1.0)
_format_panel(axes[1], ref_masked_norm, "NUFFT", vmin=0.0, vmax=1.0)
_format_panel(axes[2], masked_norm, "PyGROG MaskedFFT", vmin=0.0, vmax=1.0)

plt.show()
# sphinx_gallery_end_ignore
