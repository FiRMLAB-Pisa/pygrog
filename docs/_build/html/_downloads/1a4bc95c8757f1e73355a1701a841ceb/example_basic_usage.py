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

import torch

from brainweb_dl import get_mri

from mrinufft import display_2D_trajectory, get_operator, initialize_2D_spiral
from mrinufft.density import voronoi

from pygrog.calib import GrogInterpolator
from pygrog.operator import SparseFFT


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


# %%
# BrainWeb phantom + golden-angle spiral trajectory
# =================================================
#
# Following the same setup as the mri-nufft documentation examples:
# a BrainWeb M0 slice at full resolution and a golden-angle 2D spiral.

image = get_mri(0, "T1")
image = np.flip(image, axis=(0, 2))[90].astype(np.float32)
image /= image.max() + 1e-8
shape = image.shape
n_coils = 16  # coils for simulation

samples = initialize_2D_spiral(Nc=48, Ns=600, nb_revolutions=10).astype(
    np.float32
)
density = voronoi(samples)

plt.figure()
plt.imshow(image, cmap="gray", origin="lower")
plt.title("BrainWeb M0 phantom")
plt.axis("off")
plt.tight_layout()
plt.show()

# %%

display_2D_trajectory(samples)
plt.show()

# %%
# mri-nufft: k-space simulation + reference adjoint reconstruction
# ================================================================

smaps = _synthetic_smaps(shape, n_coils=n_coils)

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

# %%
# PyGROG: calibration, gridding, reconstruction
# =============================================
#
# **Shortcut path** — ``ret_image=True`` handles the ``sqrt_weights``
# pre-multiplication internally and returns an RSS-combined image directly.
# Use this for quick one-shot reconstructions.
#
# **Explicit sparse IFFT path** — for iterative reconstruction, call
# ``interpolate(ret_image=False)`` once to obtain the raw sparse k-space, then
# pre-multiply by ``sqrt_weights`` *before* the iterative loop so that
# :class:`~pygrog.operator.SparseFFT` ``.forward`` and ``.adjoint`` satisfy the
# adjointness condition throughout.

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

grog = GrogInterpolator(shape=shape, coords=coords, kernel_width=2, oversamp=1.25, image_shape=shape)
grog.calc_interp_table(calib_cart, lamda=0.01, precision=1)

# GrogInterpolator expects (n_coils, n_shots, n_readout).
kspace_nc_shaped = kspace_nc.astype(np.complex64).reshape(n_coils, *samples.shape[:2])

# --- Shortcut: ret_image=True applies sqrt_weights internally ----------------
image_grog = grog.interpolate(kspace_nc_shaped, ret_image=True)

# --- Explicit sparse IFFT path -----------------------------------------------
# Step 1: GROG-interpolate to raw sparse Cartesian samples (no weights applied).
kspace_sparse = grog.interpolate(kspace_nc_shaped, ret_image=False)
print(f"PyGROG sparse shape : {kspace_sparse.shape}")

# Step 2: Pre-multiply by plan.pre_weights once (caller's responsibility).
#   plan.pre_weights gives sqrt(density_compensation) in the same sample order
#   as interpolate() returns — no index arithmetic required.
sqrt_w = grog.plan.pre_weights
sparse_t = torch.as_tensor(np.asarray(kspace_sparse))
sparse_weighted = sparse_t * sqrt_w.to(sparse_t.dtype).unsqueeze(0)

# Step 3: SparseFFT.forward applies sqrt_weights again → full density compensation.
op = SparseFFT(plan=grog.plan, smaps=torch.as_tensor(smaps))
image_grog_explicit = op.forward(sparse_weighted).abs().cpu().numpy()

# %%
# Comparison
# ==========
#
# Both PyGROG paths (shortcut and explicit) should match each other and the
# mri-nufft adjoint reference.

ref_abs = np.abs(image_ref)
ref_abs /= ref_abs.max() + 1e-12

grog_abs = np.abs(image_grog)
grog_abs /= grog_abs.max() + 1e-12

grog_exp_abs = image_grog_explicit
grog_exp_abs /= grog_exp_abs.max() + 1e-12

nmse_shortcut = ((grog_abs - ref_abs) ** 2).mean() / (ref_abs**2).mean()
nmse_explicit = ((grog_exp_abs - ref_abs) ** 2).mean() / (ref_abs**2).mean()
print(f"NMSE shortcut  (ret_image=True)    : {nmse_shortcut:.3e}")
print(f"NMSE explicit  (sparse IFFT path)  : {nmse_explicit:.3e}")

# Error maps in % of the nufft reference
err_rss = 100.0 * (grog_abs - ref_abs) / (ref_abs.max() + 1e-12)
err_exp = 100.0 * (grog_exp_abs - ref_abs) / (ref_abs.max() + 1e-12)
emax = max(np.abs(err_rss).max(), np.abs(err_exp).max())

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

axes[0, 0].imshow(ref_abs, cmap="gray", origin="lower")
axes[0, 0].set_xticks([])
axes[0, 0].set_yticks([])
axes[0, 0].set_title("mri-nufft (reference)")

axes[0, 1].imshow(grog_abs, cmap="gray", origin="lower")
axes[0, 1].set_xticks([])
axes[0, 1].set_yticks([])
axes[0, 1].set_title("PyGROG RSS (ret_image=True)")

axes[0, 2].imshow(grog_exp_abs, cmap="gray", origin="lower")
axes[0, 2].set_xticks([])
axes[0, 2].set_yticks([])
axes[0, 2].set_title("PyGROG full (sparse IFFT)")

axes[1, 0].axis("off")

im1 = axes[1, 1].imshow(err_rss, cmap="bwr", origin="lower", vmin=-10, vmax=10)
axes[1, 1].set_xticks([])
axes[1, 1].set_yticks([])
axes[1, 1].set_title(f"error RSS [%]  MAE={np.abs(err_rss).mean():.2f}%")
fig.colorbar(im1, ax=axes[1, 1], fraction=0.046, pad=0.04)

im2 = axes[1, 2].imshow(err_exp, cmap="bwr", origin="lower", vmin=-10, vmax=10)
axes[1, 2].set_xticks([])
axes[1, 2].set_yticks([])
axes[1, 2].set_title(f"error full [%]  MAE={np.abs(err_exp).mean():.2f}%")
fig.colorbar(im2, ax=axes[1, 2], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()


# %%
# Multi-slice batch reconstruction
# ================================
#
# Demonstrates the new multi-axis batch (``*B``) capability: three axial
# slices share the same spiral trajectory, so we can stack them along a
# leading batch axis and have :class:`~pygrog.operator.SparseFFT` and
# :class:`~pygrog.calib.GrogInterpolator` vectorise across slices in a
# single call.

# Pull three adjacent BrainWeb slices.
vol = get_mri(0, "T1")
vol = np.flip(vol, axis=(0, 2)).astype(np.float32)
vol /= vol.max() + 1e-8
slices = vol[88:91]                           # (B=3, ny, nx)
B = slices.shape[0]

# Simulate batched k-space with the same trajectory but per-slice content.
ksp_batch = np.stack(
    [
        nufft_sim.op(s.astype(np.complex64))
        for s in slices
    ],
    axis=0,
)                                              # (B, n_coils, n_samples)
ksp_batch_shaped = ksp_batch.reshape(B, n_coils, *samples.shape[:2])

# Single batched GROG interpolation.
sparse_batch = grog.interpolate(ksp_batch_shaped, ret_image=False)
sparse_batch_t = torch.as_tensor(np.asarray(sparse_batch))
sparse_batch_w = sparse_batch_t * sqrt_w.to(sparse_batch_t.dtype)

# Single batched SparseFFT recon — same operator instance handles ``*B``.
recon_batch = op.forward(sparse_batch_w).abs().cpu().numpy()

fig, axes = plt.subplots(1, B, figsize=(4 * B, 4))
for i in range(B):
    axes[i].imshow(recon_batch[i], cmap="gray", origin="lower")
    axes[i].set_xticks([])
    axes[i].set_yticks([])
    axes[i].set_title(f"slice {i}")
fig.suptitle("Multi-slice batched PyGROG reconstruction")
plt.tight_layout()
plt.show()
