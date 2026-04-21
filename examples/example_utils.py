"""
===========================================
Utils Tour: Coil Compression and NLINV
===========================================

This example introduces two utility routines from :mod:`pygrog.utils`:

1. **Coil compression** (:func:`~pygrog.utils.coil_compress`) — reduces the
   number of receiver coils via PCA to speed up downstream processing without
   significant SNR loss.
2. **NLINV coil calibration** (:func:`~pygrog.utils.nlinv_calib`) — estimates
   coil sensitivity maps from undersampled k-space data using the
   nonlinear-inverse (NLINV) algorithm.

All data are **synthetic**: a multi-coil phantom is constructed by multiplying
a reference image with smooth sensitivity maps.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

# %%
# Synthetic multi-coil dataset
# ============================
#
# We build a simple 2-D phantom and multiply it by four synthetic coil
# sensitivity maps (smooth Gaussian weighting centred at each coil
# position).


def _phantom(shape):
    """Tiny 2-D Shepp-Logan phantom."""
    ny, nx = shape
    yy, xx = np.mgrid[-1 : 1 : ny * 1j, -1 : 1 : nx * 1j]
    img = np.zeros(shape, dtype=np.float32)
    img += 1.0 * ((xx / 0.9) ** 2 + (yy / 0.9) ** 2 < 1)
    img += 0.4 * ((xx / 0.6) ** 2 + ((yy - 0.1) / 0.7) ** 2 < 1)
    img -= 0.6 * ((xx / 0.15) ** 2 + ((yy + 0.2) / 0.2) ** 2 < 1)
    return img.clip(0, 1)


def _smooth_smaps(n_coils, shape, sigma=0.5):
    """Gaussian-weighted synthetic sensitivity maps."""
    ny, nx = shape
    yy, xx = np.mgrid[-1 : 1 : ny * 1j, -1 : 1 : nx * 1j]
    # coil positions evenly spaced on a unit circle
    angles = 2 * np.pi * np.arange(n_coils) / n_coils
    cx = np.cos(angles)
    cy = np.sin(angles)
    smaps = np.zeros((n_coils, ny, nx), dtype=np.complex64)
    for i in range(n_coils):
        r2 = (xx - cx[i]) ** 2 + (yy - cy[i]) ** 2
        smaps[i] = np.exp(-r2 / (2 * sigma**2)).astype(np.complex64)
    # Normalise so that sum-of-squares is 1
    sos = np.sqrt((np.abs(smaps) ** 2).sum(axis=0, keepdims=True))
    smaps /= np.where(sos > 0, sos, 1.0)
    return smaps


image_shape = (48, 48)
n_coils = 8
phantom = _phantom(image_shape)
smaps_gt = _smooth_smaps(n_coils, image_shape)

# Coil images: (n_coils, ny, nx)
coil_images = smaps_gt * phantom[np.newaxis].astype(np.complex64)

# Cartesian k-space: (n_coils, ny, nx)
kspace_full = np.fft.fftshift(
    np.fft.fft2(np.fft.ifftshift(coil_images, axes=(-2, -1))), axes=(-2, -1)
)

print(f"Coil images shape: {coil_images.shape}")
print(f"K-space shape    : {kspace_full.shape}")

# %%

fig, axes = plt.subplots(2, n_coils // 2, figsize=(11, 4))
axes = axes.flatten()
for i in range(n_coils):
    axes[i].imshow(np.abs(coil_images[i]), cmap="gray", origin="lower")
    axes[i].set_title(f"Coil {i + 1}")
    axes[i].axis("off")
plt.suptitle("Synthetic multi-coil images (8 coils)")
plt.tight_layout()
plt.show()

# %%
# Coil compression
# ================
#
# :func:`~pygrog.utils.coil_compress` reduces the number of virtual coils
# using PCA on the k-space data.  It returns the compressed k-space and the
# compression matrix ``W`` of shape ``(n_virtual, n_coils)``.
#
# The argument ``n_coils`` accepts:
#
# * an **integer** — exact number of virtual coils to keep, or
# * a **float** in (0, 1] — energy fraction to retain.

from pygrog.utils import coil_compress

n_virtual = 4

kspace_flat = kspace_full.reshape(n_coils, -1)  # (n_coils, n_samples)

kspace_cc, W = coil_compress(kspace_flat, n_virtual)

print(f"\nOriginal coils   : {kspace_flat.shape[0]}")
print(f"Virtual coils    : {kspace_cc.shape[0]}")
print(f"Compression matrix: {W.shape}")

# Reconstruct coil images from compressed data
kspace_cc_full = kspace_cc.reshape(n_virtual, *image_shape)
coil_images_cc = np.fft.fftshift(
    np.fft.ifft2(np.fft.ifftshift(kspace_cc_full, axes=(-2, -1))), axes=(-2, -1)
)

# RSS of compressed coils vs original
rss_orig = np.sqrt((np.abs(coil_images) ** 2).sum(0))
rss_cc = np.sqrt((np.abs(coil_images_cc) ** 2).sum(0))

# %%

fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
axes[0].imshow(phantom, cmap="gray", origin="lower")
axes[0].set_title("Reference")
axes[0].axis("off")
axes[1].imshow(rss_orig, cmap="gray", origin="lower")
axes[1].set_title(f"RSS ({n_coils} coils)")
axes[1].axis("off")
axes[2].imshow(rss_cc, cmap="gray", origin="lower")
axes[2].set_title(f"RSS ({n_virtual} virtual coils, PCA)")
axes[2].axis("off")
plt.suptitle("Coil compression")
plt.tight_layout()
plt.show()

# %%
# .. note::
#    Coil compression is lossless when ``n_coils >= original_n_coils`` and
#    near-lossless when the retained energy fraction is high (e.g. > 0.99).
#    Use a float threshold ``coil_compress(data, 0.99)`` for automatic rank
#    selection.

# %%
# NLINV coil sensitivity estimation
# ==================================
#
# :func:`~pygrog.utils.nlinv_calib` estimates coil sensitivity maps from a
# Cartesian **undersampled** acquisition using the nonlinear-inverse (NLINV)
# algorithm (Uecker et al.\ 2008).  It alternates between estimating the image
# and the sensitivities, coupled by a Sobolev-space regulariser on the maps.
#
# The function returns ``(smaps, image)`` when ``ret_image=True``, or just
# ``smaps`` otherwise.

from pygrog.utils import nlinv_calib

# Cartesian undersampling mask: keep centre lines + random outer k-space
rng = np.random.default_rng(1)
mask = np.zeros(image_shape, dtype=bool)
cal = 8  # full calibration region
mask[image_shape[0] // 2 - cal : image_shape[0] // 2 + cal, :] = True
rand_rows = rng.choice(image_shape[0], size=image_shape[0] // 4, replace=False)
mask[rand_rows, :] = True

# Apply mask to k-space
kspace_us = kspace_full * mask[np.newaxis]

print(f"\nUndersampling factor: {mask.size / mask.sum():.1f}x")
print(f"Undersampled k-space shape: {kspace_us.shape}")

# %%

# NLINV calibration (Cartesian, pass mask and ndim)
smaps_nlinv, image_nlinv = nlinv_calib(
    kspace_us,
    ndim=2,
    mask=mask,
    max_iter=8,
    cg_iter=5,
    ret_image=True,
)

print(f"\nEstimated smaps shape: {smaps_nlinv.shape}")
print(f"Reconstructed image shape: {image_nlinv.shape}")

# %%

fig, axes = plt.subplots(2, n_coils // 2 + 1, figsize=(11, 5))
axes = axes.flatten()

axes[0].imshow(phantom, cmap="gray", origin="lower")
axes[0].set_title("Reference")
axes[0].axis("off")

for i in range(n_coils // 2):
    axes[i + 1].imshow(np.abs(smaps_gt[i]), cmap="magma", origin="lower", vmin=0)
    axes[i + 1].set_title(f"GT smap {i + 1}")
    axes[i + 1].axis("off")

smaps_nlinv_np = (
    smaps_nlinv.numpy() if isinstance(smaps_nlinv, torch.Tensor) else smaps_nlinv
)
axes[n_coils // 2 + 1].imshow(np.abs(image_nlinv), cmap="gray", origin="lower")
axes[n_coils // 2 + 1].set_title("NLINV image")
axes[n_coils // 2 + 1].axis("off")

for i in range(n_coils // 2):
    axes[n_coils // 2 + 2 + i].imshow(
        np.abs(smaps_nlinv_np[i]), cmap="magma", origin="lower", vmin=0
    )
    axes[n_coils // 2 + 2 + i].set_title(f"NLINV smap {i + 1}")
    axes[n_coils // 2 + 2 + i].axis("off")

plt.suptitle("NLINV coil sensitivity estimation")
plt.tight_layout()
plt.show()

# %%
# .. note::
#    NLINV is an iterative method; accuracy improves with more ``max_iter``
#    and ``cg_iter`` at the cost of compute time.  The Sobolev-space
#    regulariser (``sobolev_width``, ``sobolev_deg``) enforces smooth
#    sensitivity maps — increase ``sobolev_deg`` for smoother maps at the
#    cost of accuracy near the FOV boundary.

plt.show()
