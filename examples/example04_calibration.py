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

All data use a **BrainWeb T1-weighted** phantom: a multi-coil dataset is
constructed by multiplying it with synthetic coil sensitivity maps.
"""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch


# sphinx_gallery_start_ignore
def _phase_mag_rgb(arr2d):
    """Complex 2-D array to RGB: hue=phase, value=magnitude."""
    mag = np.abs(arr2d)
    mag_norm = mag / (mag.max() + 1e-8)
    hue = (np.angle(arr2d) + np.pi) / (2.0 * np.pi)  # [0, 1]
    sat = np.ones_like(hue)
    return mcolors.hsv_to_rgb(np.stack([hue, sat, mag_norm], axis=-1))


def _add_phase_amp_rgb_legend(fig):
    """Single RGB legend for phase/magnitude map encoding.

    Horizontal axis encodes phase in [-pi, pi], vertical axis encodes
    normalized amplitude in [0, 1].
    """
    n_phase = 256
    n_amp = 256
    phase = np.linspace(-np.pi, np.pi, n_phase, dtype=np.float32)
    amp = np.linspace(0.0, 1.0, n_amp, dtype=np.float32)
    hue = (phase[np.newaxis, :] + np.pi) / (2.0 * np.pi)
    hue = np.repeat(hue, n_amp, axis=0)
    sat = np.ones((n_amp, n_phase), dtype=np.float32)
    val = np.repeat(amp[:, np.newaxis], n_phase, axis=1)
    rgb = mcolors.hsv_to_rgb(np.stack([hue, sat, val], axis=-1))

    cax = fig.add_axes([0.86, 0.18, 0.12, 0.62])
    cax.imshow(rgb, origin="lower", aspect="auto")
    cax.set_title("RGB key", fontsize=9)
    cax.set_xlabel("phase", fontsize=8)
    cax.set_ylabel("amplitude", fontsize=8)
    cax.set_xticks([0, n_phase - 1])
    cax.set_xticklabels([r"$-\pi$", r"$\pi$"], fontsize=8)
    cax.set_yticks([0, n_amp - 1])
    cax.set_yticklabels(["0", "1"], fontsize=8)


# sphinx_gallery_end_ignore

from brainweb_dl import get_mri


# sphinx_gallery_start_ignore
def _synthetic_smaps(shape, n_coils=4):
    """Synthetic sensitivity maps (Gaussian envelope + linear phase)."""
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


# sphinx_gallery_end_ignore

image = get_mri(0, "T1")

# sphinx_gallery_start_ignore
image = np.flip(image, axis=(0, 1, 2))[90].astype(np.float32)
image /= image.max() + 1e-8
image_shape = image.shape
n_coils = 16
smaps_gt = _synthetic_smaps(image_shape, n_coils=n_coils)

# Coil images: (n_coils, ny, nx)
coil_images = smaps_gt * image[np.newaxis].astype(np.complex64)

# Cartesian k-space: (n_coils, ny, nx)
kspace_full = np.fft.fftshift(
    np.fft.fft2(np.fft.ifftshift(coil_images, axes=(-2, -1))), axes=(-2, -1)
)

print(f"Coil images shape: {coil_images.shape}")
print(f"K-space shape    : {kspace_full.shape}")
# sphinx_gallery_end_ignore

# %%

fig, axes = plt.subplots(2, n_coils // 2, figsize=(11, 4))
# sphinx_gallery_start_ignore
axes = axes.flatten()
for i in range(n_coils):
    axes[i].imshow(np.abs(coil_images[i]), cmap="gray", origin="upper")
    axes[i].set_title(f"Coil {i + 1}")
    axes[i].axis("off")
plt.suptitle("Synthetic multi-coil images (16 coils)")
plt.tight_layout()
plt.show()
# sphinx_gallery_end_ignore

# %%
# Coil Compression via PCA
# =========================
#
# Use :func:`~pygrog.utils.coil_compress` to reduce the number of receiver coils
# via PCA on k-space data, reducing computational cost with minimal SNR loss.
# The function returns the compressed k-space and the compression matrix
# ``W`` of shape ``(n_virtual, n_coils)``.

# Prepare k-space data for compression
kspace_flat = kspace_full.reshape(n_coils, -1)  # (n_coils, n_samples)

# %%

n_virtual = 8


from pygrog.utils import coil_compress

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

# sphinx_gallery_start_ignore
axes[0].imshow(image, cmap="gray", origin="upper")
axes[0].set_title("Reference")
axes[0].axis("off")
axes[1].imshow(rss_orig, cmap="gray", origin="upper")
axes[1].set_title(f"RSS ({n_coils} coils)")
axes[1].axis("off")
axes[2].imshow(rss_cc, cmap="gray", origin="upper")
axes[2].set_title(f"RSS ({n_virtual} virtual coils, PCA)")
axes[2].axis("off")
plt.suptitle("Coil compression")
plt.tight_layout()
plt.show()
# sphinx_gallery_end_ignore

# %%
# .. note::
#    Coil compression is lossless when ``n_coils >= original_n_coils`` and
#    near-lossless when the retained energy fraction is high (e.g. > 0.99).
#    Use a float threshold ``coil_compress(data, 0.99)`` for automatic rank
#    selection.

# %%
# NLINV Sensitivity Estimation
# =============================
#
# :func:`~pygrog.utils.nlinv_calib` estimates coil sensitivity maps jointly
# with the image using the nonlinear-inverse (NLINV) algorithm
# (Uecker et al.\ 2008). It requires a small fully-sampled ACS centre
# to bootstrap, but no explicit prior knowledge of the sensitivity maps.
#
# The function supports two modes:
# * **``cal_width=None``** - full k-space calibrationless reconstruction
# * **``cal_width=24``** - fast calibration mode with central k-space cropping

# %%

from pygrog.utils import nlinv_calib

acs = 8  # fully-sampled ACS centre width (columns), as in MATLAB reference
cal_width = 24  # NLINV internal calibration resolution for Step 2

# Cartesian undersampling mask: R=2 readout subsampling + 8-col ACS centre
mask = np.zeros(image_shape, dtype=bool)
mask[:, ::2] = True  # R=2 subsampling on readout (column) direction
cx = image_shape[1] // 2
cy = image_shape[0] // 2
mask[cy - acs // 2 : cy + acs // 2, cx - acs // 2 : cx + acs // 2] = True  # ACS

kspace_us = kspace_full * mask[np.newaxis]

print(f"\nUndersampling factor : {mask.size / mask.sum():.2f}x")
print(f"ACS region           : all rows x {acs} cols")
print(f"Undersampled k-space : {kspace_us.shape}")

ncols_show = min(n_coils // 2, 4)

# Zero-filled RSS image from undersampled data (baseline)
coil_images_us = np.fft.fftshift(
    np.fft.ifft2(np.fft.ifftshift(kspace_us, axes=(-2, -1))), axes=(-2, -1)
)
rss_us = np.sqrt((np.abs(coil_images_us) ** 2).sum(0))


# %%
# NLINV Step 1: Full k-space calibrationless reconstruction
# ==========================================================
#
# Call :func:`~pygrog.utils.nlinv_calib` with ``cal_width=None`` to solve
# jointly for image and sensitivity maps using the entire undersampled k-space.
# cal_width=None -> full k-space, no cropping, matching MATLAB reference
smaps_full, _, image_full = nlinv_calib(
    kspace_us,
    cal_width=None,
    ndim=2,
    mask=mask,
    ret_cal=True,
    ret_image=True,
)

smaps_full_np = (
    smaps_full.numpy() if isinstance(smaps_full, torch.Tensor) else smaps_full
)
image_full_np = (
    image_full.numpy() if isinstance(image_full, torch.Tensor) else image_full
)

print(f"\n[Step 1] Smaps shape     : {smaps_full_np.shape}")
print(f"[Step 1] NLINV image shape: {image_full_np.shape}")

# %%

fig, axes = plt.subplots(3, ncols_show + 1, figsize=(3 * (ncols_show + 1), 8))
# sphinx_gallery_start_ignore

axes[0, 0].imshow(image, cmap="gray", origin="upper")
axes[0, 0].set_title("Reference")
axes[0, 0].axis("off")
for i in range(ncols_show):
    axes[0, i + 1].imshow(_phase_mag_rgb(smaps_gt[i]), origin="upper")
    axes[0, i + 1].set_title(f"GT smap {i + 1}")
    axes[0, i + 1].axis("off")

axes[1, 0].imshow(rss_us, cmap="gray", origin="upper")
axes[1, 0].set_title("RSS (zero-filled)")
axes[1, 0].axis("off")
for i in range(ncols_show):
    axes[1, i + 1].imshow(_phase_mag_rgb(smaps_full_np[i]), origin="upper")
    axes[1, i + 1].set_title(f"NLINV smap {i + 1}")
    axes[1, i + 1].axis("off")

axes[2, 0].imshow(np.abs(image_full_np), cmap="gray", origin="upper")
axes[2, 0].set_title("NLINV image")
axes[2, 0].axis("off")
for i in range(ncols_show):
    axes[2, i + 1].imshow(_phase_mag_rgb(smaps_full_np[i]), origin="upper")
    axes[2, i + 1].set_title(f"NLINV smap {i + 1}")
    axes[2, i + 1].axis("off")

plt.suptitle(
    f"NLINV - calibrationless image reconstruction ({image_shape[0]}x{image_shape[1]})"
)
plt.tight_layout(rect=(0.0, 0.0, 0.84, 1.0))
_add_phase_amp_rgb_legend(fig)
plt.show()
# sphinx_gallery_end_ignore

# %%
# NLINV Step 2: Calibration mode with central k-space cropping
# ============================================================
#
# Call :func:`~pygrog.utils.nlinv_calib` with ``cal_width=24`` to crop
# to the central region and solve for low-resolution sensitivity maps and
# a synthesized calibration k-space patch (useful for GRAPPA/GROG training).

smaps_cal, grappa_train, image_cal = nlinv_calib(
    kspace_us,
    cal_width=cal_width,
    ndim=2,
    mask=mask,
    ret_cal=True,
    ret_image=True,
)

smaps_cal_np = smaps_cal.numpy() if isinstance(smaps_cal, torch.Tensor) else smaps_cal
grappa_train_np = (
    grappa_train.numpy() if isinstance(grappa_train, torch.Tensor) else grappa_train
)
image_cal_np = image_cal.numpy() if isinstance(image_cal, torch.Tensor) else image_cal

print(f"\n[Step 2] Smaps shape       : {smaps_cal_np.shape}")
print(f"[Step 2] Low-res image shape: {image_cal_np.shape}")
print(f"[Step 2] Cal k-space shape  : {grappa_train_np.shape}")

# %%

# Low-res image and smaps at cal_width resolution

fig, axes = plt.subplots(1, ncols_show + 1, figsize=(3 * (ncols_show + 1), 3))
# sphinx_gallery_start_ignore

axes[0].imshow(np.abs(image_cal_np), cmap="gray", origin="upper")
axes[0].set_title(f"NLINV image ({cal_width}x{cal_width})")
axes[0].axis("off")
for i in range(ncols_show):
    axes[i + 1].imshow(_phase_mag_rgb(smaps_cal_np[i]), origin="upper")
    axes[i + 1].set_title(f"NLINV smap {i + 1}")
    axes[i + 1].axis("off")

plt.suptitle(f"NLINV calibration mode - low-res result ({cal_width}x{cal_width})")
plt.tight_layout(rect=(0.0, 0.0, 0.84, 1.0))
_add_phase_amp_rgb_legend(fig)
plt.show()
# sphinx_gallery_end_ignore

# %%

# Zero-filled vs synthesized central k-space patch

acr_zf = kspace_us[
    # sphinx_gallery_start_ignore
    :,
    cy - cal_width // 2 : cy + cal_width // 2,
    cx - cal_width // 2 : cx + cal_width // 2,
]

fig, axes = plt.subplots(2, ncols_show, figsize=(3 * ncols_show, 5.5))
for i in range(ncols_show):
    axes[0, i].imshow(np.log1p(np.abs(acr_zf[i])), cmap="inferno", origin="upper")
    axes[0, i].set_title(f"Zero-filled - coil {i + 1}")
    axes[0, i].axis("off")
    axes[1, i].imshow(
        np.log1p(np.abs(grappa_train_np[i])), cmap="inferno", origin="upper"
    )
    axes[1, i].set_title(f"Synthesized - coil {i + 1}")
    axes[1, i].axis("off")

plt.suptitle(
    f"Calibration k-space ({cal_width}x{cal_width}) - "
    f"zero-filled (top) vs NLINV-synthesized (bottom)"
)
plt.tight_layout()
plt.show()
# sphinx_gallery_end_ignore

# %%
# .. note::
#    In calibration mode (integer ``cal_width``) the sensitivity maps are
#    estimated at low resolution and zero-padded to the full FOV - they are
#    smoother but less accurate near the FOV boundary.  Use ``cal_width=None``
#    when full image reconstruction quality matters; use an integer
#    ``cal_width`` when only the synthesized k-space patch and fast coil
#    estimates are needed (e.g.\ as input to GROG/GRAPPA kernel training).

plt.show()


# %%
# Multi-slice batched NLINV calibration
# =====================================
#
# :func:`~pygrog.utils.nlinv_calib` accepts a leading batch axis and runs
# the calibration once per batch element.  Sensitivity maps and image
# reconstructions are returned per-slice; the synthesized GRAPPA training
# k-space can optionally be averaged across the batch with
# ``train_reduce='mean'`` to produce a single shared kernel.

batch_slices = np.stack([kspace_us, kspace_us[:, ::-1, :]], axis=0)
print(f"\n[Multi-slice] batched k-space shape : {batch_slices.shape}")

# Per-slice smaps + shared (mean-reduced) GRAPPA training k-space.
smaps_b, train_b_mean, image_b = nlinv_calib(
    batch_slices,
    cal_width=cal_width,
    ndim=2,
    ret_cal=True,
    ret_image=True,
    train_reduce="mean",
)
print(f"[Multi-slice] smaps shape           : {tuple(smaps_b.shape)}")
print(f"[Multi-slice] image shape           : {tuple(image_b.shape)}")
print(f"[Multi-slice] mean-train shape      : {tuple(train_b_mean.shape)}")

# %%

fig, axes = plt.subplots(2, ncols_show, figsize=(3 * ncols_show, 5.5))
# sphinx_gallery_start_ignore
for i in range(ncols_show):
    axes[0, i].imshow(_phase_mag_rgb(smaps_b[0, i]), origin="upper")
    axes[0, i].set_title(f"slice 0 - smap {i + 1}")
    axes[0, i].axis("off")
    axes[1, i].imshow(_phase_mag_rgb(smaps_b[1, i]), origin="upper")
    axes[1, i].set_title(f"slice 1 - smap {i + 1}")
    axes[1, i].axis("off")

plt.suptitle("Batched NLINV - per-slice smaps (shared GRAPPA training kernel)")
plt.tight_layout(rect=(0.0, 0.0, 0.84, 1.0))
_add_phase_amp_rgb_legend(fig)
plt.show()
# sphinx_gallery_end_ignore
