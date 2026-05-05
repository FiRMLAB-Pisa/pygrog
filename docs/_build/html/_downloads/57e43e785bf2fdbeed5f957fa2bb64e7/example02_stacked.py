"""
=========================================================
Batched and Stacked Reconstructions with PyGROG
=========================================================

This example focuses on stack/batch behavior and trajectory handling.

We demonstrate three cases:

1. Same trajectory for all images in a stack (vectorized in one call).
2. Different trajectory for each image in the stack (one plan per image).
3. Two stack axes: one shared trajectory axis and one per-image trajectory axis.
"""

import matplotlib.pyplot as plt
import numpy as np

from brainweb_dl import get_mri

from mrinufft import get_operator, initialize_2D_spiral
from mrinufft.density import voronoi

from pygrog.calib import GrogInterpolator
from pygrog.operator import SparseFFT

# sphinx_gallery_start_ignore
def _synthetic_smaps(shape, n_coils=4):
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


def _calib_from_coil_images(coil_images, calib_size=24):
    calib_cart_full = np.fft.fftshift(
        np.fft.fftn(np.fft.ifftshift(coil_images, axes=(-2, -1)), axes=(-2, -1)),
        axes=(-2, -1),
    ).astype(np.complex64)
    ny, nx = coil_images.shape[-2:]
    cy, cx = ny // 2, nx // 2
    return calib_cart_full[
        :,
        cy - calib_size // 2 : cy + calib_size // 2,
        cx - calib_size // 2 : cx + calib_size // 2,
    ]


def _normalize(x):
    return x / (x.max() + 1e-12)


def _rotation_matrix(theta):
    return np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=np.float32,
    )


def _simulate_case2_inputs(images_case2, angles, samples_base, shape, n_coils, smaps):
    b_size = images_case2.shape[0]
    n_shots, n_read = samples_base.shape[:2]
    ksp_case2 = np.zeros((b_size, n_coils, n_shots, n_read), dtype=np.complex64)
    coords_case2 = np.zeros((b_size, n_shots, n_read, 2), dtype=np.float32)

    for b, theta in enumerate(angles):
        rot = _rotation_matrix(float(theta))
        samples_b = (samples_base.reshape(-1, 2) @ rot.T).reshape(n_shots, n_read, 2)
        density_b = voronoi(samples_b)
        nufft_b = get_operator("finufft")(
            samples=samples_b,
            shape=shape,
            n_coils=n_coils,
            smaps=smaps,
            density=density_b,
            squeeze_dims=True,
        )
        ksp_case2[b] = nufft_b.op(images_case2[b].astype(np.complex64)).reshape(
            n_coils, n_shots, n_read
        )
        coords_case2[b] = (samples_b * np.asarray(shape, dtype=np.float32)).astype(
            np.float32
        )

    return ksp_case2, coords_case2


def _simulate_case3_inputs(images_case3, angles_b, samples_base, shape, n_coils, smaps):
    t_size, b_size = images_case3.shape[:2]
    n_shots, n_read = samples_base.shape[:2]
    ksp_case3 = np.zeros((t_size, b_size, n_coils, n_shots, n_read), dtype=np.complex64)
    coords_case3 = np.zeros((b_size, n_shots, n_read, 2), dtype=np.float32)

    for b, theta in enumerate(angles_b):
        rot = _rotation_matrix(float(theta))
        samples_b = (samples_base.reshape(-1, 2) @ rot.T).reshape(n_shots, n_read, 2)
        density_b = voronoi(samples_b)
        nufft_b = get_operator("finufft")(
            samples=samples_b,
            shape=shape,
            n_coils=n_coils,
            smaps=smaps,
            density=density_b,
            squeeze_dims=True,
        )
        ksp_case3[:, b] = np.stack(
            [nufft_b.op(images_case3[t, b].astype(np.complex64)) for t in range(t_size)],
            axis=0,
        ).reshape(t_size, n_coils, n_shots, n_read)
        coords_case3[b] = (samples_b * np.asarray(shape, dtype=np.float32)).astype(
            np.float32
        )

    return ksp_case3, coords_case3
# sphinx_gallery_end_ignore


# %%
vol = get_mri(0, "T1")
# sphinx_gallery_start_ignore
vol = np.flip(vol, axis=(0, 1, 2)).astype(np.float32)
vol /= vol.max() + 1e-8

shape = vol[90].shape
n_coils = 12
smaps = _synthetic_smaps(shape, n_coils=n_coils)

# Calibration patch is trajectory-independent, reused by all plans.
coil_calib = smaps * vol[90][None, ...]
calib_cart = _calib_from_coil_images(coil_calib, calib_size=24)

# A base spiral with fixed (n_shots, n_read) for all scenarios.
samples_base = initialize_2D_spiral(Nc=32, Ns=500, nb_revolutions=8).astype(np.float32)
n_shots, n_read = samples_base.shape[:2]
# sphinx_gallery_end_ignore


# %%
# Case 1: Shared trajectory across one stack axis
# ================================================
#
# When all images in a stack use the same trajectory, we can initialize a
# single :class:`~pygrog.calib.GrogInterpolator` and call
# :meth:`~pygrog.calib.GrogInterpolator.interpolate` once with batch dimension.

B = 3
images_case1 = vol[88 : 88 + B]

density_base = voronoi(samples_base)
nufft_shared = get_operator("finufft")(
    samples=samples_base,
    shape=shape,
    n_coils=n_coils,
    smaps=smaps,
    density=density_base,
    squeeze_dims=True,
)

ksp_case1 = np.stack(
    [nufft_shared.op(images_case1[b].astype(np.complex64)) for b in range(B)],
    axis=0,
)  # (B, C, n_samples)
ksp_case1 = ksp_case1.reshape(B, n_coils, n_shots, n_read)


coords_base = (samples_base * np.asarray(shape, dtype=np.float32)).astype(np.float32)
grog_shared = GrogInterpolator(
    shape=shape,
    coords=coords_base,
    kernel_width=2,
    oversamp=1.25,
    image_shape=shape,
)
grog_shared.calc_interp_table(calib_cart, lamda=0.01, precision=1)

sparse_case1 = grog_shared.interpolate(ksp_case1, ret_image=False)
sqrt_w = np.asarray(grog_shared.plan.pre_weights)
sparse_case1_w = sparse_case1 * sqrt_w

op_shared = SparseFFT(plan=grog_shared.plan, smaps=smaps)
recon_case1 = np.abs(op_shared.adjoint(sparse_case1_w))

print(f"Case 1 (shared trajectory): {recon_case1.shape}")

# %%

fig, axes = plt.subplots(2, B, figsize=(4 * B, 6), constrained_layout=True)
# sphinx_gallery_start_ignore
for b in range(B):
    axes[0, b].imshow(_normalize(images_case1[b]), cmap="gray", origin="upper")
    axes[0, b].set_title(f"GT #{b + 1}")
    axes[0, b].axis("off")
    axes[1, b].imshow(_normalize(recon_case1[b]), cmap="gray", origin="upper")
    axes[1, b].set_title(f"Shared traj recon #{b + 1}")
    axes[1, b].axis("off")
plt.show()
# sphinx_gallery_end_ignore


# %%
# Case 2: Different trajectory per image in one stack axis
# =========================================================
#
# Different trajectory per image is handled in one stacked call.
# ``coords_case2`` carries a non-singleton stack axis (B), so GROG builds one
# stacked plan and applies all trajectory variants in a single interpolate().

images_case2 = vol[92 : 92 + B]
angles = np.linspace(0.0, np.pi / 6.0, B, dtype=np.float32)

# sphinx_gallery_start_ignore
ksp_case2, coords_case2 = _simulate_case2_inputs(
    images_case2, angles, samples_base, shape, n_coils, smaps
)
# sphinx_gallery_end_ignore

grog_case2 = GrogInterpolator(
    shape=shape,
    coords=coords_case2,
    kernel_width=2,
    oversamp=1.25,
    image_shape=shape,
)
grog_case2.calc_interp_table(calib_cart, lamda=0.01, precision=1)

sparse_case2 = grog_case2.interpolate(ksp_case2, ret_image=False)
pre_w_case2 = np.asarray(grog_case2.plan.pre_weights)[:, np.newaxis, :]
sparse_case2_w = sparse_case2 * pre_w_case2

op_case2 = SparseFFT(plan=grog_case2.plan, smaps=smaps)
recon_case2 = np.abs(op_case2.adjoint(sparse_case2_w))
print(f"Case 2 (per-image trajectory): {recon_case2.shape}")

# %%

fig, axes = plt.subplots(2, B, figsize=(4 * B, 6), constrained_layout=True)
# sphinx_gallery_start_ignore
for b in range(B):
    axes[0, b].imshow(_normalize(images_case2[b]), cmap="gray", origin="upper")
    axes[0, b].set_title(f"GT #{b + 1}")
    axes[0, b].axis("off")
    axes[1, b].imshow(_normalize(recon_case2[b]), cmap="gray", origin="upper")
    axes[1, b].set_title(f"Per-image traj recon #{b + 1}")
    axes[1, b].axis("off")
plt.show()
# sphinx_gallery_end_ignore


# %%
# Case 3: Two stack axes (T, B)
# =============================
#
# Demonstrate 2D stacking: T (time/temporal) and B (trajectory variants).
# ``coords_case3`` is stacked along B only, while data carries (T, B).
# T is treated as batch and B as stacked trajectories in a single call.

T = 2
B2 = 3
images_case3 = vol[96 : 96 + T * B2].reshape(T, B2, *shape)
angles_b = np.linspace(0.0, np.pi / 5.0, B2, dtype=np.float32)

# sphinx_gallery_start_ignore
ksp_case3, coords_case3 = _simulate_case3_inputs(
    images_case3, angles_b, samples_base, shape, n_coils, smaps
)
# sphinx_gallery_end_ignore

grog_case3 = GrogInterpolator(
    shape=shape,
    coords=coords_case3,
    kernel_width=2,
    oversamp=1.25,
    image_shape=shape,
)
grog_case3.calc_interp_table(calib_cart, lamda=0.01, precision=1)

sparse_case3 = grog_case3.interpolate(ksp_case3, ret_image=False)
pre_w_case3 = np.asarray(grog_case3.plan.pre_weights)[np.newaxis, :, np.newaxis, :]
sparse_case3_w = sparse_case3 * pre_w_case3

op_case3 = SparseFFT(plan=grog_case3.plan, smaps=smaps)
recon_case3 = np.abs(op_case3.adjoint(sparse_case3_w))

print(f"Case 3 (2D stack TxB): {recon_case3.shape}")

# %%

fig, axes = plt.subplots(T, B2, figsize=(4 * B2, 3.5 * T), constrained_layout=True)
# sphinx_gallery_start_ignore
for t in range(T):
    for b in range(B2):
        axes[t, b].imshow(_normalize(recon_case3[t, b]), cmap="gray", origin="upper")
        axes[t, b].set_title(f"T={t}, B={b}")
        axes[t, b].axis("off")
plt.show()
# sphinx_gallery_end_ignore
