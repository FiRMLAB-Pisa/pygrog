"""
===================================================
Gadgets with mri-nufft Data: Subspace and B0 ORC
===================================================

This example uses a common data pipeline for both PyGROG gadgets:

1. BrainWeb phantom from ``brainweb-dl``.
2. Trajectory + k-space simulation + reference adjoint from ``mri-nufft``.
3. GROG gridding to feed :class:`~pygrog.operator.SparseFFT`.
4. Comparison against mri-nufft reference formulations.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from mrinufft import display_2D_trajectory, get_operator, initialize_2D_spiral
from mrinufft.density import voronoi
from mrinufft.extras import fse_simulation, get_brainweb_map, make_b0map
from mrinufft.operators.off_resonance import MRIFourierCorrected
from mrinufft.trajectories.utils import Acquisition
from pygrog.calib import GrogInterpolator
from pygrog.gadgets import OffResonanceCorrection, SubspaceProjection, SubspaceSparseFFT
from pygrog.operator import SparseFFT

# %%
# Shared setup
# ============


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


m0, t1, t2 = get_brainweb_map(0)
m0 = np.flip(m0, axis=(0, 1, 2))[90].astype(np.float32)
t1 = np.flip(t1, axis=(0, 1, 2))[90].astype(np.float32)
t2 = np.flip(t2, axis=(0, 1, 2))[90].astype(np.float32)
image = m0 / (m0.max() + 1e-8)
shape = image.shape
n_coils = 16

samples = initialize_2D_spiral(Nc=48, Ns=600, nb_revolutions=10).astype(
    np.float32
)
density = voronoi(samples)

display_2D_trajectory(samples)
plt.show()

# Simulate one multi-coil acquisition with ground-truth smaps.
smaps = _synthetic_smaps(shape, n_coils=n_coils)
nufft_sim = get_operator("finufft")(
    samples=samples,
    shape=shape,
    n_coils=n_coils,
    smaps=smaps,
    density=density,
    squeeze_dims=True,
)
kspace_single = nufft_sim.op(image.astype(np.complex64))  # (n_coils, n_samples)

# GROG plan and SparseFFT base operator for PyGROG gadgets.
coords = (samples * np.asarray(shape, dtype=np.float32)).astype(np.float32)
coil_calib = smaps * image[None, ...]
calib_cart_full = np.fft.fftshift(
    np.fft.fftn(np.fft.ifftshift(coil_calib, axes=(-2, -1)), axes=(-2, -1)),
    axes=(-2, -1),
).astype(np.complex64)
# Extract 24x24 calibration region from k-space centre
calib_size = 24
cy, cx = shape[0] // 2, shape[1] // 2
calib_cart = calib_cart_full[
    :,
    cy - calib_size // 2 : cy + calib_size // 2,
    cx - calib_size // 2 : cx + calib_size // 2,
]

grog = GrogInterpolator(shape=shape, coords=coords, kernel_width=2, oversamp=1.25, image_shape=shape)
grog.calc_interp_table(calib_cart, lamda=0.01, precision=1)

base_op = SparseFFT(
    plan=grog.plan, smaps=torch.as_tensor(smaps)
)

# %%
# Gadget 1: SubspaceProjection / SubspaceSparseFFT
# =================================================
# m0, t1, t2 are already loaded in the shared setup above.

etl = 8
te = np.arange(etl, dtype=np.float32) * 8.0
tr = 3000.0

frames = fse_simulation(m0, t1, t2, te, tr).astype(np.float32)
frames = np.ascontiguousarray(frames)  # (T, ny, nx)

# Simulate non-Cartesian k-space per frame using mri-nufft.
kspace_frames = np.stack(
    [nufft_sim.op(frames[t].astype(np.complex64)) for t in range(etl)],
    axis=0,
)  # (T, n_coils, n_samples)

# mri-nufft adjoint reference per frame.
nufft_ref = get_operator("finufft")(
    samples=samples,
    shape=shape,
    n_coils=n_coils,
    smaps=smaps,
    density=density,
    squeeze_dims=True,
)
frames_ref = np.stack([nufft_ref.adj_op(kspace_frames[t]) for t in range(etl)], axis=0)


# Learn subspace basis from signal dictionary sampled from BrainWeb ranges.
def _estimate_basis(train_data, rank):
    _, _, vh = np.linalg.svd(train_data, full_matrices=False)
    return vh[:rank]


t1_vals = np.linspace(float(t1[t1 > 0].min()) + 1.0, float(t1.max()), 60)
t2_vals = np.linspace(float(t2[t2 > 0].min()) + 1.0, float(t2.max()), 60)
t1_grid, t2_grid = np.meshgrid(t1_vals, t2_vals)
train = fse_simulation(1.0, t1_grid.ravel(), t2_grid.ravel(), te, tr).astype(np.float32)
rank = 4
basis = _estimate_basis(train.T, rank)

# mri-nufft subspace reference: manual adjoint (∑_t conj(φ_r(t)) * NUFFT^H y_t)
# This avoids depending on MRISubspace's internal batching convention.
coeff_ref_nufft = np.stack(
    [
        sum(
            np.conj(basis[r, t]) * nufft_ref.adj_op(kspace_frames[t])
            for t in range(etl)
        )
        for r in range(rank)
    ],
    axis=0,
)

# PyGROG subspace path: non-Cartesian -> sparse Cartesian -> SparseFFT + subspace.
#
# Mirror the MRF benchmark: build ONE GROG plan whose coords embeds T as the
# leading spatial axis.  For 2D data we add a singleton k2=1 dimension so the
# layout is (T, 1, k1, k0) matching the MRF pattern (T, spokes, readout).
# natural_shape = (T, 1, n_shots, n_read, kw), encoding_axis=-5 -> T at nat-axis 0.
n_shots, n_read = samples.shape[:2]
coords_sub = np.broadcast_to(
    coords[np.newaxis, np.newaxis],         # (1, 1, n_shots, n_read, 2)
    (etl, 1, n_shots, n_read, 2),           # (T, 1, n_shots, n_read, 2)
).copy()
grog_sub = GrogInterpolator(shape=shape, coords=coords_sub, kernel_width=2, oversamp=1.25, image_shape=shape)
grog_sub.calc_interp_table(calib_cart, lamda=0.01, precision=1)
base_op_sub = SparseFFT(plan=grog_sub.plan, smaps=torch.as_tensor(smaps))

# kspace_frames: (T, C, n_shots*n_read) -> (T, C, 1, n_shots, n_read) -> (1, C, T, 1, n_shots, n_read)
kspace_sub = (
    kspace_frames
    .reshape(etl, n_coils, 1, n_shots, n_read)  # (T, C, 1, n_shots, n_read)
    .transpose(1, 0, 2, 3, 4)[np.newaxis]        # (1, C, T, 1, n_shots, n_read)
    .astype(np.complex64)
)
sparse_sub = grog_sub.interpolate(
    torch.as_tensor(kspace_sub)
)  # (1, C, T*1*n_shots*n_read*kw) flat
# SubspaceSparseFFT expects natural shape preserved: (B, C, *natural_shape)
sparse_sub = sparse_sub.reshape(
    1, n_coils, *grog_sub.plan.natural_shape,
)  # (1, C, T, 1, n_shots, n_read, kw)

proj = SubspaceProjection(n_components=rank)
proj.fit(torch.as_tensor(train, dtype=torch.float32))
sub_op = SubspaceSparseFFT(base_op_sub, proj.basis.to(torch.complex64), encoding_axis=-5)
coeff_pygrog = sub_op.forward(sparse_sub).detach().cpu().numpy()
# Drop leading B=1 batch axis -> (rank, H, W)
coeff_pygrog = coeff_pygrog[0]

# Display all coefficients: rows = mri-nufft / PyGROG / error, cols = coefficients
fig, axes = plt.subplots(3, rank, figsize=(4 * rank, 10))
for r in range(rank):
    c_ref = np.abs(coeff_ref_nufft[r])
    c_ref /= c_ref.max() + 1e-12
    c_grog = np.abs(coeff_pygrog[r])
    c_grog /= c_grog.max() + 1e-12

    axes[0, r].imshow(c_ref, cmap="gray", origin="lower")
    axes[0, r].set_xticks([])
    axes[0, r].set_yticks([])
    axes[0, r].set_title(f"coeff #{r + 1}")

    axes[1, r].imshow(c_grog, cmap="gray", origin="lower")
    axes[1, r].set_xticks([])
    axes[1, r].set_yticks([])

    err = 100.0 * (c_grog - c_ref) / (c_ref.max() + 1e-12)
    im = axes[2, r].imshow(err, cmap="bwr", origin="lower", vmin=-10, vmax=10)
    axes[2, r].set_xticks([])
    axes[2, r].set_yticks([])
    axes[2, r].set_title(f"MAE={np.abs(err).mean():.2f}%")
    fig.colorbar(im, ax=axes[2, r], fraction=0.046, pad=0.04, label="%")

axes[0, 0].set_ylabel("mri-nufft")
axes[1, 0].set_ylabel("PyGROG")
axes[2, 0].set_ylabel("error")
plt.tight_layout()
plt.show()

# %%
# Gadget 2: OffResonanceCorrection
# ================================

# Use a T1-weighted slice for the ORC test (more high-frequency anatomy → the
# blur from B0 inhomogeneity is much more visible than on the M0 map).
from brainweb_dl import get_mri

image_orc = np.flip(get_mri(0, "T1"), axis=(0, 1, 2))[90].astype(np.float32)
# Center-crop / pad to match the GROG plan's image shape.
def _center_crop_pad(arr, target):
    out = np.zeros(target, dtype=arr.dtype)
    s_in = arr.shape
    # crop / pad each axis
    src = []
    dst = []
    for si, ti in zip(s_in, target):
        if si >= ti:
            off = (si - ti) // 2
            src.append(slice(off, off + ti))
            dst.append(slice(0, ti))
        else:
            off = (ti - si) // 2
            src.append(slice(0, si))
            dst.append(slice(off, off + si))
    out[tuple(dst)] = arr[tuple(src)]
    return out

image_orc = _center_crop_pad(image_orc, shape)
image_orc = image_orc / (image_orc.max() + 1e-8)
brain_mask = image_orc > 0.1 * image_orc.max()
b0_map, _ = make_b0map(shape, b0range=(-200, 200), mask=brain_mask)

# Per-readout timing — identical for every spiral arm (same as mri-nufft example).
t_read = (
    np.arange(samples.shape[1], dtype=np.float32) * Acquisition.default.raster_time
)
readout_time = np.repeat(t_read[None, :], samples.shape[0], axis=0)  # (n_shots, n_read)

orc_nufft = MRIFourierCorrected(
    nufft_ref,
    b0_map=b0_map,
    readout_time=readout_time,
    mask=brain_mask,
)

kspace_off = orc_nufft.op(image_orc.astype(np.complex64))
image_no_orc = np.squeeze(np.abs(nufft_ref.adj_op(kspace_off)))
image_ref_orc = np.squeeze(np.abs(orc_nufft.adj_op(kspace_off)))

# PyGROG ORC: GROG-interpolate then apply ORC-corrected SparseFFT.
# OffResonanceCorrection takes the same (n_shots, n_read) timing as mri-nufft;
# it broadcasts the temporal basis B internally against base.natural_shape.
n_shots, n_read = samples.shape[:2]

base_op_orc = SparseFFT(plan=grog.plan)  # no smaps
sqrt_w_orc = grog.plan.pre_weights        # (n_shots, n_read, kw)

orc_pygrog = OffResonanceCorrection(
    base_op_orc,
    field_map=b0_map.astype(np.float32),
    readout_time=readout_time,
    mask=brain_mask,
    n_components=-1,
    method="svd",
)

# GROG-interpolate the off-resonance k-space and pre-weight.
sparse_off = torch.as_tensor(
    np.asarray(
        grog.interpolate(
            kspace_off.astype(np.complex64).reshape(n_coils, *samples.shape[:2]),
            ret_image=False,
        )
    ),
    dtype=torch.complex64,
)
# Pre-multiply by sqrt density weights (natural shape kept; zero-weight sentinel
# entries are killed by sqrt_w = 0 inside SparseFFT).
sparse_off = sparse_off * sqrt_w_orc.to(sparse_off.dtype).unsqueeze(0)

# Adjoint returns (n_coils, *image_shape) — combine with smaps
smaps_t = torch.as_tensor(smaps)
result_orc = orc_pygrog.adjoint(sparse_off)  # (n_coils, *image_shape)
image_pygrog_orc = np.abs((result_orc * smaps_t.conj()).sum(0).detach().cpu().numpy())

image_no_orc /= image_no_orc.max() + 1e-12
image_ref_orc /= image_ref_orc.max() + 1e-12
image_pygrog_orc /= image_pygrog_orc.max() + 1e-12

err_orc = 100.0 * (image_pygrog_orc - image_ref_orc) / (image_ref_orc.max() + 1e-12)
err_orc_mae = np.abs(err_orc).mean()

# Layout: col 0 = mri-nufft w/o and with ORC; col 1 = PyGROG ORC and error
fig, axes = plt.subplots(2, 2, figsize=(8, 8))
axes[0, 0].imshow(image_no_orc, cmap="gray", origin="lower")
axes[0, 0].set_xticks([])
axes[0, 0].set_yticks([])
axes[0, 0].set_title("No correction")
axes[0, 0].set_ylabel("mri-nufft")

axes[1, 0].imshow(image_ref_orc, cmap="gray", origin="lower")
axes[1, 0].set_xticks([])
axes[1, 0].set_yticks([])
axes[1, 0].set_title("ORC")
axes[1, 0].set_ylabel("mri-nufft")

axes[0, 1].imshow(image_pygrog_orc, cmap="gray", origin="lower")
axes[0, 1].set_xticks([])
axes[0, 1].set_yticks([])
axes[0, 1].set_title("ORC")
axes[0, 1].set_ylabel("PyGROG")

err_im = axes[1, 1].imshow(
    err_orc,
    cmap="bwr",
    origin="lower",
    vmin=-10,
    vmax=10,
)
axes[1, 1].set_xticks([])
axes[1, 1].set_yticks([])
axes[1, 1].set_title(f"error [%]  MAE={err_orc_mae:.2f}%")
fig.colorbar(err_im, ax=axes[1, 1], fraction=0.046, pad=0.04, label="%")

plt.tight_layout()
plt.show()
