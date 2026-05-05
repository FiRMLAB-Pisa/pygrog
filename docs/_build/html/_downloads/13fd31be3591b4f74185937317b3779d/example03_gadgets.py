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

from mrinufft import get_operator, initialize_2D_spiral
from mrinufft.density import voronoi
from mrinufft.extras import fse_simulation, get_brainweb_map, make_b0map
from mrinufft.operators.off_resonance import MRIFourierCorrected
from mrinufft.trajectories.utils import Acquisition
from pygrog.calib import GrogInterpolator
from pygrog.gadgets import OffResonanceGadget, SubspaceGadget, with_offresonance  # noqa: F401
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


def _normalize_unit(image):
    return image / (image.max() + 1e-12)


def _format_panel(ax, image, title, *, cmap="gray", vmin=None, vmax=None):
    ax.imshow(image, cmap=cmap, origin="upper", vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)


def _estimate_basis(train_data, rank):
    _, _, vh = np.linalg.svd(train_data, full_matrices=False)
    return vh[:rank]


def _center_crop_pad(arr, target):
    out = np.zeros(target, dtype=arr.dtype)
    s_in = arr.shape
    src = []
    dst = []
    for si, ti in zip(s_in, target, strict=False):
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
# sphinx_gallery_end_ignore

m0, t1, t2 = get_brainweb_map(0)

# sphinx_gallery_start_ignore
m0 = np.flip(m0, axis=(0, 1, 2))[90].astype(np.float32)
t1 = np.flip(t1, axis=(0, 1, 2))[90].astype(np.float32)
t2 = np.flip(t2, axis=(0, 1, 2))[90].astype(np.float32)
image = m0 / (m0.max() + 1e-8)
shape = image.shape
n_coils = 16

samples = initialize_2D_spiral(Nc=48, Ns=600, nb_revolutions=10).astype(np.float32)
density = voronoi(samples)

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

grog = GrogInterpolator(
    shape=shape, coords=coords, kernel_width=2, oversamp=1.25, image_shape=shape
)
grog.calc_interp_table(calib_cart, lamda=0.01, precision=1)

base_op = SparseFFT(plan=grog.plan, smaps=smaps)
# sphinx_gallery_end_ignore

# %%
# Gadget 1: SubspaceSparseFFT for Multi-Echo FSE
# ===============================================
#
# Learn a low-rank basis from a dictionary of FSE signal evolutions,
# then project GROG-interpolated k-space onto that basis to extract
# subspace coefficients.

# sphinx_gallery_start_ignore
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
t1_vals = np.linspace(float(t1[t1 > 0].min()) + 1.0, float(t1.max()), 60)
t2_vals = np.linspace(float(t2[t2 > 0].min()) + 1.0, float(t2.max()), 60)
t1_grid, t2_grid = np.meshgrid(t1_vals, t2_vals)
train = fse_simulation(1.0, t1_grid.ravel(), t2_grid.ravel(), te, tr).astype(np.float32)
rank = 4
basis = _estimate_basis(train.T, rank)

coeff_truth = np.stack(
    [sum(np.conj(basis[r, t]) * frames[t] for t in range(etl)) for r in range(rank)],
    axis=0,
)

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
# sphinx_gallery_end_ignore

# %%
# PyGROG Subspace gadget API call
# ===============================
#
# Build a GROG plan with temporal frames as the leading stack axis,
# interpolate all frames, and project onto the learned subspace basis
# via the public decorator/wrapper API.
n_shots, n_read = samples.shape[:2]
coords_sub = np.broadcast_to(
    coords[np.newaxis, np.newaxis],  # (1, 1, n_shots, n_read, 2)
    (etl, 1, n_shots, n_read, 2),  # (T, 1, n_shots, n_read, 2)
).copy()
grog_sub = GrogInterpolator(
    shape=shape, coords=coords_sub, kernel_width=2, oversamp=1.25, image_shape=shape
)
grog_sub.calc_interp_table(calib_cart, lamda=0.01, precision=1)
base_op_sub = SparseFFT(plan=grog_sub.plan, smaps=smaps)

# kspace_frames: (T, C, n_shots*n_read) -> (T, C, 1, n_shots, n_read) -> (1, C, T, 1, n_shots, n_read)
kspace_sub = (
    kspace_frames.reshape(
        etl, n_coils, 1, n_shots, n_read
    )  # (T, C, 1, n_shots, n_read)
    .transpose(1, 0, 2, 3, 4)[np.newaxis]  # (1, C, T, 1, n_shots, n_read)
)
sparse_sub = grog_sub.interpolate(kspace_sub).reshape(1, n_coils, *grog_sub.plan.natural_shape)

# Learn a rank-K temporal basis Phi from the training dictionary (K, T).
basis_torch = torch.as_tensor(_estimate_basis(train.T, rank), dtype=torch.complex64)

# Preferred public construction via wrapper class.
sub_op = SubspaceGadget(base_op_sub, basis_torch, encoding_axis=-5)
# Equivalent decorator-style wrapping (same behavior):
# sub_op = with_subspace(base_op_sub, basis_torch, encoding_axis=-5)

coeff_pygrog = np.asarray(sub_op.adjoint(sparse_sub))
# Drop leading B=1 batch axis -> (rank, H, W)
coeff_pygrog = coeff_pygrog[0]

# %%
# Display all coefficients: one row per coefficient.

fig, axes = plt.subplots(rank, 3, figsize=(11, 3.5 * rank), squeeze=False)
# sphinx_gallery_start_ignore
for r in range(rank):
    c_truth = _normalize_unit(np.abs(coeff_truth[r]))
    c_ref = _normalize_unit(np.abs(coeff_ref_nufft[r]))
    c_grog = _normalize_unit(np.abs(coeff_pygrog[r]))

    _format_panel(axes[r, 0], c_truth, f"GT coeff #{r + 1}", vmin=0.0, vmax=1.0)
    _format_panel(axes[r, 1], c_ref, "NUFFT coeff", vmin=0.0, vmax=1.0)
    _format_panel(axes[r, 2], c_grog, "PyGROG coeff", vmin=0.0, vmax=1.0)

plt.tight_layout()
plt.show()
# sphinx_gallery_end_ignore

# %%
# Gadget 2: Off-resonance gadget
# ==============================
#
# Model and correct for B0 field inhomogeneity on GROG-gridded data.
# Create a field map and per-readout timing basis, then apply ORC to reconstruct
# sharp images despite off-resonance blurring.

# sphinx_gallery_start_ignore
from brainweb_dl import get_mri

image_orc = np.flip(get_mri(0, "T1"), axis=(0, 1, 2))[90].astype(np.float32)


image_orc = _center_crop_pad(image_orc, shape)
image_orc = image_orc / (image_orc.max() + 1e-8)
brain_mask = image_orc > 0.1 * image_orc.max()
b0_map, _ = make_b0map(shape, b0range=(-200, 200), mask=brain_mask)

# Per-readout timing — identical for every spiral arm (same as mri-nufft example).
t_read = np.arange(samples.shape[1], dtype=np.float32) * Acquisition.default.raster_time
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
# sphinx_gallery_end_ignore


# %%
# PyGROG off-resonance API call
# =============================
#
# Apply :class:`~pygrog.gadgets.OffResonanceGadget` to correct
# off-resonance blurring. The wrapper automatically constructs a low-rank
# temporal basis for the correction term.
n_shots, n_read = samples.shape[:2]

base_op_orc = SparseFFT(plan=grog.plan)  # no smaps
sqrt_w_orc = np.asarray(grog.plan.pre_weights)  # (n_shots, n_read, kw)

orc_pygrog = OffResonanceGadget(
    base_op_orc,
    field_map=b0_map,
    readout_time=readout_time,
    mask=brain_mask,
    n_components=-1,
    method="svd",
)

# GROG-interpolate the off-resonance k-space and pre-weight.
sparse_off = grog.interpolate(
    kspace_off.reshape(n_coils, n_shots, n_read),
    ret_image=False,
) * sqrt_w_orc[np.newaxis]

# Adjoint returns (n_coils, *image_shape) — combine with smaps
result_orc = orc_pygrog.adjoint(sparse_off)  # (n_coils, *image_shape)
image_pygrog_orc = np.abs((np.asarray(result_orc) * smaps.conj()).sum(0))

# Compose gadgets by double-decoration (recommended over instantiating
# internal sparse classes directly):
# op_joint = with_offresonance(
#     with_subspace(base_op_sub, basis_torch, encoding_axis=-5),
#     b0_map=b0_map,
#     readout_time=readout_time,
#     mask=brain_mask,
#     L=-1,
#     method="svd",
# )

# %%

truth_orc = _normalize_unit(np.abs(image_orc))

# sphinx_gallery_start_ignore
image_no_orc = _normalize_unit(image_no_orc)
image_ref_orc = _normalize_unit(image_ref_orc)
image_pygrog_orc = _normalize_unit(image_pygrog_orc)

fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
_format_panel(axes[0], truth_orc, "Ground truth", vmin=0.0, vmax=1.0)
_format_panel(axes[1], image_no_orc, "NUFFT non-corrected", vmin=0.0, vmax=1.0)
_format_panel(axes[2], image_ref_orc, "NUFFT corrected", vmin=0.0, vmax=1.0)
_format_panel(axes[3], image_pygrog_orc, "PyGROG corrected", vmin=0.0, vmax=1.0)

plt.show()
# sphinx_gallery_end_ignore
