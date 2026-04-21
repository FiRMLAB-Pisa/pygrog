"""
====================================================
Gadgets: Subspace Projection and Off-Resonance Correction
====================================================

This example demonstrates two reconstruction *gadgets* that can be stacked
on top of :class:`~pygrog.operator.SparseFFT`:

1. **SubspaceProjection** — projects multi-frame/multi-contrast data onto a
   low-rank temporal subspace computed by truncated SVD.  Useful for dynamic
   MRI (e.g., cardiac cine, T2-shuffling).
2. **OffResonanceCorrection** — compensates B0 field inhomogeneities via a
   low-rank factorisation of the spatiotemporal phase modulation.

Both gadgets wrap a :class:`~pygrog.operator.SparseFFT` operator and expose
the same ``forward`` / ``adjoint`` interface, so they can be used
interchangeably inside iterative solvers.

All data are **synthetic** — no scanner files are required.

.. note::

   `mri-nufft <https://mind-inria.github.io/mri-nufft/>`_ provides equivalent
   gadgets on top of standard NUFFT backends:

   - **Subspace NUFFT** — ``mri_nufft.functional.subspace_nufft``
     (see `mri-nufft subspace example
     <https://mind-inria.github.io/mri-nufft/generated/autoexamples/example_subspace.html>`_).
   - **Off-resonance corrected NUFFT** — ``mri_nufft.trajectory.MRIFourierCorrected``
     (see `mri-nufft off-resonance example
     <https://mind-inria.github.io/mri-nufft/generated/autoexamples/example_offresonance.html>`_).

   The PyGROG gadgets here share the same mathematical formulation but replace
   the fixed-kernel NUFFT with data-driven GROG gridding.  They can also be
   stacked together: wrap a :class:`~pygrog.gadgets.SubspaceSparseFFT` inside
   :class:`~pygrog.gadgets.OffResonanceCorrection` for joint subspace +
   off-resonance corrected reconstruction.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

# %%
# Shared helpers
# ==============


def _phantom(shape):
    """Tiny 2-D Shepp-Logan phantom."""
    ny, nx = shape
    yy, xx = np.mgrid[-1 : 1 : ny * 1j, -1 : 1 : nx * 1j]
    img = np.zeros(shape, dtype=np.float32)
    img += 1.0 * ((xx / 0.9) ** 2 + (yy / 0.9) ** 2 < 1)
    img += 0.4 * ((xx / 0.6) ** 2 + ((yy - 0.1) / 0.7) ** 2 < 1)
    img -= 0.6 * ((xx / 0.15) ** 2 + ((yy + 0.2) / 0.2) ** 2 < 1)
    return img.clip(0, 1)


def _make_sparse_fft(image_shape, n_samples, *, seed=0):
    """Create a SparseFFT with random Cartesian sampling."""
    from pygrog.operator import SparseFFT

    rng = np.random.default_rng(seed)
    grid_shape = tuple(s + s // 4 for s in image_shape)
    grid_size = int(np.prod(grid_shape))
    indices = rng.integers(0, grid_size, size=n_samples).astype(np.int64)
    weights = np.ones(n_samples, dtype=np.float32)
    return SparseFFT(grid_shape, image_shape, indices, weights)


image_shape = (48, 48)
n_samples = 512
rng = np.random.default_rng(0)

base_op = _make_sparse_fft(image_shape, n_samples)
phantom = _phantom(image_shape)

# %%
# Gadget 1 — Subspace Projection
# ================================
#
# Dynamic MRI acquires one k-space frame per TR.  Rather than reconstruct
# each frame independently (computationally expensive), we project the
# temporal signal onto a low-dimensional subspace spanned by ``K``
# basis vectors.
#
# :class:`~pygrog.gadgets.SubspaceProjection` fits the basis from any
# representative time-series data via ``fit()``, then provides
# ``forward()`` (frames → coefficients) and ``adjoint()`` (coefficients →
# frames).

from pygrog.gadgets import SubspaceProjection, SubspaceSparseFFT

# Simulate a 2-D time-series: T monoexponential decays on top of the phantom
T = 16  # number of temporal frames
K = 4  # subspace rank

# Exponential decay fingerprint: (T, n_spatial) where n_spatial = prod(image_shape)
t = torch.linspace(0.0, 1.0, T)
decay_constants = torch.tensor([0.5, 1.0, 2.0, 4.0])
# shape: (T, n_decay_curves) — columns are exponential decays with different rates
fingerprints = torch.exp(-t.unsqueeze(1) * decay_constants.unsqueeze(0))
# shape: (T, n_spatial) — assign random decay constants to spatial positions
torch.manual_seed(0)
assign = torch.randint(0, len(decay_constants), (int(np.prod(image_shape)),))
calib_ts = fingerprints[:, assign]  # (T, n_spatial)

# Fit subspace basis
proj = SubspaceProjection(n_components=K)
proj.fit(calib_ts)

print(f"Basis shape: {proj.basis.shape}")  # (K, T)

# %%
# Build a multi-frame image: phantom scaled by each temporal frame's first
# basis function (a simple proxy for dynamic content).

frames = torch.stack(
    [
        torch.as_tensor(phantom * proj.basis[0, t].real.item()).to(torch.complex64)
        for t in range(T)
    ]
)  # (T, *image_shape)

# Project T frames → K subspace coefficients
coeffs = proj.forward(frames)  # (K, *image_shape)
print(f"Frame stack shape    : {frames.shape}")
print(f"Coefficients shape   : {coeffs.shape}")

# Expand back: K coefficients → T frames (lossy if K < T)
frames_recon = proj.adjoint(coeffs)  # (T, *image_shape)

# %%

fig, axes = plt.subplots(2, 4, figsize=(11, 5.5))
for t_idx, ax in zip(range(4), axes[0], strict=False):
    ax.imshow(frames[t_idx].abs().numpy(), cmap="gray", origin="lower", vmin=0, vmax=1)
    ax.set_title(f"Frame {t_idx}")
    ax.axis("off")
for k_idx, ax in zip(range(4), axes[1], strict=False):
    ax.imshow(coeffs[k_idx].abs().numpy(), cmap="magma", origin="lower", vmin=0, vmax=1)
    ax.set_title(f"Coeff {k_idx}")
    ax.axis("off")
axes[0][0].set_ylabel("Frames (4 of 16)")
axes[1][0].set_ylabel("Subspace coefficients")
plt.suptitle(f"SubspaceProjection: T={T} frames → K={K} coefficients")
plt.tight_layout()
plt.show()

# %%
# SubspaceSparseFFT: fused subspace + k-space encoding
# ----------------------------------------------------
#
# :class:`~pygrog.gadgets.SubspaceSparseFFT` decorates a
# :class:`~pygrog.operator.SparseFFT` with the subspace basis, providing a
# single operator that maps subspace coefficients ``(K, *image_shape)``
# directly to multi-frame k-space ``(T, n_coils, n_samples)``.

sub_op = SubspaceSparseFFT(base_op, proj.basis)

# Forward: subspace coefficients → multi-frame k-space
kspace_sub = sub_op.adjoint(coeffs)  # (T, 1, n_samples) — 1 coil, no smaps
print(f"\nSubspaceSparseFFT adjoint shape: {kspace_sub.shape}")

# Adjoint: multi-frame k-space → subspace coefficients
coeffs_recon = sub_op.forward(kspace_sub)  # (K, *image_shape)
print(f"SubspaceSparseFFT forward shape: {coeffs_recon.shape}")

# %%
# Gadget 2 — Off-Resonance Correction
# =====================================
#
# B0 field inhomogeneities cause a spatially varying phase accumulation
# ``exp(i 2pi df(r) t)`` during the readout.  This blurs the image when
# ignored.
#
# :class:`~pygrog.gadgets.OffResonanceCorrection` factorises the phase
# modulation into ``L`` temporal and spatial components via SVD/MFI/MTI
# (Mann et al.\ 1997, Sutton et al.\ 2003) and accumulates the correction
# over each component.

from pygrog.gadgets import OffResonanceCorrection

# Synthetic B0 field map: quadratic bowl in Hz
ny, nx = image_shape
yy, xx = np.mgrid[-1 : 1 : ny * 1j, -1 : 1 : nx * 1j]
b0_hz = (50.0 * (xx**2 + yy**2)).astype(np.float32)  # -50 .. +50 Hz

# Readout timeline: n_samples points spanning 2 ms
readout_time = np.linspace(0.0, 2e-3, n_samples, dtype=np.float32)

# Build operator (SVD factorisation, auto-select L)
orc_op = OffResonanceCorrection(
    base_op,
    field_map=b0_hz,
    readout_time=readout_time,
    n_components=-1,
    method="svd",
)

print(f"\nORC number of components: {orc_op.n_components}")

# %%

# Simulate: noiseless k-space from the phantom image
image_t = torch.as_tensor(phantom[np.newaxis]).to(torch.complex64)  # (1, ny, nx)

kspace_norc = base_op.adjoint(image_t)  # without ORC
kspace_orc = orc_op.forward(image_t)  # with ORC (off-resonance encoded)

# Reconstruct ignoring off-resonance (direct adjoint)
image_no_corr = base_op.forward(kspace_orc)  # ignoring phase: blurry

# Reconstruct with off-resonance correction (ORC adjoint)
image_corr = orc_op.adjoint(kspace_orc)  # corrected: (1, ny, nx)

# %%

fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))

axes[0].imshow(b0_hz, cmap="bwr", origin="lower")
axes[0].set_title("B0 field map [Hz]")
axes[0].axis("off")

axes[1].imshow(image_no_corr.abs().numpy(), cmap="gray", origin="lower")
axes[1].set_title("No correction (blurred)")
axes[1].axis("off")

axes[2].imshow(image_corr.abs().numpy(), cmap="gray", origin="lower")
axes[2].set_title("ORC adjoint (corrected)")
axes[2].axis("off")

plt.suptitle("OffResonanceCorrection (SVD factorisation)")
plt.tight_layout()
plt.show()

# %%
# .. note::
#    For real acquisitions the B0 map is obtained from a field-mapping
#    sequence.  The ``readout_time`` must match the actual per-sample
#    dwell time of the trajectory, and the ``field_map`` must be in Hz
#    (not rad/s).

plt.show()
