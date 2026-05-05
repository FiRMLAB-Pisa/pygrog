"""
=================================================================
End-to-end CS reconstruction with sigpy, deepinv and mrpro
=================================================================

PyGROG ships *native* adapters for the full preprocessing chain:

* :func:`~pygrog.interop.sigpy.coil_compress`,
  :func:`~pygrog.interop.deepinv.coil_compress`,
  :func:`~pygrog.interop.mrpro.coil_compress`
* :func:`~pygrog.interop.sigpy.nlinv_calib`,
  :func:`~pygrog.interop.deepinv.nlinv_calib`,
  :func:`~pygrog.interop.mrpro.nlinv_calib`
* :class:`~pygrog.interop.sigpy.GrogInterpolator`,
  :class:`~pygrog.interop.deepinv.GrogInterpolator`,
  :class:`~pygrog.interop.mrpro.GrogInterpolator`

plus operator wrappers for each framework's solver:
:class:`~pygrog.interop.GrogLinop` (sigpy),
:class:`~pygrog.interop.GrogLinearPhysics` (deepinv) and
:class:`~pygrog.interop.GrogLinearOp` (mrpro).

Every adapter speaks the *native* shape convention of its target
framework, so the same noncartesian acquisition is fed to each pipeline
in its preferred container — :class:`numpy.ndarray` for sigpy, a
``(B, n_coils, n_samples)`` :class:`torch.Tensor` for deepinv and a
:class:`mrpro.data.KData` for mrpro — without any manual rearrangement.

For each framework we run the full chain

``coil_compress → nlinv_calib → GrogInterpolator → L1-wavelet FISTA``

starting from the same raw multi-coil 16-arm spiral acquisition of a T1
BrainWeb slice.  This is an interoperability showcase, not a benchmark.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

from brainweb_dl import get_mri
from mrinufft import get_operator, initialize_2D_spiral
from mrinufft.density import voronoi

# sphinx_gallery_start_ignore
def _center_crop_pad(arr, target):
    out = np.zeros(target, dtype=arr.dtype)
    src, dst = [], []
    for si, ti in zip(arr.shape, target, strict=False):
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


shape = (256, 256)
image_np = np.flip(get_mri(0, "T1"), axis=(0, 1, 2))[90].astype(np.float32)
image_np = _center_crop_pad(image_np, shape)
image_np /= image_np.max() + 1e-8

n_coils_full = 8
smaps_true = _synthetic_smaps(shape, n_coils=n_coils_full)

samples = initialize_2D_spiral(Nc=16, Ns=600, nb_revolutions=10).astype(np.float32)
density = voronoi(samples)
n_shots, n_read = samples.shape[:2]
coords = (samples * np.asarray(shape, dtype=np.float32)).astype(np.float32)

nufft = get_operator("finufft")(
    samples=samples,
    shape=shape,
    n_coils=n_coils_full,
    smaps=smaps_true,
    density=density,
    squeeze_dims=True,
)
kspace_raw = nufft.op(image_np.astype(np.complex64))  # (n_coils_full, n_samples)

rng = np.random.default_rng(0)
sigma = 0.001 * np.abs(kspace_raw).max()
kspace_raw = kspace_raw + sigma * (
    rng.standard_normal(kspace_raw.shape) + 1j * rng.standard_normal(kspace_raw.shape)
).astype(np.complex64)

n_compressed = 4

print(f"image shape       : {shape}")
print(f"# spiral arms     : {n_shots}")
print(f"raw k-space       : {kspace_raw.shape}  (numpy)")
print(f"target # coils    : {n_compressed}")
# sphinx_gallery_end_ignore

# %%
# sigpy: Full Preprocessing Pipeline via Native Adapters
# ======================================================
#
# Run the complete chain using sigpy's native :class:`numpy.ndarray`
# containers: coil compression → NLINV calibration → GROG gridding →
# L1-wavelet FISTA reconstruction.
#
# Every step is adapted from PyGROG via :mod:`pygrog.interop.sigpy`:
#
# * :func:`~pygrog.interop.sigpy.coil_compress` — PCA compression on flat
#   ``(n_coils, n_samples)`` k-space.
# * :func:`~pygrog.interop.sigpy.nlinv_calib` — self-calibration returning
#   coil sensitivities and calibration patch.
# * :class:`~pygrog.interop.sigpy.GrogInterpolator` — GROG with interpolate()
#   returning sparse k-space *and* GROG plan.
# * :class:`~pygrog.interop.GrogLinop` — wraps the SparseFFT as a sigpy
#   :class:`sigpy.linop.Linop` for use in solvers.

print("\n[sigpy] coil_compress → nlinv_calib → GROG → L1-wavelet FISTA")
import sigpy as sp

from pygrog.interop import GrogLinop
from pygrog.interop import sigpy as pg_sigpy
from pygrog.operator import SparseFFT

# 1. Coil compression: (n_coils, n_samples) → (n_v, n_samples).
ksp_s_compressed, _vh_s = pg_sigpy.coil_compress(kspace_raw, n_compressed)
ksp_s_arms = ksp_s_compressed.reshape(n_compressed, n_shots, n_read)

# 2. NLINV self-calibration on a 24x24 patch.  ``ret_cal=True`` returns
# both the coil sensitivity maps *and* the synthesised Cartesian
# calibration k-space patch needed by GROG — no need to fabricate one.
smaps_s, calib_s = pg_sigpy.nlinv_calib(
    ksp_s_compressed,
    coords.reshape(-1, 2),
    shape,
    cal_width=24,
    max_iter=12,
    ret_cal=True,
)

# 3. GROG planning + interpolation — calibration patch comes straight
# from NLINV.
grog_s = pg_sigpy.GrogInterpolator(
    coords, shape, kernel_width=2, oversamp=1.25, image_shape=shape
)
grog_s.calc_interp_table(calib_s, lamda=0.01, precision=1)
sparse_s, plan_s = grog_s.interpolate(ksp_s_arms)
sparse_s_t = torch.as_tensor(sparse_s)
# pre_weights is flat (n_samples,); reshape to natural to align with the
# natural-shape sparse output ``(n_coils, *natural_shape)``.
pre_w = plan_s.pre_weights.reshape(*plan_s.natural_shape)
sparse_s_t = sparse_s_t * pre_w

# 4. SparseFFT + sigpy linop wrapper.
op_s = SparseFFT(plan=plan_s, smaps=smaps_s)
img_zf = op_s.adjoint(sparse_s_t).numpy()

# 5. L1-wavelet FISTA via ``LinearLeastSquares``.
M_sigpy = GrogLinop(op_s).H
W_sigpy = sp.linop.Wavelet(shape, axes=(-2, -1), wave_name="db4", level=3)
A_sigpy = M_sigpy * W_sigpy.H
y_sigpy = sparse_s_t.reshape(n_compressed, op_s.indices.shape[0]).numpy()
proxg = sp.prox.L1Reg(W_sigpy.oshape, lamda=5e-4)
sigpy_lipschitz = float(
    sp.app.MaxEig(
        A_sigpy.N,
        dtype=np.complex64,
        device=sp.cpu_device,
        max_iter=80,
        show_pbar=False,
    ).run()
)
wav_hat = sp.app.LinearLeastSquares(
    A=A_sigpy,
    y=y_sigpy,
    proxg=proxg,
    alpha=0.9 / sigpy_lipschitz,
    max_iter=80,
    show_pbar=False,
).run()
img_sigpy = np.abs(W_sigpy.H(wav_hat))
print(f"  done — output shape {img_sigpy.shape}")


# %%
# deepinv: Full Preprocessing Pipeline via Native Adapters
# ========================================================
#
# Run the complete chain using deepinv's native batched
# :class:`torch.Tensor` containers: coil compression → NLINV calibration →
# GROG gridding → L1-wavelet FISTA reconstruction.
#
# Every step is adapted from PyGROG via :mod:`pygrog.interop.deepinv`:
#
# * :func:`~pygrog.interop.deepinv.coil_compress` — PCA compression on
#   ``(B, n_coils, n_samples)`` batched tensors.
# * :func:`~pygrog.interop.deepinv.nlinv_calib` — self-calibration
#   returning ``(B, n_v, *shape)`` coil sensitivities.
# * :class:`~pygrog.interop.deepinv.GrogInterpolator` — GROG with batched
#   input/output handling.
# * :class:`~pygrog.interop.GrogLinearPhysics` — wraps the SparseFFT as a
#   deepinv :class:`deepinv.physics.LinearPhysics` for solvers.

print("\n[deepinv] coil_compress → nlinv_calib → GROG → L1-wavelet FISTA")
from deepinv.physics.functional import power_method
from deepinv.optim.prior import WaveletPrior
from deepinv.optim.data_fidelity import L2
from deepinv.optim.optimizers import optim_builder

from pygrog.interop import GrogLinearPhysics
from pygrog.interop import deepinv as pg_deepinv

ksp_d_flat = torch.as_tensor(kspace_raw).unsqueeze(0)  # (1, n_coils, n_samples)

# 1. Coil compression on the batched tensor.
ksp_d_compressed, _vh_d = pg_deepinv.coil_compress(ksp_d_flat, n_compressed)
ksp_d_arms = ksp_d_compressed.reshape(1, n_compressed, n_shots, n_read)

# 2. NLINV self-calibration (returns smaps + calibration k-space patch).
smaps_d, calib_d = pg_deepinv.nlinv_calib(
    ksp_d_compressed,
    coords.reshape(-1, 2),
    shape,
    cal_width=24,
    max_iter=12,
    ret_cal=True,
)  # smaps: (1, n_v, H, W); calib: (n_v, *cal_shape)

# 3. GROG using the NLINV-derived calibration patch.
grog_d = pg_deepinv.GrogInterpolator(
    coords, shape, kernel_width=2, oversamp=1.25, image_shape=shape
)
grog_d.calc_interp_table(calib_d[0], lamda=0.01, precision=1)
sparse_d, plan_d = grog_d.interpolate(ksp_d_arms)  # (1, n_v, *natural, kw)
sparse_d = sparse_d * plan_d.pre_weights.reshape(*plan_d.natural_shape).to(
    sparse_d.dtype
)

# 4. SparseFFT + deepinv physics wrapper.
op_d = SparseFFT(plan=plan_d, smaps=smaps_d[0])

# 5. L1-wavelet FISTA via ``optim_builder``.
physics = GrogLinearPhysics(op_d)
y_di = sparse_d.reshape(1, n_compressed, op_d.indices.shape[0]).clone()
x_dagger = physics.A_adjoint(y_di)  # (1, 1, H, W)

with torch.no_grad():
    x0 = x_dagger / (x_dagger.norm() + 1e-12)
    deepinv_lipschitz = power_method(
        lambda v: physics.A_adjoint(physics.A(v)),
        x0,
        max_iter=100,
    ).item()

prior = WaveletPrior(wv="db4", wvdim=2, level=3, is_complex=True)
prior.explicit_prior = False
data_fidelity = L2()
params = {"stepsize": 0.9 / deepinv_lipschitz, "lambda": 1e-2, "a": 3}
recon = optim_builder(
    iteration="FISTA",
    prior=prior,
    data_fidelity=data_fidelity,
    early_stop=False,
    max_iter=80,
    params_algo=params,
    verbose=False,
)
x_di = recon(y_di, physics)
img_deepinv = np.abs(x_di.squeeze().detach().cpu().numpy())
print(f"  done — output shape {img_deepinv.shape}")


# %%
# mrpro: Full Preprocessing Pipeline via Native Adapters
# ======================================================
#
# Run the complete chain using mrpro's native :class:`mrpro.data.KData`
# containers: coil compression → NLINV calibration → GROG gridding →
# L1-wavelet FISTA reconstruction.
#
# Every step is adapted from PyGROG via :mod:`pygrog.interop.mrpro`:
#
# * :func:`~pygrog.interop.mrpro.coil_compress` — dispatches to mrpro's PCA
#   compressor on :class:`KData`.
# * :func:`~pygrog.interop.mrpro.nlinv_calib` — self-calibration returning
#   mrpro-ordered smaps and calibration patch.
# * :class:`~pygrog.interop.mrpro.GrogInterpolator` — GROG gridding with
#   trajectory snapping on :class:`KData`.
# * :class:`~pygrog.interop.GrogLinearOp` — wraps the SparseFFT as an
#   :class:`mrpro.operators.LinearOperator` for solver composition.

print("\n[mrpro] coil_compress → nlinv_calib → GROG → L1-wavelet FISTA")
from mrpro.algorithms.optimizers import pgd
from mrpro.data import KData, KHeader, KTrajectory, SpatialDimension
from mrpro.operators import WaveletOp
from mrpro.operators.functionals import L1NormViewAsReal, L2NormSquared

from pygrog.interop import GrogLinearOp
from pygrog.interop import mrpro as pg_mrpro

# Build a minimal KData wrapping the raw acquisition.
kx_t = torch.as_tensor(coords[..., 1]).reshape(1, 1, 1, n_shots, n_read)
ky_t = torch.as_tensor(coords[..., 0]).reshape(1, 1, 1, n_shots, n_read)
kz_t = torch.zeros_like(kx_t)
traj = KTrajectory(kz=kz_t, ky=ky_t, kx=kx_t)
data_t = (
    torch.as_tensor(kspace_raw).reshape(n_coils_full, 1, n_shots, n_read).unsqueeze(0)
)
spatial = SpatialDimension(z=1, y=shape[0], x=shape[1])
header = KHeader(
    recon_matrix=spatial,
    encoding_matrix=spatial,
    recon_fov=SpatialDimension(z=1.0, y=1.0, x=1.0),
    encoding_fov=SpatialDimension(z=1.0, y=1.0, x=1.0),
)
kdata = KData(header=header, data=data_t, traj=traj)

# 1. Coil compression — dispatches to mrpro's PCA compressor.
kdata = pg_mrpro.coil_compress(kdata, n_compressed)

# 2. NLINV self-calibration through pygrog (returns smaps + cal patch).
smaps_m, calib_m = pg_mrpro.nlinv_calib(
    kdata,
    cal_width=24,
    max_iter=12,
    ret_cal=True,
)  # smaps: (n_v, 1, H, W); calib: (n_v, *cal_shape)

# 3. GROG — interpolation snaps the trajectory onto the grid and fuses
# the kernel-width axis into ``k0``; calibration patch comes from NLINV.
grog_m = pg_mrpro.GrogInterpolator(kdata, kernel_width=2, oversamp=1.25)
grog_m.calc_interp_table(calib_m, lamda=0.01, precision=1)
kdata_grog, plan_m = grog_m.interpolate(kdata)

# 4. SparseFFT + mrpro linop wrapper.
op_m = SparseFFT(plan=plan_m, smaps=smaps_m[:, 0])

# 5. L1-wavelet FISTA via ``pgd``.
mrpro_op = GrogLinearOp(op_m)
A_mr = mrpro_op.H
wavelet_op = WaveletOp(domain_shape=shape, dim=(-2, -1), wavelet_name="db4", level=3)
acq = A_mr @ wavelet_op.H

# k-space layout from the adapter: (other=1, coils, k2=1, k1, k0*kw)
y_mr = kdata_grog.data.squeeze(0).squeeze(1)  # (coils, k1, k0*kw)
y_mr = y_mr * plan_m.pre_weights.to(y_mr.dtype).reshape(1, n_shots, -1)

f = 0.5 * L2NormSquared(target=y_mr, divide_by_n=False) @ acq
g = 1e-3 * L1NormViewAsReal(divide_by_n=False)
(init_wave,) = wavelet_op(torch.zeros(shape, dtype=torch.complex64))
mrpro_norm_init = torch.randn_like(init_wave)
mrpro_lipschitz = acq.operator_norm(
    initial_value=mrpro_norm_init,
    dim=None,
    max_iterations=80,
).item()
stepsize = 0.9 / mrpro_lipschitz
(wave_hat,) = pgd(
    f=f,
    g=g,
    initial_value=init_wave,
    stepsize=stepsize,
    max_iterations=80,
    backtrack_factor=1.0,
)
(img_mr_t,) = wavelet_op.H(wave_hat)
img_mrpro = np.abs(img_mr_t.detach().cpu().numpy())
print(f"  done — output shape {img_mrpro.shape}")


# %%
# Display
# =======

# sphinx_gallery_start_ignore
def _norm(x):
    return x / (x.max() + 1e-12)

fig, axes = plt.subplots(1, 5, figsize=(16, 4))
panels = [
    ("Ground truth", _norm(image_np)),
    ("Zero-filled (adjoint, sigpy)", _norm(np.abs(img_zf))),
    ("sigpy (L1-wavelet)", _norm(img_sigpy)),
    ("deepinv (L1-wavelet)", _norm(img_deepinv)),
    ("mrpro (L1-wavelet)", _norm(img_mrpro)),
]
for ax, (title, im) in zip(axes, panels, strict=False):
    ax.imshow(im, cmap="gray", origin="upper", vmin=0.0, vmax=1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
plt.tight_layout()
plt.show()
# sphinx_gallery_end_ignore
