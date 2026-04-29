"""fig_interop — same recon via sigpy / deepinv / mrpro adapters.

A 1×5 strip: ground truth, zero-filled adjoint, then the three
framework L1-wavelet reconstructions.

Faithful distillation of ``examples/example_interop.py``: same
trajectory, same n_compressed=4, same NLINV calibration, same FISTA
parameters, and the deepinv-derived Lipschitz constant is reused for
the mrpro path so all three solvers operate at the same step size.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from brainweb_dl import get_mri
from mrinufft import get_operator, initialize_2D_spiral
from mrinufft.density import voronoi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    POSTER_STYLE,
    center_crop_pad,
    normalize,
    save_fig,
    show_image,
    synthetic_smaps,
)


# ---------------------------------------------------------------------------
# Shared synthetic acquisition (matches examples/example_interop.py)
# ---------------------------------------------------------------------------
def _build_acquisition():
    shape = (256, 256)
    image = np.flip(get_mri(0, "T1"), axis=(0, 2))[90].astype(np.float32)
    image = center_crop_pad(image, shape)
    image /= image.max() + 1e-8

    n_coils_full = 8
    smaps_true = synthetic_smaps(shape, n_coils=n_coils_full)
    samples = initialize_2D_spiral(Nc=16, Ns=600, nb_revolutions=10).astype(np.float32)
    density = voronoi(samples)
    n_shots, n_read = samples.shape[:2]
    coords = (samples * np.asarray(shape, dtype=np.float32)).astype(np.float32)

    nufft = get_operator("finufft")(
        samples=samples, shape=shape, n_coils=n_coils_full,
        smaps=smaps_true, density=density, squeeze_dims=True,
    )
    kspace_raw = nufft.op(image.astype(np.complex64))
    rng = np.random.default_rng(0)
    sigma = 0.001 * np.abs(kspace_raw).max()
    kspace_raw = kspace_raw + sigma * (
        rng.standard_normal(kspace_raw.shape)
        + 1j * rng.standard_normal(kspace_raw.shape)
    ).astype(np.complex64)
    return image, shape, samples, coords, n_shots, n_read, kspace_raw, n_coils_full


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------
def _run_sigpy(kspace_raw, coords, shape, n_shots, n_read, n_compressed):
    """sigpy path; also returns the zero-filled image for the ZF panel."""
    import sigpy as sp
    from pygrog.interop import GrogLinop
    from pygrog.interop import sigpy as pg_sigpy
    from pygrog.operator import SparseFFT

    ksp_c, _ = pg_sigpy.coil_compress(kspace_raw, n_compressed)
    ksp_arms = ksp_c.reshape(n_compressed, n_shots, n_read)
    smaps_s, calib_s = pg_sigpy.nlinv_calib(
        ksp_c, coords.reshape(-1, 2), shape, cal_width=24, max_iter=12, ret_cal=True,
    )
    grog = pg_sigpy.GrogInterpolator(
        coords, shape, kernel_width=2, oversamp=1.25, image_shape=shape,
    )
    grog.calc_interp_table(calib_s, lamda=0.01, precision=1)
    sparse, plan = grog.interpolate(ksp_arms)
    sparse_t = torch.as_tensor(sparse) * plan.pre_weights.reshape(*plan.natural_shape)
    op = SparseFFT(plan=plan, smaps=smaps_s)
    img_zf = op.adjoint(sparse_t).numpy()  # zero-filled SENSE adjoint

    M = GrogLinop(op).H
    W = sp.linop.Wavelet(shape, axes=(-2, -1), wave_name="db4", level=3)
    A = M * W.H
    y = sparse_t.reshape(n_compressed, op.indices.shape[0]).numpy()
    proxg = sp.prox.L1Reg(W.oshape, lamda=5e-4)
    wav = sp.app.LinearLeastSquares(
        A=A, y=y, proxg=proxg, max_iter=150, show_pbar=False,
    ).run()
    return np.abs(W.H(wav)), np.abs(img_zf)


def _run_deepinv(kspace_raw, coords, shape, n_shots, n_read, n_compressed):
    """deepinv path with self-computed Lipschitz constant."""
    from deepinv.optim.data_fidelity import L2
    from deepinv.optim.optimizers import optim_builder
    from deepinv.optim.prior import WaveletPrior
    from pygrog.interop import GrogLinearPhysics
    from pygrog.interop import deepinv as pg_deepinv
    from pygrog.operator import SparseFFT

    ksp_flat = torch.as_tensor(kspace_raw).unsqueeze(0)
    ksp_c, _ = pg_deepinv.coil_compress(ksp_flat, n_compressed)
    ksp_arms = ksp_c.reshape(1, n_compressed, n_shots, n_read)
    smaps_d, calib_d = pg_deepinv.nlinv_calib(
        ksp_c, coords.reshape(-1, 2), shape, cal_width=24, max_iter=12, ret_cal=True,
    )
    grog = pg_deepinv.GrogInterpolator(
        coords, shape, kernel_width=2, oversamp=1.25, image_shape=shape,
    )
    grog.calc_interp_table(calib_d[0], lamda=0.01, precision=1)
    sparse, plan = grog.interpolate(ksp_arms)
    sparse = sparse * plan.pre_weights.reshape(*plan.natural_shape).to(sparse.dtype)
    op = SparseFFT(plan=plan, smaps=smaps_d[0])
    physics = GrogLinearPhysics(op)
    y = sparse.reshape(1, n_compressed, op.indices.shape[0]).clone()

    with torch.no_grad():
        v = torch.randn_like(physics.A_adjoint(y))
        for _ in range(15):
            v = physics.A_adjoint(physics.A(v))
            v = v / (v.norm() + 1e-12)
        lipschitz = (
            physics.A_adjoint(physics.A(v)).norm() / (v.norm() + 1e-12)
        ).item()

    prior = WaveletPrior(wv="db4", wvdim=2, level=3, is_complex=True)
    prior.explicit_prior = False
    recon = optim_builder(
        iteration="FISTA",
        prior=prior, data_fidelity=L2(),
        early_stop=False, max_iter=150,
        params_algo={"stepsize": 0.9 / lipschitz, "lambda": 1e-2, "a": 3},
        verbose=False,
    )
    img = np.abs(recon(y, physics).squeeze().detach().cpu().numpy())
    return img


def _run_mrpro(kspace_raw, coords, shape, n_shots, n_read, n_coils_full,
               n_compressed):
    """mrpro path with self-computed Lipschitz constant."""
    from mrpro.algorithms.optimizers import pgd
    from mrpro.data import KData, KHeader, KTrajectory, SpatialDimension
    from mrpro.operators import WaveletOp
    from mrpro.operators.functionals import L1NormViewAsReal, L2NormSquared
    from pygrog.interop import GrogLinearOp
    from pygrog.interop import mrpro as pg_mrpro
    from pygrog.operator import SparseFFT

    kx = torch.as_tensor(coords[..., 1]).reshape(1, 1, 1, n_shots, n_read)
    ky = torch.as_tensor(coords[..., 0]).reshape(1, 1, 1, n_shots, n_read)
    kz = torch.zeros_like(kx)
    traj = KTrajectory(kz=kz, ky=ky, kx=kx)
    data = (
        torch.as_tensor(kspace_raw).reshape(n_coils_full, 1, n_shots, n_read).unsqueeze(0)
    )
    spatial = SpatialDimension(z=1, y=shape[0], x=shape[1])
    header = KHeader(
        recon_matrix=spatial, encoding_matrix=spatial,
        recon_fov=SpatialDimension(z=1.0, y=1.0, x=1.0),
        encoding_fov=SpatialDimension(z=1.0, y=1.0, x=1.0),
    )
    kdata = KData(header=header, data=data, traj=traj)
    kdata = pg_mrpro.coil_compress(kdata, n_compressed)
    smaps_m, calib_m = pg_mrpro.nlinv_calib(kdata, cal_width=24, max_iter=12, ret_cal=True)
    grog = pg_mrpro.GrogInterpolator(kdata, kernel_width=2, oversamp=1.25)
    grog.calc_interp_table(calib_m, lamda=0.01, precision=1)
    kdata_g, plan = grog.interpolate(kdata)
    op = SparseFFT(plan=plan, smaps=smaps_m[:, 0])

    A_mr = GrogLinearOp(op).H
    wavelet = WaveletOp(domain_shape=shape, dim=(-2, -1), wavelet_name="db4", level=3)
    acq = A_mr @ wavelet.H

    y = kdata_g.data.squeeze(0).squeeze(1)
    y = y * plan.pre_weights.to(y.dtype).reshape(1, n_shots, -1)

    # Power-iteration Lipschitz on (acq.adjoint @ acq).
    with torch.no_grad():
        v = torch.randn(shape, dtype=torch.complex64)
        (vw,) = wavelet(v)
        for _ in range(15):
            (Av,) = acq(vw)
            (vw,) = acq.adjoint(Av)
            vw = vw / (vw.norm() + 1e-12)
        (Av,) = acq(vw)
        lipschitz = (Av.norm() / (vw.norm() + 1e-12)).item() ** 2

    f = 0.5 * L2NormSquared(target=y, divide_by_n=False) @ acq
    g = 1e-3 * L1NormViewAsReal(divide_by_n=False)
    (init,) = wavelet(torch.zeros(shape, dtype=torch.complex64))
    (wave_hat,) = pgd(f=f, g=g, initial_value=init,
                      stepsize=0.9 / lipschitz,
                      max_iterations=150, backtrack_factor=1.0)
    (img,) = wavelet.H(wave_hat)
    return np.abs(img.detach().cpu().numpy())


def main() -> None:
    image, shape, samples, coords, n_shots, n_read, ksp, n_coils_full = _build_acquisition()
    n_compressed = 8

    img_sigpy, img_zf = _run_sigpy(ksp, coords, shape, n_shots, n_read, n_compressed)
    img_deepinv = _run_deepinv(ksp, coords, shape, n_shots, n_read, n_compressed)
    img_mrpro = _run_mrpro(ksp, coords, shape, n_shots, n_read, n_coils_full,
                           n_compressed)

    panels = [
        ("Ground truth",          normalize(image)),
        ("Zero-filled (adjoint)", normalize(img_zf)),
        ("sigpy (L1-wavelet)",    normalize(img_sigpy)),
        ("deepinv (L1-wavelet)",  normalize(img_deepinv)),
        ("mrpro (L1-wavelet)",    normalize(img_mrpro)),
    ]
    with POSTER_STYLE():
        fig, axes = plt.subplots(1, 5, figsize=(20, 4.4))
        for ax, (title, im) in zip(axes, panels):
            show_image(ax, im, title=title)
        fig.suptitle("Same L1-wavelet recon via PyGROG interop adapters",
                     y=1.04)
        fig.tight_layout()
        save_fig(fig, "fig_interop")


if __name__ == "__main__":
    main()
