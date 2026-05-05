"""fig_subspace — subspace coefficient images + temporal basis curves.

Top row: |coefficient| images for K basis functions (PyGROG SubspaceSparseFFT).
Bottom row: corresponding temporal basis curves φ_k(t).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from mrinufft import get_operator, initialize_2D_spiral
from mrinufft.density import voronoi
from mrinufft.extras import fse_simulation, get_brainweb_map

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    CMAP_GRAY,
    POSTER_STYLE,
    normalize,
    save_fig,
    synthetic_smaps,
)

from pygrog.calib import GrogInterpolator
from pygrog.gadgets import SubspaceProjection, SubspaceSparseFFT
from pygrog.operator import SparseFFT


def _estimate_basis(train, rank):
    _, _, vh = np.linalg.svd(train, full_matrices=False)
    return vh[:rank]


def main() -> None:
    m0, t1, t2 = get_brainweb_map(0)
    m0 = np.flip(m0, axis=(0, 2))[90].astype(np.float32)
    t1 = np.flip(t1, axis=(0, 2))[90].astype(np.float32)
    t2 = np.flip(t2, axis=(0, 2))[90].astype(np.float32)
    image = m0 / (m0.max() + 1e-8)
    shape = image.shape
    n_coils = 16

    samples = initialize_2D_spiral(Nc=48, Ns=600, nb_revolutions=10).astype(np.float32)
    density = voronoi(samples)
    smaps = synthetic_smaps(shape, n_coils=n_coils)

    nufft = get_operator("finufft")(
        samples=samples,
        shape=shape,
        n_coils=n_coils,
        smaps=smaps,
        density=density,
        squeeze_dims=True,
    )

    etl = 8
    te = np.arange(etl, dtype=np.float32) * 8.0
    tr = 3000.0
    frames = fse_simulation(m0, t1, t2, te, tr).astype(np.float32)

    kspace_frames = np.stack(
        [nufft.op(frames[t].astype(np.complex64)) for t in range(etl)],
        axis=0,
    )

    # Subspace basis from physiological-range training signals.
    t1_v = np.linspace(float(t1[t1 > 0].min()) + 1.0, float(t1.max()), 60)
    t2_v = np.linspace(float(t2[t2 > 0].min()) + 1.0, float(t2.max()), 60)
    t1g, t2g = np.meshgrid(t1_v, t2_v)
    train = fse_simulation(1.0, t1g.ravel(), t2g.ravel(), te, tr).astype(np.float32)
    rank = 4
    _estimate_basis(train.T, rank)

    # GROG plan with T baked into natural shape.
    coords = (samples * np.asarray(shape, dtype=np.float32)).astype(np.float32)
    n_shots, n_read = samples.shape[:2]
    coords_sub = np.broadcast_to(
        coords[None, None],
        (etl, 1, n_shots, n_read, 2),
    ).copy()
    coil = smaps * image[None, ...]
    cart = np.fft.fftshift(
        np.fft.fftn(np.fft.ifftshift(coil, axes=(-2, -1)), axes=(-2, -1)),
        axes=(-2, -1),
    ).astype(np.complex64)
    cy, cx = shape[0] // 2, shape[1] // 2
    cs = 24
    calib = cart[:, cy - cs // 2 : cy + cs // 2, cx - cs // 2 : cx + cs // 2]
    grog = GrogInterpolator(
        shape=shape,
        coords=coords_sub,
        kernel_width=2,
        oversamp=1.25,
        image_shape=shape,
    )
    grog.calc_interp_table(calib, lamda=0.01, precision=1)

    base_op = SparseFFT(plan=grog.plan, smaps=smaps)
    kspace_sub = (
        kspace_frames.reshape(etl, n_coils, 1, n_shots, n_read)
        .transpose(1, 0, 2, 3, 4)[None]
        .astype(np.complex64)
    )
    sparse_sub = torch.as_tensor(grog.interpolate(kspace_sub)).reshape(
        1, n_coils, *grog.plan.natural_shape
    )
    proj = SubspaceProjection(n_components=rank)
    proj.fit(torch.as_tensor(train, dtype=torch.float32))
    sub_op = SubspaceSparseFFT(
        base_op, proj.basis.to(torch.complex64), encoding_axis=-5
    )
    coeff = np.asarray(sub_op.adjoint(sparse_sub))[0]  # (rank, H, W)

    with POSTER_STYLE():
        fig, axes = plt.subplots(
            2,
            rank,
            figsize=(4.0 * rank, 8.5),
            gridspec_kw={"height_ratios": [1.4, 1.0]},
        )
        for r in range(rank):
            img = normalize(np.abs(coeff[r]))
            axes[0, r].imshow(img, cmap=CMAP_GRAY, origin="lower", vmin=0.0, vmax=1.0)
            axes[0, r].set_xticks([])
            axes[0, r].set_yticks([])
            axes[0, r].set_title(f"|coefficient #{r + 1}|")

            phi = np.asarray(proj.basis[r])
            axes[1, r].plot(te, np.real(phi), "-o", label="Re")
            axes[1, r].plot(te, np.imag(phi), "--s", label="Im")
            axes[1, r].axhline(0.0, color="k", lw=0.6, alpha=0.4)
            axes[1, r].set_xlabel("TE [ms]")
            axes[1, r].set_title(f"$\\varphi_{{{r + 1}}}(t)$")
            if r == 0:
                axes[1, r].legend(loc="best", frameon=False)
        fig.tight_layout()
        save_fig(fig, "fig_subspace")


if __name__ == "__main__":
    main()
