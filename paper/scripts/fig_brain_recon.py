"""fig_brain_recon — BrainWeb spiral, NUFFT reference vs PyGROG.

Three columns:
1. mri-nufft adjoint reference (zero-filled SENSE)
2. PyGROG GROG → SparseFFT reconstruction
3. signed difference (×5) with NRMSE annotation
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
    nrmse,
    normalize,
    save_fig,
    show_diff,
    show_image,
    synthetic_smaps,
)

from pygrog.calib import GrogInterpolator  # noqa: E402
from pygrog.operator import SparseFFT  # noqa: E402


def main() -> None:
    image = np.flip(get_mri(0, "T1"), axis=(0, 2))[90].astype(np.float32)
    image /= image.max() + 1e-8
    shape = image.shape
    n_coils = 16

    samples = initialize_2D_spiral(Nc=48, Ns=600, nb_revolutions=10).astype(np.float32)
    density = voronoi(samples)
    smaps = synthetic_smaps(shape, n_coils=n_coils)

    nufft = get_operator("finufft")(
        samples=samples, shape=shape, n_coils=n_coils,
        smaps=smaps, density=density, squeeze_dims=True,
    )
    kspace = nufft.op(image.astype(np.complex64))
    img_ref = np.abs(nufft.adj_op(kspace))

    # GROG calibration from a 24x24 Cartesian patch of the coil-modulated image.
    coil = smaps * image[None, ...]
    cart = np.fft.fftshift(
        np.fft.fftn(np.fft.ifftshift(coil, axes=(-2, -1)), axes=(-2, -1)),
        axes=(-2, -1),
    ).astype(np.complex64)
    cy, cx = shape[0] // 2, shape[1] // 2
    cs = 24
    calib = cart[:, cy - cs // 2:cy + cs // 2, cx - cs // 2:cx + cs // 2]

    coords = (samples * np.asarray(shape, dtype=np.float32)).astype(np.float32)
    grog = GrogInterpolator(
        shape=shape, coords=coords, kernel_width=2, oversamp=1.25, image_shape=shape,
    )
    grog.calc_interp_table(calib, lamda=0.01, precision=1)

    n_shots, n_read = samples.shape[:2]
    sparse = grog.interpolate(
        kspace.astype(np.complex64).reshape(n_coils, n_shots, n_read),
        ret_image=False,
    )
    sqrt_w = np.asarray(grog.plan.pre_weights)
    op = SparseFFT(plan=grog.plan, smaps=smaps)
    img_grog = np.abs(op.adjoint(torch.as_tensor(sparse * sqrt_w[None])))

    img_ref_n = normalize(img_ref)
    img_grog_n = normalize(img_grog)
    diff5 = 5.0 * (img_grog_n - img_ref_n)
    metric = nrmse(img_grog_n, img_ref_n)

    with POSTER_STYLE():
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
        show_image(axes[0], img_ref_n, title="mri-nufft (reference)")
        show_image(axes[1], img_grog_n, title="PyGROG (GROG → SparseFFT)")
        show_diff(axes[2], diff5, vlim=0.5,
                  title=f"5 × difference   NRMSE={metric:.3f}")
        fig.tight_layout()
        save_fig(fig, "fig_brain_recon")


if __name__ == "__main__":
    main()
