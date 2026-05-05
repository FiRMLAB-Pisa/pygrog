"""fig_solvers — CG vs PCG vs LSMR convergence + final-image NRMSE bar.

Left  : convergence curves (relative residual vs iteration).
Right : NRMSE bar chart of final reconstruction vs ground truth.
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
from _common import (
    POSTER_STYLE,
    nrmse,
    normalize,
    save_fig,
    synthetic_smaps,
)

from pygrog import PolynomialPreconditioner
from pygrog.calib import GrogInterpolator
from pygrog.operator import MaskedFFT


def main() -> None:
    image = np.flip(get_mri(0, "T1"), axis=(0, 2))[90].astype(np.float32)
    image /= image.max() + 1e-8
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
    kspace = nufft.op(image.astype(np.complex64))

    grog = GrogInterpolator(
        shape=shape,
        coords=samples.reshape(48, 600, 2),
        kernel_width=2,
        oversamp=2.0,
        image_shape=shape,
    )
    calib_full = (smaps * image[None, ...]).astype(np.complex64)
    grog.calc_interp_table(calib_full, lamda=0.01, precision=1)

    kspace_arms = np.asarray(kspace).reshape(n_coils, 48, 600).astype(np.complex64)
    kgrid, mplan = grog.interpolate(kspace_arms, grid=True)
    op = MaskedFFT(plan=mplan, smaps=torch.as_tensor(smaps))
    b = torch.as_tensor(np.asarray(kgrid))

    res_cg, res_lsmr, res_pcg = [], [], []

    img_cg = op.solve(
        b,
        method="cg",
        max_iter=20,
        damp=1e-3,
        callback=lambda _k, _x, r: res_cg.append(r),
    )
    img_lsmr = op.solve(
        b,
        method="lsmr",
        max_iter=20,
        callback=lambda _k, _x, r: res_lsmr.append(r),
    )

    pc = PolynomialPreconditioner(op, degree=3, n_power_iter=10)
    img_pcg = op.solve(
        b,
        method="cg",
        max_iter=20,
        damp=1e-3,
        preconditioner=pc,
        callback=lambda _k, _x, r: res_pcg.append(r),
    )

    ref = normalize(image)
    metrics = {
        "CG": nrmse(normalize(np.abs(img_cg.cpu().numpy())), ref),
        "LSMR": nrmse(normalize(np.abs(img_lsmr.cpu().numpy())), ref),
        f"PCG (deg={pc.degree})": nrmse(normalize(np.abs(img_pcg.cpu().numpy())), ref),
    }

    with POSTER_STYLE():
        fig, axes = plt.subplots(
            1, 2, figsize=(13, 5.0), gridspec_kw={"width_ratios": [1.4, 1.0]}
        )
        axes[0].semilogy(res_cg, "-o", label="CG", lw=2)
        axes[0].semilogy(res_lsmr, "-s", label="LSMR", lw=2)
        axes[0].semilogy(res_pcg, "-^", label=f"PCG (deg={pc.degree})", lw=2)
        axes[0].set_xlabel("iteration")
        axes[0].set_ylabel(r"$\|r_k\| / \|r_0\|$")
        axes[0].set_title("Solver convergence (MaskedFFT)")
        axes[0].grid(True, which="both", alpha=0.3)
        axes[0].legend(frameon=False)

        names = list(metrics.keys())
        vals = list(metrics.values())
        bars = axes[1].bar(names, vals, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
        axes[1].set_ylabel("NRMSE vs ground truth")
        axes[1].set_title("Final-image accuracy")
        for b_, v in zip(bars, vals, strict=False):
            axes[1].text(
                b_.get_x() + b_.get_width() / 2.0,
                v,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=12,
            )
        axes[1].tick_params(axis="x", rotation=15)

        fig.tight_layout()
        save_fig(fig, "fig_solvers")


if __name__ == "__main__":
    main()
