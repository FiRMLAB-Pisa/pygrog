"""fig_orc — Off-resonance correction (no-ORC | mri-nufft | PyGROG | B0 map).

Mirrors the ORC half of ``examples/example_gadgets.py`` but uses the
sparse (non-Cartesian) PyGROG ORC path — the canonical recipe.
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
from mrinufft.extras import make_b0map
from mrinufft.operators.off_resonance import MRIFourierCorrected
from mrinufft.trajectories.utils import Acquisition

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    CMAP_DIFF,
    POSTER_STYLE,
    center_crop_pad,
    nrmse,
    normalize,
    save_fig,
    show_image,
    synthetic_smaps,
)

from pygrog.calib import GrogInterpolator  # noqa: E402
from pygrog.gadgets import OffResonanceCorrection  # noqa: E402
from pygrog.operator import SparseFFT  # noqa: E402


def main() -> None:
    image = np.flip(get_mri(0, "T1"), axis=(0, 2))[90].astype(np.float32)
    shape = image.shape[-2:]
    image = center_crop_pad(image, shape)
    image /= image.max() + 1e-8
    n_coils = 16

    samples = initialize_2D_spiral(Nc=48, Ns=600, nb_revolutions=10).astype(np.float32)
    density = voronoi(samples)
    smaps = synthetic_smaps(shape, n_coils=n_coils)

    nufft = get_operator("finufft")(
        samples=samples, shape=shape, n_coils=n_coils,
        smaps=smaps, density=density, squeeze_dims=True,
    )

    brain_mask = image > 0.1 * image.max()
    b0_map, _ = make_b0map(shape, b0range=(-200, 200), mask=brain_mask)

    t_read = np.arange(samples.shape[1], dtype=np.float32) * Acquisition.default.raster_time
    readout_time = np.repeat(t_read[None, :], samples.shape[0], axis=0)

    orc_nufft = MRIFourierCorrected(
        nufft, b0_map=b0_map, readout_time=readout_time, mask=brain_mask,
    )
    kspace_off = orc_nufft.op(image.astype(np.complex64))
    img_no_orc = np.squeeze(np.abs(nufft.adj_op(kspace_off)))
    img_ref_orc = np.squeeze(np.abs(orc_nufft.adj_op(kspace_off)))

    # GROG plan + ORC SparseFFT (PyGROG path, no smaps for ORC operator).
    coords = (samples * np.asarray(shape, dtype=np.float32)).astype(np.float32)
    coil = smaps * image[None, ...]
    cart = np.fft.fftshift(
        np.fft.fftn(np.fft.ifftshift(coil, axes=(-2, -1)), axes=(-2, -1)),
        axes=(-2, -1),
    ).astype(np.complex64)
    cy, cx = shape[0] // 2, shape[1] // 2
    cs = 24
    calib = cart[:, cy - cs // 2:cy + cs // 2, cx - cs // 2:cx + cs // 2]
    grog = GrogInterpolator(
        shape=shape, coords=coords, kernel_width=2, oversamp=1.25, image_shape=shape,
    )
    grog.calc_interp_table(calib, lamda=0.01, precision=1)

    n_shots, n_read = samples.shape[:2]
    base_op = SparseFFT(plan=grog.plan)
    sqrt_w = grog.plan.pre_weights

    orc_pg = OffResonanceCorrection(
        base_op, field_map=b0_map.astype(np.float32),
        readout_time=readout_time, mask=brain_mask,
        n_components=-1, method="svd",
    )
    sparse_off = torch.as_tensor(
        grog.interpolate(
            kspace_off.astype(np.complex64).reshape(n_coils, n_shots, n_read),
            ret_image=False,
        )
    ) * sqrt_w.unsqueeze(0)
    coil_imgs = orc_pg.adjoint(sparse_off)
    img_pg_orc = np.abs(
        (coil_imgs * torch.as_tensor(smaps).conj()).sum(0).cpu().numpy()
    )

    img_no_orc = normalize(img_no_orc)
    img_ref_orc = normalize(img_ref_orc)
    img_pg_orc = normalize(img_pg_orc)
    err = nrmse(img_pg_orc, img_ref_orc)

    with POSTER_STYLE():
        fig, axes = plt.subplots(1, 4, figsize=(17, 4.6))
        im = axes[0].imshow(b0_map, cmap=CMAP_DIFF, origin="lower",
                            vmin=-200, vmax=200)
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        axes[0].set_title("B$_0$ field map [Hz]")
        cb = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=12)
        show_image(axes[1], img_no_orc, title="No correction")
        show_image(axes[2], img_ref_orc, title="mri-nufft ORC (reference)")
        show_image(axes[3], img_pg_orc,
                   title=f"PyGROG ORC   NRMSE={err:.3f}")
        fig.tight_layout()
        save_fig(fig, "fig_orc")


if __name__ == "__main__":
    main()
