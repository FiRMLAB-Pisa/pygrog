#!/usr/bin/env python3
"""Estimate sensitivity maps and calibration k-space via NLINV for the MRF benchmark dataset.

Loads kspace.npy, trajectory.npy and dcf.npy from the benchmark data directory,
runs NLINV calibration using the full non-Cartesian dataset (which internally selects
the low-resolution calibration region based on the requested cal_width), then saves:

  - smaps.npy       : complex sensitivity maps of shape (n_coils, ny, nx)
  - smaps_qc.png    : QC figure showing map magnitudes and calibration k-space
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("benchmark/data"),
        help="Directory containing kspace.npy, trajectory.npy, dcf.npy.",
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs="+",
        default=[220, 220, 220],
        help="Target image shape. Default: 220 220 220.",
    )
    parser.add_argument(
        "--cal-width",
        type=int,
        default=24,
        help="Calibration region width in pixels. Default: 24.",
    )
    return parser.parse_args()


def main() -> None:
    from pygrog.utils import nlinv_calib

    args = parse_args()
    data_dir = args.data_dir
    shape = tuple(args.shape)
    cal_width = args.cal_width

    print(f"Loading benchmark data from {data_dir} ...")
    kspace = np.load(data_dir / "kspace.npy")          # (T, C, spokes, readout)
    trajectory = np.load(data_dir / "trajectory.npy")  # (T, spokes, readout, ndim)
    dcf = np.load(data_dir / "dcf.npy")                # (T, spokes, readout)

    n_frames, n_coils, n_spokes, n_readout = kspace.shape
    ndim = trajectory.shape[-1]
    print(
        f"  kspace:     {kspace.shape}  (T={n_frames}, C={n_coils}, spokes={n_spokes}, readout={n_readout})"
    )
    print(f"  trajectory: {trajectory.shape}")
    print(f"  dcf:        {dcf.shape}")

    # Flatten all frames into a single non-Cartesian samples dimension.
    # nlinv_calib internally selects the low-res calibration region from coords.
    y = kspace.swapaxes(0, 1).reshape(n_coils, -1).astype(np.complex64)
    coords = trajectory.reshape(-1, ndim).astype(np.float32)
    weights = dcf.reshape(-1).astype(np.float32)

    print(f"Running NLINV calibration (shape={shape}, cal_width={cal_width}) ...")
    smaps, grappa_train, image = nlinv_calib(
        y,
        cal_width=cal_width,
        shape=shape,
        coords=coords,
        weights=weights,
        ret_cal=True,
        ret_image=True,
    )

    print(f"  smaps:        {smaps.shape}  dtype={smaps.dtype}")
    print(f"  grappa_train: {grappa_train.shape}  dtype={grappa_train.dtype}")

    # Save sensitivity maps
    np.save(data_dir / "smaps.npy", smaps)
    print(f"Saved smaps.npy -> {data_dir / 'smaps.npy'}")

    # Save QC figure
    # For 3D volumes, take the central slice along the first spatial axis
    def _central_slice(arr):
        """Return 2D central slice of a ND array (last two dims are displayed)."""
        while arr.ndim > 2:
            arr = arr[arr.shape[0] // 2]
        return arr

    n_show = min(n_coils, 8)
    ncols = n_show
    fig, axes = plt.subplots(3, ncols, figsize=(2.5 * ncols, 7))
    fig.suptitle(
        f"NLINV calibration QC  |  cal_width={cal_width}  |  shape={shape}", fontsize=11
    )

    for c in range(n_show):
        # Row 0: sensitivity map magnitude (central slice)
        ax = axes[0, c]
        ax.imshow(np.abs(_central_slice(smaps[c])), cmap="gray")
        ax.set_title(f"smap {c}", fontsize=8)
        ax.axis("off")

        # Row 1: sensitivity map phase (central slice)
        ax = axes[1, c]
        ax.imshow(np.angle(_central_slice(smaps[c])), cmap="hsv", vmin=-np.pi, vmax=np.pi)
        ax.set_title(f"phase {c}", fontsize=8)
        ax.axis("off")

        # Row 2: grappa_train log-magnitude (central slice)
        ax = axes[2, c]
        cal_mag = np.log1p(np.abs(_central_slice(grappa_train[c])))
        ax.imshow(cal_mag, cmap="viridis")
        ax.set_title(f"cal {c}", fontsize=8)
        ax.axis("off")

    axes[0, 0].set_ylabel("smap |·|", fontsize=8)
    axes[1, 0].set_ylabel("smap ∠", fontsize=8)
    axes[2, 0].set_ylabel("log|cal k-space|", fontsize=8)

    # Also show the reconstructed image if available
    if image is not None:
        fig2, ax2 = plt.subplots(1, 1, figsize=(4, 4))
        ax2.imshow(np.abs(_central_slice(np.asarray(image))), cmap="gray")
        ax2.set_title("NLINV image estimate")
        ax2.axis("off")
        fig2.tight_layout()
        fig2.savefig(data_dir / "smaps_image_qc.png", dpi=100)
        plt.close(fig2)
        print(f"Saved smaps_image_qc.png -> {data_dir / 'smaps_image_qc.png'}")

    fig.tight_layout()
    fig.savefig(data_dir / "smaps_qc.png", dpi=100)
    plt.close(fig)
    print(f"Saved smaps_qc.png -> {data_dir / 'smaps_qc.png'}")


if __name__ == "__main__":
    main()
