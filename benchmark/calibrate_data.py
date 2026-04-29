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
import time
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark/results"),
        help="Directory for output figures. Default: benchmark/results.",
    )
    return parser.parse_args()


def main() -> None:
    from pygrog.utils import nlinv_calib

    args = parse_args()
    data_dir = args.data_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    shape = tuple(args.shape)
    cal_width = args.cal_width

    print(f"Loading benchmark data from {data_dir} ...")
    kspace = np.load(data_dir / "kspace.npy")        # (T, C, k1, k0)
    trajectory = np.load(data_dir / "trajectory.npy")  # (T, k1, k0, ndim)
    dcf = np.load(data_dir / "dcf.npy")              # (T, k1, k0)

    n_frames, n_coils, n_spokes, n_readout = kspace.shape
    ndim = trajectory.shape[-1]
    print(f"  kspace:     {kspace.shape}  (T={n_frames}, C={n_coils}, spokes={n_spokes}, readout={n_readout})")
    print(f"  trajectory: {trajectory.shape}")
    print(f"  dcf:        {dcf.shape}")

    # Flatten all frames into a single non-Cartesian samples dimension —
    # i.e. treat T as part of the trajectory (the union of every frame's
    # spokes).  nlinv_calib internally selects the low-res calibration
    # region from coords.
    y = kspace.swapaxes(0, 1).reshape(n_coils, -1).astype(np.complex64)
    coords = trajectory.reshape(-1, ndim).astype(np.float32)
    weights = dcf.reshape(-1).astype(np.float32)

    print(f"Running NLINV calibration (shape={shape}, cal_width={cal_width}) ...")
    t0 = time.time()
    smaps, grappa_train, image = nlinv_calib(
        y,
        cal_width=cal_width,
        shape=shape,
        coords=coords,
        weights=weights,
        ret_cal=True,
        ret_image=True,
    )
    elapsed = time.time() - t0
    print(f"  NLINV calibration completed in {elapsed:.1f}s")

    print(f"  smaps:        {smaps.shape}  dtype={smaps.dtype}")
    print(f"  grappa_train: {grappa_train.shape}  dtype={grappa_train.dtype}")

    # Save sensitivity maps
    np.save(data_dir / "smaps.npy", smaps)
    print(f"Saved smaps.npy -> {data_dir / 'smaps.npy'}")

    # -----------------------------------------------------------------------
    # QC figure: 3 rows (axial, sagittal, coronal) × (1 + n_coils) columns
    #   col 0       : NLINV image estimate — magnitude, grey
    #   col 1..C    : sensitivity maps — RGB hue=phase, value=magnitude
    # -----------------------------------------------------------------------
    import matplotlib.colors as mcolors

    def _slice(vol, axis):
        """Return the central 2D slice of *vol* perpendicular to *axis*, CCW rotated."""
        if vol.ndim < 3:
            return np.rot90(np.squeeze(vol), k=1)
        idx = vol.shape[axis] // 2
        sl = np.take(vol, idx, axis=axis)
        return np.rot90(sl, k=1)

    def _phase_mag_rgb(arr2d):
        """Complex 2D array → RGB image: hue=phase, value=magnitude."""
        mag = np.abs(arr2d)
        mag_norm = mag / (mag.max() + 1e-8)
        hue = (np.angle(arr2d) + np.pi) / (2 * np.pi)  # [0, 1]
        sat = np.ones_like(hue)
        return mcolors.hsv_to_rgb(np.stack([hue, sat, mag_norm], axis=-1))

    n_show = min(n_coils, 6)
    ncols = 1 + n_show
    row_labels = ["Axial", "Coronal", "Sagittal"]
    axes_idx = [2, 1, 0]  # axis2=axial (x,y), axis1=coronal (x,z), axis0=sagittal (y,z)

    img_vol = np.abs(np.asarray(image)) if image is not None else None

    fig, axes = plt.subplots(3, ncols, figsize=(2.5 * ncols, 7))
    # Ensure axes is always 2-D (handles ncols == 1 edge-case)
    axes = np.atleast_2d(axes)

    fig.suptitle(
        f"NLINV QC  |  cal_width={cal_width}  |  shape={shape}  |  t={elapsed:.1f}s",
        fontsize=11,
    )

    for row, (label, ax_idx) in enumerate(zip(row_labels, axes_idx, strict=False)):
        # col 0: NLINV image (magnitude)
        ax = axes[row, 0]
        if img_vol is not None:
            sl = _slice(img_vol, ax_idx)
            ax.imshow(sl, cmap="gray", interpolation="nearest")
        ax.set_title("image" if row == 0 else "", fontsize=8)
        ax.set_ylabel(label, fontsize=8)
        ax.axis("off")

        # cols 1…n_show: sensitivity maps (phase → RGB hue, magnitude → value)
        for c in range(n_show):
            ax = axes[row, 1 + c]
            sl = _slice(smaps[c], ax_idx)
            ax.imshow(_phase_mag_rgb(sl), interpolation="nearest")
            ax.set_title(f"coil {c}" if row == 0 else "", fontsize=8)
            ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_dir / "smaps_qc.png", dpi=120)
    plt.close(fig)
    print(f"Saved smaps_qc.png -> {output_dir / 'smaps_qc.png'}")


if __name__ == "__main__":
    main()
