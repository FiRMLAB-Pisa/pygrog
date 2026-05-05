"""fig_mrf_qualitative — in-vivo MRF subspace coefficients (PyGROG vs FINUFFT).

Headline real-data figure for the paper write-up. Shows central axial slices
of the K=5 subspace coefficient images recovered from the in-vivo MRF dataset
shipped under ``pygrog/benchmark/data/``.

By default reads cached arrays from ``pygrog/benchmark/results/`` produced by
``benchmark/run_benchmarks.py``. Pass ``--rerun`` to recompute (slow,
GPU-recommended) — see ``pygrog/benchmark/README.md``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    BENCHMARK_DIR,
    BENCHMARK_RESULTS_DIR,
    CMAP_GRAY,
    POSTER_STYLE,
    nrmse,
    normalize,
    save_fig,
)


def _ensure_cache(rerun: bool) -> tuple[Path, Path]:
    grog = BENCHMARK_RESULTS_DIR / "coeff_grog.npy"
    nufft = BENCHMARK_RESULTS_DIR / "coeff_nufft.npy"
    if rerun or not (grog.exists() and nufft.exists()):
        print("[fig_mrf_qualitative] (re)running benchmark to refresh coeffs…")
        subprocess.run(
            [sys.executable, "run_benchmarks.py"],
            cwd=BENCHMARK_DIR,
            check=True,
        )
    return grog, nufft


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Re-run benchmark to refresh cached coefficients.",
    )
    parser.add_argument(
        "--slice",
        type=int,
        default=None,
        help="Slice index along axis 1 (default: central).",
    )
    args = parser.parse_args()

    grog_p, nufft_p = _ensure_cache(args.rerun)
    coeff_grog = np.load(grog_p)
    coeff_nufft = np.load(nufft_p)
    assert (
        coeff_grog.shape == coeff_nufft.shape
    ), f"shape mismatch: {coeff_grog.shape} vs {coeff_nufft.shape}"
    K = coeff_grog.shape[0]
    if coeff_grog.ndim == 4:
        z = args.slice if args.slice is not None else coeff_grog.shape[1] // 2
        slabs_g = np.abs(coeff_grog[:, z, :, :])
        slabs_n = np.abs(coeff_nufft[:, z, :, :])
    else:  # 2D fallback
        slabs_g = np.abs(coeff_grog)
        slabs_n = np.abs(coeff_nufft)

    metric = nrmse(slabs_g, slabs_n)

    with POSTER_STYLE():
        fig, axes = plt.subplots(2, K, figsize=(3.4 * K, 7.2))
        for k in range(K):
            ref = normalize(slabs_n[k])
            est = normalize(slabs_g[k])
            axes[0, k].imshow(ref, cmap=CMAP_GRAY, origin="lower", vmin=0.0, vmax=1.0)
            axes[1, k].imshow(est, cmap=CMAP_GRAY, origin="lower", vmin=0.0, vmax=1.0)
            for ax in (axes[0, k], axes[1, k]):
                ax.set_xticks([])
                ax.set_yticks([])
            axes[0, k].set_title(f"coefficient #{k + 1}")
        axes[0, 0].set_ylabel("FINUFFT", rotation=90)
        axes[1, 0].set_ylabel("PyGROG", rotation=90)
        fig.suptitle(
            f"In-vivo MRF: subspace coefficients   (volume NRMSE = {metric:.3f})",
            y=1.02,
        )
        fig.tight_layout()
        save_fig(fig, "fig_mrf_qualitative")


if __name__ == "__main__":
    main()
