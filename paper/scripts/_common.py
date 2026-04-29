"""Shared utilities for the PyGROG poster figures.

All figure scripts in this directory import from ``_common`` so the visual
style (fonts, dpi, colormaps, transparent background) and the input data
paths stay in sync.

Usage
-----
>>> from _common import POSTER_STYLE, save_fig, synthetic_smaps, nrmse
>>> with POSTER_STYLE:
...     fig, ax = plt.subplots()
...     ...
...     save_fig(fig, "my_figure")
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PAPER_DIR = Path(__file__).resolve().parents[1]
FIGURES_DIR = PAPER_DIR / "figures"
PYGROG_ROOT = PAPER_DIR.parent
BENCHMARK_DIR = PYGROG_ROOT / "benchmark"
BENCHMARK_DATA_DIR = BENCHMARK_DIR / "data"
BENCHMARK_RESULTS_DIR = BENCHMARK_DIR / "results"

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
# Poster-friendly: large fonts (>=14 pt), colorblind-safe colormaps,
# transparent background so panels can drop onto any slide background.
_POSTER_RC = {
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 20,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "image.cmap": "viridis",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.transparent": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
}

# Magnitude images use viridis; signed difference maps use RdBu_r.
CMAP_MAG = "viridis"
CMAP_GRAY = "gray"
CMAP_DIFF = "RdBu_r"


@contextmanager
def POSTER_STYLE() -> Iterator[None]:
    """Context manager applying the poster matplotlib style."""
    with mpl.rc_context(_POSTER_RC):
        yield


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def save_fig(fig: plt.Figure, name: str, dpi: int = 300) -> Path:
    """Save *fig* as ``paper/figures/<name>.png`` and return the path."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / f"{name}.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight", transparent=False)
    plt.close(fig)
    print(f"[paper] wrote {out}")
    return out


# ---------------------------------------------------------------------------
# Synthetic acquisition helpers (lifted from pygrog/examples/*)
# ---------------------------------------------------------------------------
def synthetic_smaps(shape: tuple[int, int], n_coils: int = 8) -> np.ndarray:
    """Smooth multi-coil maps reused across the BrainWeb examples."""
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


def center_crop_pad(arr: np.ndarray, target: tuple[int, ...]) -> np.ndarray:
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


def normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    return x / (np.abs(x).max() + 1e-12)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def nrmse(estimate: np.ndarray, reference: np.ndarray) -> float:
    estimate = np.abs(estimate)
    reference = np.abs(reference)
    num = np.sqrt(((estimate - reference) ** 2).mean())
    den = np.sqrt((reference**2).mean()) + 1e-12
    return float(num / den)


def psnr(estimate: np.ndarray, reference: np.ndarray) -> float:
    estimate = np.abs(estimate)
    reference = np.abs(reference)
    mse = ((estimate - reference) ** 2).mean()
    if mse <= 0:
        return float("inf")
    peak = float(reference.max())
    return float(20.0 * np.log10(peak / np.sqrt(mse)))


# ---------------------------------------------------------------------------
# Panel helpers
# ---------------------------------------------------------------------------
def show_image(ax: plt.Axes, img: np.ndarray, *, title: str | None = None,
               cmap: str = CMAP_GRAY, vmin: float | None = 0.0,
               vmax: float | None = 1.0):
    ax.imshow(np.abs(img), cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    if title is not None:
        ax.set_title(title)


def show_diff(ax: plt.Axes, diff: np.ndarray, *, vlim: float = 0.1,
              title: str | None = None, cbar: bool = True):
    im = ax.imshow(diff, cmap=CMAP_DIFF, origin="lower", vmin=-vlim, vmax=vlim)
    ax.set_xticks([])
    ax.set_yticks([])
    if title is not None:
        ax.set_title(title)
    if cbar:
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=12)
    return im
