#!/usr/bin/env python3
"""Plot benchmark figures from benchmark JSON outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _format_runtime(value_sec: float) -> str:
    if value_sec >= 1.0:
        return f"{value_sec:.2f} s"
    if value_sec >= 1e-3:
        return f"{value_sec * 1e3:.2f} ms"
    return f"{value_sec * 1e6:.2f} us"


def _format_memory(value_gb: float) -> str:
    if value_gb >= 1.0:
        return f"{value_gb:.2f} GB"
    return f"{value_gb * 1024.0:.2f} MB"


def _annotate_bars(ax, bars, formatter, x_shift: float = 0.0, stagger: bool = True):
    vals = [float(bar.get_height()) for bar in bars]
    ymax = max(vals) if vals else 0.0
    offset = max(0.01 * ymax, 0.003)
    n = len(vals)
    fontsize = 8 if n <= 8 else 7
    for i, (bar, val) in enumerate(zip(bars, vals, strict=False)):
        x = bar.get_x() + bar.get_width() / 2.0 + x_shift
        y = val + offset
        if stagger and i % 2 == 1:
            y += offset * 1.8
        ax.text(
            x,
            y,
            formatter(val),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            rotation=0,
        )


def _extract_case_value(case: dict, path: tuple[str, ...], default=np.nan):
    cur = case
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    if cur is None:
        return default
    return float(cur)


def _scaling_cases(results: dict) -> list[dict]:
    scaling = results.get("scaling", {})
    cases = scaling.get("cases", [])
    if not cases:
        raise ValueError(
            "results.json does not include scaling cases. Re-run run_benchmarks.py."
        )
    return cases


def _size_labels(cases: list[dict]) -> list[str]:
    labels = []
    for case in cases:
        k = case["samples_per_frame"] / 1000.0
        labels.append(f"{case['label']}\n({k:.1f}k)")
    return labels


def plot_preprocessing(results: dict, out: Path) -> None:
    cases = _scaling_cases(results)
    labels = _size_labels(cases)
    n = len(cases)
    x = np.arange(n)
    width = 0.36

    stages = [("planning", "Planning"), ("interpolation", "Interpolation")]

    def _stage_val(case: dict, device: str, stage: str, metric: str):
        val = _extract_case_value(
            case, ("preprocessing", device, stage, metric), np.nan
        )
        # Backward compatibility with older results.json that stored only combined preprocessing.
        if np.isnan(val) and stage == "interpolation":
            return _extract_case_value(case, ("preprocessing", device, metric), np.nan)
        if np.isnan(val) and stage == "planning":
            return 0.0
        return val

    fig, axes = plt.subplots(2, 2, figsize=(max(15, 2.4 * n + 3), 8.8), sharex=True)

    for col, (stage_key, stage_name) in enumerate(stages):
        ax_rt = axes[0, col]
        ax_mem = axes[1, col]

        runtime_cpu = [
            _stage_val(case, "cpu", stage_key, "runtime_sec") for case in cases
        ]
        runtime_gpu = [
            _stage_val(case, "gpu", stage_key, "runtime_sec") for case in cases
        ]
        bars_cpu = ax_rt.bar(
            x - width / 2, runtime_cpu, width, color="#4E79A7", label="CPU"
        )
        bars_gpu = ax_rt.bar(
            x + width / 2, runtime_gpu, width, color="#E15759", label="GPU"
        )
        _annotate_bars(ax_rt, bars_cpu, _format_runtime, x_shift=-0.04, stagger=False)
        _annotate_bars(ax_rt, bars_gpu, _format_runtime, x_shift=0.04, stagger=False)

        mem_cpu_ram = [_stage_val(case, "cpu", stage_key, "ram_gb") for case in cases]
        mem_gpu_ram = [_stage_val(case, "gpu", stage_key, "ram_gb") for case in cases]
        mem_gpu_vram = [_stage_val(case, "gpu", stage_key, "vram_gb") for case in cases]
        mem_cpu_vram = [0.0 for _ in cases]

        ax_mem.bar(x - width / 2, mem_cpu_ram, width, color="#4E79A7", label="CPU RAM")
        ax_mem.bar(x + width / 2, mem_gpu_ram, width, color="#E15759", label="GPU RAM")
        ax_mem.bar(
            x + width / 2,
            mem_gpu_vram,
            width,
            bottom=mem_gpu_ram,
            color="#E15759",
            hatch="///",
            alpha=0.9,
            label="GPU VRAM",
        )

        total_cpu = np.asarray(mem_cpu_ram, dtype=float) + np.asarray(
            mem_cpu_vram, dtype=float
        )
        total_gpu = np.asarray(mem_gpu_ram, dtype=float) + np.nan_to_num(
            np.asarray(mem_gpu_vram, dtype=float), nan=0.0
        )
        bars_mem_cpu = ax_mem.bar(x - width / 2, total_cpu, width, alpha=0.0)
        bars_mem_gpu = ax_mem.bar(x + width / 2, total_gpu, width, alpha=0.0)
        _annotate_bars(
            ax_mem, bars_mem_cpu, _format_memory, x_shift=-0.04, stagger=False
        )
        _annotate_bars(
            ax_mem, bars_mem_gpu, _format_memory, x_shift=0.04, stagger=False
        )

        ax_rt.set_title(f"{stage_name} Runtime")
        ax_mem.set_title(f"{stage_name} Memory")
        if col == 0:
            ax_rt.set_ylabel("Runtime (s)")
            ax_mem.set_ylabel("Memory (GB)")

    for panel in axes.ravel():
        panel.axvline(n - 1.5, color="#777777", linestyle="--", linewidth=1.0)
        panel.set_xticks(x)
        panel.set_xticklabels(labels, rotation=20, ha="right")

    runtime_handles = [
        Patch(facecolor="#4E79A7", label="CPU"),
        Patch(facecolor="#E15759", label="GPU"),
    ]
    memory_handles = [
        Patch(facecolor="#4E79A7", label="CPU RAM"),
        Patch(facecolor="#E15759", label="GPU RAM"),
        Patch(facecolor="#E15759", hatch="///", label="GPU VRAM"),
    ]
    fig.legend(handles=runtime_handles + memory_handles, loc="upper center", ncol=5)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _to_2d_slice(arr: np.ndarray, axis: int = 0) -> np.ndarray:
    """Extract a 2-D slice from an N-D array along the given spatial axis."""
    out = np.asarray(arr)
    ndim = out.ndim
    if ndim <= 2:
        return out
    mid = out.shape[axis] // 2
    return np.take(out, mid, axis=axis)


def _three_views(vol: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (axial, coronal, sagittal) 2-D slices from a 3-D volume.

    For 2-D input the same slice is returned for all three views.
    """
    vol = np.asarray(vol)
    if vol.ndim == 2:
        return vol, vol, vol
    if vol.ndim == 3:
        mz, my, mx = vol.shape[0] // 2, vol.shape[1] // 2, vol.shape[2] // 2
        return vol[mz, :, :], vol[:, my, :], vol[:, :, mx]
    # > 3-D: collapse leading dims first
    while vol.ndim > 3:
        vol = vol[vol.shape[0] // 2]
    return _three_views(vol)


def _normalize_unit(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32)
    maxv = float(out.max())
    if maxv <= 0.0:
        return np.zeros_like(out, dtype=np.float32)
    return out / maxv


def plot_subspace(
    coeff_nufft: np.ndarray, coeff_grog: np.ndarray, out: Path, label: str = "CPU"
) -> None:
    """Three-row coefficient comparison figure.

    Row 1: NUFFT reference magnitudes (gray)
    Row 2: GROG magnitudes (gray)
    Row 3: per-pixel MAPE between NUFFT and GROG (hot colormap)
    Each column is one subspace coefficient.  Average MAPE is annotated in the
    bottom-right corner of the MAPE panel.
    """
    k = min(coeff_nufft.shape[0], coeff_grog.shape[0])

    top_tiles = []  # NUFFT
    mid_tiles = []  # GROG
    mape_tiles = []  # MAPE

    eps = 1e-8
    for i in range(k):
        nufft_2d = np.abs(_to_2d_slice(coeff_nufft[i]))
        grog_2d = np.abs(_to_2d_slice(coeff_grog[i]))
        top_tiles.append(_normalize_unit(nufft_2d))
        mid_tiles.append(_normalize_unit(grog_2d))
        mape = np.abs(nufft_2d - grog_2d) / (nufft_2d + eps) * 100.0
        mape_tiles.append(mape.astype(np.float32))

    top_row = np.concatenate(top_tiles, axis=1)
    mid_row = np.concatenate(mid_tiles, axis=1)
    mape_row = np.concatenate(mape_tiles, axis=1)

    avg_mape = float(np.mean(mape_row))
    h = top_row.shape[0]

    # Two axes: grayscale canvas (NUFFT + GROG) and MAPE canvas.
    fig, (ax_gray, ax_mape) = plt.subplots(
        2,
        1,
        figsize=(3.2 * k, 9.6),
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.06},
    )

    gray_canvas = np.concatenate([top_row, mid_row], axis=0)
    ax_gray.imshow(gray_canvas, cmap="gray", origin="upper", vmin=0.0, vmax=1.0)
    tile_widths = [tile.shape[1] for tile in top_tiles]
    col_centers = []
    cursor = 0
    for w in tile_widths:
        col_centers.append(cursor + w / 2.0)
        cursor += w

    ax_gray.set_xticks(col_centers)
    ax_gray.set_xticklabels([rf"$\phi_{{{i}}}$" for i in range(k)], fontsize=10)
    ax_gray.xaxis.tick_top()
    ax_gray.tick_params(axis="x", length=0, pad=3)
    ax_gray.set_yticks([h / 2.0, h + h / 2.0])
    ax_gray.set_yticklabels(["NUFFT reference", "GROG"], fontsize=11)
    for tick in ax_gray.get_yticklabels():
        tick.set_rotation(90)
        tick.set_verticalalignment("center")
        tick.set_horizontalalignment("center")
    ax_gray.set_ylabel("", fontsize=11)

    cursor = 0
    for w in tile_widths[:-1]:
        cursor += w
        ax_gray.axvline(
            cursor - 0.5, color="#666666", linewidth=0.5, linestyle="--", alpha=0.6
        )
        ax_mape.axvline(
            cursor - 0.5, color="#666666", linewidth=0.5, linestyle="--", alpha=0.6
        )

    for spine in ["top", "right", "bottom", "left"]:
        ax_gray.spines[spine].set_visible(False)

    mape_vmax = float(np.percentile(mape_row, 95))
    im = ax_mape.imshow(
        mape_row, cmap="hot", origin="upper", vmin=0.0, vmax=max(mape_vmax, 1.0)
    )
    ax_mape.set_xticks([])
    ax_mape.set_yticks([mape_row.shape[0] / 2.0])
    ax_mape.set_yticklabels(["MAPE (%)"], fontsize=11)
    ax_mape.set_ylabel("")
    for spine in ["top", "right", "bottom", "left"]:
        ax_mape.spines[spine].set_visible(False)

    # Annotate average MAPE in the bottom-right corner of the MAPE panel.
    ax_mape.text(
        0.98,
        0.05,
        f"avg MAPE = {avg_mape:.1f}%",
        transform=ax_mape.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        color="white",
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "black", "alpha": 0.5},
    )

    # Shared colourbar for MAPE.
    cbar = fig.colorbar(
        im, ax=ax_mape, orientation="vertical", fraction=0.015, pad=0.01
    )
    cbar.set_label("MAPE (%)", fontsize=9)

    fig.suptitle(label, fontsize=14, fontweight="bold")
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_grog_views(coeff_grog: np.ndarray, out: Path, label: str = "CPU") -> None:
    """GROG-only panel: 3 orthogonal views x k coefficients.

    Rows:  axial / coronal / sagittal
    Cols:  each subspace coefficient
    Normalization is per-coefficient (shared across the three views so that
    relative intensity between planes is preserved).
    """
    k = coeff_grog.shape[0]
    view_names = ["Axial", "Coronal", "Sagittal"]

    tiles: list[list[np.ndarray]] = [[], [], []]  # [view_idx][coeff_idx]

    for i in range(k):
        vol = np.abs(coeff_grog[i])
        vmax = float(vol.max())
        scale = vmax if vmax > 0.0 else 1.0
        ax_sl, cor_sl, sag_sl = _three_views(vol)
        tiles[0].append((ax_sl / scale).astype(np.float32))
        tiles[1].append((cor_sl / scale).astype(np.float32))
        tiles[2].append((sag_sl / scale).astype(np.float32))

    # Each row has k tiles; pad narrower tiles to the same height per row.
    def _make_row(row_tiles: list[np.ndarray]) -> np.ndarray:
        # Tiles in the same column may differ in shape across views —
        # just concatenate; they are already per-coefficient-normalised.
        return np.concatenate(row_tiles, axis=1)

    rows = [_make_row(tiles[v]) for v in range(3)]

    # Pad rows to the same width (they should already match column-by-column).
    max_w = max(r.shape[1] for r in rows)
    rows = [
        np.pad(r, ((0, 0), (0, max_w - r.shape[1]))) if r.shape[1] < max_w else r
        for r in rows
    ]

    # Build column-divider positions for x-tick labels (coefficient index).
    tile_widths = [tiles[0][i].shape[1] for i in range(k)]
    col_centers = []
    cx = 0
    for w in tile_widths:
        col_centers.append(cx + w / 2.0)
        cx += w

    n_views = 3
    row_heights = [rows[v].shape[0] for v in range(n_views)]
    row_centers = []
    ry = 0
    for h in row_heights:
        row_centers.append(ry + h / 2.0)
        ry += h

    canvas = np.concatenate(rows, axis=0)
    fig_h = max(2.0 * n_views, 1.0 * canvas.shape[0] / canvas.shape[1] * 3.2 * k)

    fig, ax = plt.subplots(figsize=(3.2 * k, fig_h))
    ax.imshow(canvas, cmap="gray", origin="upper", vmin=0.0, vmax=1.0)

    ax.set_xticks(col_centers)
    ax.set_xticklabels([rf"$\phi_{{{i}}}$" for i in range(k)], fontsize=10)
    ax.set_yticks(row_centers)
    ax.set_yticklabels(view_names, fontsize=10)
    ax.tick_params(length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    # Draw faint dividers between columns.
    cx = 0
    for w in tile_widths[:-1]:
        cx += w
        ax.axvline(cx - 0.5, color="#555555", linewidth=0.5, linestyle="--")
    # Draw faint dividers between rows.
    ry = 0
    for h in row_heights[:-1]:
        ry += h
        ax.axhline(ry - 0.5, color="#555555", linewidth=0.5, linestyle="--")

    fig.tight_layout(pad=0.3)
    fig.suptitle(label, fontsize=14, fontweight="bold")
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_linop(results: dict, out: Path) -> None:
    cases = _scaling_cases(results)
    labels = _size_labels(cases)
    n = len(cases)
    x = np.arange(n)

    column_specs = [
        (
            "forward",
            "CPU",
            [("finufft_cpu", "NUFFT", "#4E79A7"), ("grog_cpu", "GROG", "#F28E2B")],
        ),
        (
            "forward",
            "GPU",
            [("cufinufft_gpu", "NUFFT", "#59A14F"), ("grog_gpu", "GROG", "#E15759")],
        ),
        (
            "adjoint",
            "CPU",
            [("finufft_cpu", "NUFFT", "#4E79A7"), ("grog_cpu", "GROG", "#F28E2B")],
        ),
        (
            "adjoint",
            "GPU",
            [("cufinufft_gpu", "NUFFT", "#59A14F"), ("grog_gpu", "GROG", "#E15759")],
        ),
    ]
    width = 0.35
    offsets = [-0.5 * width, 0.5 * width]

    fig, axes = plt.subplots(2, 4, figsize=(max(19, 2.7 * n + 7), 9.4), sharex=True)

    for col, (op_name, device_name, methods) in enumerate(column_specs):
        ax_rt = axes[0, col]
        ax_mem = axes[1, col]

        for (method_key, method_name, color), off in zip(
            methods, offsets, strict=False
        ):
            runtime_vals = np.array(
                [
                    _extract_case_value(
                        case, ("linop", op_name, method_key, "runtime_sec"), np.nan
                    )
                    for case in cases
                ],
                dtype=float,
            )
            ram_vals = np.array(
                [
                    _extract_case_value(
                        case, ("linop", op_name, method_key, "ram_gb"), np.nan
                    )
                    for case in cases
                ],
                dtype=float,
            )
            if method_key.endswith("_cpu"):
                vram_vals = np.zeros_like(ram_vals)
            else:
                vram_vals = np.array(
                    [
                        _extract_case_value(
                            case, ("linop", op_name, method_key, "vram_gb"), np.nan
                        )
                        for case in cases
                    ],
                    dtype=float,
                )

            bars_rt = ax_rt.bar(
                x + off,
                runtime_vals,
                width=width,
                color=color,
                label=f"{device_name} {method_name}",
            )
            ax_mem.bar(x + off, ram_vals, width=width, color=color)
            ax_mem.bar(
                x + off,
                vram_vals,
                width=width,
                bottom=ram_vals,
                color=color,
                hatch="///",
                alpha=0.9,
            )

            total_mem = np.where(
                np.isfinite(ram_vals),
                ram_vals + np.nan_to_num(vram_vals, nan=0.0),
                np.nan,
            )
            bars_mem_total = ax_mem.bar(x + off, total_mem, width=width, alpha=0.0)
            _annotate_bars(
                ax_rt, bars_rt, _format_runtime, x_shift=off * 0.35, stagger=False
            )
            _annotate_bars(
                ax_mem,
                bars_mem_total,
                _format_memory,
                x_shift=off * 0.35,
                stagger=False,
            )

        ax_rt.set_title(f"{op_name.capitalize()} ({device_name})")
        if col == 0:
            ax_rt.set_ylabel("Runtime (s)")
            ax_mem.set_ylabel("Memory (GB)")

        for panel in [ax_rt, ax_mem]:
            panel.axvline(n - 1.5, color="#777777", linestyle="--", linewidth=1.0)
            panel.set_xticks(x)
            panel.set_xticklabels(labels, rotation=20, ha="right")

    method_handles = [
        Patch(facecolor="#4E79A7", label="CPU NUFFT"),
        Patch(facecolor="#F28E2B", label="CPU GROG"),
        Patch(facecolor="#59A14F", label="GPU NUFFT"),
        Patch(facecolor="#E15759", label="GPU GROG"),
    ]
    component_handles = [
        Patch(facecolor="#BBBBBB", label="RAM"),
        Patch(facecolor="#BBBBBB", hatch="///", label="VRAM"),
    ]
    fig.legend(handles=method_handles + component_handles, loc="upper center", ncol=6)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-json", type=Path, default=Path("benchmark/results/results.json")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark/results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = _load_json(args.results_json)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    coeff_nufft = np.load(args.output_dir / "coeff_nufft.npy")
    coeff_grog = np.load(args.output_dir / "coeff_grog.npy")

    plot_preprocessing(results, args.output_dir / "figure_preprocessing.png")
    plot_linop(results, args.output_dir / "figure_linop.png")
    plot_subspace(
        coeff_nufft, coeff_grog, args.output_dir / "figure_coeffs_cpu.png", label="CPU"
    )
    plot_grog_views(
        coeff_grog, args.output_dir / "figure_grog_views_cpu.png", label="CPU"
    )

    cuda_nufft_path = args.output_dir / "coeff_nufft_cuda.npy"
    cuda_grog_path = args.output_dir / "coeff_grog_cuda.npy"
    if cuda_nufft_path.exists() and cuda_grog_path.exists():
        coeff_nufft_cuda = np.load(cuda_nufft_path)
        coeff_grog_cuda = np.load(cuda_grog_path)
        plot_subspace(
            coeff_nufft_cuda,
            coeff_grog_cuda,
            args.output_dir / "figure_coeffs_cuda.png",
            label="CUDA",
        )
        plot_grog_views(
            coeff_grog_cuda,
            args.output_dir / "figure_grog_views_cuda.png",
            label="CUDA",
        )

    print(f"Saved figures in {args.output_dir}")


if __name__ == "__main__":
    main()
