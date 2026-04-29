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


def _scaling_cases(results: dict) -> list[dict] | None:
    scaling = results.get("scaling", {})
    if scaling.get("skipped"):
        return None
    cases = scaling.get("cases", [])
    if not cases:
        return None
    return cases


def _size_labels(cases: list[dict]) -> list[str]:
    labels = []
    for case in cases:
        k = case["samples_per_frame"] / 1000.0
        labels.append(f"{case['label']}\n({k:.1f}k)")
    return labels


def plot_preprocessing(results: dict, out: Path) -> bool:
    """Return False and skip if scaling cases are absent (--skip-scaling run)."""
    cases = _scaling_cases(results)
    if cases is None:
        print(
            "Skipping figure_preprocessing.png — no scaling data (run without --skip-scaling to include)."
        )
        return False
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

    The trajectory/image axis convention here is ``(x, y, z)`` (the first
    axis varies along readout — sagittal direction in the dataset), so:

    * ``vol[:, :, mz]`` → (x, y) → **axial**
    * ``vol[:, my, :]`` → (x, z) → **coronal**
    * ``vol[mx, :, :]`` → (y, z) → **sagittal**

    All slices are rotated 90° counter-clockwise so anatomical orientations
    match standard radiology display.

    For 2-D input the same slice is returned for all three views.
    """
    vol = np.asarray(vol)
    if vol.ndim == 2:
        s = np.rot90(vol, k=1)
        return s, s, s
    if vol.ndim == 3:
        mx, my, mz = vol.shape[0] // 2, vol.shape[1] // 2, vol.shape[2] // 2
        axial = np.rot90(vol[:, :, mz], k=1)
        coronal = np.rot90(vol[:, my, :], k=1)
        sagittal = np.rot90(vol[mx, :, :], k=1)
        return axial, coronal, sagittal
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

    Row 1: NUFFT reference magnitudes (gray, per-coeff normalized)
    Row 2: GROG magnitudes (gray, per-coeff normalized)
    Row 3: signed error (GROG - NUFFT) on bwr colormap, ±10% of max
    Each column is one subspace coefficient.  Per-coefficient NRMSE annotated.
    """
    k = min(coeff_nufft.shape[0], coeff_grog.shape[0])

    top_tiles = []  # NUFFT
    mid_tiles = []  # GROG
    err_tiles = []  # signed error
    nrmse_vals = []

    for i in range(k):
        # Use the CCW-rotated axial view (first of _three_views)
        nufft_2d = _normalize_unit(_three_views(np.abs(coeff_nufft[i]))[0])
        grog_2d = _normalize_unit(_three_views(np.abs(coeff_grog[i]))[0])
        top_tiles.append(nufft_2d)
        mid_tiles.append(grog_2d)
        err = grog_2d - nufft_2d  # signed difference, both in [0,1]
        err_tiles.append(err.astype(np.float32))
        nrmse = float(np.sqrt(np.mean(err**2)))
        nrmse_vals.append(nrmse)

    top_row = np.concatenate(top_tiles, axis=1)
    mid_row = np.concatenate(mid_tiles, axis=1)
    err_row = np.concatenate(err_tiles, axis=1)

    avg_nrmse = float(np.mean(nrmse_vals))
    h = top_row.shape[0]

    fig, (ax_gray, ax_err) = plt.subplots(
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
    ax_gray.set_xticklabels([rf"$\phi_{{{i}}}$" for i in range(k)], fontsize=13)
    ax_gray.xaxis.tick_top()
    ax_gray.tick_params(axis="x", length=0, pad=3)
    ax_gray.set_yticks([h / 2.0, h + h / 2.0])
    ax_gray.set_yticklabels(["NUFFT reference", "GROG"], fontsize=13)
    for tick in ax_gray.get_yticklabels():
        tick.set_rotation(90)
        tick.set_verticalalignment("center")
        tick.set_horizontalalignment("center")

    cursor = 0
    for w in tile_widths[:-1]:
        cursor += w
        ax_gray.axvline(
            cursor - 0.5, color="#666666", linewidth=0.5, linestyle="--", alpha=0.6
        )
        ax_err.axvline(
            cursor - 0.5, color="#666666", linewidth=0.5, linestyle="--", alpha=0.6
        )

    for spine in ["top", "right", "bottom", "left"]:
        ax_gray.spines[spine].set_visible(False)

    err_vmax = 0.10  # ±10% of normalized max
    im = ax_err.imshow(
        err_row, cmap="bwr", origin="upper", vmin=-err_vmax, vmax=err_vmax
    )
    ax_err.set_xticks([])
    ax_err.set_yticks([err_row.shape[0] / 2.0])
    ax_err.set_yticklabels(["Error (GROG\u2212NUFFT)"], fontsize=13)
    for tick in ax_err.get_yticklabels():
        tick.set_rotation(90)
        tick.set_verticalalignment("center")
        tick.set_horizontalalignment("center")
    for spine in ["top", "right", "bottom", "left"]:
        ax_err.spines[spine].set_visible(False)

    # Annotate per-coefficient NRMSE above each error tile
    cursor = 0
    for i, (w, nrmse) in enumerate(zip(tile_widths, nrmse_vals, strict=False)):
        cx = cursor + w / 2.0
        ax_err.text(
            cx,
            -2,
            f"NRMSE={nrmse:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="black",
            transform=ax_err.transData,
        )
        cursor += w

    # Global NRMSE in corner
    ax_err.text(
        0.98,
        0.05,
        f"avg NRMSE = {avg_nrmse:.3f}",
        transform=ax_err.transAxes,
        ha="right",
        va="bottom",
        fontsize=12,
        color="white",
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "black", "alpha": 0.5},
    )

    cbar = fig.colorbar(im, ax=ax_err, orientation="vertical", fraction=0.015, pad=0.01)
    cbar.set_label("Signed error", fontsize=11)

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
    ax.set_xticklabels([rf"$\phi_{{{i}}}$" for i in range(k)], fontsize=13)
    ax.set_yticks(row_centers)
    ax.set_yticklabels(view_names, fontsize=13)
    for tick in ax.get_yticklabels():
        tick.set_rotation(90)
        tick.set_verticalalignment("center")
        tick.set_horizontalalignment("center")
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
    if cases is None:
        print(
            "Skipping figure_linop.png — no scaling data (run without --skip-scaling to include)."
        )
        return
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


def plot_runtime(results: dict, out: Path) -> None:
    """2×2 panel: runtime / speedup factor / memory footprint / preprocessing.

    Panel (0,0): SubspaceSparseFFT runtime per operation (adj/fwd), all devices.
    Panel (0,1): Speedup factor NUFFT/GROG per device.
    Panel (1,0): Memory footprint (RAM + VRAM) per operation.
    Panel (1,1): GROG preprocessing one-time cost (plan + interpolation).

    Always available regardless of ``--skip-scaling``.
    """
    cpu = results.get("steps", {}).get("runtime_cpu", {})
    gpu = results.get("steps", {}).get("runtime_gpu", {})
    plan_s = results.get("steps", {}).get("grog_plan_creation", {})
    interp_s = results.get("steps", {}).get("grog_interpolation", {})

    # ── helpers ──────────────────────────────────────────────────────────────
    def _rt(d, key):
        e = d.get(key, {})
        return float(e.get("runtime_mean_sec", float("nan"))), float(
            e.get("runtime_std_sec", 0.0)
        )

    def _ram(d, key):
        v = d.get(key, {}).get("peak_ram_gb")
        return float(v) if v is not None else float("nan")

    def _rt_g(grp, op):
        return _rt(gpu.get(grp, {}), op)

    def _ram_g(grp, op):
        return _ram(gpu.get(grp, {}), op)

    def _vram_g(grp, op):
        d = gpu.get(grp, {}).get(op, {})
        for k in (
            "peak_gpu_mem_gb_nvml",
            "peak_gpu_mem_gb_torch",
            "peak_gpu_mem_gb_cupy",
        ):
            v = d.get(k)
            if v:
                return float(v)
        return 0.0

    ops = [("adjoint", "Adjoint\n(k→img)"), ("forward", "Forward\n(img→k)")]
    x = np.arange(len(ops))

    nufft_gpu_avail = not np.isnan(_rt_g("nufft_cufinufft_gpu", "nufft_adjoint")[0])
    grog_full_avail = not np.isnan(_rt_g("grog_full_gpu", "grog_adjoint")[0])
    grog_dual_avail = not np.isnan(_rt_g("grog_dual_stream_gpu", "grog_adjoint")[0])

    # (label, color, [(mean,std) per op], [ram per op], [vram per op])
    specs: list[tuple] = [
        (
            "NUFFT CPU",
            "#4E79A7",
            [_rt(cpu, "nufft_finufft_adjoint"), _rt(cpu, "nufft_finufft_forward")],
            [_ram(cpu, "nufft_finufft_adjoint"), _ram(cpu, "nufft_finufft_forward")],
            [0.0, 0.0],
        ),
        (
            "GROG CPU",
            "#F28E2B",
            [_rt(cpu, "grog_adjoint"), _rt(cpu, "grog_forward")],
            [_ram(cpu, "grog_adjoint"), _ram(cpu, "grog_forward")],
            [0.0, 0.0],
        ),
    ]
    if nufft_gpu_avail:
        specs.append(
            (
                "NUFFT GPU",
                "#59A14F",
                [
                    _rt_g("nufft_cufinufft_gpu", "nufft_adjoint"),
                    _rt_g("nufft_cufinufft_gpu", "nufft_forward"),
                ],
                [
                    _ram_g("nufft_cufinufft_gpu", "nufft_adjoint"),
                    _ram_g("nufft_cufinufft_gpu", "nufft_forward"),
                ],
                [
                    _vram_g("nufft_cufinufft_gpu", "nufft_adjoint"),
                    _vram_g("nufft_cufinufft_gpu", "nufft_forward"),
                ],
            )
        )
    if grog_full_avail:
        specs.append(
            (
                "GROG GPU (full)",
                "#E15759",
                [
                    _rt_g("grog_full_gpu", "grog_adjoint"),
                    _rt_g("grog_full_gpu", "grog_forward"),
                ],
                [
                    _ram_g("grog_full_gpu", "grog_adjoint"),
                    _ram_g("grog_full_gpu", "grog_forward"),
                ],
                [
                    _vram_g("grog_full_gpu", "grog_adjoint"),
                    _vram_g("grog_full_gpu", "grog_forward"),
                ],
            )
        )
    if grog_dual_avail:
        specs.append(
            (
                "GROG GPU (dual)",
                "#B07AA1",
                [
                    _rt_g("grog_dual_stream_gpu", "grog_adjoint"),
                    _rt_g("grog_dual_stream_gpu", "grog_forward"),
                ],
                [
                    _ram_g("grog_dual_stream_gpu", "grog_adjoint"),
                    _ram_g("grog_dual_stream_gpu", "grog_forward"),
                ],
                [
                    _vram_g("grog_dual_stream_gpu", "grog_adjoint"),
                    _vram_g("grog_dual_stream_gpu", "grog_forward"),
                ],
            )
        )

    n_m = len(specs)
    bar_w = min(0.7 / n_m, 0.18)
    bar_off = -(n_m - 1) / 2.0 * bar_w

    def _finalize(ax, bar_ann, formatter, *, rot=False, bold=False):
        """Set ylim with 40% headroom then annotate above errorbar caps."""
        valid = [(b, m, s) for b, m, s in bar_ann if not np.isnan(m)]
        if not valid:
            return
        ymax = max(m + s for _, m, s in valid)
        ax.set_ylim(0, ymax * 1.45)
        margin = ymax * 0.03
        for bar, m, std in valid:
            kw: dict = {"ha": "center", "va": "bottom"}
            if bold:
                kw.update(fontsize=10, fontweight="bold")
            else:
                kw["fontsize"] = 7
                if rot:
                    kw["rotation"] = 35
            ax.text(
                bar.get_x() + bar.get_width() / 2, m + std + margin, formatter(m), **kw
            )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_rt, ax_sp, ax_mem, ax_prep = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # ── (0,0) SubspaceSparseFFT runtime ──────────────────────────────────
    bar_ann_rt: list[tuple] = []
    for mi, (label, color, rt_pairs, _ram_vals, _vram_vals) in enumerate(specs):
        means = [v[0] for v in rt_pairs]
        stds = [v[1] for v in rt_pairs]
        bars = ax_rt.bar(
            x + bar_off + mi * bar_w,
            means,
            bar_w,
            color=color,
            label=label,
            yerr=stds,
            capsize=3,
            error_kw={"elinewidth": 1},
        )
        for bar, m, std in zip(bars, means, stds, strict=False):
            if not np.isnan(m):
                bar_ann_rt.append((bar, m, std))
    ax_rt.set_xticks(x)
    ax_rt.set_xticklabels([lb for _, lb in ops])
    ax_rt.set_ylabel("Runtime (s)")
    ax_rt.set_title("SubspaceSparseFFT runtime")
    ax_rt.legend(fontsize=8, loc="upper right")
    _finalize(ax_rt, bar_ann_rt, _format_runtime, rot=True)

    # ── (0,1) Speedup factor ─────────────────────────────────────────────
    def _sp(n_rt, g_rt):
        return n_rt / g_rt if (not np.isnan(n_rt + g_rt) and g_rt > 0) else float("nan")

    sp_types = [("CPU", "#4E79A7")]
    sp_vals_map: dict[str, list[float]] = {
        "CPU": [
            _sp(_rt(cpu, "nufft_finufft_adjoint")[0], _rt(cpu, "grog_adjoint")[0]),
            _sp(_rt(cpu, "nufft_finufft_forward")[0], _rt(cpu, "grog_forward")[0]),
        ],
    }
    if nufft_gpu_avail and grog_full_avail:
        sp_types.append(("GPU (full)", "#E15759"))
        sp_vals_map["GPU (full)"] = [
            _sp(
                _rt_g("nufft_cufinufft_gpu", "nufft_adjoint")[0],
                _rt_g("grog_full_gpu", "grog_adjoint")[0],
            ),
            _sp(
                _rt_g("nufft_cufinufft_gpu", "nufft_forward")[0],
                _rt_g("grog_full_gpu", "grog_forward")[0],
            ),
        ]
    if nufft_gpu_avail and grog_dual_avail:
        sp_types.append(("GPU (dual)", "#B07AA1"))
        sp_vals_map["GPU (dual)"] = [
            _sp(
                _rt_g("nufft_cufinufft_gpu", "nufft_adjoint")[0],
                _rt_g("grog_dual_stream_gpu", "grog_adjoint")[0],
            ),
            _sp(
                _rt_g("nufft_cufinufft_gpu", "nufft_forward")[0],
                _rt_g("grog_dual_stream_gpu", "grog_forward")[0],
            ),
        ]

    n_sp = len(sp_types)
    sp_w = min(0.7 / n_sp, 0.30)
    sp_off = -(n_sp - 1) / 2.0 * sp_w
    bar_ann_sp: list[tuple] = []
    for si, (slabel, scolor) in enumerate(sp_types):
        vals = sp_vals_map[slabel]
        bars = ax_sp.bar(x + sp_off + si * sp_w, vals, sp_w, color=scolor, label=slabel)
        for bar, v in zip(bars, vals, strict=False):
            if not np.isnan(v):
                bar_ann_sp.append((bar, v, 0.0))
    ax_sp.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    ax_sp.set_xticks(x)
    ax_sp.set_xticklabels([lb for _, lb in ops])
    ax_sp.set_ylabel("Speedup  (NUFFT / GROG)")
    ax_sp.set_title("Speedup factor")
    if sp_types:
        ax_sp.legend(fontsize=8)
    if bar_ann_sp:
        ymax_sp = max(v for _, v, _ in bar_ann_sp if not np.isnan(v))
        ax_sp.set_ylim(0, max(ymax_sp * 1.45, 2.0))
        margin_sp = max(ymax_sp * 0.03, 0.05)
        for bar, v, _ in bar_ann_sp:
            if not np.isnan(v):
                ax_sp.text(
                    bar.get_x() + bar.get_width() / 2,
                    v + margin_sp,
                    f"{v:.1f}×",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )

    # ── (1,0) Memory footprint ────────────────────────────────────────────
    bar_ann_mem: list[tuple] = []
    any_vram = False
    for mi, (label, color, _rt_pairs, ram_list, vram_list) in enumerate(specs):
        xpos = x + bar_off + mi * bar_w
        total = [
            r + v if not np.isnan(r) else float("nan")
            for r, v in zip(ram_list, vram_list, strict=False)
        ]
        ax_mem.bar(xpos, ram_list, bar_w, color=color, label=label)
        if any(v > 0.0 for v in vram_list):
            ax_mem.bar(
                xpos,
                vram_list,
                bar_w,
                bottom=ram_list,
                color=color,
                hatch="///",
                alpha=0.85,
            )
            any_vram = True
        ghost = ax_mem.bar(xpos, total, bar_w, alpha=0.0)
        for bar, m in zip(ghost, total, strict=False):
            if not np.isnan(m):
                bar_ann_mem.append((bar, m, 0.0))
    ax_mem.set_xticks(x)
    ax_mem.set_xticklabels([lb for _, lb in ops])
    ax_mem.set_ylabel("Peak memory (GB)")
    title_mem = (
        "Memory footprint  (RAM + VRAM)" if any_vram else "Memory footprint  (peak RAM)"
    )
    ax_mem.set_title(title_mem)
    legend_handles = list(ax_mem.get_legend_handles_labels()[0])
    legend_labels  = list(ax_mem.get_legend_handles_labels()[1])
    if any_vram:
        legend_handles.append(Patch(facecolor="#888888", hatch="///"))
        legend_labels.append("VRAM (on top)")
    ax_mem.legend(legend_handles, legend_labels, fontsize=8, loc="upper right")
    _finalize(ax_mem, bar_ann_mem, _format_memory)

    # ── (1,1) GROG preprocessing ─────────────────────────────────────────
    prep_labels = ["Plan\ncreation", "Interpolation"]
    plan_rt = float(plan_s.get("runtime_mean_sec", float("nan")))
    interp_rt = float(interp_s.get("runtime_mean_sec", float("nan")))
    plan_std = float(plan_s.get("runtime_std_sec", 0.0))
    interp_std = float(interp_s.get("runtime_std_sec", 0.0))
    plan_ram = float(plan_s.get("peak_ram_gb", float("nan")))
    interp_ram = float(interp_s.get("peak_ram_gb", float("nan")))

    prep_x = np.arange(len(prep_labels))
    ax_prep2 = ax_prep.twinx()
    bar_prt = ax_prep.bar(
        prep_x - 0.2,
        [plan_rt, interp_rt],
        0.35,
        color="#76B7B2",
        label="Runtime (s)",
        yerr=[plan_std, interp_std],
        capsize=3,
        error_kw={"elinewidth": 1},
    )
    bar_pram = ax_prep2.bar(
        prep_x + 0.2,
        [plan_ram, interp_ram],
        0.35,
        color="#EDC948",
        label="Peak RAM (GB)",
        alpha=0.8,
    )
    _finalize(
        ax_prep,
        [
            (b, m, s)
            for b, m, s in zip(bar_prt, [plan_rt, interp_rt], [plan_std, interp_std], strict=False)
        ],
        _format_runtime,
    )
    _finalize(
        ax_prep2,
        [(b, m, 0.0) for b, m in zip(bar_pram, [plan_ram, interp_ram], strict=False)],
        _format_memory,
    )
    ax_prep.set_xticks(prep_x)
    ax_prep.set_xticklabels(prep_labels)
    ax_prep.set_ylabel("Runtime (s)", color="#76B7B2")
    ax_prep2.set_ylabel("Peak RAM (GB)", color="#EDC948")
    ax_prep.set_title("GROG preprocessing  (one-time cost)")
    lines1, lbls1 = ax_prep.get_legend_handles_labels()
    lines2, lbls2 = ax_prep2.get_legend_handles_labels()
    ax_prep.legend(lines1 + lines2, lbls1 + lbls2, fontsize=8)

    fig.suptitle("3D subspace MRF benchmark", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-json", type=Path, default=Path("benchmark/results/results.json")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark/results"))
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    results = _load_json(args.results_json)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    coeff_nufft = np.load(args.output_dir / "coeff_nufft.npy")
    coeff_grog = np.load(args.output_dir / "coeff_grog.npy")

    plot_preprocessing(results, args.output_dir / "figure_preprocessing.png")
    plot_linop(results, args.output_dir / "figure_linop.png")
    plot_runtime(results, args.output_dir / "figure_runtime.png")
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
