"""fig_benchmarks — runtime + peak-memory bars from results.json.

Reads ``pygrog/benchmark/results/results.json`` produced by
``benchmark/run_benchmarks.py`` and re-renders in poster style. Refreshing
the underlying numbers is done separately via ``run_benchmarks.py``.

Adaptive layout (2×2).  Top row shows per-call bars; bottom row shows
PyGROG preprocessing (plan creation + interpolation).

When GPU results are present in results.json the top-row bars are extended
to include cuFINUFFT GPU, PyGROG GPU (full, smaps on GPU) and PyGROG GPU
(dual-stream, smaps on CPU).  Absent entries render as absent bars so the
figure degrades gracefully on CPU-only runs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import BENCHMARK_RESULTS_DIR, POSTER_STYLE, save_fig  # noqa: E402


def _step(d: dict, key: str = "runtime_mean_sec") -> tuple[float, float]:
    """Return (mean, std) for *key* in metrics dict; (nan, nan) if absent."""
    if not isinstance(d, dict) or d.get(key) is None:
        return float("nan"), float("nan")
    mean = float(d[key])
    std_key = key.replace("mean", "std") if "mean" in key else None
    std = float(d.get(std_key) or 0.0) if std_key else 0.0
    return mean, std


def _vram(d: dict) -> float:
    """Extract best available VRAM reading [GB] from a metrics dict."""
    if not isinstance(d, dict):
        return float("nan")
    for key in ("peak_gpu_mem_gb_nvml", "peak_gpu_mem_gb_torch",
                "peak_gpu_mem_gb_cupy"):
        v = d.get(key)
        if v is not None:
            return float(v)
    return float("nan")


def _annotate(ax, bars, vals, fmt="{v:.2g}"):
    for b_, v in zip(bars, vals):
        if np.isfinite(v):
            ax.text(b_.get_x() + b_.get_width() / 2.0, v,
                    "  " + fmt.format(v=v),
                    ha="center", va="bottom", fontsize=11)


def _bar_group(ax, x, offsets, series, w):
    """Draw one group of bars.

    *series* is a list of (vals, label, color) tuples aligned with *offsets*.
    Returns list of bar containers.
    """
    bars = []
    for off, (vals, label, color) in zip(offsets, series):
        b = ax.bar(x + off, vals, width=w, label=label, color=color)
        bars.append(b)
    return bars


def main() -> None:
    rj = BENCHMARK_RESULTS_DIR / "results.json"
    if not rj.exists():
        raise FileNotFoundError(
            f"{rj} missing — run benchmark/run_benchmarks.py first."
        )
    res = json.loads(rj.read_text())
    steps = res.get("steps", {})
    cpu = steps.get("runtime_cpu", {})
    gpu = steps.get("runtime_gpu", {})

    # ---- CPU per-call values -------------------------------------------
    nu_fwd_rt,  _ = _step(cpu.get("nufft_finufft_forward", {}))
    nu_adj_rt,  _ = _step(cpu.get("nufft_finufft_adjoint", {}))
    g_fwd_rt,   _ = _step(cpu.get("grog_forward", {}))
    g_adj_rt,   _ = _step(cpu.get("grog_adjoint", {}))
    nu_fwd_ram, _ = _step(cpu.get("nufft_finufft_forward", {}), "peak_ram_gb")
    nu_adj_ram, _ = _step(cpu.get("nufft_finufft_adjoint", {}), "peak_ram_gb")
    g_fwd_ram,  _ = _step(cpu.get("grog_forward", {}), "peak_ram_gb")
    g_adj_ram,  _ = _step(cpu.get("grog_adjoint", {}), "peak_ram_gb")

    # ---- GPU per-call values (nan when absent / skipped) ----------------
    cuf      = gpu.get("nufft_cufinufft_gpu", {})
    grog_full = gpu.get("grog_full_gpu", {})
    grog_dual = gpu.get("grog_dual_stream_gpu", {})

    cuf_fwd_rt,  _ = _step(cuf.get("nufft_forward", {}))
    cuf_adj_rt,  _ = _step(cuf.get("nufft_adjoint", {}))
    gf_fwd_rt,   _ = _step(grog_full.get("grog_forward", {}))
    gf_adj_rt,   _ = _step(grog_full.get("grog_adjoint", {}))
    gd_fwd_rt,   _ = _step(grog_dual.get("grog_forward", {}))
    gd_adj_rt,   _ = _step(grog_dual.get("grog_adjoint", {}))

    cuf_fwd_vram  = _vram(cuf.get("nufft_forward", {}))
    cuf_adj_vram  = _vram(cuf.get("nufft_adjoint", {}))
    gf_fwd_vram   = _vram(grog_full.get("grog_forward", {}))
    gf_adj_vram   = _vram(grog_full.get("grog_adjoint", {}))
    gd_fwd_vram   = _vram(grog_dual.get("grog_forward", {}))
    gd_adj_vram   = _vram(grog_dual.get("grog_adjoint", {}))

    have_gpu = any(np.isfinite(v) for v in (
        cuf_fwd_rt, gf_fwd_rt, gd_fwd_rt,
    ))

    # ---- Preprocessing --------------------------------------------------
    plan_rt, plan_rt_std     = _step(steps.get("grog_plan_creation", {}))
    interp_rt, interp_rt_std = _step(steps.get("grog_interpolation", {}))
    plan_ram, _   = _step(steps.get("grog_plan_creation", {}),  "peak_ram_gb")
    interp_ram, _ = _step(steps.get("grog_interpolation", {}), "peak_ram_gb")

    # ---- Layout ---------------------------------------------------------
    groups = ["forward", "adjoint"]
    x = np.arange(len(groups))

    if have_gpu:
        # 5 series: FINUFFT-CPU | cuFINUFFT-GPU | PyGROG-CPU | PyGROG-GPU-full | PyGROG-GPU-dual
        series_rt = [
            ([nu_fwd_rt,  nu_adj_rt],  "FINUFFT (CPU)",              "#9aa6b2"),
            ([cuf_fwd_rt, cuf_adj_rt], "cuFINUFFT (GPU)",            "#aec7e8"),
            ([g_fwd_rt,   g_adj_rt],   "PyGROG (CPU)",               "#1f77b4"),
            ([gf_fwd_rt,  gf_adj_rt],  "PyGROG GPU (full)",          "#ff7f0e"),
            ([gd_fwd_rt,  gd_adj_rt],  "PyGROG GPU (dual-stream)",   "#d62728"),
        ]
        # For memory: top row = VRAM (GPU series), RAM (CPU series)
        series_mem = [
            ([nu_fwd_ram,  nu_adj_ram],  "FINUFFT (CPU) — RAM",           "#9aa6b2"),
            ([cuf_fwd_vram, cuf_adj_vram], "cuFINUFFT (GPU) — VRAM",      "#aec7e8"),
            ([g_fwd_ram,   g_adj_ram],   "PyGROG (CPU) — RAM",            "#1f77b4"),
            ([gf_fwd_vram, gf_adj_vram], "PyGROG GPU full — VRAM",        "#ff7f0e"),
            ([gd_fwd_vram, gd_adj_vram], "PyGROG GPU dual — VRAM",        "#d62728"),
        ]
        n_series = 5
    else:
        series_rt = [
            ([nu_fwd_rt, nu_adj_rt], "FINUFFT (CPU)", "#9aa6b2"),
            ([g_fwd_rt,  g_adj_rt],  "PyGROG (CPU)",  "#1f77b4"),
        ]
        series_mem = [
            ([nu_fwd_ram, nu_adj_ram], "FINUFFT — RAM", "#9aa6b2"),
            ([g_fwd_ram,  g_adj_ram],  "PyGROG — RAM",  "#1f77b4"),
        ]
        n_series = 2

    w = min(0.38, 0.85 / n_series)
    span = w * n_series
    offsets = np.linspace(-span / 2 + w / 2, span / 2 - w / 2, n_series)

    figw = 15 if not have_gpu else 20
    with POSTER_STYLE():
        fig, axes = plt.subplots(2, 2, figsize=(figw, 10.5))

        # ---- (0,0) per-call runtime -------------------------------------
        ax = axes[0, 0]
        all_bars_rt = _bar_group(ax, x, offsets, series_rt, w)
        ax.set_xticks(x); ax.set_xticklabels(groups)
        ax.set_ylabel("per-call runtime [s]")
        ax.set_title("In-vivo MRF: per-call runtime")
        ax.legend(frameon=False, fontsize=12, loc="upper right")
        for bars_c, (vals, *_) in zip(all_bars_rt, series_rt):
            _annotate(ax, bars_c, vals, "{v:.1f} s")
        # CPU speedup callout (FINUFFT-CPU / PyGROG-CPU)
        for i, (vn, vg) in enumerate(zip([nu_fwd_rt, nu_adj_rt],
                                          [g_fwd_rt,  g_adj_rt])):
            if np.isfinite(vn) and np.isfinite(vg) and vg > 0:
                ymax = max(v for s in series_rt for v in s[0] if np.isfinite(v))
                ax.text(i, ymax * 1.18, f"×{vn / vg:.1f}",
                        ha="center", va="bottom",
                        fontsize=13, color="#1f77b4", fontweight="bold")
        all_vals_rt = [v for s in series_rt for v in s[0] if np.isfinite(v)]
        ax.set_ylim(top=max(all_vals_rt) * 1.38)

        # ---- (0,1) per-call peak memory ---------------------------------
        ax = axes[0, 1]
        all_bars_mem = _bar_group(ax, x, offsets, series_mem, w)
        ax.set_xticks(x); ax.set_xticklabels(groups)
        ax.set_ylabel("peak RAM / VRAM [GB]")
        ax.set_title("In-vivo MRF: per-call peak memory")
        ax.legend(frameon=False, fontsize=12, loc="upper right")
        for bars_c, (vals, *_) in zip(all_bars_mem, series_mem):
            _annotate(ax, bars_c, vals, "{v:.1f} GB")

        # ---- (1,0) preprocessing runtime --------------------------------
        ax = axes[1, 0]
        labels = ["plan creation\n(one-shot)", "interpolation\n(per-call)"]
        means  = np.array([plan_rt, interp_rt])
        stds   = np.array([plan_rt_std, interp_rt_std])
        bars = ax.bar(labels, means, yerr=stds, capsize=6,
                      color=["#2ca02c", "#ff7f0e"])
        ax.set_ylabel("runtime [s]")
        ax.set_title("PyGROG preprocessing: runtime")
        _annotate(ax, bars, means, "{v:.1f} s")

        # ---- (1,1) preprocessing RAM ------------------------------------
        ax = axes[1, 1]
        rams = np.array([plan_ram, interp_ram])
        bars = ax.bar(labels, rams, color=["#2ca02c", "#ff7f0e"])
        ax.set_ylabel("peak RAM [GB]")
        ax.set_title("PyGROG preprocessing: peak memory")
        _annotate(ax, bars, rams, "{v:.1f} GB")

        meta = res.get("input", {})
        env  = res.get("environment", {})
        n_cores  = env.get("cpu_count")
        n_frames = meta.get("n_frames") or meta.get("used_frames")
        note_bits = ["data: in-vivo MRF"]
        if n_frames:
            note_bits.append(f"{n_frames} frames")
        if n_cores:
            note_bits.append(f"{n_cores} CPU cores")
        if have_gpu:
            gpu_name = env.get("gpu_name") or "GPU"
            note_bits.append(gpu_name)
        fig.text(0.5, -0.01, "  ·  ".join(note_bits),
                 ha="center", fontsize=12, color="#444")

        fig.tight_layout()
        save_fig(fig, "fig_benchmarks")


if __name__ == "__main__":
    main()
