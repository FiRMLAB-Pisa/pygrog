#!/usr/bin/env python3
"""Plot benchmark figures from benchmark JSON outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _runtime_bar(ax, labels, values, title, ylabel=False):
    x = np.arange(len(labels))
    ax.bar(x, values, color=["#4E79A7", "#F28E2B", "#59A14F", "#E15759"][: len(labels)])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    if ylabel:
        ax.set_ylabel("Runtime (s)")
    ax.set_title(title)


def _to_2d_slice(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr)
    while out.ndim > 2:
        out = out[out.shape[0] // 2]
    return out


def plot_subspace(coeff_nufft: np.ndarray, coeff_grog: np.ndarray, out: Path) -> None:
    k = min(4, coeff_nufft.shape[0])

    fig, axes = plt.subplots(2, k, figsize=(3.5 * k, 6))
    for i in range(k):
        n = _to_2d_slice(np.abs(coeff_nufft[i]))
        g = _to_2d_slice(np.abs(coeff_grog[i]))
        vmax = max(float(n.max()), float(g.max()), 1e-8)

        axes[0, i].imshow(n, cmap="magma", origin="lower", vmin=0, vmax=vmax)
        axes[0, i].set_title(f"NUFFT coeff {i}")
        axes[0, i].axis("off")

        axes[1, i].imshow(g, cmap="magma", origin="lower", vmin=0, vmax=vmax)
        axes[1, i].set_title(f"GROG coeff {i}")
        axes[1, i].axis("off")

    plt.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_runtime(results: dict, out: Path) -> None:
    cpu = results["steps"]["runtime_cpu"]
    gpu = results["steps"].get("runtime_gpu", {})

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    cpu_labels = ["NUFFT fwd", "NUFFT adj", "GROG fwd", "GROG adj"]
    cpu_vals = [
        cpu["nufft_finufft_forward"]["runtime_mean_sec"],
        cpu["nufft_finufft_adjoint"]["runtime_mean_sec"],
        cpu["grog_forward"]["runtime_mean_sec"],
        cpu["grog_adjoint"]["runtime_mean_sec"],
    ]
    _runtime_bar(axes[0], cpu_labels, cpu_vals, "CPU (FINUFFT vs GROG)", ylabel=True)

    gpu_labels = []
    gpu_vals = []
    if "nufft_cufinufft_gpu" in gpu:
        gpu_labels.extend(["CUFINUFFT fwd", "CUFINUFFT adj"])
        gpu_vals.extend(
            [
                gpu["nufft_cufinufft_gpu"]["nufft_forward"]["runtime_mean_sec"],
                gpu["nufft_cufinufft_gpu"]["nufft_adjoint"]["runtime_mean_sec"],
            ]
        )
    if "grog_full_gpu" in gpu:
        gpu_labels.extend(["GROG full fwd", "GROG full adj"])
        gpu_vals.extend(
            [
                gpu["grog_full_gpu"]["grog_forward"]["runtime_mean_sec"],
                gpu["grog_full_gpu"]["grog_adjoint"]["runtime_mean_sec"],
            ]
        )
    if "grog_dual_stream_gpu" in gpu:
        gpu_labels.extend(["GROG dual fwd", "GROG dual adj"])
        gpu_vals.extend(
            [
                gpu["grog_dual_stream_gpu"]["grog_forward"]["runtime_mean_sec"],
                gpu["grog_dual_stream_gpu"]["grog_adjoint"]["runtime_mean_sec"],
            ]
        )

    if gpu_labels:
        _runtime_bar(axes[1], gpu_labels, gpu_vals, "GPU (CUFINUFFT vs GROG)")
    else:
        axes[1].text(0.5, 0.5, "No GPU results available", ha="center", va="center")
        axes[1].set_axis_off()

    plt.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_memory(results: dict, out: Path) -> None:
    cpu = results["steps"]["runtime_cpu"]
    gpu = results["steps"].get("runtime_gpu", {})

    labels = []
    ram_vals = []
    vram_vals = []

    for key, name in [
        ("nufft_finufft_forward", "NUFFT CPU fwd"),
        ("nufft_finufft_adjoint", "NUFFT CPU adj"),
        ("grog_forward", "GROG CPU fwd"),
        ("grog_adjoint", "GROG CPU adj"),
    ]:
        labels.append(name)
        ram_vals.append(cpu[key].get("peak_ram_gb", 0.0))
        vram_vals.append(0.0)

    if "nufft_cufinufft_gpu" in gpu:
        for key, name in [("nufft_forward", "CUFINUFFT fwd"), ("nufft_adjoint", "CUFINUFFT adj")]:
            m = gpu["nufft_cufinufft_gpu"][key]
            labels.append(name)
            ram_vals.append(m.get("peak_ram_gb", 0.0))
            vram_vals.append(
                m.get("peak_gpu_mem_gb_nvml")
                or m.get("peak_gpu_mem_gb_torch")
                or m.get("peak_gpu_mem_gb_cupy")
                or 0.0
            )

    for grp, prefix in [("grog_full_gpu", "GROG full"), ("grog_dual_stream_gpu", "GROG dual")]:
        if grp in gpu:
            for key, suf in [("grog_forward", "fwd"), ("grog_adjoint", "adj")]:
                m = gpu[grp][key]
                labels.append(f"{prefix} {suf}")
                ram_vals.append(m.get("peak_ram_gb", 0.0))
                vram_vals.append(
                    m.get("peak_gpu_mem_gb_nvml")
                    or m.get("peak_gpu_mem_gb_torch")
                    or m.get("peak_gpu_mem_gb_cupy")
                    or 0.0
                )

    x = np.arange(len(labels))
    width = 0.42

    fig, ax = plt.subplots(figsize=(max(10, 0.8 * len(labels)), 4.8))
    ax.bar(x - width / 2, ram_vals, width, label="Peak RAM (GB)", color="#4E79A7")
    ax.bar(x + width / 2, vram_vals, width, label="Peak VRAM (GB)", color="#F28E2B")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("GB")
    ax.set_title("Memory footprint by benchmark step")
    ax.legend()

    plt.tight_layout()
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

    plot_subspace(coeff_nufft, coeff_grog, args.output_dir / "figure_subspace_coeffs.png")
    plot_runtime(results, args.output_dir / "figure_runtime_cpu_gpu.png")
    plot_memory(results, args.output_dir / "figure_memory_profile.png")

    print(f"Saved figures in {args.output_dir}")


if __name__ == "__main__":
    main()
