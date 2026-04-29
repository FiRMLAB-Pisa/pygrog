"""Micro-benchmark for SubspaceSparseFFT adjoint/forward.

Times the gadget alone (no GROG interp, no NUFFT comparison) on synthetic
data shaped like the real 3D MRF case, and reports peak VRAM via
torch.cuda.max_memory_allocated for the chosen `k_chunk`.

Usage:
    python _micro_subspace.py [--shape S S S] [--T T] [--K K] [--C C]
                              [--n-readout N] [--n-spokes M] [--k-chunk J]
                              [--device cpu|cuda]
"""

from __future__ import annotations

import argparse
import gc
import time

import numpy as np
import torch

from pygrog.gadgets import SubspaceSparseFFT
from pygrog.operator import SparseFFT


def make_op(shape, T, K, C, n_readout, n_spokes, device, k_chunk, oversamp=1.25):
    grid_shape = tuple(int(round(s * oversamp / 2) * 2) for s in shape)
    n_per_t = n_spokes * n_readout
    n_total = T * n_per_t
    rng = np.random.default_rng(0)
    flat = int(np.prod(grid_shape))
    indices = torch.from_numpy(rng.integers(0, flat, size=n_total).astype(np.int64))
    weights = torch.from_numpy(rng.random(n_total, dtype=np.float32) + 0.1)
    smaps = torch.from_numpy(
        rng.standard_normal((C, *shape), dtype=np.float32)
        + 1j * rng.standard_normal((C, *shape), dtype=np.float32)
    ).to(torch.complex64)

    base = SparseFFT(
        grid_shape=grid_shape,
        image_shape=shape,
        indices=indices,
        weights=weights,
        smaps=smaps,
        device=device,
        toeplitz=False,
    )
    # Override flat natural_shape with multi-dim natural to expose T axis.
    base.natural_shape = (T, n_spokes, n_readout)

    basis = torch.from_numpy(
        rng.standard_normal((K, T), dtype=np.float32)
        + 1j * rng.standard_normal((K, T), dtype=np.float32)
    ).to(torch.complex64)

    op = SubspaceSparseFFT(base, basis, encoding_axis=-3, k_chunk=k_chunk)
    return op, T, K, C, n_spokes, n_readout, grid_shape


def time_run(fn, *args, repeat=3, warmup=1, device="cpu"):
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        out = fn(*args)
    if device == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn(*args)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    peak = (
        torch.cuda.max_memory_allocated() / (1024**3) if device == "cuda" else 0.0
    )
    return out, float(np.mean(times)), float(np.std(times)), peak


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--shape", type=int, nargs=3, default=[64, 64, 32])
    p.add_argument("--T", type=int, default=100)
    p.add_argument("--K", type=int, default=5)
    p.add_argument("--C", type=int, default=6)
    p.add_argument("--n-readout", type=int, default=128)
    p.add_argument("--n-spokes", type=int, default=12)
    p.add_argument("--k-chunk", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    op, T, K, C, ns, nr, grid_shape = make_op(
        tuple(args.shape),
        args.T,
        args.K,
        args.C,
        args.n_readout,
        args.n_spokes,
        args.device,
        args.k_chunk,
    )
    nat = op._base.natural_shape
    rng = np.random.default_rng(1)
    coeffs = torch.from_numpy(
        rng.standard_normal((K, *args.shape), dtype=np.float32)
        + 1j * rng.standard_normal((K, *args.shape), dtype=np.float32)
    ).to(torch.complex64)
    ksp = torch.from_numpy(
        rng.standard_normal((C, *nat), dtype=np.float32)
        + 1j * rng.standard_normal((C, *nat), dtype=np.float32)
    ).to(torch.complex64)
    if args.device == "cuda":
        coeffs = coeffs.cuda()
        ksp = ksp.cuda()

    print(
        f"shape={tuple(args.shape)} grid={grid_shape} T={T} K={K} C={C} "
        f"n_per_t={ns*nr} k_chunk={args.k_chunk} device={args.device}"
    )

    _, mean, sd, peak = time_run(
        op.adjoint, ksp, repeat=args.repeat, warmup=args.warmup, device=args.device
    )
    print(f"  adjoint : {mean*1e3:8.1f} ± {sd*1e3:5.1f} ms   peak VRAM {peak:6.2f} GB")

    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    _, mean, sd, peak = time_run(
        op.forward, coeffs, repeat=args.repeat, warmup=args.warmup, device=args.device
    )
    print(f"  forward : {mean*1e3:8.1f} ± {sd*1e3:5.1f} ms   peak VRAM {peak:6.2f} GB")
    del op
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
