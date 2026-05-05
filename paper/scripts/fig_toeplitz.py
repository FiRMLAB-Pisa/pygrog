"""fig_toeplitz — Toeplitz vs nested A^H A timing across PyGROG operators.

Bars compare per-call runtime of ``op.normal`` with ``toeplitz=True`` vs
``toeplitz=False`` for SparseFFT, OffResonanceSparseFFT, and
SubspaceSparseFFT. Speed-ups annotated; max relative error printed under
each operator name to advertise the equivalence.

CPU-only timing — matches the benchmark suite default.
"""

from __future__ import annotations

import sys
import time
import types
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import POSTER_STYLE, save_fig

from pygrog.gadgets._off_resonance import OffResonanceSparseFFT
from pygrog.gadgets._subspace import SubspaceSparseFFT
from pygrog.operator import SparseFFT


def _bench(fn, x, n_warmup=2, n_iter=8):
    for _ in range(n_warmup):
        out = fn(x)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        out = fn(x)
    return (time.perf_counter() - t0) / n_iter, out


def _rel_err(a, b):
    a = a.detach().cpu().numpy()
    b = b.detach().cpu().numpy()
    return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-30))


def _make_sparse(grid, n_samples, n_coils, *, toeplitz, seed=0):
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, int(np.prod(grid)), n_samples).astype(np.int64)
    weights = rng.random(n_samples).astype(np.float32) + 0.1
    smaps = (
        rng.standard_normal((n_coils, *grid))
        + 1j * rng.standard_normal((n_coils, *grid))
    ).astype(np.complex64) * 0.5
    return SparseFFT(
        grid_shape=grid,
        image_shape=grid,
        indices=indices,
        weights=weights,
        smaps=smaps,
        toeplitz=toeplitz,
    )


def main() -> None:
    grid = (192, 192)
    # Single shared trajectory size and coil count for all three operators so
    # that the Toeplitz-vs-nested comparison is fair across panels.
    # Larger n_samples favours Toeplitz (whose cost is independent of n_samples)
    # over nested NUFFT (whose interp cost is linear in n_samples).
    n_samples = 192 * 1600
    n_coils = 8
    L = 4

    # SparseFFT
    op_t = _make_sparse(grid, n_samples, n_coils, toeplitz=True, seed=0)
    op_n = _make_sparse(grid, n_samples, n_coils, toeplitz=False, seed=0)
    x = torch.randn(*grid, dtype=torch.complex64)
    t_sp_t, y_t = _bench(op_t.normal, x)
    t_sp_n, y_n = _bench(op_n.normal, x)
    err_sp = _rel_err(y_t, y_n)

    # ORC
    op_b_t = _make_sparse(grid, n_samples, n_coils, toeplitz=True, seed=1)
    op_b_n = _make_sparse(grid, n_samples, n_coils, toeplitz=False, seed=1)
    rng = np.random.default_rng(11)
    B = (
        rng.standard_normal((n_samples, L)) + 1j * rng.standard_normal((n_samples, L))
    ).astype(np.complex64)
    C = (rng.standard_normal((L, *grid)) + 1j * rng.standard_normal((L, *grid))).astype(
        np.complex64
    )
    orc_t = OffResonanceSparseFFT(op_b_t, B, C, toeplitz=True)
    orc_n = OffResonanceSparseFFT(op_b_n, B, C, toeplitz=False)
    x_orc = torch.randn(*grid, dtype=torch.complex64)
    t_orc_t, y_t = _bench(orc_t.normal, x_orc, n_iter=4)
    t_orc_n, y_n = _bench(orc_n.normal, x_orc, n_iter=4)
    err_orc = _rel_err(y_t, y_n)

    # Subspace — same total n_samples as the other operators, factorised as T_*n_pts.
    T_, K = 12, 4
    n_pts = n_samples // T_
    n_samples_sub = T_ * n_pts
    rng = np.random.default_rng(13)
    indices = rng.integers(0, int(np.prod(grid)), n_samples_sub).astype(np.int64)
    weights = rng.random(n_samples_sub).astype(np.float32) + 0.1
    smaps_sub = (
        rng.standard_normal((n_coils, *grid))
        + 1j * rng.standard_normal((n_coils, *grid))
    ).astype(np.complex64) * 0.5
    sort_perm = torch.argsort(torch.as_tensor(indices))
    inv_perm = torch.empty_like(sort_perm)
    inv_perm[sort_perm] = torch.arange(n_samples_sub)
    indices_t = torch.as_tensor(indices)
    weights_t = torch.as_tensor(weights)
    plan = types.SimpleNamespace(
        grid_shape=grid,
        image_shape=grid,
        grid_size=int(np.prod(grid)),
        indices=indices_t[sort_perm],
        sqrt_weights=torch.sqrt(weights_t)[sort_perm],
        sort_perm=sort_perm,
        inv_perm=inv_perm,
        natural_shape=(T_, n_pts),
        n_samples=n_samples_sub,
    )
    base_t = SparseFFT(plan=plan, smaps=smaps_sub, toeplitz=True)
    base_n = SparseFFT(plan=plan, smaps=smaps_sub, toeplitz=False)
    basis = (rng.standard_normal((K, T_)) + 1j * rng.standard_normal((K, T_))).astype(
        np.complex64
    )
    sub_t = SubspaceSparseFFT(base_t, basis, encoding_axis=-2)
    sub_n = SubspaceSparseFFT(base_n, basis, encoding_axis=-2)
    x_sub = torch.randn(K, *grid, dtype=torch.complex64)
    t_sub_t, y_t = _bench(sub_t.normal, x_sub, n_iter=4)
    t_sub_n, y_n = _bench(sub_n.normal, x_sub, n_iter=4)
    err_sub = _rel_err(y_t, y_n)

    labels = ["SparseFFT", f"ORC (L={L})", f"Subspace (K={K}, T={T_})"]
    nested = np.array([t_sp_n, t_orc_n, t_sub_n]) * 1e3
    toep = np.array([t_sp_t, t_orc_t, t_sub_t]) * 1e3
    speedups = nested / toep
    errs = [err_sp, err_orc, err_sub]

    with POSTER_STYLE():
        fig, ax = plt.subplots(figsize=(11, 5.5))
        x_pos = np.arange(len(labels))
        w = 0.38
        b1 = ax.bar(
            x_pos - w / 2,
            nested,
            width=w,
            label="nested  forward(adjoint(·))",
            color="#9aa6b2",
        )
        b2 = ax.bar(
            x_pos + w / 2, toep, width=w, label="Toeplitz  op.normal", color="#1f77b4"
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            [f"{lab}\n(rel-err {e:.1e})" for lab, e in zip(labels, errs, strict=False)]
        )
        ax.set_ylabel("per-call runtime [ms]  (CPU)")
        ax.set_title("Toeplitz $A^H A$ vs nested forward+adjoint")
        ax.legend(frameon=False, loc="upper left")
        for _i, (_bn, bt, sp) in enumerate(zip(b1, b2, speedups, strict=False)):
            ax.text(
                bt.get_x() + bt.get_width() / 2.0,
                bt.get_height(),
                f"x{sp:.1f}",
                ha="center",
                va="bottom",
                fontsize=12,
            )
        fig.tight_layout()
        save_fig(fig, "fig_toeplitz")


if __name__ == "__main__":
    main()
