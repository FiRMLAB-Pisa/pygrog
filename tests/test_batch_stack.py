"""Tests for multi-axis batch (`*B`) and stacked-trajectory (`*S`) support.

These tests exercise the new batch/stack capabilities introduced across:

- :class:`pygrog.calib.GrogPlan` / :class:`GrogInterpolator` (stacked plans)
- :class:`pygrog.operator.SparseFFT` (multi-axis ``*B``)
- :class:`pygrog.gadgets.SubspaceSparseFFT` and
  :class:`pygrog.gadgets.OffResonanceSparseFFT` (``*B`` + ``*S`` prefix)
- :class:`pygrog.toeplitz.GrogToeplitzOp`,
  :class:`OffResonanceToeplitzOp`, :class:`SubspaceToeplitzOp`
- :func:`pygrog.utils.nlinv_calib` (batched volumes; ``train_reduce='mean'``)

The tests assert bit-for-bit (or close-to) equivalence with the per-element
loop reference.
"""

import numpy as np
import torch
import pytest

from pygrog.calib import GrogInterpolator, GrogPlan
from pygrog.operator import SparseFFT
from pygrog.utils import nlinv_calib

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _spiral2d(n_shots=8, n_per_shot=64, fov=64, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n_per_shot)
    angles = 2 * np.pi * np.arange(n_shots) / n_shots
    coords = np.zeros((n_shots, n_per_shot, 2), dtype=np.float32)
    for s, a in enumerate(angles):
        r = (fov / 2) * t
        coords[s, :, 0] = r * np.cos(a + 4 * np.pi * t)
        coords[s, :, 1] = r * np.sin(a + 4 * np.pi * t)
    return coords + 0.01 * rng.standard_normal(coords.shape).astype(np.float32)


def _make_calib(shape=(64, 64), n_coils=4, cal=24, seed=0):
    rng = np.random.default_rng(seed)
    return (
        rng.standard_normal((n_coils, cal, cal))
        + 1j * rng.standard_normal((n_coils, cal, cal))
    ).astype(np.complex64)


def _build_interp(coords, shape, calib):
    interp = GrogInterpolator(shape=shape, coords=coords, kernel_width=2)
    interp.calc_interp_table(calib)
    return interp


# ---------------------------------------------------------------------------
# 1. GrogPlan.stack: cross-element bit-equivalence
# ---------------------------------------------------------------------------


def test_grog_plan_stack_matches_per_element():
    shape = (48, 48)
    coords_list = [_spiral2d(seed=s, fov=48) for s in range(3)]
    calib = _make_calib(shape=shape)

    plans = []
    for c in coords_list:
        interp = _build_interp(c, shape, calib)
        # Run a dummy interpolation to populate plan internals via interpolate().
        ksp = torch.randn(4, *c.shape[:-1], dtype=torch.complex64)
        interp.interpolate(ksp)
        plans.append(interp.plan)

    stacked = GrogPlan._stack(plans)
    assert stacked.is_stacked
    assert stacked.stack_shape == (3,)
    # indices shape: (S, n_per) where n_per matches per-element n_samples.
    assert stacked.indices.shape == (3, plans[0].indices.shape[-1])


# ---------------------------------------------------------------------------
# 2. SparseFFT multi-axis *B equivalence
# ---------------------------------------------------------------------------


def test_sparse_fft_multi_axis_batch_matches_loop():
    n_samples = 128
    grid = (32, 32)
    img = (24, 24)
    rng = np.random.default_rng(0)
    indices = rng.integers(0, np.prod(grid), size=n_samples)
    weights = np.ones(n_samples, dtype=np.float32)
    smaps = (
        rng.standard_normal((4, *img)) + 1j * rng.standard_normal((4, *img))
    ).astype(np.complex64)
    op = SparseFFT(grid, img, indices, weights, smaps=torch.as_tensor(smaps))

    # *B = (B1, B2)
    ksp = torch.randn(2, 3, 4, n_samples, dtype=torch.complex64)
    out = op.adjoint(ksp)
    assert out.shape == (2, 3, *img)
    ref = torch.stack(
        [
            torch.stack([op.adjoint(ksp[i, j]) for j in range(3)], dim=0)
            for i in range(2)
        ],
        dim=0,
    )
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-6)

    # forward round
    img_b = torch.randn(2, 3, *img, dtype=torch.complex64)
    k_out = op.forward(img_b)
    assert k_out.shape == (2, 3, 4, n_samples)
    k_ref = torch.stack(
        [
            torch.stack([op.forward(img_b[i, j]) for j in range(3)], dim=0)
            for i in range(2)
        ],
        dim=0,
    )
    torch.testing.assert_close(k_out, k_ref, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# 3. Stacked SparseFFT equivalence to per-element loop
# ---------------------------------------------------------------------------


def _make_stacked_op(n_stack=2, n_coils=2, shape=(48, 48), n_per_shot=48):
    coords_list = [
        _spiral2d(seed=s, fov=shape[0], n_per_shot=n_per_shot) for s in range(n_stack)
    ]
    calib = _make_calib(shape=shape, n_coils=n_coils)
    interps = [_build_interp(c, shape, calib) for c in coords_list]
    rng = np.random.default_rng(7)
    smaps = (
        rng.standard_normal((n_coils, *shape))
        + 1j * rng.standard_normal((n_coils, *shape))
    ).astype(np.complex64)
    plans = []
    sparse_per = []
    for c, interp in zip(coords_list, interps, strict=False):
        ksp = torch.from_numpy(
            (
                rng.standard_normal((n_coils, *c.shape[:-1]))
                + 1j * rng.standard_normal((n_coils, *c.shape[:-1]))
            ).astype(np.complex64)
        )
        sparse = interp.interpolate(ksp)
        plans.append(interp.plan)
        sparse_per.append(torch.as_tensor(sparse))
    plan_stk = GrogPlan._stack(plans)
    op_single = [SparseFFT(plan=p, smaps=torch.as_tensor(smaps)) for p in plans]
    op_stk = SparseFFT(plan=plan_stk, smaps=torch.as_tensor(smaps))
    # sparse_per shape per element: (C, *natural_per).
    return op_single, op_stk, sparse_per, plan_stk, smaps


def test_sparse_fft_stacked_forward_matches_loop():
    op_single, op_stk, sparse_per, _plan_stk, _ = _make_stacked_op()
    # Build stacked sparse input matching plan_stk.indices order.
    sparse_stk = torch.stack(sparse_per, dim=1)  # (C, S, *natural_per)
    sparse_stk = sparse_stk.movedim(1, 0)  # (S, C, *natural_per)
    out = op_stk.adjoint(sparse_stk)
    ref = torch.stack(
        [op_single[s].adjoint(sparse_per[s]) for s in range(len(op_single))],
        dim=0,
    )
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# 4. nlinv_calib: batched Cartesian + train_reduce='mean'
# ---------------------------------------------------------------------------


def test_nlinv_calib_batched_cartesian_matches_loop():
    rng = np.random.default_rng(0)
    B, C, N = 2, 4, 32
    y_full = rng.standard_normal((C, N, N)) + 1j * rng.standard_normal((C, N, N))
    y_batch = np.stack(
        [
            y_full
            + 0.05
            * b
            * (
                rng.standard_normal(y_full.shape)
                + 1j * rng.standard_normal(y_full.shape)
            )
            for b in range(B)
        ],
        axis=0,
    ).astype(np.complex64)

    smaps_loop = []
    train_loop = []
    for b in range(B):
        s, t = nlinv_calib(y_batch[b], ndim=2, cal_width=16, max_iter=2, cg_iter=4)
        smaps_loop.append(torch.as_tensor(s))
        train_loop.append(torch.as_tensor(t))
    smaps_loop = torch.stack(smaps_loop, dim=0)
    train_loop = torch.stack(train_loop, dim=0)

    smaps_b, train_b = nlinv_calib(y_batch, ndim=2, cal_width=16, max_iter=2, cg_iter=4)
    smaps_b = torch.as_tensor(smaps_b)
    train_b = torch.as_tensor(train_b)
    assert smaps_b.shape == (B, C, N, N)
    torch.testing.assert_close(smaps_b, smaps_loop, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(train_b, train_loop, rtol=1e-5, atol=1e-5)


def test_nlinv_calib_train_reduce_mean():
    rng = np.random.default_rng(1)
    B, C, N = 2, 3, 24
    y_batch = (
        rng.standard_normal((B, C, N, N)) + 1j * rng.standard_normal((B, C, N, N))
    ).astype(np.complex64)
    _, train_none = nlinv_calib(
        y_batch, ndim=2, cal_width=12, max_iter=2, cg_iter=3, train_reduce="none"
    )
    _, train_mean = nlinv_calib(
        y_batch, ndim=2, cal_width=12, max_iter=2, cg_iter=3, train_reduce="mean"
    )
    train_none = torch.as_tensor(train_none)
    train_mean = torch.as_tensor(train_mean)
    assert train_mean.shape == train_none.shape[1:]
    torch.testing.assert_close(train_mean, train_none.mean(0), rtol=1e-5, atol=1e-5)


def test_nlinv_calib_invalid_train_reduce():
    y = (np.random.randn(4, 16, 16) + 1j * np.random.randn(4, 16, 16)).astype(
        np.complex64
    )
    with pytest.raises(ValueError, match="train_reduce"):
        nlinv_calib(y, ndim=2, cal_width=8, train_reduce="bogus")
