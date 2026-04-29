"""Tests for the :class:`MaskedFFT` operator family and the
:meth:`GrogInterpolator.interpolate` ``grid=True`` path.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pygrog.calib import GrogInterpolator
from pygrog.operator import MaskedFFT, MaskedFFTPlan, SparseFFT


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------
@pytest.fixture(scope="module")
def grog_setup():
    rng = np.random.default_rng(42)
    shape = (16, 16)
    image_shape = (12, 12)
    nc, nv, nr = 3, 8, 16
    coords = rng.standard_normal((nv, nr, 2)).astype(np.float32)
    grog = GrogInterpolator(
        shape=shape,
        coords=coords,
        kernel_width=2,
        oversamp=2.0,
        image_shape=image_shape,
    )
    calib = (
        rng.standard_normal((nc, *shape)) + 1j * rng.standard_normal((nc, *shape))
    ).astype(np.complex64)
    grog.calc_interp_table(calib, lamda=0.01, precision=1)

    data = (
        rng.standard_normal((nc, nv, nr)) + 1j * rng.standard_normal((nc, nv, nr))
    ).astype(np.complex64)
    smaps = (
        rng.standard_normal((nc, *image_shape))
        + 1j * rng.standard_normal((nc, *image_shape))
    ).astype(np.complex64)
    return {
        "grog": grog,
        "data": data,
        "smaps": torch.as_tensor(smaps),
        "image_shape": image_shape,
        "n_coils": nc,
    }


# ---------------------------------------------------------------------
# interpolate(grid=True) shape contracts
# ---------------------------------------------------------------------
def test_interpolate_grid_returns_plan_and_kspace(grog_setup):
    g = grog_setup["grog"]
    data = grog_setup["data"]
    nc = grog_setup["n_coils"]

    kgrid, plan = g.interpolate(data, grid=True)

    assert isinstance(plan, MaskedFFTPlan)
    assert tuple(plan.image_shape) == grog_setup["image_shape"]
    grid_shape = plan.grid_shape
    # Output kspace: (n_coils, *grid_shape)
    assert tuple(kgrid.shape) == (nc, *grid_shape)
    assert plan.mask.shape == plan.density.shape == grid_shape


def test_interpolate_grid_and_ret_image_mutually_exclusive(grog_setup):
    g = grog_setup["grog"]
    data = grog_setup["data"]
    with pytest.raises(ValueError):
        g.interpolate(data, grid=True, ret_image=True)


# ---------------------------------------------------------------------
# MaskedFFT construction parity (plan vs raw arguments)
# ---------------------------------------------------------------------
def test_maskedfft_plan_constructor(grog_setup):
    g = grog_setup["grog"]
    data = grog_setup["data"]

    _, plan = g.interpolate(data, grid=True)

    op_plan = MaskedFFT(plan=plan)
    op_raw = MaskedFFT(
        grid_shape=plan.grid_shape,
        image_shape=plan.image_shape,
        mask=plan.mask,
        density=plan.density,
    )
    assert op_plan.grid_shape == op_raw.grid_shape
    assert op_plan.image_shape == op_raw.image_shape
    assert torch.equal(op_plan.mask, op_raw.mask)
    assert torch.equal(op_plan.density, op_raw.density)


def test_maskedfft_requires_plan_or_explicit_args():
    with pytest.raises(ValueError, match="plan.*required"):
        MaskedFFT()


# ---------------------------------------------------------------------
# Forward / adjoint shape contracts
# ---------------------------------------------------------------------
def test_maskedfft_forward_shape_with_smaps(grog_setup):
    g = grog_setup["grog"]
    data = grog_setup["data"]
    smaps = grog_setup["smaps"]

    kgrid, plan = g.interpolate(data, grid=True)
    op = MaskedFFT(plan=plan, smaps=smaps)
    img = op.adjoint(kgrid)
    assert img.shape == tuple(grog_setup["image_shape"])
    assert img.dtype == kgrid.dtype


def test_maskedfft_forward_shape_no_smaps(grog_setup):
    g = grog_setup["grog"]
    data = grog_setup["data"]

    kgrid, plan = g.interpolate(data, grid=True)
    op = MaskedFFT(plan=plan)
    img = op.adjoint(kgrid)
    assert img.shape == (grog_setup["n_coils"], *grog_setup["image_shape"])


def test_maskedfft_adjoint_shape_with_smaps(grog_setup):
    g = grog_setup["grog"]
    data = grog_setup["data"]
    smaps = grog_setup["smaps"]

    _, plan = g.interpolate(data, grid=True)
    op = MaskedFFT(plan=plan, smaps=smaps)
    img = torch.randn(*grog_setup["image_shape"], dtype=torch.complex64)
    kgrid = op.forward(img)
    assert kgrid.shape == (grog_setup["n_coils"], *plan.grid_shape)


# ---------------------------------------------------------------------
# Adjoint dot-product test (true adjointness)
# ---------------------------------------------------------------------
def test_maskedfft_dot_product(grog_setup):
    g = grog_setup["grog"]
    data = grog_setup["data"]
    smaps = grog_setup["smaps"]

    _, plan = g.interpolate(data, grid=True)
    op = MaskedFFT(plan=plan, smaps=smaps)

    torch.manual_seed(0)
    x = torch.randn(*grog_setup["image_shape"], dtype=torch.complex64)
    y = torch.randn(grog_setup["n_coils"], *plan.grid_shape, dtype=torch.complex64)

    Ax = op.forward(x)  # image → kspace  (A x)
    AHy = op.adjoint(y)  # kspace → image  (A^H y)

    lhs = torch.vdot(Ax.reshape(-1), y.reshape(-1))
    rhs = torch.vdot(x.reshape(-1), AHy.reshape(-1))
    assert torch.allclose(lhs, rhs, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------
# Toeplitz vs explicit forward(adjoint(.)) equivalence
# ---------------------------------------------------------------------
def test_maskedfft_toeplitz_matches_compose(grog_setup):
    g = grog_setup["grog"]
    data = grog_setup["data"]
    smaps = grog_setup["smaps"]

    _, plan = g.interpolate(data, grid=True)
    op = MaskedFFT(plan=plan, smaps=smaps)
    assert op.toeplitz is True  # CPU default

    img = torch.randn(*grog_setup["image_shape"], dtype=torch.complex64)
    y_toep = op.normal(img)

    op.toeplitz = False
    op._toep_op = None
    y_compose = op.normal(img)

    assert torch.allclose(y_toep, y_compose, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------
# Batched/stacked forward
# ---------------------------------------------------------------------
def test_maskedfft_batched_forward(grog_setup):
    g = grog_setup["grog"]
    data = grog_setup["data"]
    smaps = grog_setup["smaps"]

    kgrid, plan = g.interpolate(data, grid=True)
    op = MaskedFFT(plan=plan, smaps=smaps)

    # Build a batch of 2 by stacking along leading axis.
    kgrid_t = torch.as_tensor(np.asarray(kgrid))
    batched = torch.stack([kgrid_t, kgrid_t + 0.1])
    img_batched = op.adjoint(batched)
    assert img_batched.shape == (2, *grog_setup["image_shape"])

    img0 = op.adjoint(kgrid_t)
    assert torch.allclose(img_batched[0], img0)


# ---------------------------------------------------------------------
# SparseFFT vs MaskedFFT numerical agreement
# ---------------------------------------------------------------------
def test_sparsefft_vs_maskedfft_image_close(grog_setup):
    """Image reconstructed via the sparse path should be close to the one
    via the masked/grid path (they use the same density-compensated
    gridding under the hood)."""
    g = grog_setup["grog"]
    data = grog_setup["data"]
    smaps = grog_setup["smaps"]

    # Sparse path
    sparse = g.interpolate(data, ret_image=False)
    sparse_t = torch.as_tensor(np.asarray(sparse))
    op_sp = SparseFFT(plan=g.plan, smaps=smaps)
    b_sp = sparse_t * torch.as_tensor(g.plan.pre_weights)
    img_sp = op_sp.adjoint(b_sp)

    # Masked/grid path
    kgrid, plan = g.interpolate(data, grid=True)
    op_m = MaskedFFT(plan=plan, smaps=smaps)
    img_m = op_m.adjoint(kgrid)

    # They should agree closely (same scattering math, only ordering differs).
    rel = (
        torch.linalg.vector_norm(img_sp - img_m) / torch.linalg.vector_norm(img_sp)
    ).item()
    assert rel < 1e-4
