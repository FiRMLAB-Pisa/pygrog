"""Smoke + agreement tests for ``pygrog.cg`` / ``pygrog.lsmr`` /
``PolynomialPreconditioner`` exposed through the operator ``solve()`` mixin.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pygrog import PolynomialPreconditioner
from pygrog._solve._mixin import SolveMixin
from pygrog.calib import GrogInterpolator
from pygrog.gadgets._off_resonance import (
    OffResonanceMaskedFFT,
    OffResonanceSparseFFT,
)
from pygrog.gadgets._subspace import SubspaceMaskedFFT, SubspaceSparseFFT
from pygrog.operator import MaskedFFT, SparseFFT


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------
@pytest.fixture(scope="module")
def grog_setup():
    rng = np.random.default_rng(0)
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
# Mixin attachment
# ---------------------------------------------------------------------
@pytest.mark.parametrize(
    "cls",
    [
        SparseFFT,
        MaskedFFT,
        SubspaceSparseFFT,
        SubspaceMaskedFFT,
        OffResonanceSparseFFT,
        OffResonanceMaskedFFT,
    ],
)
def test_solvemixin_inherited(cls):
    assert issubclass(cls, SolveMixin)
    assert callable(cls.solve)


# ---------------------------------------------------------------------
# CG / LSMR run cleanly and produce same shape
# ---------------------------------------------------------------------
def test_sparsefft_cg_lsmr_shapes(grog_setup):
    g = grog_setup["grog"]
    data = grog_setup["data"]
    smaps = grog_setup["smaps"]

    op = SparseFFT(plan=g.plan, smaps=smaps)
    sparse = g.interpolate(data, ret_image=False)
    sparse_t = torch.as_tensor(np.asarray(sparse))
    b = sparse_t * torch.as_tensor(g.plan.pre_weights)
    b = b.reshape(*b.shape[:-1], *op.natural_shape)

    x_cg = op.solve(b, method="cg", max_iter=10, damp=1e-3)
    x_lsmr = op.solve(b, method="lsmr", max_iter=10)
    assert x_cg.shape == tuple(grog_setup["image_shape"])
    assert x_lsmr.shape == tuple(grog_setup["image_shape"])
    assert torch.isfinite(x_cg).all()
    assert torch.isfinite(x_lsmr).all()


def test_maskedfft_cg_lsmr_shapes(grog_setup):
    g = grog_setup["grog"]
    data = grog_setup["data"]
    smaps = grog_setup["smaps"]

    kgrid, mplan = g.interpolate(data, grid=True)
    mop = MaskedFFT(plan=mplan, smaps=smaps)
    b = torch.as_tensor(np.asarray(kgrid))

    x_cg = mop.solve(b, method="cg", max_iter=10, damp=1e-3)
    x_lsmr = mop.solve(b, method="lsmr", max_iter=10)
    assert x_cg.shape == tuple(grog_setup["image_shape"])
    assert x_lsmr.shape == tuple(grog_setup["image_shape"])
    assert torch.isfinite(x_cg).all()
    assert torch.isfinite(x_lsmr).all()


# ---------------------------------------------------------------------
# Default method dispatch
# ---------------------------------------------------------------------
def test_default_method_picks_cg_when_toeplitz(grog_setup):
    g = grog_setup["grog"]
    data = grog_setup["data"]
    smaps = grog_setup["smaps"]

    op = SparseFFT(plan=g.plan, smaps=smaps)
    assert op.toeplitz is True  # CPU default
    sparse_t = torch.as_tensor(np.asarray(g.interpolate(data, ret_image=False)))
    b = sparse_t * torch.as_tensor(g.plan.pre_weights)
    b = b.reshape(*b.shape[:-1], *op.natural_shape)

    x_default = op.solve(b, max_iter=5)
    x_cg = op.solve(b, method="cg", max_iter=5)
    # Default 'cg' path should match explicit 'cg' bit-for-bit.
    assert torch.allclose(x_default, x_cg)


# ---------------------------------------------------------------------
# Polynomial preconditioner
# ---------------------------------------------------------------------
def test_polynomial_preconditioner_smoke(grog_setup):
    g = grog_setup["grog"]
    smaps = grog_setup["smaps"]
    op = SparseFFT(plan=g.plan, smaps=smaps)

    pc = PolynomialPreconditioner(op, degree=3, n_power_iter=5)
    assert pc.degree == 3
    assert len(pc.coeffs) == 4
    assert pc.spectrum[0] == 0.0
    assert pc.spectrum[1] > 0.0

    # Apply on a sample image — shape preserved.
    x = torch.randn(*grog_setup["image_shape"], dtype=torch.complex64)
    y = pc.apply(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_pcg_runs_and_matches_cg_shape(grog_setup):
    g = grog_setup["grog"]
    data = grog_setup["data"]
    smaps = grog_setup["smaps"]

    op = SparseFFT(plan=g.plan, smaps=smaps)
    sparse_t = torch.as_tensor(np.asarray(g.interpolate(data, ret_image=False)))
    b = sparse_t * torch.as_tensor(g.plan.pre_weights)
    b = b.reshape(*b.shape[:-1], *op.natural_shape)

    pc = PolynomialPreconditioner(op, degree=2, n_power_iter=5)
    x_cg = op.solve(b, method="cg", max_iter=10, damp=1e-3)
    x_pcg = op.solve(b, method="cg", max_iter=10, damp=1e-3, preconditioner=pc)
    assert x_pcg.shape == x_cg.shape
    assert torch.isfinite(x_pcg).all()


def test_polynomial_preconditioner_explicit_sample_shape(grog_setup):
    """When `op` lacks `smaps` and `n_coils`, user must pass `sample_shape`."""
    g = grog_setup["grog"]
    op = SparseFFT(plan=g.plan)  # no smaps → operator has no n_coils

    with pytest.raises(RuntimeError, match="sample_shape"):
        PolynomialPreconditioner(op, degree=2, n_power_iter=3)

    pc = PolynomialPreconditioner(
        op,
        degree=2,
        n_power_iter=3,
        sample_shape=(grog_setup["n_coils"], *grog_setup["image_shape"]),
    )
    assert pc.spectrum[1] > 0.0


# ---------------------------------------------------------------------
# Unknown method
# ---------------------------------------------------------------------
def test_unknown_method_raises(grog_setup):
    g = grog_setup["grog"]
    smaps = grog_setup["smaps"]
    op = SparseFFT(plan=g.plan, smaps=smaps)
    sparse_t = torch.as_tensor(
        np.asarray(g.interpolate(grog_setup["data"], ret_image=False))
    )
    b = sparse_t * torch.as_tensor(g.plan.pre_weights)
    b = b.reshape(*b.shape[:-1], *op.natural_shape)

    with pytest.raises(ValueError, match="Unknown method"):
        op.solve(b, method="bogus")
