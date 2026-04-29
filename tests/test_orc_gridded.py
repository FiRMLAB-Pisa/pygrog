"""Test the gridded ORC path: ``with_off_resonance`` over a ``MaskedFFT``.

Verifies that when ``coords`` and ``dcf`` are supplied, the temporal basis
is re-evaluated on the Cartesian grid (shape ``(*grid_shape, L)``) so that
:class:`OffResonanceMaskedFFT` broadcasts it correctly against gridded
k-space.  Also checks that omitting them raises a clear error.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pygrog.calib import GrogInterpolator
from pygrog.gadgets._off_resonance import (
    OffResonanceMaskedFFT,
    with_off_resonance,
)
from pygrog.operator import MaskedFFT


@pytest.fixture(scope="module")
def gridded_setup():
    rng = np.random.default_rng(0)
    shape = (16, 16)
    image_shape = (12, 12)
    nv, nr = 8, 16
    nc = 2
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
    _, mplan = grog.interpolate(data, grid=True)

    # Per-sample readout times and a simple radial-style DCF (|k|).
    readout_time = np.linspace(0, 1e-3, nv * nr, dtype=np.float32)
    dcf = np.linalg.norm(coords.reshape(-1, 2), axis=-1).astype(np.float32) + 1e-3

    b0_map = (rng.standard_normal(image_shape).astype(np.float32) * 5.0)
    return {
        "mplan": mplan,
        "image_shape": image_shape,
        "grid_shape": tuple(mplan.grid_shape),
        "coords": coords,
        "dcf": dcf,
        "readout_time": readout_time,
        "b0_map": b0_map,
        "n_coils": nc,
    }


def test_gridded_orc_basis_shape_matches_grid(gridded_setup):
    """B is reshaped to (*grid_shape, L) so existing reshape branch fires."""
    op_m = MaskedFFT(plan=gridded_setup["mplan"])
    orc = with_off_resonance(
        op_m,
        b0_map=gridded_setup["b0_map"],
        readout_time=gridded_setup["readout_time"],
        coords=gridded_setup["coords"],
        dcf=gridded_setup["dcf"],
        method="svd",
        L=3,
    )
    assert isinstance(orc, OffResonanceMaskedFFT)
    grid_shape = gridded_setup["grid_shape"]
    assert orc.B.shape[:-1] == grid_shape, (
        f"expected B shape (*grid_shape, L) = {(*grid_shape, 3)}, got {tuple(orc.B.shape)}"
    )
    assert orc.L == 3


def test_gridded_orc_recenter_at_k0(gridded_setup):
    """t_grid at center index ≈ readout_time[k≈0]."""
    from pygrog.gadgets._off_resonance import _gridded_orc_basis
    from mrinufft.extras.field_map import get_complex_fieldmap_rad

    s = gridded_setup
    field_map = get_complex_fieldmap_rad(s["b0_map"], None)
    mask = np.ones(s["image_shape"], dtype=bool)
    # Manufacture a tiny B/C of the right shapes (only their dtypes/shapes
    # are inspected by the helper to recover L and bin counts).
    L = 2
    n_samples = s["readout_time"].size
    B = np.zeros((n_samples, L), dtype=np.complex64)
    C = np.zeros((L, *s["image_shape"]), dtype=np.complex64)

    _, t_grid = _gridded_orc_basis(
        B_orig=B,
        C=C,
        readout_time=s["readout_time"],
        field_map_complex=field_map,
        mask=mask,
        coords=s["coords"],
        dcf=s["dcf"],
        grid_shape=s["grid_shape"],
        n_bins=128,
    )
    center = tuple(g // 2 for g in s["grid_shape"])
    k_norms = np.linalg.norm(s["coords"].reshape(-1, 2), axis=-1)
    expected = float(s["readout_time"][int(np.argmin(k_norms))])
    assert np.isclose(t_grid[center], expected, atol=1e-5), (
        f"t_grid[center]={t_grid[center]:.6e}, expected={expected:.6e}"
    )


def test_with_off_resonance_masked_requires_coords_dcf(gridded_setup):
    op_m = MaskedFFT(plan=gridded_setup["mplan"])
    with pytest.raises(ValueError, match="coords"):
        with_off_resonance(
            op_m,
            b0_map=gridded_setup["b0_map"],
            readout_time=gridded_setup["readout_time"],
            method="svd",
            L=2,
        )


def test_coords_dcf_propagate_from_plan(gridded_setup):
    s = gridded_setup
    s["mplan"]._coords = torch.as_tensor(s["coords"])
    s["mplan"]._dcf = torch.as_tensor(s["dcf"]).float()
    op_m = MaskedFFT(plan=s["mplan"])
    assert op_m._coords is not None
    assert op_m._dcf is not None
    # No coords/dcf kwargs — should pull from operator.
    orc = with_off_resonance(
        op_m,
        b0_map=s["b0_map"],
        readout_time=s["readout_time"],
        method="svd",
        L=2,
    )
    assert orc.B.shape[:-1] == s["grid_shape"]


def test_gridded_orc_forward_adjoint_shapes(gridded_setup):
    """Smoke test: forward / adjoint produce the right shapes and finite values."""
    s = gridded_setup
    op_m = MaskedFFT(plan=s["mplan"])
    orc = with_off_resonance(
        op_m,
        b0_map=s["b0_map"],
        readout_time=s["readout_time"],
        coords=s["coords"],
        dcf=s["dcf"],
        method="svd",
        L=2,
    )
    n_coils = s["n_coils"]
    img = torch.randn(n_coils, *s["image_shape"], dtype=torch.complex64) * 0.1
    kgrid = orc.forward(img)
    assert kgrid.shape == (n_coils, *s["grid_shape"])
    assert torch.isfinite(kgrid).all()
    img_back = orc.adjoint(kgrid)
    assert img_back.shape == (n_coils, *s["image_shape"])
    assert torch.isfinite(img_back).all()
