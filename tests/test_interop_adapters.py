"""Tests for the per-framework GROG / NLINV / coil-compression adapters.

These exercise the new ``GrogInterpolator``, ``nlinv_calib``, and
``coil_compress`` functions exposed under ``pygrog.interop.{mrpro,
sigpy, deepinv}`` sub-namespaces.

A small synthetic phantom + 8-arm spiral acquisition serves as the
shared fixture.  The point of these tests is to verify the *I/O
contracts* of the adapters (native-shape inputs and outputs); accuracy
of GROG / NLINV / coil compression themselves is covered elsewhere.
"""

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def fixture():
    """Return a dict carrying the shared synthetic acquisition."""
    pytest.importorskip("mrinufft")
    from mrinufft import get_operator, initialize_2D_spiral

    shape = (96, 96)
    n_coils = 4
    samples = initialize_2D_spiral(
        Nc=8, Ns=200, nb_revolutions=4
    ).astype(np.float32)
    coords_np = (samples * np.asarray(shape, np.float32)).astype(np.float32)

    yy, xx = np.mgrid[-1 : 1 : shape[0] * 1j, -1 : 1 : shape[1] * 1j]
    phantom = (xx**2 + yy**2 < 0.6**2).astype(np.complex64)
    smaps = np.stack(
        [
            np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 0.5)
            for cx, cy in [(0.5, 0), (-0.5, 0), (0, 0.5), (0, -0.5)]
        ]
    ).astype(np.complex64)
    smaps /= np.sqrt((np.abs(smaps) ** 2).sum(0)) + 1e-12

    nufft = get_operator("finufft")(
        samples=samples,
        shape=shape,
        n_coils=n_coils,
        smaps=smaps,
        squeeze_dims=True,
    )
    ksp = nufft.op(phantom).astype(np.complex64)  # (n_coils, n_samples)

    calib_full = np.fft.fftshift(
        np.fft.fftn(
            np.fft.ifftshift(smaps * phantom[None], axes=(-2, -1)),
            axes=(-2, -1),
        ),
        axes=(-2, -1),
    ).astype(np.complex64)
    cy, cx = shape[0] // 2, shape[1] // 2
    calib = calib_full[:, cy - 12 : cy + 12, cx - 12 : cx + 12]

    return {
        "shape": shape,
        "n_coils": n_coils,
        "samples": samples,
        "coords": coords_np,
        "ksp": ksp,
        "smaps": smaps,
        "calib": calib,
    }


def _make_kdata(fix):
    """Build a minimal mrpro ``KData`` from the fixture."""
    pytest.importorskip("mrpro")
    from mrpro.data import KData, KHeader, KTrajectory, SpatialDimension

    shape = fix["shape"]
    samples = fix["samples"]
    n_coils = fix["n_coils"]
    coords_np = fix["coords"]
    n_shots, n_read = samples.shape[:2]

    kx_t = torch.as_tensor(coords_np[..., 1]).reshape(1, 1, 1, n_shots, n_read)
    ky_t = torch.as_tensor(coords_np[..., 0]).reshape(1, 1, 1, n_shots, n_read)
    kz_t = torch.zeros_like(kx_t)
    traj = KTrajectory(kz=kz_t, ky=ky_t, kx=kx_t)
    data_t = (
        torch.as_tensor(fix["ksp"])
        .reshape(n_coils, 1, n_shots, n_read)
        .unsqueeze(0)
    )
    spatial = SpatialDimension(z=1, y=shape[0], x=shape[1])
    header = KHeader(
        recon_matrix=spatial,
        encoding_matrix=spatial,
        recon_fov=SpatialDimension(z=1.0, y=1.0, x=1.0),
        encoding_fov=SpatialDimension(z=1.0, y=1.0, x=1.0),
    )
    return KData(header=header, data=data_t, traj=traj)


# ---------------------------------------------------------------------------
# sigpy adapter
# ---------------------------------------------------------------------------
def test_sigpy_grog_interpolator(fixture):
    pytest.importorskip("sigpy")
    from pygrog.interop import sigpy as pg_sigpy

    fix = fixture
    grog = pg_sigpy.GrogInterpolator(
        fix["coords"],
        fix["shape"],
        kernel_width=2,
        oversamp=1.25,
    )
    grog.calc_interp_table(fix["calib"], lamda=0.01)
    sparse, plan = grog.interpolate(
        fix["ksp"].reshape(fix["n_coils"], *fix["samples"].shape[:2])
    )
    assert isinstance(sparse, np.ndarray)
    assert sparse.shape[0] == fix["n_coils"]
    assert sparse.shape[1:-1] == fix["samples"].shape[:2]
    assert sparse.shape[-1] >= 1  # kw axis present
    assert plan.natural_shape[:-1] == fix["samples"].shape[:2]


def test_sigpy_nlinv_calib(fixture):
    pytest.importorskip("sigpy")
    from pygrog.interop import sigpy as pg_sigpy

    fix = fixture
    smaps = pg_sigpy.nlinv_calib(
        fix["ksp"],
        fix["coords"].reshape(-1, 2),
        fix["shape"],
        cal_width=20,
        max_iter=3,
    )
    assert isinstance(smaps, np.ndarray)
    assert smaps.shape == (fix["n_coils"], *fix["shape"])


def test_sigpy_coil_compress(fixture):
    pytest.importorskip("sigpy")
    from pygrog.interop import sigpy as pg_sigpy

    fix = fixture
    n_v = 2
    compressed, matrix = pg_sigpy.coil_compress(fix["ksp"], n_v)
    assert isinstance(compressed, np.ndarray)
    assert compressed.shape == (n_v, fix["ksp"].shape[1])
    assert matrix.shape == (n_v, fix["n_coils"])


# ---------------------------------------------------------------------------
# deepinv adapter
# ---------------------------------------------------------------------------
def test_deepinv_grog_interpolator(fixture):
    pytest.importorskip("deepinv")
    from pygrog.interop import deepinv as pg_deepinv

    fix = fixture
    ksp_t = torch.as_tensor(
        fix["ksp"].reshape(fix["n_coils"], *fix["samples"].shape[:2])
    ).unsqueeze(0)  # (B=1, coils, k1, k0)
    grog = pg_deepinv.GrogInterpolator(
        fix["coords"],
        fix["shape"],
        kernel_width=2,
        oversamp=1.25,
    )
    grog.calc_interp_table(fix["calib"], lamda=0.01)
    sparse, plan = grog.interpolate(ksp_t)
    assert isinstance(sparse, torch.Tensor)
    assert sparse.shape[0] == 1
    assert sparse.shape[1] == fix["n_coils"]
    assert sparse.shape[2:-1] == fix["samples"].shape[:2]


def test_deepinv_nlinv_calib(fixture):
    pytest.importorskip("deepinv")
    from pygrog.interop import deepinv as pg_deepinv

    fix = fixture
    ksp_t = torch.as_tensor(fix["ksp"]).unsqueeze(0)  # (1, coils, n_samples)
    smaps = pg_deepinv.nlinv_calib(
        ksp_t,
        fix["coords"].reshape(-1, 2),
        fix["shape"],
        cal_width=20,
        max_iter=3,
    )
    assert isinstance(smaps, torch.Tensor)
    assert smaps.shape == (1, fix["n_coils"], *fix["shape"])


def test_deepinv_coil_compress(fixture):
    pytest.importorskip("deepinv")
    from pygrog.interop import deepinv as pg_deepinv

    fix = fixture
    ksp_t = torch.as_tensor(fix["ksp"]).unsqueeze(0)
    compressed, matrix = pg_deepinv.coil_compress(ksp_t, 2)
    assert compressed.shape == (1, 2, fix["ksp"].shape[1])
    assert matrix.shape == (2, fix["n_coils"])


# ---------------------------------------------------------------------------
# mrpro adapter
# ---------------------------------------------------------------------------
def test_mrpro_grog_interpolator(fixture):
    pytest.importorskip("mrpro")
    from pygrog.interop import mrpro as pg_mrpro

    fix = fixture
    kdata = _make_kdata(fix)
    grog = pg_mrpro.GrogInterpolator(kdata, kernel_width=2, oversamp=1.25)
    grog.calc_interp_table(fix["calib"], lamda=0.01)
    new_kdata, plan = grog.interpolate(kdata)
    # mrpro layout: (other=1, coils, k2=1, k1=n_shots, k0=n_read*kw)
    assert new_kdata.data.shape[0] == 1  # other
    assert new_kdata.data.shape[1] == fix["n_coils"]
    assert new_kdata.data.shape[2] == 1  # k2 (2D)
    assert new_kdata.data.shape[3] == fix["samples"].shape[0]  # k1 = n_shots
    assert new_kdata.data.shape[4] == fix["samples"].shape[1] * plan.natural_shape[-1]
    # Trajectory carries new gridded sample positions.
    assert new_kdata.traj.kx.shape[-2:] == new_kdata.data.shape[-2:]


def test_mrpro_nlinv_calib(fixture):
    pytest.importorskip("mrpro")
    from pygrog.interop import mrpro as pg_mrpro

    fix = fixture
    kdata = _make_kdata(fix)
    smaps = pg_mrpro.nlinv_calib(kdata, cal_width=20, max_iter=3)
    # mrpro smap layout: (n_coils, z=1, y, x) for 2D
    assert smaps.shape == (fix["n_coils"], 1, *fix["shape"])


def test_mrpro_coil_compress(fixture):
    pytest.importorskip("mrpro")
    from pygrog.interop import mrpro as pg_mrpro

    fix = fixture
    kdata = _make_kdata(fix)
    new = pg_mrpro.coil_compress(kdata, 2)
    # KData with reduced coil dim (axis -4).
    assert new.data.shape[-4] == 2
    # All other axes preserved.
    assert new.data.shape[:-4] == kdata.data.shape[:-4]
    assert new.data.shape[-3:] == kdata.data.shape[-3:]


# ===========================================================================
# Toeplitz `.normal()` short-circuit hooks
# ===========================================================================
def _make_toeplitz_op():
    from pygrog.operator import SparseFFT
    torch.manual_seed(0)
    nx = ny = 32
    n_coils = 4
    n_samples = 800
    indices = torch.randint(0, nx * ny, (n_samples,), dtype=torch.int64)
    weights = torch.rand(n_samples, dtype=torch.float32) + 0.1
    smaps = torch.randn(n_coils, ny, nx, dtype=torch.complex64) * 0.5
    op_t = SparseFFT(grid_shape=(ny, nx), image_shape=(ny, nx),
                     indices=indices, weights=weights, smaps=smaps,
                     toeplitz=True)
    op_n = SparseFFT(grid_shape=(ny, nx), image_shape=(ny, nx),
                     indices=indices, weights=weights, smaps=smaps,
                     toeplitz=False)
    return op_t, op_n


def _rel_err(a, b):
    return (a - b).abs().max().item() / (b.abs().max().item() + 1e-30)


def test_sigpy_grog_normal_linop():
    pytest.importorskip("sigpy")
    from pygrog.interop import GrogNormalLinop
    op_t, op_n = _make_toeplitz_op()
    x = torch.randn(*op_t.image_shape, dtype=torch.complex64)
    ref = op_n.normal(x)
    AHA = GrogNormalLinop(op_t)
    out = AHA(x.numpy())
    assert _rel_err(torch.as_tensor(out), ref) < 1e-4


def test_deepinv_grog_a_adjoint_a():
    pytest.importorskip("deepinv")
    from pygrog.interop import GrogLinearPhysics
    op_t, op_n = _make_toeplitz_op()
    x = torch.randn(*op_t.image_shape, dtype=torch.complex64)
    ref = op_n.normal(x)
    phys = GrogLinearPhysics(op_t)
    out = phys.A_adjoint_A(x.unsqueeze(0).unsqueeze(0))
    assert _rel_err(out[0, 0], ref) < 1e-4


def test_mrpro_h_gram_normal():
    pytest.importorskip("mrpro")
    import types
    from pygrog.operator import SparseFFT
    from pygrog.interop import GrogLinearOp

    # mrpro adapter requires a multi-axis natural_shape (kw fusion).
    torch.manual_seed(0)
    nx = ny = 32
    n_coils = 4
    n_shots, n_pts = 16, 50
    n_samples = n_shots * n_pts
    indices = torch.randint(0, nx * ny, (n_samples,), dtype=torch.int64)
    weights = torch.rand(n_samples, dtype=torch.float32) + 0.1
    smaps = torch.randn(n_coils, ny, nx, dtype=torch.complex64) * 0.5
    sort_perm = torch.argsort(indices)
    inv_perm = torch.empty_like(sort_perm)
    inv_perm[sort_perm] = torch.arange(n_samples)
    plan = types.SimpleNamespace(
        grid_shape=(ny, nx), image_shape=(ny, nx), grid_size=ny * nx,
        indices=indices[sort_perm],
        sqrt_weights=torch.sqrt(weights)[sort_perm],
        sort_perm=sort_perm, inv_perm=inv_perm,
        natural_shape=(n_shots, n_pts), n_samples=n_samples,
    )
    op_t = SparseFFT(plan=plan, smaps=smaps, toeplitz=True)
    op_n = SparseFFT(plan=plan, smaps=smaps, toeplitz=False)
    x = torch.randn(ny, nx, dtype=torch.complex64)
    ref = op_n.normal(x)

    m = GrogLinearOp(op_t)
    (out,) = m.H.gram(x)
    assert _rel_err(out, ref) < 1e-4
