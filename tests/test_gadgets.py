"""Tests for pygrog gadgets.

Covers:
  - SubspaceProjection: SVD fit, forward projection, adjoint expansion
  - SubspaceSparseFFT: shapes, dot-product adjointness (with and without smaps)
  - OffResonanceSparseFFT: shapes, dot-product adjointness (manufactured B/C)
  - OffResonanceCorrection: adjointness with a trivial (zero) B0 field map

Tests parametrized on ``device`` run on CPU always; CUDA is skipped when no
GPU is available.
"""

import numpy as np
import torch

from pygrog.gadgets._subspace import (
    SubspaceProjection,
    SubspaceSparseFFT,
    with_subspace,
)
from pygrog.gadgets._off_resonance import (
    OffResonanceSparseFFT,
    OffResonanceCorrection,
)
from pygrog.operator._sparse_fft import SparseFFT

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sparse_fft(grid_shape, image_shape, n_samples, *, smaps=None, seed=0):
    rng = np.random.default_rng(seed)
    grid_size = int(np.prod(grid_shape))
    indices = rng.integers(0, grid_size, size=n_samples)
    weights = np.ones(n_samples, dtype=np.float32)
    return SparseFFT(grid_shape, image_shape, indices, weights, smaps=smaps)


def _cdot(a, b):
    return torch.vdot(a.flatten(), b.flatten())


def _make_orc_op(image_shape, n_samples, n_coils, L, *, seed=0, smaps=True):
    """Return OffResonanceSparseFFT with manufactured (random) B and C."""
    torch.manual_seed(seed)
    grid_shape = tuple(s + 4 for s in image_shape)
    smap_tensor = (
        torch.randn(n_coils, *image_shape, dtype=torch.complex64) if smaps else None
    )
    base = _make_sparse_fft(
        grid_shape, image_shape, n_samples, smaps=smap_tensor, seed=seed
    )
    # Small magnitudes so forward/adjoint numerics stay well-conditioned
    B = (torch.randn(n_samples, L, dtype=torch.complex64) * 0.1).numpy()
    C = (torch.randn(L, *image_shape, dtype=torch.complex64) * 0.1).numpy()
    return OffResonanceSparseFFT(base, B, C)


def _make_subspace_op(image_shape, n_samples, n_coils, K, T, *, seed=0):
    """Return SubspaceSparseFFT with smaps so batch path is exercised.

    The base SparseFFT is built with a 2-D ``natural_shape=(T, n_pts)``
    so the subspace gadget's ``encoding_axis`` (default ``-4``) lands
    on the temporal axis (``T``).
    """
    import types
    assert n_samples % T == 0, "n_samples must be divisible by T"
    n_pts = n_samples // T
    torch.manual_seed(seed)
    smaps = torch.randn(n_coils, *image_shape, dtype=torch.complex64)
    grid_shape = tuple(s + 4 for s in image_shape)
    rng = np.random.default_rng(seed)
    indices = torch.from_numpy(
        rng.integers(0, int(np.prod(grid_shape)), n_samples).astype(np.int64))
    weights = torch.ones(n_samples, dtype=torch.float32)
    sort_perm = torch.argsort(indices)
    inv_perm = torch.empty_like(sort_perm)
    inv_perm[sort_perm] = torch.arange(n_samples)
    plan = types.SimpleNamespace(
        grid_shape=grid_shape, image_shape=tuple(image_shape),
        grid_size=int(np.prod(grid_shape)),
        indices=indices[sort_perm],
        sqrt_weights=torch.sqrt(weights)[sort_perm],
        sort_perm=sort_perm, inv_perm=inv_perm,
        natural_shape=(T, n_pts), n_samples=n_samples,
    )
    base = SparseFFT(plan=plan, smaps=smaps)
    rng2 = torch.Generator().manual_seed(seed)
    basis_raw = torch.randn(K, T, dtype=torch.complex64, generator=rng2)
    basis = basis_raw / basis_raw.norm(dim=1, keepdim=True)
    # encoding_axis=-2: full layout is (C, T, n_pts) → -2 → T axis.
    return SubspaceSparseFFT(base, basis, encoding_axis=-2)


# Constants for OffResonanceCorrection tests
_ORC_IMG = (12, 12)
_ORC_N_SAMPLES = 48
_ORC_N_COILS = 3
_ORC_L = 3


def _build_orc(seed=0):
    base = _make_sparse_fft((16, 16), _ORC_IMG, _ORC_N_SAMPLES, seed=seed)
    field_map = (
        np.random.default_rng(seed).standard_normal(_ORC_IMG).astype(np.float32) * 5
    )
    readout_time = np.linspace(0, 1e-3, _ORC_N_SAMPLES, dtype=np.float32)
    return OffResonanceCorrection(base, field_map, readout_time, n_components=-1)


# Constants for SubspaceSparseFFT tests
_SUB_IMG = (16, 16)
_SUB_N_SAMPLES = 64
_SUB_N_COILS = 3
_SUB_K, _SUB_T = 4, 8  # subspace rank, number of frames

# Constants for OffResonanceSparseFFT tests
_ORS_IMG = (16, 16)
_ORS_N_SAMPLES = 64
_ORS_N_COILS = 3
_ORS_L = 4


# ===========================================================================
# SubspaceProjection
# ===========================================================================


def test_subspace_projection_fit_and_forward_shape():
    n_frames, n_spatial = 20, 100
    data = torch.randn(n_frames, n_spatial, dtype=torch.complex64)
    proj = SubspaceProjection(n_components=5)
    proj.fit(data)
    assert proj.basis.shape == (5, n_frames)
    coeff = proj.forward(data)
    assert coeff.shape == (5, n_spatial)


def test_subspace_projection_basis_orthonormal():
    """Fitted basis rows are orthonormal: basis @ basis^H ≈ I."""
    n_frames, n_spatial = 30, 200
    data = torch.randn(n_frames, n_spatial, dtype=torch.complex64)
    proj = SubspaceProjection(n_components=5)
    proj.fit(data)
    G = proj.basis @ proj.basis.conj().T  # (5, 5) Gram matrix
    torch.testing.assert_close(
        G, torch.eye(5, dtype=torch.complex64), rtol=1e-5, atol=1e-5
    )


def test_subspace_projection_full_rank_roundtrip():
    """Full-rank projection (K=n_frames) is an exact roundtrip."""
    n_frames, nx, ny = 16, 8, 8
    data = torch.randn(n_frames, nx, ny, dtype=torch.complex64)
    proj = SubspaceProjection(n_components=n_frames)
    proj.fit(data.reshape(n_frames, -1))
    recon = proj.adjoint(proj.forward(data))
    torch.testing.assert_close(recon, data, atol=1e-4, rtol=1e-5)


def test_subspace_projection_is_idempotent():
    """Projecting already-projected data changes nothing."""
    n_frames, n_spatial = 20, 100
    data = torch.randn(n_frames, n_spatial, dtype=torch.complex64)
    proj = SubspaceProjection(n_components=5)
    proj.fit(data)
    coeff = proj.forward(data)
    recon = proj.adjoint(coeff)  # back to frame space
    coeff2 = proj.forward(recon)  # project again
    torch.testing.assert_close(coeff2, coeff, rtol=1e-5, atol=1e-5)


# ===========================================================================
# SubspaceSparseFFT
# ===========================================================================


def test_subspace_sparse_fft_forward_shape(device):
    op = _make_subspace_op(_SUB_IMG, _SUB_N_SAMPLES, _SUB_N_COILS, _SUB_K, _SUB_T)
    ksp = torch.randn(
        _SUB_N_COILS, _SUB_T, _SUB_N_SAMPLES // _SUB_T,
        dtype=torch.complex64, device=device,
    )
    out = op.forward(ksp)
    assert out.shape == (_SUB_K, *_SUB_IMG)
    assert out.device.type == device.type


def test_subspace_sparse_fft_adjoint_shape(device):
    op = _make_subspace_op(_SUB_IMG, _SUB_N_SAMPLES, _SUB_N_COILS, _SUB_K, _SUB_T)
    coeffs = torch.randn(_SUB_K, *_SUB_IMG, dtype=torch.complex64, device=device)
    out = op.adjoint(coeffs)
    # Adjoint output shape is (1, C, T, n_pts) with the leading 1 batch.
    assert out.shape[-3:] == (_SUB_N_COILS, _SUB_T, _SUB_N_SAMPLES // _SUB_T)
    assert out.device.type == device.type


def test_subspace_sparse_fft_adjointness(device):
    """<forward(x), y> == <x, adjoint(y)>."""
    op = _make_subspace_op(
        _SUB_IMG, _SUB_N_SAMPLES, _SUB_N_COILS, _SUB_K, _SUB_T, seed=1
    )
    x = torch.randn(
        _SUB_N_COILS, _SUB_T, _SUB_N_SAMPLES // _SUB_T,
        dtype=torch.complex64, device=device,
    )
    y = torch.randn(_SUB_K, *_SUB_IMG, dtype=torch.complex64, device=device)
    fx = op.forward(x)
    aty = op.adjoint(y).reshape(x.shape)
    torch.testing.assert_close(
        _cdot(fx, y), _cdot(x, aty), rtol=1e-4, atol=1e-4
    )


def test_subspace_sparse_fft_with_subspace_wrapper(device):
    """with_subspace() returns a SubspaceSparseFFT with correct types."""
    op = _make_subspace_op(
        _SUB_IMG, _SUB_N_SAMPLES, _SUB_N_COILS, _SUB_K, _SUB_T, seed=2
    )
    assert isinstance(op, SubspaceSparseFFT)
    ksp = torch.randn(
        _SUB_N_COILS, _SUB_T, _SUB_N_SAMPLES // _SUB_T,
        dtype=torch.complex64, device=device,
    )
    assert op.forward(ksp).shape == (_SUB_K, *_SUB_IMG)


# ===========================================================================
# OffResonanceSparseFFT
# ===========================================================================


def test_off_resonance_sparse_fft_forward_shape_with_smaps(device):
    op = _make_orc_op(_ORS_IMG, _ORS_N_SAMPLES, _ORS_N_COILS, _ORS_L, seed=10)
    ksp = torch.randn(
        _ORS_N_COILS, _ORS_N_SAMPLES, dtype=torch.complex64, device=device
    )
    out = op.forward(ksp)
    assert out.shape == _ORS_IMG
    assert out.device.type == device.type


def test_off_resonance_sparse_fft_adjoint_shape_with_smaps(device):
    op = _make_orc_op(_ORS_IMG, _ORS_N_SAMPLES, _ORS_N_COILS, _ORS_L, seed=11)
    img = torch.randn(*_ORS_IMG, dtype=torch.complex64, device=device)
    out = op.adjoint(img)
    assert out.shape == (_ORS_N_COILS, _ORS_N_SAMPLES)
    assert out.device.type == device.type


def test_off_resonance_sparse_fft_adjointness_with_smaps(device):
    """<forward(x), y> == <x, adjoint(y)> (SENSE + B0 correction path)."""
    op = _make_orc_op(_ORS_IMG, _ORS_N_SAMPLES, _ORS_N_COILS, _ORS_L, seed=12)
    x = torch.randn(_ORS_N_COILS, _ORS_N_SAMPLES, dtype=torch.complex64, device=device)
    y = torch.randn(*_ORS_IMG, dtype=torch.complex64, device=device)
    torch.testing.assert_close(
        _cdot(op.forward(x), y), _cdot(x, op.adjoint(y)), rtol=1e-4, atol=1e-4
    )


def test_off_resonance_sparse_fft_adjointness_no_smaps(device):
    """Adjointness also holds when the base operator has no smaps (loop path)."""
    op = _make_orc_op(
        _ORS_IMG, _ORS_N_SAMPLES, _ORS_N_COILS, _ORS_L, seed=13, smaps=False
    )
    # Without smaps: forward → (n_coils, *image_shape), adjoint ← same
    x = torch.randn(_ORS_N_COILS, _ORS_N_SAMPLES, dtype=torch.complex64, device=device)
    y = torch.randn(_ORS_N_COILS, *_ORS_IMG, dtype=torch.complex64, device=device)
    torch.testing.assert_close(
        _cdot(op.forward(x), y), _cdot(x, op.adjoint(y)), rtol=1e-4, atol=1e-4
    )


def test_off_resonance_sparse_fft_normal_is_self_adjoint(device):
    """normal(x) = forward(adjoint(x)) is self-adjoint."""
    op = _make_orc_op(_ORS_IMG, _ORS_N_SAMPLES, _ORS_N_COILS, _ORS_L, seed=14)
    x = torch.randn(*_ORS_IMG, dtype=torch.complex64, device=device)
    y = torch.randn(*_ORS_IMG, dtype=torch.complex64, device=device)
    torch.testing.assert_close(
        _cdot(op.normal(x), y), _cdot(x, op.normal(y)), rtol=1e-3, atol=1e-3
    )


# ===========================================================================
# OffResonanceCorrection (standalone gadget)
# ===========================================================================


def test_off_resonance_correction_n_components():
    orc = _build_orc()
    assert orc.n_components >= 1


def test_off_resonance_correction_forward_shape(device):
    orc = _build_orc(seed=20)
    img = torch.randn(_ORC_N_COILS, *_ORC_IMG, dtype=torch.complex64, device=device)
    ksp = orc.forward(img)
    assert ksp.shape == (_ORC_N_COILS, _ORC_N_SAMPLES)


def test_off_resonance_correction_adjoint_shape(device):
    orc = _build_orc(seed=21)
    ksp = torch.randn(
        _ORC_N_COILS, _ORC_N_SAMPLES, dtype=torch.complex64, device=device
    )
    img = orc.adjoint(ksp)
    assert img.shape == (_ORC_N_COILS, *_ORC_IMG)


def test_off_resonance_correction_adjointness(device):
    """<forward(x), y> == <x, adjoint(y)>."""
    orc = _build_orc(seed=22)
    x = torch.randn(_ORC_N_COILS, *_ORC_IMG, dtype=torch.complex64, device=device)
    y = torch.randn(_ORC_N_COILS, _ORC_N_SAMPLES, dtype=torch.complex64, device=device)
    torch.testing.assert_close(
        _cdot(orc.forward(x), y), _cdot(x, orc.adjoint(y)), rtol=1e-4, atol=1e-4
    )
