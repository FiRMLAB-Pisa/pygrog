"""
Tests for scatter_add and gather — the C++/CUDA binning primitives.

CPU path:  sorted-index clean-partition OMP scatter + auto-vectorised gather.
           SIMD level (SSE2 / AVX2 / AVX-512) is selected transparently at
           load time via target_clones (GCC/Clang) or CPUID dispatch (MSVC).

CUDA path: simple global-atomic scatter_add *and* shared-memory binned
           scatter_add_binned (called by SparseFFT._scatter when on GPU).

All public functions live in ``pygrog.operator``.
"""

import pytest
import torch


def _make_sorted(n_pts, grid_size, device):
    """Return (indices, weights, data) with indices sorted ascending."""
    idx = torch.randint(0, grid_size, (n_pts,), dtype=torch.int64, device=device)
    idx, _ = idx.sort()
    w = torch.rand(n_pts, dtype=torch.float32, device=device)
    data = torch.randn(n_pts, dtype=torch.complex64, device=device)
    return idx, w, data


# ---------------------------------------------------------------------------
# Import / availability
# ---------------------------------------------------------------------------


def test_scatter_add_importable():
    """scatter_add and gather must be importable from pygrog.operator."""
    from pygrog.operator import scatter_add, gather  # noqa: F401


def test_extension_loads():
    """The C++ extension (_pygrog_torch) must load without error."""
    from pygrog.operator._sparse_fft import _get_torch_ext

    ext = _get_torch_ext()
    assert hasattr(ext, "scatter_add")
    assert hasattr(ext, "gather")
    assert hasattr(ext, "scatter_add_binned")


# ---------------------------------------------------------------------------
# scatter_add correctness (CPU and CUDA)
# ---------------------------------------------------------------------------


def test_scatter_add_matches_torch_reference(device):
    """scatter_add must equal torch.scatter_add reference."""
    from pygrog.operator import scatter_add

    grid_size, n_pts = 64, 512
    idx, w, data = _make_sorted(n_pts, grid_size, device)

    grid = torch.zeros(grid_size, dtype=torch.complex64, device=device)
    scatter_add(grid, data, idx, w)

    # Reference via torch.scatter_add on real/imag separately
    weighted = data * w.to(torch.complex64)
    ref = torch.zeros(grid_size, dtype=torch.complex64, device=device)
    ref.scatter_add_(0, idx, weighted)

    torch.testing.assert_close(grid, ref, rtol=1e-4, atol=1e-4)


def test_scatter_add_accumulates_into_existing(device):
    """scatter_add must *add* to existing grid values (in-place +=)."""
    from pygrog.operator import scatter_add

    grid_size, n_pts = 32, 128
    idx, w, data = _make_sorted(n_pts, grid_size, device)

    grid = torch.ones(grid_size, dtype=torch.complex64, device=device)
    grid_ref = grid.clone()

    scatter_add(grid, data, idx, w)

    weighted = data * w.to(torch.complex64)
    grid_ref.scatter_add_(0, idx, weighted)

    torch.testing.assert_close(grid, grid_ref, rtol=1e-4, atol=1e-4)


def test_scatter_add_empty_data(device):
    """scatter_add with zero points must leave grid unchanged."""
    from pygrog.operator import scatter_add

    grid = torch.zeros(16, dtype=torch.complex64, device=device)
    idx = torch.empty(0, dtype=torch.int64, device=device)
    w = torch.empty(0, dtype=torch.float32, device=device)
    data = torch.empty(0, dtype=torch.complex64, device=device)

    scatter_add(grid, data, idx, w)
    assert grid.abs().max() == 0.0


def test_scatter_add_single_point(device):
    """Scatter a single weighted point to a known bin."""
    from pygrog.operator import scatter_add

    grid = torch.zeros(10, dtype=torch.complex64, device=device)
    idx = torch.tensor([3], dtype=torch.int64, device=device)
    w = torch.tensor([0.5], dtype=torch.float32, device=device)
    data = torch.tensor([1.0 + 2.0j], dtype=torch.complex64, device=device)

    scatter_add(grid, data, idx, w)

    assert abs(grid[3].item() - (0.5 + 1.0j)) < 1e-5
    assert grid[:3].abs().max() == 0.0
    assert grid[4:].abs().max() == 0.0


def test_scatter_add_all_same_bin(device):
    """All points into one bin — result equals sum of weighted values."""
    from pygrog.operator import scatter_add

    n_pts, grid_size = 256, 16
    target = 7
    idx = torch.full((n_pts,), target, dtype=torch.int64, device=device)
    w = torch.ones(n_pts, dtype=torch.float32, device=device)
    data = torch.ones(n_pts, dtype=torch.complex64, device=device)

    grid = torch.zeros(grid_size, dtype=torch.complex64, device=device)
    scatter_add(grid, data, idx, w)

    assert abs(grid[target].real.item() - n_pts) < 1e-3
    assert grid[target].imag.abs().item() < 1e-5


# ---------------------------------------------------------------------------
# gather correctness (CPU and CUDA)
# ---------------------------------------------------------------------------


def test_gather_matches_index_select(device):
    """gather must equal weights * grid[indices]."""
    from pygrog.operator import gather

    grid_size, n_pts = 64, 256
    idx, w, _ = _make_sorted(n_pts, grid_size, device)
    grid = torch.randn(grid_size, dtype=torch.complex64, device=device)

    out = gather(grid, idx, w)

    ref = grid[idx] * w.to(torch.complex64)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_gather_empty(device):
    """gather with zero points must return empty tensor."""
    from pygrog.operator import gather

    grid = torch.randn(16, dtype=torch.complex64, device=device)
    idx = torch.empty(0, dtype=torch.int64, device=device)
    w = torch.empty(0, dtype=torch.float32, device=device)

    out = gather(grid, idx, w)
    assert out.numel() == 0


def test_gather_zero_weight(device):
    """Zero weights must produce zero output regardless of grid values."""
    from pygrog.operator import gather

    grid = torch.ones(8, dtype=torch.complex64, device=device) * 99.0
    idx = torch.arange(8, dtype=torch.int64, device=device)
    w = torch.zeros(8, dtype=torch.float32, device=device)

    out = gather(grid, idx, w)
    assert out.abs().max() < 1e-6


# ---------------------------------------------------------------------------
# scatter → gather round-trip
# ---------------------------------------------------------------------------


def test_scatter_gather_round_trip_identity(device):
    """scatter then gather with weight=1 recovers weighted sum correctly."""
    from pygrog.operator import scatter_add, gather

    grid_size = 32
    n_pts = grid_size  # one point per bin, sorted
    idx = torch.arange(n_pts, dtype=torch.int64, device=device)
    w = torch.ones(n_pts, dtype=torch.float32, device=device)
    data = torch.randn(n_pts, dtype=torch.complex64, device=device)

    grid = torch.zeros(grid_size, dtype=torch.complex64, device=device)
    scatter_add(grid, data, idx, w)

    # With unit weights and one-point-per-bin, gather == original data
    out = gather(grid, idx, w)
    torch.testing.assert_close(out, data, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Binned CUDA scatter (SparseFFT._scatter with bin_starts)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_binned_scatter_matches_simple():
    """Shared-memory binned scatter must equal the simple scatter."""
    from pygrog.operator._sparse_fft import _scatter_add

    device = torch.device("cuda")
    grid_size = 1024
    n_pts = 4096
    bin_size = 256

    idx, w, data = _make_sorted(n_pts, grid_size, device)

    # Simple scatter
    grid_simple = torch.zeros(grid_size, dtype=torch.complex64, device=device)
    _scatter_add(grid_simple, data, idx, w)

    # Binned scatter — build bin_starts
    n_bins = (grid_size + bin_size - 1) // bin_size
    bin_edges = torch.arange(n_bins + 1, dtype=torch.int64, device=device) * bin_size
    bin_edges[-1] = grid_size
    bin_starts = torch.searchsorted(idx, bin_edges)

    grid_binned = torch.zeros(grid_size, dtype=torch.complex64, device=device)
    _scatter_add(grid_binned, data, idx, w, bin_starts=bin_starts, bin_size=bin_size)

    torch.testing.assert_close(grid_simple, grid_binned, rtol=1e-4, atol=1e-4)
