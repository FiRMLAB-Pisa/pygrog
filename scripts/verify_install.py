#!/usr/bin/env python3
"""
verify_install.py — Verify a pygrog installation is functional.

Checks:
  1. Package import and version
  2. C++ extension (_pygrog_torch) availability
  3. CUDA availability (optional, reported only)
  4. A short end-to-end smoke test (scatter/gather round-trip on CPU)

Exit codes:
  0 — all required checks passed
  1 — one or more required checks failed
"""

import sys

_PASS = "\033[1;32m[PASS]\033[0m"  # noqa: S105
_FAIL = "\033[1;31m[FAIL]\033[0m"
_INFO = "\033[1;34m[INFO]\033[0m"
_WARN = "\033[1;33m[WARN]\033[0m"

errors = []


def check(label, fn):
    """Run *fn*, print result, collect failures."""
    try:
        result = fn()
        msg = f"  {result}" if result else ""
        print(f"{_PASS} {label}{msg}")
        return True
    except Exception as exc:
        errors.append(label)
        print(f"{_FAIL} {label}: {exc}")
        return False


# ── 1. Package import ────────────────────────────────────────────────────────
def _import_pygrog():
    import pygrog

    return f"version {pygrog.__version__}"


check("Import pygrog", _import_pygrog)


# ── 2. PyTorch ───────────────────────────────────────────────────────────────
def _import_torch():
    import torch

    return f"version {torch.__version__}"


check("Import torch", _import_torch)


# ── 3. C++ extension ─────────────────────────────────────────────────────────
def _ext_loaded():
    import pygrog._pygrog_torch  # noqa: F401

    return "pre-built wheel extension"


if not check("C++ extension (_pygrog_torch)", _ext_loaded):
    # Warn but don't give up — the JIT path may still work at runtime.
    print(f"{_WARN} The pre-built extension was not found.")
    print(f"{_WARN} pygrog will attempt JIT compilation at first use.")
    print(f"{_WARN} Run the smoke test below to confirm end-to-end functionality.")
    errors.pop()  # remove from hard failures

# ── 4. CUDA (informational) ──────────────────────────────────────────────────
try:
    import torch

    if torch.cuda.is_available():
        dev = torch.cuda.get_device_name(0)
        print(f"{_INFO} CUDA available: {dev}")
    else:
        print(f"{_INFO} CUDA not available (CPU-only mode)")
except Exception as exc:
    print(f"{_WARN} Could not query CUDA status: {exc}")


# ── 5. Smoke test: end-to-end scatter/gather on CPU ─────────────────────────
def _smoke_test():
    """Small scatter-add → gather round-trip through the C++ extension."""
    import torch
    from pygrog.operator import SparseFFT  # noqa: F401 — exercises the ext load path

    # Minimal scatter_add round-trip via the extension
    import pygrog._pygrog_torch as _ext

    grid = torch.zeros(16, dtype=torch.complex64)  # 1-D complex grid
    data = torch.ones(4, dtype=torch.complex64)  # 4 source points
    indices = torch.tensor([0, 3, 7, 12], dtype=torch.int64)
    weights = torch.ones(4)

    _ext.scatter_add(grid, data, indices, weights)

    assert grid[0].real.item() > 0, "scatter_add produced no output"

    gathered = _ext.gather(grid, indices, weights)
    assert gathered.shape == (4,), f"gather shape mismatch: {gathered.shape}"
    return "scatter_add + gather OK"


check("Smoke test (CPU scatter/gather)", _smoke_test)


# ── 6. CUDA smoke test (skipped if not available) ────────────────────────────
def _cuda_smoke_test():
    import torch
    import pygrog._pygrog_torch as _ext

    grid = torch.zeros(16, dtype=torch.complex64, device="cuda")
    data = torch.ones(4, dtype=torch.complex64, device="cuda")
    indices = torch.tensor([0, 3, 7, 12], dtype=torch.int64, device="cuda")
    weights = torch.ones(4, device="cuda")

    _ext.scatter_add(grid, data, indices, weights)
    assert grid[0].real.item() > 0
    return "CUDA scatter_add OK"


try:
    import torch

    if torch.cuda.is_available():
        check("Smoke test (CUDA scatter/gather)", _cuda_smoke_test)
    else:
        print(f"{_INFO} Skipping CUDA smoke test (no GPU)")
except ImportError:
    pass

# ── Summary ──────────────────────────────────────────────────────────────────
print()
if errors:
    print(f"{_FAIL} {len(errors)} check(s) failed: {', '.join(errors)}")
    sys.exit(1)
else:
    print(f"{_PASS} All checks passed — pygrog is ready to use.")
    sys.exit(0)
