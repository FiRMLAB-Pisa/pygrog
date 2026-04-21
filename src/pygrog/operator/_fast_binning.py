"""Fast scatter-add and gather operations for sparse FFT.

Thin wrappers around the ``_pygrog_torch`` C++ extension (CPU + CUDA).
"""

__all__ = ["scatter_add", "gather"]

import pathlib

import torch

# ---------------------------------------------------------------------------
# Torch C++ extension (lazy, cached)
# ---------------------------------------------------------------------------
_torch_ext = None
_torch_ext_checked = False


def _get_torch_ext():
    """Return the ``_pygrog_torch`` extension module.

    Raises
    ------
    RuntimeError
        If neither the pre-built extension nor JIT compilation succeeds.
    """
    global _torch_ext, _torch_ext_checked
    if _torch_ext_checked:
        return _torch_ext

    # 1. try pre-built wheel extension
    try:
        import pygrog._pygrog_torch as _ext

        _torch_ext = _ext
        _torch_ext_checked = True
        return _torch_ext
    except ImportError:
        pass

    # 2. JIT compilation
    jit_error = None
    try:
        import torch.utils.cpp_extension as _cpp_ext

        _HERE = pathlib.Path(__file__).parent.parent.parent.parent.parent
        _CSRC = _HERE / "csrc" / "torch"

        sources = [
            str(_CSRC / "module.cpp"),
            str(_CSRC / "grog_interp.cpp"),
            str(_CSRC / "sparse_ops.cpp"),
            str(_CSRC / "sparse_ops_avx2.cpp"),
            str(_CSRC / "sparse_ops_avx512.cpp"),
        ]
        define_macros = []
        if torch.cuda.is_available() and (_CSRC / "sparse_ops_cuda.cu").exists():
            sources.append(str(_CSRC / "grog_interp_cuda.cu"))
            sources.append(str(_CSRC / "sparse_ops_cuda.cu"))
            define_macros.append(("COMPILE_WITH_CUDA", "1"))

        _torch_ext = _cpp_ext.load(
            name="_pygrog_torch_jit",
            sources=sources,
            extra_cflags=[
                "-O3",
                "-std=c++17",
                "-fopenmp",
                "-march=native",
                "-DPYGROG_MARCH_NATIVE",
            ],
            extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"],
            extra_ldflags=["-fopenmp"],
            define_macros=define_macros,
            verbose=False,
        )
        _torch_ext_checked = True
        return _torch_ext
    except Exception as exc:
        jit_error = exc

    raise RuntimeError(
        "pygrog requires the _pygrog_torch C++ extension but it could not "
        "be loaded.  Install from a precompiled wheel (`pip install pygrog`) "
        "or build from source with a C++17 compiler "
        "(`pip install --no-build-isolation -e .`).\n"
        f"JIT compilation error: {jit_error}"
    )


def scatter_add(
    grid: torch.Tensor,
    data: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
) -> None:
    """Scatter-add: ``grid[indices[i]] += weights[i] * data[i]`` (in-place).

    Parameters
    ----------
    grid : torch.Tensor
        Flat output grid (complex), modified in-place.
    data : torch.Tensor
        Input data values (complex), 1-D.
    indices : torch.Tensor
        Target grid indices (int64), 1-D.
    weights : torch.Tensor
        Per-sample real weights (float), 1-D.

    Notes
    -----
    For optimal CPU performance, pass *sorted* indices (ascending) so
    the C++ clean-partition kernel can avoid synchronisation.
    GPU binned scatter is available internally via
    :class:`~pygrog.operator.SparseFFT` which auto-computes bins from
    the GROG plan.
    """
    ext = _get_torch_ext()
    ext.scatter_add(grid, data, indices, weights)


def gather(
    grid: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Gather: ``out[i] = weights[i] * grid[indices[i]]``.

    Parameters
    ----------
    grid : torch.Tensor
        Flat input grid (complex), 1-D.
    indices : torch.Tensor
        Source indices (int64), 1-D.
    weights : torch.Tensor
        Per-sample real weights (float), 1-D.

    Returns
    -------
    torch.Tensor
        Gathered values (complex), 1-D.
    """
    ext = _get_torch_ext()
    return ext.gather(grid, indices, weights)
