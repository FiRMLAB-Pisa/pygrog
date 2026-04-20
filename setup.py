"""
setup.py — Build the _pygrog_torch C++/CUDA extension as part of
``pip install .`` or ``python setup.py build_ext --inplace``.

The extension is **required**: the build will fail if a C++17 compiler
or torch is not available.  Precompiled wheels are provided for all
major platforms so end-users typically do not need a local compiler.

CUDA support is auto-detected via ``nvcc`` availability.  Set the
environment variable ``PYGROG_FORCE_CUDA=1`` to require it, or
``PYGROG_NO_CUDA=1`` to skip it.
"""

import os
import subprocess
import sys
from pathlib import Path

from setuptools import setup

from torch.utils.cpp_extension import BuildExtension


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _nvcc_available() -> bool:
    """Return True if nvcc is on PATH and functional."""
    try:
        r = subprocess.run(
            ["nvcc", "--version"], capture_output=True, timeout=10
        )
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _has_cuda() -> bool:
    """Decide whether to build CUDA sources."""
    if os.environ.get("PYGROG_NO_CUDA", ""):
        return False
    if os.environ.get("PYGROG_FORCE_CUDA", ""):
        return True
    try:
        import torch
        return torch.cuda.is_available() and _nvcc_available()
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Extension definition
# ---------------------------------------------------------------------------
HERE = Path(__file__).parent
CSRC = Path("csrc") / "torch"

sources_cpp = [
    str(CSRC / "module.cpp"),
    str(CSRC / "grog_interp.cpp"),
    str(CSRC / "sparse_ops.cpp"),
    str(CSRC / "sparse_ops_avx2.cpp"),
    str(CSRC / "sparse_ops_avx512.cpp"),
]

sources_cuda = [
    str(CSRC / "grog_interp_cuda.cu"),
    str(CSRC / "sparse_ops_cuda.cu"),
]


class TorchBuildExt(BuildExtension):
    """BuildExtension with per-file MSVC SIMD flags."""

    def build_extensions(self):
        # MSVC per-file SIMD flags
        if sys.platform == "win32" and self.compiler is not None:
            _original_compile = self.compiler._compile

            def _compile_simd(obj, src, ext, cc_args, extra_postargs, pp_opts):
                extra = list(extra_postargs)
                if "sparse_ops_avx2" in src:
                    extra.append("/arch:AVX2")
                elif "sparse_ops_avx512" in src:
                    extra.append("/arch:AVX512")
                return _original_compile(obj, src, ext, cc_args, extra, pp_opts)

            self.compiler._compile = _compile_simd

        super().build_extensions()


def _make_extension():
    """Create the CppExtension or CUDAExtension for _pygrog_torch."""
    from torch.utils.cpp_extension import CppExtension, CUDAExtension

    use_cuda = _has_cuda()

    if use_cuda:
        return CUDAExtension(
            name="pygrog._pygrog_torch",
            sources=sources_cpp + sources_cuda,
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17", "-fopenmp"]
                if sys.platform != "win32"
                else ["/O2", "/std:c++17", "/openmp"],
                "nvcc": [
                    "-O3",
                    "--expt-relaxed-constexpr",
                    "-std=c++17",
                    "-gencode=arch=compute_70,code=sm_70",
                    "-gencode=arch=compute_80,code=sm_80",
                    "-gencode=arch=compute_86,code=sm_86",
                    "-gencode=arch=compute_89,code=sm_89",
                    "-gencode=arch=compute_90,code=sm_90",
                ],
            },
            extra_link_args=["-fopenmp"] if sys.platform != "win32" else [],
            define_macros=[("COMPILE_WITH_CUDA", "1")],
        )
    else:
        cxx_flags = (
            ["-O3", "-std=c++17", "-fopenmp"]
            if sys.platform != "win32"
            else ["/O2", "/std:c++17", "/openmp"]
        )
        link_flags = ["-fopenmp"] if sys.platform != "win32" else []
        return CppExtension(
            name="pygrog._pygrog_torch",
            sources=sources_cpp,
            extra_compile_args={"cxx": cxx_flags},
            extra_link_args=link_flags,
        )


setup(
    ext_modules=[_make_extension()],
    cmdclass={"build_ext": TorchBuildExt},
)