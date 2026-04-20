"""Sparse FFT operator for GROG-gridded data.

Provides forward (sparse-to-dense / adjoint NUFFT) and adjoint (dense-to-sparse
/ forward NUFFT) transforms using pre-computed GROG plan metadata (indices,
weights, grid/image shapes).

Coil combination is performed inside the operator:
- If sensitivity maps are provided, SENSE-style combination (forward) or
  expansion (adjoint) is used.
- Otherwise, root-sum-of-squares combination is applied (forward only;
  adjoint assumes single-channel input).

Both paths process data **coil-by-coil** to limit peak memory.

Weights follow the convention that GROG-interpolated data are pre-multiplied
by ``weights**0.5`` once right after interpolation, and the same ``weights**0.5``
is applied inside both forward and adjoint for orthonormality.

Optional dual-stream GPU pipelining: when ``device`` is set and differs from the
data device, each coil is transferred asynchronously on alternating CUDA streams
while the previous coil's FFT executes concurrently.
"""

__all__ = ["SparseFFT"]

import pathlib

import numpy as np
import torch

from .._utils import resize
from .._base._fftc import fft, ifft


# ---------------------------------------------------------------------------
# Torch C++ extension (lazy, cached)
# ---------------------------------------------------------------------------
_torch_ext = None
_torch_ext_checked = False


def _get_torch_ext():
    """Return the ``_pygrog_torch`` extension module.

    Load priority:
      1. Pre-compiled wheel extension (``pygrog._pygrog_torch``)
      2. JIT compilation via ``torch.utils.cpp_extension.load()``

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
            extra_cflags=["-O3", "-std=c++17", "-fopenmp", "-march=native",
                          "-DPYGROG_MARCH_NATIVE"],
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


# ---------------------------------------------------------------------------
# scatter_add / gather wrappers
# ---------------------------------------------------------------------------
def _scatter_add(grid, data, indices, weights, bin_starts=None, bin_size=0):
    """grid[indices[i]] += weights[i] * data[i], in-place."""
    ext = _get_torch_ext()
    if bin_starts is not None:
        ext.scatter_add_binned(grid, data, indices, weights,
                               bin_starts, bin_size)
    else:
        ext.scatter_add(grid, data, indices, weights)


def _gather(grid, indices, weights):
    """out[i] = weights[i] * grid[indices[i]]."""
    ext = _get_torch_ext()
    return ext.gather(grid, indices, weights)


# =====================================================================
# SparseFFT
# =====================================================================
class SparseFFT:
    """Sparse FFT / IFFT operator with coil combination.

    Accepts either a pre-built plan (from
    :meth:`~pygrog.grog.GrogInterpolator.fft_plan`) or raw arrays.
    When a plan is provided, sorted indices, sqrt-weights, and
    permutation arrays are reused directly; otherwise they are computed
    from the raw ``indices`` / ``weights`` arguments.

    Parameters
    ----------
    plan : SimpleNamespace | None
        Pre-built plan from ``GrogInterpolator.fft_plan()``.  If given,
        *grid_shape*, *image_shape*, *indices*, and *weights* are ignored.
    grid_shape : tuple[int, ...] | None
        Oversampled Cartesian k-space grid, e.g. ``(nz, ny, nx)``.
    image_shape : tuple[int, ...] | None
        Target image shape (center-crop), e.g. ``(nz, ny, nx)``.
    indices : array-like | None
        Flat grid indices ``(n_samples,)`` int64.
    weights : array-like | None
        Density-compensation weights ``(n_samples,)`` float32.
        ``sqrt(weights)`` is applied in both directions.
    smaps : torch.Tensor | None
        ``(n_coils, *image_shape)`` sensitivity maps.  *None* → RSS.
    device : str | torch.device | None
        Compute device.  When ``'cuda'`` with CPU data, dual-stream
        pipelining is enabled.
    """

    def __init__(
        self,
        plan=None,
        grid_shape=None,
        image_shape=None,
        indices=None,
        weights=None,
        smaps=None,
        device=None,
    ):
        # --- Accept plan or raw arguments ----------------------------------
        if plan is not None:
            self.grid_shape = tuple(plan.grid_shape)
            self.image_shape = tuple(plan.image_shape)
            self.grid_size = int(plan.grid_size)
            self.indices = plan.indices
            self.sqrt_weights = plan.sqrt_weights
            self.sort_perm = plan.sort_perm
            self.inv_perm = plan.inv_perm
        else:
            if grid_shape is None or image_shape is None:
                raise ValueError(
                    "Either 'plan' or both 'grid_shape'/'image_shape' required"
                )
            self.grid_shape = tuple(grid_shape)
            self.image_shape = tuple(image_shape)
            self.grid_size = int(np.prod(grid_shape))

            idx = torch.as_tensor(indices).ravel().to(torch.int64)
            w = torch.as_tensor(weights).ravel().to(torch.float32)
            sqrt_w = torch.sqrt(w)

            sort_perm = torch.argsort(idx)
            self.sort_perm = sort_perm
            self.indices = idx[sort_perm]
            self.sqrt_weights = sqrt_w[sort_perm]

            inv_perm = torch.empty_like(sort_perm)
            inv_perm[sort_perm] = torch.arange(len(sort_perm))
            self.inv_perm = inv_perm

        self.ndim = len(self.grid_shape)
        self.fft_axes = tuple(range(-self.ndim, 0))

        # GPU binning — computed lazily on first use
        self._bin_starts = None
        self._bin_size = 0

        # Sensitivity maps — pre-compute conjugate (view, free)
        if smaps is not None:
            self.smaps = torch.as_tensor(smaps)
            self._conj_smaps = self.smaps.conj()
        else:
            self.smaps = None
            self._conj_smaps = None

        # Pre-compute center-slice for adjoint zero-pad
        self._pad_slices = tuple(
            slice((gs - is_) // 2, (gs - is_) // 2 + is_)
            for gs, is_ in zip(self.grid_shape, self.image_shape)
        )

        self.device = torch.device(device) if device is not None else None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_bins(self, device):
        """Compute GPU bins on first use for the given device."""
        if device.type != "cuda" or self._bin_starts is not None:
            return
        self._bin_size = 256
        n_bins = (self.grid_size + self._bin_size - 1) // self._bin_size
        bin_edges = torch.arange(n_bins + 1, dtype=torch.int64) * self._bin_size
        bin_edges[-1] = self.grid_size
        sorted_idx = self.indices.to(device)
        self._bin_starts = torch.searchsorted(sorted_idx, bin_edges.to(device))

    def _scatter(self, grid, data, indices, sqrt_w):
        """Weighted scatter-add using binned kernel when available."""
        _scatter_add(grid, data, indices, sqrt_w,
                     bin_starts=self._bin_starts, bin_size=self._bin_size)

    # ------------------------------------------------------------------
    # Forward: sparse k-space -> image  (adjoint NUFFT direction)
    # ------------------------------------------------------------------
    def forward(self, sparse_kspace: torch.Tensor) -> torch.Tensor:
        """Sparse k-space to image.

        Parameters
        ----------
        sparse_kspace : torch.Tensor
            ``(n_coils, n_samples)`` complex.

        Returns
        -------
        torch.Tensor
            ``(*image_shape,)`` combined image (complex).
        """
        n_coils = sparse_kspace.shape[0]
        src_device = sparse_kspace.device
        comp_device = self.device if self.device is not None else src_device
        use_pipeline = (comp_device.type == "cuda" and src_device.type == "cpu")

        dtype = sparse_kspace.dtype
        indices = self.indices.to(comp_device)
        sqrt_w = self.sqrt_weights.to(comp_device)
        sort_perm = self.sort_perm.to(comp_device)
        self._ensure_bins(comp_device)

        # Pre-allocate reusable grid buffer (one alloc, reused per coil)
        grid = torch.empty(self.grid_size, dtype=dtype, device=comp_device)

        if self.smaps is not None:
            conj_smaps = self._conj_smaps.to(comp_device, dtype=dtype)
            accum = torch.zeros(self.image_shape, dtype=dtype, device=comp_device)
        else:
            accum = torch.zeros(self.image_shape, dtype=torch.float32, device=comp_device)

        if use_pipeline:
            self._forward_pipeline(
                sparse_kspace, indices, sqrt_w, sort_perm, grid,
                conj_smaps if self.smaps is not None else None, accum, dtype,
            )
        else:
            for c in range(n_coils):
                coil_data = sparse_kspace[c].to(comp_device)[sort_perm]
                img_c = self._scatter_ifft_crop(coil_data, indices, sqrt_w, grid, dtype)
                if self.smaps is not None:
                    # Fused multiply-accumulate: accum += img_c * conj_smaps[c]
                    accum.addcmul_(img_c, conj_smaps[c])
                else:
                    accum += img_c.abs().square()

        if self.smaps is None:
            accum = accum.sqrt().to(dtype)

        return accum.to(src_device)

    # ------------------------------------------------------------------
    # Adjoint: image -> sparse k-space  (forward NUFFT direction)
    # ------------------------------------------------------------------
    def adjoint(self, image: torch.Tensor) -> torch.Tensor:
        """Image to sparse k-space.

        Parameters
        ----------
        image : torch.Tensor
            ``(*image_shape,)`` complex.

        Returns
        -------
        torch.Tensor
            ``(n_coils, n_samples)`` complex.
        """
        src_device = image.device
        comp_device = self.device if self.device is not None else src_device
        use_pipeline = (comp_device.type == "cuda" and src_device.type == "cpu")

        dtype = image.dtype
        indices = self.indices.to(comp_device)
        sqrt_w = self.sqrt_weights.to(comp_device)
        inv_perm = self.inv_perm.to(comp_device)
        image_d = image.to(comp_device)

        if self.smaps is not None:
            smaps = self.smaps.to(comp_device, dtype=dtype)
            n_coils = smaps.shape[0]
        else:
            n_coils = 1

        n_samples = indices.shape[0]
        output = torch.zeros(n_coils, n_samples, dtype=dtype, device=comp_device)

        # Pre-allocate reusable padded grid buffer
        padded = torch.empty(*self.grid_shape, dtype=dtype, device=comp_device)

        if use_pipeline and self.smaps is not None:
            self._adjoint_pipeline(
                image_d, indices, sqrt_w, inv_perm, smaps, padded, output, dtype,
            )
        else:
            for c in range(n_coils):
                coil_img = image_d * smaps[c] if self.smaps is not None else image_d
                output[c] = self._fft_pad_gather(
                    coil_img, indices, sqrt_w, inv_perm, padded, dtype,
                )

        return output.to(src_device)

    def __call__(self, x, adjoint=False):
        if adjoint:
            return self.adjoint(x)
        return self.forward(x)

    # ------------------------------------------------------------------
    # Core single-coil ops (fused)
    # ------------------------------------------------------------------
    def _scatter_ifft_crop(self, coil_data, indices, sqrt_w, grid, dtype):
        """Scatter -> IFFT -> crop for one coil.  Reuses *grid* buffer."""
        grid.zero_()
        self._scatter(grid, coil_data, indices, sqrt_w)
        return ifft(grid.reshape(self.grid_shape),
                    oshape=self.image_shape, axes=self.fft_axes)

    def _fft_pad_gather(self, coil_img, indices, sqrt_w, inv_perm, padded, dtype):
        """FFT -> zero-pad -> gather -> unpermute for one coil.

        Reuses the *padded* buffer to avoid a fresh allocation per coil.
        """
        fft_result = fft(coil_img, axes=self.fft_axes)
        # In-place zero-pad: write FFT result into centre of pre-zeroed grid
        padded.zero_()
        padded[self._pad_slices] = fft_result
        return _gather(padded.reshape(-1), indices, sqrt_w)[inv_perm]

    # ------------------------------------------------------------------
    # Dual-stream GPU pipelining  (CPU input -> GPU compute)
    # ------------------------------------------------------------------
    def _forward_pipeline(self, sparse_kspace, indices, sqrt_w, sort_perm,
                          grid, conj_smaps, accum, dtype):
        """Dual-stream forward: overlap H2D transfer with FFT."""
        n_coils = sparse_kspace.shape[0]
        device = accum.device

        s1 = torch.cuda.Stream(device=device)
        s2 = torch.cuda.Stream(device=device)

        buf = [
            sparse_kspace[0].pin_memory().to(device, non_blocking=True),
            None,
        ]

        for c in range(n_coils):
            cur = c % 2
            nxt = 1 - cur
            stream = s1 if cur == 0 else s2

            if c + 1 < n_coils:
                other = s1 if nxt == 0 else s2
                with torch.cuda.stream(other):
                    buf[nxt] = sparse_kspace[c + 1].pin_memory().to(
                        device, non_blocking=True)

            with torch.cuda.stream(stream):
                coil_data = buf[cur][sort_perm]
                img_c = self._scatter_ifft_crop(coil_data, indices, sqrt_w, grid, dtype)
                if conj_smaps is not None:
                    accum.addcmul_(img_c, conj_smaps[c])
                else:
                    accum += img_c.abs().square()

        torch.cuda.synchronize(device)

    def _adjoint_pipeline(self, image_d, indices, sqrt_w, inv_perm,
                          smaps, padded, output, dtype):
        """Dual-stream adjoint: overlap FFT with D2H transfer."""
        n_coils = smaps.shape[0]
        device = image_d.device

        s1 = torch.cuda.Stream(device=device)
        s2 = torch.cuda.Stream(device=device)

        for c in range(n_coils):
            stream = s1 if c % 2 == 0 else s2
            with torch.cuda.stream(stream):
                coil_img = image_d * smaps[c]
                output[c] = self._fft_pad_gather(
                    coil_img, indices, sqrt_w, inv_perm, padded, dtype,
                )

        torch.cuda.synchronize(device)
