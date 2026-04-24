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

__all__ = ["SparseFFT", "gather", "scatter_add"]

import pathlib

import numpy as np
import torch
from mrinufft._array_compat import with_torch

from .._base._fftc import fft, ifft
from .._utils import resize

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


# ---------------------------------------------------------------------------
# scatter_add / gather wrappers
# ---------------------------------------------------------------------------
def _scatter_add(grid, data, indices, weights, bin_starts=None, bin_size=0):
    """grid[indices[i]] += weights[i] * data[i], in-place."""
    ext = _get_torch_ext()
    if bin_starts is not None:
        ext.scatter_add_binned(grid, data, indices, weights, bin_starts, bin_size)
    else:
        ext.scatter_add(grid, data, indices, weights)


def _gather(grid, indices, weights):
    """out[i] = weights[i] * grid[indices[i]]."""
    ext = _get_torch_ext()
    return ext.gather(grid, indices, weights)


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
    """
    _scatter_add(grid, data, indices, weights)


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
    return _gather(grid, indices, weights)


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
        grid_shape=None,
        image_shape=None,
        indices=None,
        weights=None,
        smaps=None,
        device=None,
        *,
        plan=None,
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
            for gs, is_ in zip(self.grid_shape, self.image_shape, strict=False)
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
        _scatter_add(
            grid,
            data,
            indices,
            sqrt_w,
            bin_starts=self._bin_starts,
            bin_size=self._bin_size,
        )

    # ------------------------------------------------------------------
    # Forward: sparse k-space -> image  (adjoint NUFFT direction)
    # ------------------------------------------------------------------
    @with_torch
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
        use_pipeline = comp_device.type == "cuda" and src_device.type == "cpu"

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
            accum = torch.zeros(
                (n_coils, *self.image_shape), dtype=dtype, device=comp_device
            )

        if use_pipeline:
            self._forward_pipeline(
                sparse_kspace,
                indices,
                sqrt_w,
                sort_perm,
                grid,
                conj_smaps if self.smaps is not None else None,
                accum,
                dtype,
            )
        else:
            for c in range(n_coils):
                coil_data = sparse_kspace[c].to(comp_device)[sort_perm]
                img_c = self._scatter_ifft_crop(coil_data, indices, sqrt_w, grid, dtype)
                if self.smaps is not None:
                    # Fused multiply-accumulate: accum += img_c * conj_smaps[c]
                    accum.addcmul_(img_c, conj_smaps[c])
                else:
                    accum[c] = img_c

        return accum.to(src_device)

    # ------------------------------------------------------------------
    # Adjoint: image -> sparse k-space  (forward NUFFT direction)
    # ------------------------------------------------------------------
    @with_torch
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
        use_pipeline = comp_device.type == "cuda" and src_device.type == "cpu"

        dtype = image.dtype
        indices = self.indices.to(comp_device)
        sqrt_w = self.sqrt_weights.to(comp_device)
        inv_perm = self.inv_perm.to(comp_device)
        image_d = image.to(comp_device)

        if self.smaps is not None:
            smaps = self.smaps.to(comp_device, dtype=dtype)
            n_coils = smaps.shape[0]
        else:
            n_coils = image_d.shape[0]

        n_samples = indices.shape[0]
        output = torch.zeros(n_coils, n_samples, dtype=dtype, device=comp_device)

        # Pre-allocate reusable padded grid buffer
        padded = torch.empty(*self.grid_shape, dtype=dtype, device=comp_device)

        if use_pipeline and self.smaps is not None:
            self._adjoint_pipeline(
                image_d,
                indices,
                sqrt_w,
                inv_perm,
                smaps,
                padded,
                output,
                dtype,
            )
        else:
            for c in range(n_coils):
                coil_img = image_d * smaps[c] if self.smaps is not None else image_d[c]
                output[c] = self._fft_pad_gather(
                    coil_img,
                    indices,
                    sqrt_w,
                    inv_perm,
                    padded,
                    dtype,
                )

        return output.to(src_device)

    def __call__(self, x, adjoint=False):
        if adjoint:
            return self.adjoint(x)
        return self.forward(x)

    # ------------------------------------------------------------------
    # Batch helpers for decorators  (Tier-1 FFT fusion)
    # ------------------------------------------------------------------
    def _scatter_ifft_crop_batch(self, batch_kspace: torch.Tensor) -> torch.Tensor:
        """Scatter (per-component loop) → **one batched IFFT** → crop.

        Reduces ``B x n_coils`` IFFT calls to a single
        ``torch.fft.ifftn((B, *grid_shape))`` call.  The scatter loop is
        unchanged (needs a batched C++ kernel for full fusion).

        Parameters
        ----------
        batch_kspace : torch.Tensor
            ``(B, n_samples)`` complex, **unsorted**.  Each row is an
            independently-weighted k-space vector (e.g. one ORC component
            x one coil, or one subspace frame x one coil).

        Returns
        -------
        torch.Tensor
            ``(B, *image_shape)`` complex, on the same device as input.
        """
        B = batch_kspace.shape[0]
        src_device = batch_kspace.device
        comp_device = self.device if self.device is not None else src_device
        dtype = batch_kspace.dtype

        indices = self.indices.to(comp_device)
        sqrt_w = self.sqrt_weights.to(comp_device)
        sort_perm = self.sort_perm.to(comp_device)
        self._ensure_bins(comp_device)

        # Sort all B inputs at once: (B, n_samples) — one indexing op
        sorted_ksp = batch_kspace.to(comp_device)[:, sort_perm]

        # Scatter B times — C++ kernel, bottleneck until batched kernel exists
        grids = torch.zeros(B, self.grid_size, dtype=dtype, device=comp_device)
        for b in range(B):
            self._scatter(grids[b], sorted_ksp[b], indices, sqrt_w)

        # ONE batched IFFT + center-crop over last ``ndim`` dims
        # axes=(-k,...,-1) work correctly on any batch prefix
        grids_nd = grids.reshape(B, *self.grid_shape)
        # Full-size IFFT first, then center-crop image; cropping in k-space
        # (via oshape) would be wrong when the grid is oversampled.
        full_imgs = ifft(grids_nd, axes=self.fft_axes)  # (B, *grid_shape)
        imgs = resize(full_imgs, (B, *self.image_shape))
        return imgs.to(src_device)

    def _fft_pad_gather_batch(self, batch_imgs: torch.Tensor) -> torch.Tensor:
        """ONE batched FFT -> zero-pad -> gather (per-component loop).

        Reduces ``B x n_coils`` FFT calls to a single
        ``torch.fft.fftn((B, *image_shape))`` call.  The gather loop is
        unchanged (needs a batched C++ kernel for full fusion).

        Parameters
        ----------
        batch_imgs : torch.Tensor
            ``(B, *image_shape)`` complex.  Each slice ``[b]`` is an
            independently-weighted image (e.g. one ORC component x one coil).

        Returns
        -------
        torch.Tensor
            ``(B, n_samples)`` complex, in original (unsorted) k-space order.
        """
        B = batch_imgs.shape[0]
        src_device = batch_imgs.device
        comp_device = self.device if self.device is not None else src_device
        dtype = batch_imgs.dtype

        indices = self.indices.to(comp_device)
        sqrt_w = self.sqrt_weights.to(comp_device)
        inv_perm = self.inv_perm.to(comp_device)
        imgs_d = batch_imgs.to(comp_device)

        # Zero-pad images in image space (adjoint of center-crop in image space)
        padded = torch.zeros(B, *self.grid_shape, dtype=dtype, device=comp_device)
        padded[(slice(None), *self._pad_slices)] = imgs_d
        # ONE batched FFT at full grid_shape
        fft_results = fft(padded, axes=self.fft_axes)  # (B, *grid_shape)
        padded_flat = fft_results.reshape(B, -1)

        # Gather B times — C++ kernel, bottleneck until batched kernel exists
        output = torch.stack(
            [_gather(padded_flat[b], indices, sqrt_w)[inv_perm] for b in range(B)]
        )
        return output.to(src_device)  # (B, n_samples)

    # ------------------------------------------------------------------
    # Core single-coil ops (fused)
    # ------------------------------------------------------------------
    def _scatter_ifft_crop(self, coil_data, indices, sqrt_w, grid, _dtype):
        """Scatter -> full-size IFFT -> image-space center-crop for one coil.

        The IFFT is performed at the full (oversampled) ``grid_shape``; the
        result is then center-cropped in **image space** to ``image_shape``.
        Cropping k-space first (the old behaviour) is wrong when
        ``grid_shape != image_shape`` (oversampling > 1).
        """
        grid.zero_()
        self._scatter(grid, coil_data, indices, sqrt_w)
        full_img = ifft(grid.reshape(self.grid_shape), axes=self.fft_axes)
        return resize(full_img, self.image_shape)

    def _fft_pad_gather(self, coil_img, indices, sqrt_w, inv_perm, padded, _dtype):
        """Zero-pad image -> FFT at grid_shape -> gather -> unpermute for one coil.

        Adjoint of: scatter -> IFFT at grid_shape -> center-crop image.
        Reuses the *padded* buffer to avoid a fresh allocation per coil.
        """
        # Zero-pad image in image space (adjoint of center-crop in image space)
        padded.zero_()
        padded[self._pad_slices] = coil_img
        fft_result = fft(padded, axes=self.fft_axes)  # FFT at full grid_shape
        return _gather(fft_result.reshape(-1), indices, sqrt_w)[inv_perm]

    # ------------------------------------------------------------------
    # Dual-stream GPU pipelining  (CPU input -> GPU compute)
    # ------------------------------------------------------------------
    def _forward_pipeline(
        self, sparse_kspace, indices, sqrt_w, sort_perm, grid, conj_smaps, accum, dtype
    ):
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
                    buf[nxt] = (
                        sparse_kspace[c + 1].pin_memory().to(device, non_blocking=True)
                    )

            with torch.cuda.stream(stream):
                coil_data = buf[cur][sort_perm]
                img_c = self._scatter_ifft_crop(coil_data, indices, sqrt_w, grid, dtype)
                if conj_smaps is not None:
                    accum.addcmul_(img_c, conj_smaps[c])
                else:
                    accum += img_c.abs().square()

        torch.cuda.synchronize(device)

    def _adjoint_pipeline(
        self, image_d, indices, sqrt_w, inv_perm, smaps, padded, output, dtype
    ):
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
                    coil_img,
                    indices,
                    sqrt_w,
                    inv_perm,
                    padded,
                    dtype,
                )

        torch.cuda.synchronize(device)
