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
from .._solve._mixin import SolveMixin

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
class SparseFFT(SolveMixin):
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
    toeplitz : bool | None, optional
        Use Toeplitz embedding (PSF on `grid_shape`) for the self-adjoint
        operator :meth:`normal`.  ``None`` → auto: enabled on CPU,
        disabled on CUDA (matches :func:`pygrog.utils.nlinv` policy).
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
        toeplitz=None,
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
            self.natural_shape = tuple(
                int(s) for s in getattr(plan, "natural_shape", (int(plan.n_samples),))
            )
            self.stack_shape = tuple(
                int(s) for s in getattr(plan, "stack_shape", ()) or ()
            )
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
            self.natural_shape = (int(idx.numel()),)
            self.stack_shape = ()

        # n_samples is the per-stack-element sample count.  When stacked,
        # self.indices has shape (*stack_shape, n_samples); else (n_samples,).
        self.n_samples = int(self.indices.shape[-1])
        self.ndim = len(self.grid_shape)
        self.fft_axes = tuple(range(-self.ndim, 0))

        # GPU binning — computed lazily on first use
        self._bin_starts = None
        self._bin_size = 0

        # Packed-stack arrays — built lazily on first stacked op (see
        # ``_packed_arrays``).  Cached per compute device.
        self._packed_cache = None

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

        # --- Toeplitz acceleration -----------------------------------------
        # Auto-toggle: ON when target compute is CPU, OFF on CUDA (mirrors
        # `pygrog.utils.nlinv` policy).  User can force True/False.
        if toeplitz is None:
            target = self.device if self.device is not None else torch.device("cpu")
            toeplitz = target.type == "cpu"
        self.toeplitz = bool(toeplitz)
        self._toep_op = None  # lazily built on first .normal() call

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_bins(self, device):
        """Compute GPU bins on first use for the given device."""
        if device.type != "cuda" or self._bin_starts is not None:
            return
        if self.stack_shape:
            # No bin caching for stacked plans (per-stack bins would need
            # rebuilding inside the forward loop).  This only affects CUDA
            # micro-optimisation; correctness is unchanged.
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
    # Per-stack 1-D selectors (for stacked plans)
    # ------------------------------------------------------------------
    def _stack_arrays(self, s_flat_idx: int):
        """Return ``(indices, sqrt_weights, sort_perm, inv_perm)`` for one
        flattened stack element.  If unstacked, returns the global arrays.
        """
        if not self.stack_shape:
            return self.indices, self.sqrt_weights, self.sort_perm, self.inv_perm
        s_total = int(np.prod(self.stack_shape))
        idx = self.indices.reshape(s_total, self.n_samples)[s_flat_idx]
        sqw = self.sqrt_weights.reshape(s_total, self.n_samples)[s_flat_idx]
        sp = self.sort_perm.reshape(s_total, self.n_samples)[s_flat_idx]
        ip = self.inv_perm.reshape(s_total, self.n_samples)[s_flat_idx]
        return idx, sqw, sp, ip

    # ------------------------------------------------------------------
    # Packed-stack arrays — fuse all stack elements into a single sorted
    # 1-D problem on a flat super-grid of shape ``(S_total * grid_size,)``.
    # Per-stack indices are offset by ``s * grid_size`` so the concatenated
    # array is globally sorted (each stack lives in its own disjoint slab),
    # which lets the single-trajectory C++ kernels handle the whole stack
    # in one call without naive Python looping.
    # ------------------------------------------------------------------
    def _packed_arrays(self, comp_device: torch.device):
        """Return packed ``(indices, sqrt_w, sort_perm, inv_perm, S, n_per)``.

        Each output is a 1-D tensor of length ``S_total * n_per`` on
        ``comp_device``; ``indices`` are offset so writes to the flat super-
        grid land in disjoint per-stack slabs.  Cached per device.
        """
        if not self.stack_shape:
            raise RuntimeError("_packed_arrays called on an unstacked plan")

        cache = getattr(self, "_packed_cache", None)
        if cache is not None and cache[0] == comp_device:
            return cache[1]

        S_total = int(np.prod(self.stack_shape))
        n_per = int(self.n_samples)

        idx2d = self.indices.reshape(S_total, n_per).to(comp_device)
        sqw2d = self.sqrt_weights.reshape(S_total, n_per).to(comp_device)
        sp2d = self.sort_perm.reshape(S_total, n_per).to(comp_device)
        ip2d = self.inv_perm.reshape(S_total, n_per).to(comp_device)

        grid_off = (
            torch.arange(S_total, device=comp_device, dtype=idx2d.dtype)
            * self.grid_size
        ).unsqueeze(-1)
        data_off = (
            torch.arange(S_total, device=comp_device, dtype=sp2d.dtype) * n_per
        ).unsqueeze(-1)

        packed = (
            (idx2d + grid_off).reshape(-1).contiguous(),
            sqw2d.reshape(-1).contiguous(),
            (sp2d + data_off).reshape(-1).contiguous(),
            (ip2d + data_off).reshape(-1).contiguous(),
            S_total,
            n_per,
        )
        self._packed_cache = (comp_device, packed)
        return packed

    # ------------------------------------------------------------------
    # Forward: sparse k-space -> image  (adjoint NUFFT direction)
    # ------------------------------------------------------------------
    @with_torch
    def adjoint(self, sparse_kspace: torch.Tensor) -> torch.Tensor:
        """Sparse k-space to image.

        Accepted input layouts (with optional leading ``*B`` batch and, for
        stacked plans, leading ``*S`` stack axes inserted between batch and
        single-frame dims):

        - ``(*B, *S, n_coils, *natural_shape)``
        - ``(*B, *S, n_coils, n_samples)`` (legacy / flat form)

        Output: ``(*B, *S, *image_shape)`` if smaps are set (SENSE-combined),
        else ``(*B, *S, n_coils, *image_shape)``.
        """
        # Natural-shape (multi-dim) input → fold trailing dims into n_samples.
        nat = self.natural_shape
        if (
            len(nat) > 1
            and tuple(int(s) for s in sparse_kspace.shape[-len(nat) :]) == nat
        ):
            flat_shape = (*tuple(int(s) for s in sparse_kspace.shape[:-len(nat)]), self.n_samples)
            sparse_kspace = sparse_kspace.reshape(flat_shape)

        # x now has trailing (n_coils, n_samples).  Split prefix into (*B, *S).
        s_shape = self.stack_shape
        s_ndim = len(s_shape)
        prefix = tuple(int(s) for s in sparse_kspace.shape[:-2])
        if s_ndim:
            if len(prefix) < s_ndim or tuple(prefix[-s_ndim:]) != s_shape:
                raise ValueError(
                    f"sparse_kspace prefix {prefix} must end with stack_shape {s_shape}"
                )
            B_shape = prefix[:-s_ndim]
        else:
            B_shape = prefix

        # Single-frame fast path (no batch, no stack).
        if not prefix:
            return self._forward_single(sparse_kspace, 0)

        # Loop over flattened (*B, *S).
        S_total = int(np.prod(s_shape)) if s_shape else 1
        B_total = int(np.prod(B_shape)) if B_shape else 1
        # Flatten to (B_total, S_total, n_coils, n_samples)
        n_coils = int(sparse_kspace.shape[-2])
        flat = sparse_kspace.reshape(B_total, S_total, n_coils, self.n_samples)
        # Stacked path: fuse all S stack elements into a single packed
        # scatter call (one C++ kernel invocation handles every stack
        # element at once via offset indices into a flat super-grid).
        if s_ndim:
            outs = []
            for b in range(B_total):
                outs.append(self._forward_packed(flat[b]))
        else:
            outs = []
            for b in range(B_total):
                outs.append(self._forward_single(flat[b, 0], 0).unsqueeze(0))
        # Single-frame output shape:
        single_out_shape = (
            tuple(self.image_shape)
            if self.smaps is not None
            else (n_coils, *self.image_shape)
        )
        stacked = torch.stack(outs, dim=0)  # (B_total, S_total, *single_out)
        return stacked.reshape(*B_shape, *s_shape, *single_out_shape)

    def _forward_single(self, sparse_kspace: torch.Tensor, s_flat_idx: int = 0):
        """Single-frame forward (one stack element). ``sparse_kspace`` shape:
        ``(n_coils, n_samples)``."""
        n_coils = sparse_kspace.shape[0]
        src_device = sparse_kspace.device
        comp_device = self.device if self.device is not None else src_device
        use_pipeline = comp_device.type == "cuda" and src_device.type == "cpu"

        dtype = sparse_kspace.dtype
        idx_s, sqw_s, sp_s, _ = self._stack_arrays(s_flat_idx)
        indices = idx_s.to(comp_device)
        sqrt_w = sqw_s.to(comp_device)
        sort_perm = sp_s.to(comp_device)
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
    # Packed-stack forward — one C++ scatter call covers all S elements
    # ------------------------------------------------------------------
    def _forward_packed(self, sparse_kspace: torch.Tensor) -> torch.Tensor:
        """Stacked forward over all stack elements at once.

        Parameters
        ----------
        sparse_kspace : torch.Tensor
            ``(S_total, n_coils, n_per)`` complex.

        Returns
        -------
        torch.Tensor
            ``(S_total, *image_shape)`` if smaps are set (SENSE-combined),
            else ``(S_total, n_coils, *image_shape)``.
        """
        src_device = sparse_kspace.device
        comp_device = self.device if self.device is not None else src_device
        dtype = sparse_kspace.dtype

        idx_p, sqw_p, sp_p, _, S_total, _n_per = self._packed_arrays(comp_device)

        n_coils = int(sparse_kspace.shape[-2])
        # Bring data on compute device, fold (S, n_coils, n_per) -> (n_coils, S*n_per)
        ksp_d = sparse_kspace.to(comp_device).permute(1, 0, 2).reshape(n_coils, -1)
        # Globally sort once via packed_sort_perm (one indexing op per coil).
        sorted_all = ksp_d[:, sp_p]  # (n_coils, S*n_per)

        super_grid = torch.empty(
            S_total * self.grid_size, dtype=dtype, device=comp_device
        )

        if self.smaps is not None:
            conj_smaps = self._conj_smaps.to(comp_device, dtype=dtype)
            accum = torch.zeros(
                S_total, *self.image_shape, dtype=dtype, device=comp_device
            )
        else:
            accum = torch.zeros(
                S_total, n_coils, *self.image_shape, dtype=dtype, device=comp_device
            )

        for c in range(n_coils):
            super_grid.zero_()
            # Single C++ kernel call covers every stack element.
            _scatter_add(super_grid, sorted_all[c], idx_p, sqw_p)
            full_imgs = ifft(
                super_grid.reshape(S_total, *self.grid_shape),
                axes=self.fft_axes,
            )
            imgs = resize(full_imgs, (S_total, *self.image_shape))
            if self.smaps is not None:
                accum.addcmul_(imgs, conj_smaps[c].unsqueeze(0))
            else:
                accum[:, c] = imgs

        return accum.to(src_device)

    # ------------------------------------------------------------------
    # Adjoint: image -> sparse k-space  (forward NUFFT direction)
    # ------------------------------------------------------------------
    @with_torch
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Image to sparse k-space.

        Accepted input layouts (with optional leading ``*B`` batch and, for
        stacked plans, leading ``*S`` stack axes):

        - ``(*B, *S, *image_shape)`` if smaps are set
        - ``(*B, *S, n_coils, *image_shape)`` otherwise

        Output: ``(*B, *S, n_coils, *natural_shape)``.
        """
        out = self._adjoint_flat(image)
        nat = self.natural_shape
        if len(nat) > 1:
            out = out.reshape(*out.shape[:-1], *nat)
        return out

    @with_torch
    def _adjoint_flat(self, image: torch.Tensor) -> torch.Tensor:
        """Flat-output adjoint: returns (*B, *S, n_coils, n_samples)."""
        s_shape = self.stack_shape
        s_ndim = len(s_shape)
        # Single-frame ndim (no batch, no stack):
        single_ndim = len(self.image_shape) + (0 if self.smaps is not None else 1)
        prefix = tuple(int(s) for s in image.shape[: image.ndim - single_ndim])

        if s_ndim:
            if len(prefix) < s_ndim or tuple(prefix[-s_ndim:]) != s_shape:
                raise ValueError(
                    f"image prefix {prefix} must end with stack_shape {s_shape}"
                )
            B_shape = prefix[:-s_ndim]
        else:
            B_shape = prefix

        # Single-frame fast path.
        if not prefix:
            return self._adjoint_single(image, 0)

        S_total = int(np.prod(s_shape)) if s_shape else 1
        B_total = int(np.prod(B_shape)) if B_shape else 1
        # Reshape image to (B_total, S_total, *single_shape)
        single_shape = tuple(image.shape[image.ndim - single_ndim :])
        flat = image.reshape(B_total, S_total, *single_shape)
        if s_ndim:
            outs = [self._adjoint_packed(flat[b]) for b in range(B_total)]
        else:
            outs = [
                self._adjoint_single(flat[b, 0], 0).unsqueeze(0) for b in range(B_total)
            ]
        # Each entry has shape (S_total, n_coils, n_samples)
        n_coils = outs[0].shape[1]
        stacked = torch.stack(outs, dim=0)  # (B_total, S_total, n_coils, n_samples)
        return stacked.reshape(*B_shape, *s_shape, n_coils, self.n_samples)

    def _adjoint_single(self, image: torch.Tensor, s_flat_idx: int = 0):
        """Single-frame adjoint (one stack element)."""
        src_device = image.device
        comp_device = self.device if self.device is not None else src_device
        use_pipeline = comp_device.type == "cuda" and src_device.type == "cpu"

        dtype = image.dtype
        idx_s, sqw_s, _, ip_s = self._stack_arrays(s_flat_idx)
        indices = idx_s.to(comp_device)
        sqrt_w = sqw_s.to(comp_device)
        inv_perm = ip_s.to(comp_device)
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

    # ------------------------------------------------------------------
    # Packed-stack adjoint — one C++ gather call covers all S elements
    # ------------------------------------------------------------------
    def _adjoint_packed(self, image: torch.Tensor) -> torch.Tensor:
        """Stacked adjoint over all stack elements at once.

        Parameters
        ----------
        image : torch.Tensor
            ``(S_total, *image_shape)`` if smaps are set, otherwise
            ``(S_total, n_coils, *image_shape)``.

        Returns
        -------
        torch.Tensor
            ``(S_total, n_coils, n_per)`` complex.
        """
        src_device = image.device
        comp_device = self.device if self.device is not None else src_device
        dtype = image.dtype

        idx_p, sqw_p, _, ip_p, S_total, n_per = self._packed_arrays(comp_device)

        image_d = image.to(comp_device)
        if self.smaps is not None:
            smaps = self.smaps.to(comp_device, dtype=dtype)
            n_coils = smaps.shape[0]
        else:
            n_coils = image_d.shape[1]

        # Pre-allocate a packed (S, *grid_shape) zero-padded buffer reused
        # per coil; one batched FFT per coil over the full stack.
        padded = torch.zeros(S_total, *self.grid_shape, dtype=dtype, device=comp_device)
        slc = (slice(None), *self._pad_slices)

        # Output is built sorted in packed layout, then unsorted via inv_perm
        # at the end (single indexing op per coil).
        output_unsorted = torch.empty(
            n_coils, S_total * n_per, dtype=dtype, device=comp_device
        )

        for c in range(n_coils):
            if self.smaps is not None:
                coil_imgs = image_d * smaps[c]  # (S, *image_shape)
            else:
                coil_imgs = image_d[:, c]  # (S, *image_shape)
            padded.zero_()
            padded[slc] = coil_imgs
            kgrid = fft(padded, axes=self.fft_axes)  # (S, *grid_shape)
            super_flat = kgrid.reshape(-1)  # (S*grid_size,)
            # Single C++ kernel call across the whole stack.
            sorted_packed = _gather(super_flat, idx_p, sqw_p)  # (S*n_per,)
            # Undo per-stack sort with one indexing op.
            output_unsorted[c] = sorted_packed[ip_p]

        out = (
            output_unsorted.reshape(n_coils, S_total, n_per)
            .permute(1, 0, 2)
            .contiguous()
        )
        return out.to(src_device)

    def __call__(self, x, adjoint=False):
        if adjoint:
            return self.adjoint(x)
        return self.forward(x)

    # ------------------------------------------------------------------
    # Normal operator: A^H A
    # ------------------------------------------------------------------
    @with_torch
    def normal(self, image: torch.Tensor) -> torch.Tensor:
        """Self-adjoint application: ``A^H A image``.

        When ``self.toeplitz`` is True, uses a pre-computed PSF on
        ``grid_shape`` (built lazily on first call) and applies
        pad / FFT / PSF / IFFT / crop per coil.  Otherwise dispatches to
        a fused ``forward(adjoint(.))`` helper that *skips* the
        ``inv_perm`` (in adjoint) and ``sort_perm`` (in forward) round-
        trip — the intermediate sparse k-space is kept in sorted order
        end-to-end ("sort-once" optimisation).

        Parameters
        ----------
        image : torch.Tensor
            Same shape as the output of :meth:`forward`.

        Returns
        -------
        torch.Tensor
            Same shape as input.
        """
        if self.toeplitz:
            if self._toep_op is None:
                # Local import to avoid circular import at module load.
                from .._toep._grog_toep import GrogToeplitzOp

                self._toep_op = GrogToeplitzOp(self, device=self.device)
            return self._toep_op(image)
        return self._normal_no_toep(image)

    # ------------------------------------------------------------------
    # Sort-once non-Toeplitz normal
    # ------------------------------------------------------------------
    @with_torch
    def _normal_no_toep(self, image: torch.Tensor) -> torch.Tensor:
        """Batched/stacked dispatcher for :meth:`_normal_*_no_toep`.

        Mirrors the dispatch logic of :meth:`forward` / :meth:`adjoint`
        but routes per-frame work to fused helpers that omit the
        ``inv_perm`` / ``sort_perm`` round-trip in the intermediate
        sparse-k-space buffer.
        """
        s_shape = self.stack_shape
        s_ndim = len(s_shape)
        single_ndim = (
            len(self.image_shape)
            if self.smaps is not None
            else len(self.image_shape) + 1
        )
        prefix = tuple(int(s) for s in image.shape[: image.ndim - single_ndim])
        if s_ndim:
            if len(prefix) < s_ndim or tuple(prefix[-s_ndim:]) != s_shape:
                raise ValueError(
                    f"image prefix {prefix} must end with stack_shape {s_shape}"
                )
            B_shape = prefix[:-s_ndim]
        else:
            B_shape = prefix

        if not prefix:
            return self._normal_single_no_toep(image, 0)

        S_total = int(np.prod(s_shape)) if s_shape else 1
        B_total = int(np.prod(B_shape)) if B_shape else 1
        single_shape = tuple(image.shape[image.ndim - single_ndim :])
        flat = image.reshape(B_total, S_total, *single_shape)
        outs = []
        if s_ndim:
            for b in range(B_total):
                outs.append(self._normal_packed_no_toep(flat[b]))
        else:
            for b in range(B_total):
                outs.append(self._normal_single_no_toep(flat[b, 0], 0).unsqueeze(0))
        stacked = torch.stack(outs, dim=0)
        return stacked.reshape(*B_shape, *s_shape, *single_shape)

    def _normal_single_no_toep(self, image: torch.Tensor, s_flat_idx: int = 0):
        """Sort-once single-frame ``A^H A``; intermediate stays sorted."""
        src_device = image.device
        comp_device = self.device if self.device is not None else src_device
        dtype = image.dtype

        idx_s, sqw_s, _, _ = self._stack_arrays(s_flat_idx)
        indices = idx_s.to(comp_device)
        sqrt_w = sqw_s.to(comp_device)
        self._ensure_bins(comp_device)

        image_d = image.to(comp_device)
        if self.smaps is not None:
            smaps = self.smaps.to(comp_device, dtype=dtype)
            n_coils = smaps.shape[0]
            accum = torch.zeros(self.image_shape, dtype=dtype, device=comp_device)
        else:
            n_coils = image_d.shape[0]
            accum = torch.zeros(
                (n_coils, *self.image_shape),
                dtype=dtype,
                device=comp_device,
            )

        padded = torch.empty(*self.grid_shape, dtype=dtype, device=comp_device)
        grid = torch.empty(self.grid_size, dtype=dtype, device=comp_device)

        for c in range(n_coils):
            coil_img = image_d * smaps[c] if self.smaps is not None else image_d[c]
            # adjoint: pad -> FFT -> gather (sorted; skip [inv_perm])
            padded.zero_()
            padded[self._pad_slices] = coil_img
            kgrid = fft(padded, axes=self.fft_axes).reshape(-1)
            sorted_kspace = _gather(kgrid, indices, sqrt_w)
            # forward: scatter (presorted; skip [sort_perm]) -> IFFT -> crop
            img_c = self._scatter_ifft_crop(
                sorted_kspace,
                indices,
                sqrt_w,
                grid,
                dtype,
            )
            if self.smaps is not None:
                accum.addcmul_(img_c, smaps[c].conj())
            else:
                accum[c] = img_c

        return accum.to(src_device)

    def _normal_packed_no_toep(self, image: torch.Tensor) -> torch.Tensor:
        """Sort-once packed-stack ``A^H A`` over all S elements at once."""
        src_device = image.device
        comp_device = self.device if self.device is not None else src_device
        dtype = image.dtype

        idx_p, sqw_p, _, _, S_total, _n_per = self._packed_arrays(comp_device)

        image_d = image.to(comp_device)
        if self.smaps is not None:
            smaps = self.smaps.to(comp_device, dtype=dtype)
            n_coils = smaps.shape[0]
            accum = torch.zeros(
                S_total,
                *self.image_shape,
                dtype=dtype,
                device=comp_device,
            )
        else:
            n_coils = image_d.shape[1]
            accum = torch.zeros(
                S_total,
                n_coils,
                *self.image_shape,
                dtype=dtype,
                device=comp_device,
            )

        padded = torch.zeros(S_total, *self.grid_shape, dtype=dtype, device=comp_device)
        super_grid = torch.empty(
            S_total * self.grid_size,
            dtype=dtype,
            device=comp_device,
        )
        slc = (slice(None), *self._pad_slices)

        for c in range(n_coils):
            coil_imgs = image_d * smaps[c] if self.smaps is not None else image_d[:, c]
            # adj packed: pad/FFT/gather (sorted-packed; skip [ip_p])
            padded.zero_()
            padded[slc] = coil_imgs
            kgrid_super = fft(padded, axes=self.fft_axes).reshape(-1)
            sorted_packed = _gather(kgrid_super, idx_p, sqw_p)  # (S*n_per,)
            # fwd packed presorted: scatter (skip [:, sp_p]) / IFFT / crop
            super_grid.zero_()
            _scatter_add(super_grid, sorted_packed, idx_p, sqw_p)
            full_imgs = ifft(
                super_grid.reshape(S_total, *self.grid_shape),
                axes=self.fft_axes,
            )
            imgs = resize(full_imgs, (S_total, *self.image_shape))
            if self.smaps is not None:
                accum.addcmul_(imgs, smaps[c].conj().unsqueeze(0))
            else:
                accum[:, c] = imgs

        return accum.to(src_device)

    # ------------------------------------------------------------------
    # Iterative solve
    # ------------------------------------------------------------------
    # Provided by SolveMixin (attached at module import time).

    # ------------------------------------------------------------------
    # Batch helpers for decorators  (Tier-1 FFT fusion)
    # ------------------------------------------------------------------
    def _scatter_ifft_crop_batch(
        self,
        batch_kspace: torch.Tensor,
        s_flat_idx: int = 0,
    ) -> torch.Tensor:
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
        s_flat_idx : int
            Flattened stack-element index (ignored for unstacked plans).

        Returns
        -------
        torch.Tensor
            ``(B, *image_shape)`` complex, on the same device as input.
        """
        B = batch_kspace.shape[0]
        src_device = batch_kspace.device
        comp_device = self.device if self.device is not None else src_device
        dtype = batch_kspace.dtype

        idx_s, sqw_s, sp_s, _ = self._stack_arrays(s_flat_idx)
        indices = idx_s.to(comp_device)
        sqrt_w = sqw_s.to(comp_device)
        sort_perm = sp_s.to(comp_device)
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

    def _fft_pad_gather_batch(
        self,
        batch_imgs: torch.Tensor,
        s_flat_idx: int = 0,
    ) -> torch.Tensor:
        """ONE batched FFT -> zero-pad -> gather (per-component loop).

        Reduces ``B x n_coils`` FFT calls to a single
        ``torch.fft.fftn((B, *image_shape))`` call.  The gather loop is
        unchanged (needs a batched C++ kernel for full fusion).

        Parameters
        ----------
        batch_imgs : torch.Tensor
            ``(B, *image_shape)`` complex.  Each slice ``[b]`` is an
            independently-weighted image (e.g. one ORC component x one coil).
        s_flat_idx : int
            Flattened stack-element index (ignored for unstacked plans).

        Returns
        -------
        torch.Tensor
            ``(B, n_samples)`` complex, in original (unsorted) k-space order.
        """
        B = batch_imgs.shape[0]
        src_device = batch_imgs.device
        comp_device = self.device if self.device is not None else src_device
        dtype = batch_imgs.dtype

        idx_s, sqw_s, _, ip_s = self._stack_arrays(s_flat_idx)
        indices = idx_s.to(comp_device)
        sqrt_w = sqw_s.to(comp_device)
        inv_perm = ip_s.to(comp_device)
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
