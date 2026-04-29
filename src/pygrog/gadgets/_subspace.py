"""Subspace projection gadget and SparseFFT decorator.

Provides two complementary views of low-rank temporal/contrast subspace
compression:

* :class:`SubspaceProjection` — standalone projection via truncated SVD,
  operates on dense (n_frames, *spatial) tensors.
* :func:`with_subspace` / :class:`SubspaceSparseFFT` — decorator that wraps
  a :class:`~pygrog.operator.SparseFFT` and fuses the subspace projection
  directly into the k-space ↔ image transform.

The subspace basis ``Phi`` has shape ``(K, T)`` where ``K`` is the subspace
rank and ``T`` is the number of temporal frames or contrasts.

Data conventions::

    Sparse k-space: (*batch, n_coils, *natural_shape) — natural_shape comes
        from the GROG plan, e.g. (T, k1, k0, kw) for 3D MRF.
    Image space:    (*batch_image, K, *image_shape) — K subspace coefficients.

The ``encoding_axis`` argument identifies which axis of the sparse tensor
carries the temporal/contrast dimension ``T``; the gadget broadcasts the
basis along that axis.
"""

__all__ = [
    "SubspaceMaskedFFT",
    "SubspaceProjection",
    "SubspaceSparseFFT",
    "with_subspace",
]

import torch
import numpy as np

from mrinufft._array_compat import with_torch

from .._solve._mixin import SolveMixin


# =====================================================================
# Standalone gadget
# =====================================================================
class SubspaceProjection:
    """Low-rank temporal subspace projection via truncated SVD.

    Given multi-frame data ``(n_frames, *spatial)``, projects onto the
    leading ``n_components`` left singular vectors.

    Parameters
    ----------
    n_components : int
        Number of subspace components to retain.
    """

    def __init__(self, n_components: int):
        self.n_components = n_components
        self._basis = None

    def fit(self, calib_data: torch.Tensor) -> "SubspaceProjection":
        U, _S, _Vh = torch.linalg.svd(calib_data, full_matrices=False)
        self._basis = U[:, : self.n_components].T.conj()
        return self

    @property
    def basis(self) -> torch.Tensor:
        if self._basis is None:
            raise RuntimeError("Call fit() first.")
        return self._basis

    @with_torch
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        spatial_shape = data.shape[1:]
        flat = data.reshape(data.shape[0], -1)
        coeff = self.basis @ flat
        return coeff.reshape(self.n_components, *spatial_shape)

    @with_torch
    def adjoint(self, coefficients: torch.Tensor) -> torch.Tensor:
        spatial_shape = coefficients.shape[1:]
        flat = coefficients.reshape(self.n_components, -1)
        frames = self.basis.conj().T @ flat
        return frames.reshape(-1, *spatial_shape)


# =====================================================================
# SparseFFT decorator
# =====================================================================
def with_subspace(base_op, subspace_basis, encoding_axis: int = -4, *, toeplitz=None):
    """Wrap a SparseFFT or MaskedFFT operator with subspace projection.

    Parameters
    ----------
    base_op : SparseFFT | MaskedFFT
        Underlying operator with a multi-dim ``natural_shape`` containing
        the temporal axis.
    subspace_basis : array-like, complex
        ``(K, T)`` subspace basis matrix.
    encoding_axis : int
        Axis (in the full sparse-tensor layout) carrying ``T``.  Default
        ``-4`` matches ``(*batch, C, T, k1, k0, kw)``.
    toeplitz : bool | None, optional
        Use Toeplitz embedding for :meth:`normal`.  ``None`` inherits
        from ``base_op.toeplitz``.
    """
    from ..operator._masked_fft import MaskedFFT

    if isinstance(base_op, MaskedFFT):
        return SubspaceMaskedFFT(
            base_op,
            subspace_basis,
            encoding_axis=encoding_axis,
            toeplitz=toeplitz,
        )
    return SubspaceSparseFFT(
        base_op,
        subspace_basis,
        encoding_axis=encoding_axis,
        toeplitz=toeplitz,
    )


class SubspaceSparseFFT(SolveMixin):
    """SparseFFT with low-rank subspace projection (loop-fused).

    Adjoint (sparse → image), per coil:
        1. weight by ``sqrt_w`` once on the input;
        2. for each ``k``: multiply by ``basis[k]`` along the T axis,
           scatter into the per-K oversampled grid;
        3. ONE batched K-IFFT + center-crop;
        4. fused FMA with ``smaps[c].conj()`` into the ``(K, *image)`` accumulator.

    Forward (image → sparse), per coil:
        1. multiply ``coeffs`` by ``smaps[c]``;
        2. ONE batched K-FFT + center-pad;
        3. for each ``k``: gather; accumulate ``basis.conj()[k] * gathered``
           into the per-coil ``(*natural)`` accumulator;
        4. write into the output coil slot.

    Parameters
    ----------
    base_op : SparseFFT
        Must have a multi-dim ``natural_shape`` covering the sparse layout
        (e.g. ``(T, k1, k0, kw)``) and SENSE maps (``smaps``) attached.
    subspace_basis : torch.Tensor
        ``(K, T)`` complex basis.
    encoding_axis : int
        Axis (in full sparse layout) of the temporal dimension ``T``.
        Default ``-4`` (last four axes are natural ``(T, k1, k0, kw)``).
    """

    def __init__(
        self, base_op, subspace_basis, encoding_axis: int = -4, *, toeplitz=None
    ):
        self._base = base_op
        self.basis = torch.as_tensor(subspace_basis)  # (K, T)
        self.K, self.T = self.basis.shape
        self.encoding_axis = encoding_axis

        self.grid_shape = base_op.grid_shape
        self.image_shape = base_op.image_shape
        self.smaps = getattr(base_op, "smaps", None)

        # Position of T inside `natural_shape` (positive index).
        nat_ndim = len(base_op.natural_shape)
        # Full sparse layout: (*batch, C, *natural).  encoding_axis is given
        # relative to that layout; we need the position inside `natural`.
        # E.g. encoding_axis=-4, nat_ndim=4 → axis_in_nat = -4 + nat_ndim = 0 ✓.
        ax = encoding_axis if encoding_axis >= 0 else encoding_axis + (1 + nat_ndim)
        # `ax` now indexes (C, *natural); subtract the leading C dim.
        self._t_axis_in_nat = ax - 1
        if not (0 <= self._t_axis_in_nat < nat_ndim):
            raise ValueError(
                f"encoding_axis={encoding_axis} does not land inside natural_shape "
                f"{base_op.natural_shape} (computed nat-axis {self._t_axis_in_nat})"
            )
        if base_op.natural_shape[self._t_axis_in_nat] != self.T:
            raise ValueError(
                f"basis T={self.T} does not match natural_shape"
                f"[{self._t_axis_in_nat}]={base_op.natural_shape[self._t_axis_in_nat]}"
            )

        # Toeplitz flag inherits from base unless overridden.
        if toeplitz is None:
            toeplitz = bool(getattr(base_op, "toeplitz", False))
        self.toeplitz = bool(toeplitz)
        self._toep_op = None  # lazily built

    # ------------------------------------------------------------------
    # adjoint: sparse k-space → subspace coefficient images  (A^H)
    # ------------------------------------------------------------------
    @with_torch
    def adjoint(self, sparse_kspace: torch.Tensor) -> torch.Tensor:
        """Sparse k-space → subspace coefficient images (``A^H``)."""
        return self._adjoint_impl(sparse_kspace)

    # ------------------------------------------------------------------
    # forward: subspace coefficient images → sparse k-space  (A)
    # ------------------------------------------------------------------
    @with_torch
    def forward(self, coeffs: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(coeffs)

    # ==================================================================
    # implementation
    # ==================================================================
    def _adjoint_impl(self, sparse_kspace: torch.Tensor) -> torch.Tensor:
        """Sparse → coefficients.

        Accepted layouts:

        - ``(*B, *S, C, *natural)``  (general)
        - ``(C, *natural)``          (single frame, no batch / stack)

        Output: ``(*B, *S, K, *image_shape)`` (or ``(K, *image_shape)``
        for the no-batch / no-stack case).
        """
        base = self._base
        nat = base.natural_shape
        nat_ndim = len(nat)
        s_shape = tuple(getattr(base, "stack_shape", ()) or ())
        s_ndim = len(s_shape)

        # Identify leading prefix (*B, *S) before (C, *natural).
        prefix = tuple(
            int(s) for s in sparse_kspace.shape[: sparse_kspace.ndim - (1 + nat_ndim)]
        )
        if s_ndim:
            if len(prefix) < s_ndim or tuple(prefix[-s_ndim:]) != s_shape:
                raise ValueError(
                    f"sparse_kspace prefix {prefix} must end with stack_shape {s_shape}"
                )
            B_shape = prefix[:-s_ndim]
        else:
            B_shape = prefix
        if sparse_kspace.ndim < 1 + nat_ndim:
            raise ValueError(
                f"Expected (...{(1 + nat_ndim)}D)=(C, *natural)={('C', *tuple(nat))}; "
                f"got {tuple(sparse_kspace.shape)}"
            )

        # No batch, no stack → single-frame fast path.
        if not prefix:
            return self._adjoint_single(sparse_kspace, 0)

        B_total = int(np.prod(B_shape)) if B_shape else 1
        S_total = int(np.prod(s_shape)) if s_shape else 1
        flat = sparse_kspace.reshape(
            B_total, S_total, *sparse_kspace.shape[-(1 + nat_ndim) :]
        )
        outs = []
        for b in range(B_total):
            for s in range(S_total):
                outs.append(self._adjoint_single(flat[b, s], s))
        # outs[i]: (K, *image_shape)
        stacked = torch.stack(outs, dim=0)
        return stacked.reshape(*B_shape, *s_shape, self.K, *base.image_shape)

    def _adjoint_single(self, sparse_kspace: torch.Tensor, s_flat_idx: int = 0):
        """Single-frame, single-stack-element adjoint.  Input: ``(C, *natural)``."""
        base = self._base

        # Dispatch to dual-stream pipeline when input lives on CPU but the
        # base operator computes on CUDA.  Overlaps per-coil H2D with the
        # K-batched scatter+IFFT compute.
        if self._use_dual_stream(sparse_kspace):
            return self._adjoint_single_dual(sparse_kspace, s_flat_idx)

        nat = base.natural_shape
        nat_ndim = len(nat)
        device = sparse_kspace.device
        dtype = sparse_kspace.dtype
        n_coils = int(sparse_kspace.shape[0])

        if base.smaps is None:
            raise NotImplementedError("SubspaceSparseFFT requires base_op.smaps")
        smaps = base.smaps.to(device, dtype=dtype)

        basis = self.basis.to(device, dtype=dtype)  # (K, T)
        T = self.T
        K = self.K

        phi_shape = [1] * nat_ndim
        phi_shape[self._t_axis_in_nat] = T

        output = torch.zeros(K, *base.image_shape, dtype=dtype, device=device)

        # Per-stack pre-weights via the operator's _stack_arrays.
        _idx_s, sqw_s, _, ip_s = base._stack_arrays(s_flat_idx)
        pre_w = sqw_s[ip_s].to(device=device, dtype=dtype).view(*nat)

        for c in range(n_coils):
            sw_c = sparse_kspace[c] * pre_w  # (*nat) pre-weighted
            weighted = basis.view(K, *phi_shape) * sw_c.unsqueeze(0)
            weighted_flat = weighted.reshape(K, -1)  # (K, n_samples)
            imgs = base._scatter_ifft_crop_batch(weighted_flat, s_flat_idx=s_flat_idx)
            output.addcmul_(imgs, smaps[c].conj().unsqueeze(0))

        return output

    def _forward_impl(self, coeffs: torch.Tensor) -> torch.Tensor:
        """Coefficients → sparse.

        Accepted layouts:

        - ``(*B, *S, K, *image_shape)`` (general)
        - ``(K, *image_shape)``         (single frame, no batch / stack)

        Output: ``(*B, *S, C, *natural)`` (or ``(C, *natural)`` for the
        no-batch / no-stack case).
        """
        base = self._base
        nat = base.natural_shape
        s_shape = tuple(getattr(base, "stack_shape", ()) or ())
        s_ndim = len(s_shape)

        img_ndim = len(base.image_shape)
        single_ndim = 1 + img_ndim  # K + *image_shape
        prefix = tuple(int(s) for s in coeffs.shape[: coeffs.ndim - single_ndim])
        if s_ndim:
            if len(prefix) < s_ndim or tuple(prefix[-s_ndim:]) != s_shape:
                raise ValueError(
                    f"coeffs prefix {prefix} must end with stack_shape {s_shape}"
                )
            B_shape = prefix[:-s_ndim]
        else:
            B_shape = prefix

        if not prefix:
            return self._forward_single(coeffs, 0)

        B_total = int(np.prod(B_shape)) if B_shape else 1
        S_total = int(np.prod(s_shape)) if s_shape else 1
        flat = coeffs.reshape(B_total, S_total, *coeffs.shape[-single_ndim:])
        outs = []
        for b in range(B_total):
            for s in range(S_total):
                outs.append(self._forward_single(flat[b, s], s))
        # outs[i]: (C, *nat)
        n_coils = outs[0].shape[0]
        stacked = torch.stack(outs, dim=0)
        return stacked.reshape(*B_shape, *s_shape, n_coils, *nat)

    def _forward_single(
        self, coeffs: torch.Tensor, s_flat_idx: int = 0
    ) -> torch.Tensor:
        """Single-frame, single-stack-element forward.  Input: ``(K, *image_shape)``,
        Output: ``(C, *natural)``."""
        base = self._base

        if self._use_dual_stream(coeffs):
            return self._forward_single_dual(coeffs, s_flat_idx)

        nat = base.natural_shape
        nat_ndim = len(nat)

        if coeffs.shape[0] != self.K:
            raise ValueError(f"coeffs.shape[0]={coeffs.shape[0]} != K={self.K}")
        if tuple(int(s) for s in coeffs.shape[1:]) != tuple(base.image_shape):
            raise ValueError(
                f"coeffs spatial {tuple(coeffs.shape[1:])} != image_shape {base.image_shape}"
            )

        device = coeffs.device
        dtype = coeffs.dtype

        if base.smaps is None:
            raise NotImplementedError("SubspaceSparseFFT requires base_op.smaps")
        smaps = base.smaps.to(device, dtype=dtype)
        n_coils = int(smaps.shape[0])

        basis_conj = self.basis.conj().to(device, dtype=dtype)  # (K, T)
        T = self.T
        K = self.K

        phi_shape = [1] * nat_ndim
        phi_shape[self._t_axis_in_nat] = T

        output = torch.empty(n_coils, *nat, dtype=dtype, device=device)

        _idx_s, sqw_s, _, ip_s = base._stack_arrays(s_flat_idx)
        pre_w = sqw_s[ip_s].to(device=device, dtype=dtype).view(*nat)

        for c in range(n_coils):
            coil_imgs = coeffs * smaps[c].unsqueeze(0)  # (K, *image)
            gathered = base._fft_pad_gather_batch(coil_imgs, s_flat_idx=s_flat_idx)
            gathered_nat = gathered.reshape(K, *nat)
            ksp_c = (basis_conj.view(K, *phi_shape) * gathered_nat).sum(dim=0)
            output[c] = ksp_c * pre_w

        return output

    # ------------------------------------------------------------------
    # Dual-stream GPU pipeline (CPU input -> GPU compute, overlapped)
    # ------------------------------------------------------------------
    def _use_dual_stream(self, x: torch.Tensor) -> bool:
        """Return True when ``x`` lives on CPU and the base operator
        computes on a CUDA device — the only configuration where coil-level
        stream overlap pays off."""
        base = self._base
        comp = getattr(base, "device", None)
        return (
            comp is not None
            and getattr(comp, "type", None) == "cuda"
            and x.device.type == "cpu"
            and torch.cuda.is_available()
        )

    def _adjoint_single_dual(
        self, sparse_kspace: torch.Tensor, s_flat_idx: int = 0
    ) -> torch.Tensor:
        """Coil-pipelined adjoint: H2D of next coil overlapped with the
        K-batched scatter+IFFT+FMA of the current coil.

        Input: ``(C, *natural)`` on CPU.  Output: ``(K, *image_shape)`` on CPU
        (mirrors the synchronous fast-path return device)."""
        base = self._base
        nat = base.natural_shape
        nat_ndim = len(nat)
        comp_device = base.device
        n_coils = int(sparse_kspace.shape[0])
        K, T = self.K, self.T
        dtype = sparse_kspace.dtype

        if base.smaps is None:
            raise NotImplementedError("SubspaceSparseFFT requires base_op.smaps")

        phi_shape = [1] * nat_ndim
        phi_shape[self._t_axis_in_nat] = T

        # Pre-stage constants once on the compute device.
        basis_gpu = self.basis.to(comp_device, dtype=dtype)  # (K, T)
        _idx_s, sqw_s, _, ip_s = base._stack_arrays(s_flat_idx)
        pre_w_gpu = (
            sqw_s[ip_s].to(device=comp_device, dtype=dtype).view(*nat)
        )  # (*nat)

        smaps_cpu = base.smaps.to(dtype=dtype)
        if not smaps_cpu.is_pinned() and smaps_cpu.device.type == "cpu":
            try:
                smaps_cpu = smaps_cpu.pin_memory()
            except RuntimeError:
                pass
        sparse_pin = sparse_kspace
        if (
            sparse_pin.device.type == "cpu"
            and not sparse_pin.is_pinned()
        ):
            try:
                sparse_pin = sparse_pin.pin_memory()
            except RuntimeError:
                pass

        output_gpu = torch.zeros(
            K, *base.image_shape, dtype=dtype, device=comp_device
        )

        s_data = torch.cuda.Stream(device=comp_device)
        s_comp = torch.cuda.Stream(device=comp_device)

        # Double-buffer the per-coil sparse + smaps slabs on the data stream.
        buf_sparse: list[torch.Tensor | None] = [None, None]
        buf_smaps: list[torch.Tensor | None] = [None, None]

        with torch.cuda.stream(s_data):
            buf_sparse[0] = sparse_pin[0].to(
                comp_device, dtype=dtype, non_blocking=True
            )
            buf_smaps[0] = smaps_cpu[0].to(
                comp_device, dtype=dtype, non_blocking=True
            )

        for c in range(n_coils):
            cur = c % 2
            nxt = 1 - cur
            if c + 1 < n_coils:
                with torch.cuda.stream(s_data):
                    buf_sparse[nxt] = sparse_pin[c + 1].to(
                        comp_device, dtype=dtype, non_blocking=True
                    )
                    buf_smaps[nxt] = smaps_cpu[c + 1].to(
                        comp_device, dtype=dtype, non_blocking=True
                    )

            # Compute on s_comp; ensure cur transfer is visible there.
            s_comp.wait_stream(s_data)
            with torch.cuda.stream(s_comp):
                sw_c = buf_sparse[cur] * pre_w_gpu  # (*nat)
                weighted = basis_gpu.view(K, *phi_shape) * sw_c.unsqueeze(0)
                weighted_flat = weighted.reshape(K, -1)
                # Input is already on comp_device, so the helper's .to(...)
                # is a no-op and the IFFT executes on s_comp.
                imgs = base._scatter_ifft_crop_batch(
                    weighted_flat, s_flat_idx=s_flat_idx
                )
                output_gpu.addcmul_(imgs, buf_smaps[cur].conj().unsqueeze(0))

        torch.cuda.synchronize(comp_device)
        return output_gpu.to(sparse_kspace.device)

    def _forward_single_dual(
        self, coeffs: torch.Tensor, s_flat_idx: int = 0
    ) -> torch.Tensor:
        """Coil-pipelined forward: per-coil K-batched FFT+gather on the
        compute stream, async D2H of the result on the data stream.

        Input: ``(K, *image_shape)`` on CPU.  Output: ``(C, *natural)`` on CPU."""
        base = self._base
        nat = base.natural_shape
        nat_ndim = len(nat)
        comp_device = base.device
        K, T = self.K, self.T
        dtype = coeffs.dtype

        if base.smaps is None:
            raise NotImplementedError("SubspaceSparseFFT requires base_op.smaps")

        phi_shape = [1] * nat_ndim
        phi_shape[self._t_axis_in_nat] = T

        basis_conj_gpu = self.basis.conj().to(comp_device, dtype=dtype)
        _idx_s, sqw_s, _, ip_s = base._stack_arrays(s_flat_idx)
        pre_w_gpu = sqw_s[ip_s].to(device=comp_device, dtype=dtype).view(*nat)

        # Coeffs are constant across coils — stage once.
        coeffs_gpu = coeffs.to(comp_device, dtype=dtype, non_blocking=True)

        smaps_cpu = base.smaps.to(dtype=dtype)
        if smaps_cpu.device.type == "cpu" and not smaps_cpu.is_pinned():
            try:
                smaps_cpu = smaps_cpu.pin_memory()
            except RuntimeError:
                pass
        n_coils = int(smaps_cpu.shape[0])

        # Pinned destination so D2H copy_(non_blocking=True) is truly async.
        try:
            output_cpu = torch.empty(
                n_coils, *nat, dtype=dtype, pin_memory=True
            )
        except RuntimeError:
            output_cpu = torch.empty(n_coils, *nat, dtype=dtype)

        s_data = torch.cuda.Stream(device=comp_device)
        s_comp = torch.cuda.Stream(device=comp_device)

        buf_smaps: list[torch.Tensor | None] = [None, None]
        ksp_buf: list[torch.Tensor | None] = [None, None]

        with torch.cuda.stream(s_data):
            buf_smaps[0] = smaps_cpu[0].to(
                comp_device, dtype=dtype, non_blocking=True
            )

        for c in range(n_coils):
            cur = c % 2
            nxt = 1 - cur
            if c + 1 < n_coils:
                with torch.cuda.stream(s_data):
                    buf_smaps[nxt] = smaps_cpu[c + 1].to(
                        comp_device, dtype=dtype, non_blocking=True
                    )

            s_comp.wait_stream(s_data)
            with torch.cuda.stream(s_comp):
                coil_imgs = coeffs_gpu * buf_smaps[cur].unsqueeze(0)
                gathered = base._fft_pad_gather_batch(
                    coil_imgs, s_flat_idx=s_flat_idx
                )
                gathered_nat = gathered.reshape(K, *nat)
                ksp_c = (
                    basis_conj_gpu.view(K, *phi_shape) * gathered_nat
                ).sum(dim=0) * pre_w_gpu
                ksp_buf[cur] = ksp_c

            # Async D2H on the data stream while the next coil computes.
            s_data.wait_stream(s_comp)
            with torch.cuda.stream(s_data):
                output_cpu[c].copy_(ksp_buf[cur], non_blocking=True)

        torch.cuda.synchronize(comp_device)
        return output_cpu.to(coeffs.device)

    @with_torch
    def normal(self, coeffs):
        if self.toeplitz:
            if self._toep_op is None:
                from .._toep._sub_toep import SubspaceToeplitzOp

                self._toep_op = SubspaceToeplitzOp(
                    self,
                    device=self._base.device,
                )
            return self._toep_op(coeffs)
        return self._adjoint_impl(self._forward_impl(coeffs))

    def __call__(self, x, adjoint=False):
        if adjoint:
            return self.adjoint(x)
        return self.forward(x)


# =====================================================================
# MaskedFFT decorator
# =====================================================================
class SubspaceMaskedFFT(SolveMixin):
    """MaskedFFT with low-rank subspace projection (loop-fused).

    Mirrors :class:`SubspaceSparseFFT` but operates on pre-gridded
    k-space data via :class:`~pygrog.operator.MaskedFFT`.

    Adjoint (gridded k-space → subspace coefficients), per coil:
        1. for each ``k``: multiply by ``basis[k]`` along the T axis;
        2. ONE batched K-IFFT + mask + center-crop;
        3. fused FMA with ``smaps[c].conj()`` into the accumulator.

    Forward (subspace coefficients → gridded k-space), per coil:
        1. multiply coefficients by ``smaps[c]``;
        2. ONE batched K-FFT + center-pad + mask;
        3. for each ``k``: accumulate ``basis.conj()[k] * masked_grid``.

    Parameters
    ----------
    base_op : MaskedFFT
        Must have sensitivity maps (``smaps``) attached and a multi-dim
        ``natural_shape`` covering the grid layout (e.g. ``(T, gy, gx)``
        for a 2D+T acquisition).
    subspace_basis : torch.Tensor
        ``(K, T)`` complex basis.
    encoding_axis : int
        Axis (in full grid layout) of the temporal dimension ``T``.
        Default ``-3`` (last three axes are ``(T, gy, gx)`` for 2D).
    """

    def __init__(
        self, base_op, subspace_basis, encoding_axis: int = -3, *, toeplitz=None
    ):
        self._base = base_op
        self.basis = torch.as_tensor(subspace_basis)  # (K, T)
        self.K, self.T = self.basis.shape
        self.encoding_axis = encoding_axis

        self.grid_shape = base_op.grid_shape
        self.image_shape = base_op.image_shape
        self.smaps = getattr(base_op, "smaps", None)

        # Position of T inside natural_shape (i.e. grid_shape for MaskedFFT).
        nat_ndim = len(base_op.natural_shape)
        ax = encoding_axis if encoding_axis >= 0 else encoding_axis + (1 + nat_ndim)
        self._t_axis_in_nat = ax - 1
        if not (0 <= self._t_axis_in_nat < nat_ndim):
            raise ValueError(
                f"encoding_axis={encoding_axis} does not land inside natural_shape "
                f"{base_op.natural_shape} (computed nat-axis {self._t_axis_in_nat})"
            )
        if base_op.natural_shape[self._t_axis_in_nat] != self.T:
            raise ValueError(
                f"basis T={self.T} does not match natural_shape"
                f"[{self._t_axis_in_nat}]={base_op.natural_shape[self._t_axis_in_nat]}"
            )

        if toeplitz is None:
            toeplitz = bool(getattr(base_op, "toeplitz", False))
        self.toeplitz = bool(toeplitz)
        self._toep_op = None

    # ------------------------------------------------------------------
    # adjoint: gridded k-space → subspace coefficient images  (A^H)
    # ------------------------------------------------------------------
    @with_torch
    def adjoint(self, kspace_grid: torch.Tensor) -> torch.Tensor:
        """Gridded k-space → subspace coefficient images (``A^H``)."""
        return self._adjoint_impl(kspace_grid)

    @with_torch
    def forward(self, coeffs: torch.Tensor) -> torch.Tensor:
        """Subspace coefficient images → gridded k-space (``A``)."""
        return self._forward_impl(coeffs)

    # ==================================================================
    # implementation
    # ==================================================================
    def _adjoint_impl(self, kspace_grid: torch.Tensor) -> torch.Tensor:
        """Gridded k-space → subspace coefficients.

        Accepted layouts:
        - ``(*B, *S, C, *grid_shape)``
        - ``(C, *grid_shape)`` (single frame)

        Output: ``(*B, *S, K, *image_shape)``.
        """
        base = self._base
        nat = base.natural_shape  # == grid_shape for MaskedFFT
        nat_ndim = len(nat)
        s_shape = tuple(getattr(base, "stack_shape", ()) or ())
        s_ndim = len(s_shape)

        expected_trailing = 1 + nat_ndim  # (C, *grid_shape)
        prefix = tuple(int(s) for s in kspace_grid.shape[:-expected_trailing])
        if s_ndim:
            if len(prefix) < s_ndim or tuple(prefix[-s_ndim:]) != s_shape:
                raise ValueError(
                    f"kspace_grid prefix {prefix} must end with stack_shape {s_shape}"
                )
            B_shape = prefix[:-s_ndim]
        else:
            B_shape = prefix

        if not prefix:
            return self._adjoint_single(kspace_grid, 0)

        B_total = int(np.prod(B_shape)) if B_shape else 1
        S_total = int(np.prod(s_shape)) if s_shape else 1
        flat = kspace_grid.reshape(
            B_total, S_total, *kspace_grid.shape[-expected_trailing:]
        )
        outs = []
        for b in range(B_total):
            for s in range(S_total):
                outs.append(self._adjoint_single(flat[b, s], s))
        stacked = torch.stack(outs, dim=0)
        return stacked.reshape(*B_shape, *s_shape, self.K, *base.image_shape)

    def _adjoint_single(self, kspace_grid: torch.Tensor, s_flat_idx: int = 0):
        """Single-frame adjoint.  Input: ``(C, *grid_shape)``."""
        base = self._base
        nat = base.natural_shape  # == grid_shape
        nat_ndim = len(nat)
        device = kspace_grid.device
        dtype = kspace_grid.dtype
        n_coils = int(kspace_grid.shape[0])

        if base.smaps is None:
            raise NotImplementedError("SubspaceMaskedFFT requires base_op.smaps")
        smaps = base.smaps.to(device, dtype=dtype)
        basis = self.basis.to(device, dtype=dtype)  # (K, T)
        K = self.K

        phi_shape = [1] * nat_ndim
        phi_shape[self._t_axis_in_nat] = self.T

        output = torch.zeros(K, *base.image_shape, dtype=dtype, device=device)

        for c in range(n_coils):
            # kspace_grid[c]: (*grid_shape), expand with K-basis along T axis
            kg_c = kspace_grid[c]  # (*grid_shape)
            # (K, *grid_shape): multiply each basis vector against the grid
            weighted = basis.view(K, *phi_shape) * kg_c.unsqueeze(0)
            # weighted: (K, *grid_shape) — already on the grid
            imgs = base._mask_ifft_crop_batch(weighted, s_flat_idx=s_flat_idx)
            output.addcmul_(imgs, smaps[c].conj().unsqueeze(0))

        return output

    def _forward_impl(self, coeffs: torch.Tensor) -> torch.Tensor:
        """Subspace coefficients → gridded k-space.

        Accepted layouts:
        - ``(*B, *S, K, *image_shape)``
        - ``(K, *image_shape)`` (single frame)

        Output: ``(*B, *S, C, *grid_shape)``.
        """
        base = self._base
        nat = base.natural_shape
        s_shape = tuple(getattr(base, "stack_shape", ()) or ())
        s_ndim = len(s_shape)

        img_ndim = len(base.image_shape)
        single_ndim = 1 + img_ndim
        prefix = tuple(int(s) for s in coeffs.shape[: coeffs.ndim - single_ndim])
        if s_ndim:
            if len(prefix) < s_ndim or tuple(prefix[-s_ndim:]) != s_shape:
                raise ValueError(
                    f"coeffs prefix {prefix} must end with stack_shape {s_shape}"
                )
            B_shape = prefix[:-s_ndim]
        else:
            B_shape = prefix

        if not prefix:
            return self._forward_single(coeffs, 0)

        B_total = int(np.prod(B_shape)) if B_shape else 1
        S_total = int(np.prod(s_shape)) if s_shape else 1
        flat = coeffs.reshape(B_total, S_total, *coeffs.shape[-single_ndim:])
        outs = []
        for b in range(B_total):
            for s in range(S_total):
                outs.append(self._forward_single(flat[b, s], s))
        n_coils = outs[0].shape[0]
        stacked = torch.stack(outs, dim=0)
        return stacked.reshape(*B_shape, *s_shape, n_coils, *nat)

    def _forward_single(self, coeffs: torch.Tensor, s_flat_idx: int = 0):
        """Single-frame forward.  Input: ``(K, *image_shape)``, output: ``(C, *grid_shape)``."""
        base = self._base
        nat = base.natural_shape  # == grid_shape
        nat_ndim = len(nat)

        if coeffs.shape[0] != self.K:
            raise ValueError(f"coeffs.shape[0]={coeffs.shape[0]} != K={self.K}")
        if tuple(int(s) for s in coeffs.shape[1:]) != tuple(base.image_shape):
            raise ValueError(
                f"coeffs spatial {tuple(coeffs.shape[1:])} != image_shape {base.image_shape}"
            )

        device = coeffs.device
        dtype = coeffs.dtype

        if base.smaps is None:
            raise NotImplementedError("SubspaceMaskedFFT requires base_op.smaps")
        smaps = base.smaps.to(device, dtype=dtype)
        n_coils = int(smaps.shape[0])

        basis_conj = self.basis.conj().to(device, dtype=dtype)  # (K, T)
        K = self.K

        phi_shape = [1] * nat_ndim
        phi_shape[self._t_axis_in_nat] = self.T

        output = torch.empty(n_coils, *nat, dtype=dtype, device=device)

        for c in range(n_coils):
            coil_imgs = coeffs * smaps[c].unsqueeze(0)  # (K, *image_shape)
            # FFT + pad + mask → (K, *grid_shape)
            kgrids = base._fft_pad_mask_batch(coil_imgs, s_flat_idx=s_flat_idx)
            # Accumulate over K: sum_k basis_conj[k, T-dim] * kgrids[k]
            # basis_conj: (K, T) reshaped to (K, *phi_shape)
            ksp_c = (basis_conj.view(K, *phi_shape) * kgrids).sum(dim=0)
            output[c] = ksp_c

        return output

    @with_torch
    def normal(self, coeffs):
        """Normal operator: ``A^H A x``."""
        if self.toeplitz:
            if self._toep_op is None:
                from .._toep._sub_toep import SubspaceToeplitzOp

                self._toep_op = SubspaceToeplitzOp(self, device=self._base.device)
            return self._toep_op(coeffs)
        return self._adjoint_impl(self._forward_impl(coeffs))

    def __call__(self, x, adjoint=False):
        if adjoint:
            return self.adjoint(x)
        return self.forward(x)
