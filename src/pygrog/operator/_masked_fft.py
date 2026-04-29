"""Masked FFT operator for gridded (Cartesian / undersampled) k-space.

Provides forward (gridded k-space → image) and adjoint (image → gridded
k-space) transforms for k-space data that have already been placed on a
Cartesian oversampled grid, i.e. the output of
:meth:`~pygrog.calib.GrogInterpolator.interpolate` with ``grid=True``.

Unlike :class:`~pygrog.operator.SparseFFT`, which works with *sparse*
non-Cartesian samples via scatter/gather, this operator applies a binary (or
density-weighted) mask directly on the oversampled k-space grid, making the
forward/adjoint transforms simple FFT + element-wise multiplication.  This
is faster in low-dimensional (2D, 2D+t, multislice) settings where the full
oversampled grid fits comfortably in memory.

Coil combination is performed inside the operator:

- If sensitivity maps are provided, SENSE-style combination (forward) or
  expansion (adjoint) is used.
- Otherwise, the output keeps the coil axis intact.

Both paths process data **coil-by-coil** to limit peak memory.

Weighting convention
--------------------
The ``density`` tensor is the sum of squared GROG weights at each grid cell:

    ``density[j] = Σ_i  sqrt_weights[i]² · δ(indices[i] == j)``

The binary ``mask`` is derived from density by thresholding at zero:

    ``mask[j] = (density[j] > 0).to(real_dtype)``

For the forward operator (gridded k-space → image) the ``mask`` selects which
grid cells carry valid signal; the gridded k-space is expected to have been
density-compensated (multiplied by ``pre_weights`` before scattering) so no
additional weight multiplication is needed here.

For the Toeplitz normal operator the ``density`` grid **is** the PSF in the
oversampled k-space domain, identical to what :class:`~pygrog._toep.GrogToeplitzOp`
computes via scatter.
"""

__all__ = ["MaskedFFTPlan", "MaskedFFT"]

import numpy as np
import torch
from mrinufft._array_compat import with_torch

from .._base._fftc import fft, ifft
from .._utils import resize


class MaskedFFTPlan:
    """Plan for :class:`MaskedFFT` — the Cartesian/masked counterpart of
    :class:`~pygrog.calib.GrogPlan`.

    Returned by :meth:`~pygrog.calib.GrogInterpolator.interpolate` with
    ``grid=True`` and accepted by :class:`MaskedFFT` via its *plan* argument,
    giving the same one-liner workflow as the sparse path::

        # Sparse path
        sparse = grog.interpolate(kspace)
        op = SparseFFT(plan=grog.plan, smaps=smaps)
        image = op.forward(sparse * grog.plan.pre_weights)

        # Dense/grid path — symmetric API
        kgrid, plan = grog.interpolate(kspace, grid=True)
        op = MaskedFFT(plan=plan, smaps=smaps)
        image = op.forward(kgrid)

    Parameters
    ----------
    grid_shape : tuple[int, ...]
        Oversampled Cartesian k-space grid shape (per stack element).
    image_shape : tuple[int, ...]
        Target image shape (center-crop target).
    stack_shape : tuple[int, ...]
        Leading stack axes; ``()`` for unstacked data.
    mask : torch.Tensor
        Real float binary sampling mask, shape ``(*stack_shape, *grid_shape)``.
    density : torch.Tensor
        Real float density grid (sum of squared GROG weights per cell),
        same shape as *mask*.  Used as the Toeplitz PSF by
        :class:`~pygrog._toep.GrogToeplitzOp`.
    """

    def __init__(self, grid_shape, image_shape, stack_shape, mask, density):
        self.grid_shape = tuple(int(s) for s in grid_shape)
        self.image_shape = tuple(int(s) for s in image_shape)
        self.stack_shape = tuple(int(s) for s in stack_shape)
        self.mask = torch.as_tensor(mask).float()
        self.density = torch.as_tensor(density).float()

    def __repr__(self):
        return (
            f"MaskedFFTPlan(grid_shape={self.grid_shape}, "
            f"image_shape={self.image_shape}, "
            f"stack_shape={self.stack_shape}, "
            f"mask={tuple(self.mask.shape)}, "
            f"density={tuple(self.density.shape)})"
        )


class MaskedFFT:
    """Masked FFT / IFFT operator for gridded k-space data.

    Accepts either a pre-built plan (from
    :meth:`~pygrog.calib.GrogInterpolator.interpolate` with ``grid=True``)
    together with explicit *mask* / *density* tensors, or raw shapes and masks
    (for users who already have a Cartesian undersampling pattern and skip
    GROG entirely).

    Parameters
    ----------
    grid_shape : tuple[int, ...]
        Oversampled Cartesian k-space grid, e.g. ``(nz, ny, nx)``.
    image_shape : tuple[int, ...]
        Target image shape (center-crop), e.g. ``(nz, ny, nx)``.
    mask : torch.Tensor
        Binary sampling mask, real-valued, shape ``(*stack_shape, *grid_shape)``
        or ``(*grid_shape,)`` for unstacked plans.  1 = sampled, 0 = not sampled.
    density : torch.Tensor | None, optional
        Density-compensation grid (sum of squared GROG weights), real-valued,
        same shape as *mask*.  Required for
        :meth:`~pygrog._toep.GrogToeplitzOp` construction; if ``None`` the
        Toeplitz normal operator is disabled.
    smaps : torch.Tensor | None, optional
        ``(n_coils, *image_shape)`` sensitivity maps.  *None* → no coil
        combination.
    stack_shape : tuple[int, ...], optional
        Stack prefix, e.g. ``(n_slices,)`` or ``(T, n_slices)``.  Inferred
        from the leading axes of *mask* that are not part of *grid_shape*
        when not provided.
    device : str | torch.device | None, optional
        Compute device.
    toeplitz : bool | None, optional
        Use Toeplitz embedding for :meth:`normal`.  ``None`` → auto: enabled
        on CPU, disabled on CUDA.
    plan : object | None, optional
        Pre-built plan (ignored except for logging; all required metadata
        is passed via the other arguments).
    """

    def __init__(
        self,
        grid_shape=None,
        image_shape=None,
        mask=None,
        density=None,
        smaps=None,
        stack_shape=None,
        device=None,
        *,
        toeplitz=None,
        plan=None,
    ):
        # --- Accept MaskedFFTPlan or raw arguments -------------------------
        if plan is not None and isinstance(plan, MaskedFFTPlan):
            grid_shape = plan.grid_shape
            image_shape = plan.image_shape
            mask = plan.mask
            density = plan.density
            if stack_shape is None:
                stack_shape = plan.stack_shape
        elif plan is not None:
            # Legacy: GrogPlan or other namespace passed as hint — ignore.
            pass

        if grid_shape is None or image_shape is None or mask is None:
            raise ValueError(
                "Either 'plan' (MaskedFFTPlan) or explicit 'grid_shape', "
                "'image_shape', and 'mask' are required."
            )

        self.grid_shape = tuple(int(s) for s in grid_shape)
        self.image_shape = tuple(int(s) for s in image_shape)
        self.ndim = len(self.grid_shape)
        self.grid_size = int(np.prod(self.grid_shape))
        self.fft_axes = tuple(range(-self.ndim, 0))

        # Mask: real float, shape (*stack_shape, *grid_shape) or (*grid_shape,)
        mask_t = torch.as_tensor(mask).float()
        # Infer stack_shape from leading dims if not supplied.
        if stack_shape is None:
            n_grid_dims = len(self.grid_shape)
            if mask_t.ndim > n_grid_dims:
                stack_shape = tuple(int(s) for s in mask_t.shape[:-n_grid_dims])
            else:
                stack_shape = ()
        self.stack_shape = tuple(int(s) for s in stack_shape)

        self.mask = mask_t  # (*stack_shape, *grid_shape) or (*grid_shape,)

        if density is not None:
            self.density = torch.as_tensor(density).float()
        else:
            self.density = None

        # natural_shape: for a MaskedFFT the "natural" k-space shape is
        # (*grid_shape,) since input data is already gridded. This attribute
        # is kept for API parity with SparseFFT so that decorators that read
        # base.natural_shape work unchanged.
        self.natural_shape = self.grid_shape
        self.n_samples = self.grid_size  # fictitious; kept for API parity

        # Sensitivity maps
        if smaps is not None:
            self.smaps = torch.as_tensor(smaps)
            self._conj_smaps = self.smaps.conj()
        else:
            self.smaps = None
            self._conj_smaps = None

        # Pre-compute center-crop slice (adjoint zero-pad bookkeeping)
        self._pad_slices = tuple(
            slice((gs - is_) // 2, (gs - is_) // 2 + is_)
            for gs, is_ in zip(self.grid_shape, self.image_shape, strict=False)
        )

        self.device = torch.device(device) if device is not None else None

        # Toeplitz: auto ON for CPU, OFF for CUDA (mirrors SparseFFT policy).
        if toeplitz is None:
            target = self.device if self.device is not None else torch.device("cpu")
            toeplitz = target.type == "cpu"
        # Toeplitz requires a density grid.
        if toeplitz and self.density is None:
            toeplitz = False
        self.toeplitz = bool(toeplitz)
        self._toep_op = None

        # Stash plan reference (informational only).
        self._plan = plan

    # ------------------------------------------------------------------
    # Stack helper
    # ------------------------------------------------------------------
    def _stack_mask(self, s_flat_idx: int):
        """Return ``(mask, density)`` for one flattened stack element.

        For unstacked operators returns the full mask/density.
        """
        if not self.stack_shape:
            return self.mask, self.density
        S_total = int(np.prod(self.stack_shape))
        m = self.mask.reshape(S_total, *self.grid_shape)[s_flat_idx]
        d = (
            self.density.reshape(S_total, *self.grid_shape)[s_flat_idx]
            if self.density is not None
            else None
        )
        return m, d

    # ------------------------------------------------------------------
    # Forward: gridded k-space → image  (adjoint NUFFT direction)
    # ------------------------------------------------------------------
    @with_torch
    def forward(self, kspace_grid: torch.Tensor) -> torch.Tensor:
        """Gridded k-space to image.

        Parameters
        ----------
        kspace_grid : torch.Tensor
            Input gridded k-space, shape ``(*B, *S, n_coils, *grid_shape)``
            (or without coil axis if smaps are set and caller pre-combined).

        Returns
        -------
        torch.Tensor
            ``(*B, *S, *image_shape)`` if smaps are set (SENSE-combined),
            else ``(*B, *S, n_coils, *image_shape)``.
        """
        s_shape = self.stack_shape
        s_ndim = len(s_shape)
        grid_ndim = self.ndim
        # Input trailing axes: (n_coils, *grid_shape)
        expected_trailing = 1 + grid_ndim
        prefix = tuple(int(s) for s in kspace_grid.shape[:-expected_trailing])

        if s_ndim:
            if len(prefix) < s_ndim or tuple(prefix[-s_ndim:]) != s_shape:
                raise ValueError(
                    f"kspace_grid prefix {prefix} must end with stack_shape {s_shape}"
                )
            B_shape = prefix[:-s_ndim]
        else:
            B_shape = prefix

        n_coils = int(kspace_grid.shape[-grid_ndim - 1])

        # Single-frame fast path (no batch, no stack).
        if not prefix:
            return self._forward_single(kspace_grid, 0)

        S_total = int(np.prod(s_shape)) if s_shape else 1
        B_total = int(np.prod(B_shape)) if B_shape else 1
        flat = kspace_grid.reshape(B_total, S_total, n_coils, *self.grid_shape)

        outs = []
        for b in range(B_total):
            frame_outs = []
            for s in range(S_total):
                frame_outs.append(self._forward_single(flat[b, s], s))
            outs.append(torch.stack(frame_outs, dim=0))
        stacked = torch.stack(outs, dim=0)  # (B, S, ...)

        single_out_shape = (
            tuple(self.image_shape) if self.smaps is not None
            else (n_coils, *self.image_shape)
        )
        return stacked.reshape(*B_shape, *s_shape, *single_out_shape)

    def _forward_single(self, kspace_grid: torch.Tensor, s_flat_idx: int = 0):
        """Single-frame forward.  Input: ``(n_coils, *grid_shape)``."""
        n_coils = int(kspace_grid.shape[0])
        src_device = kspace_grid.device
        comp_device = self.device if self.device is not None else src_device
        dtype = kspace_grid.dtype

        mask_s, _ = self._stack_mask(s_flat_idx)
        mask_s = mask_s.to(comp_device, dtype=dtype.to_real() if dtype.is_complex else dtype)
        kgrid = kspace_grid.to(comp_device)

        if self.smaps is not None:
            conj_smaps = self._conj_smaps.to(comp_device, dtype=dtype)
            accum = torch.zeros(self.image_shape, dtype=dtype, device=comp_device)
        else:
            accum = torch.zeros(n_coils, *self.image_shape, dtype=dtype, device=comp_device)

        for c in range(n_coils):
            # Apply mask, IFFT, center-crop
            masked = kgrid[c] * mask_s
            full_img = ifft(masked, axes=self.fft_axes)
            img_c = resize(full_img, self.image_shape)
            if self.smaps is not None:
                accum.addcmul_(img_c, conj_smaps[c])
            else:
                accum[c] = img_c

        return accum.to(src_device)

    # ------------------------------------------------------------------
    # Adjoint: image → gridded k-space  (forward NUFFT direction)
    # ------------------------------------------------------------------
    @with_torch
    def adjoint(self, image: torch.Tensor) -> torch.Tensor:
        """Image to gridded k-space.

        Parameters
        ----------
        image : torch.Tensor
            Input image, shape ``(*B, *S, *image_shape)`` if smaps set,
            else ``(*B, *S, n_coils, *image_shape)``.

        Returns
        -------
        torch.Tensor
            ``(*B, *S, n_coils, *grid_shape)``.
        """
        s_shape = self.stack_shape
        s_ndim = len(s_shape)
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

        if not prefix:
            return self._adjoint_single(image, 0)

        S_total = int(np.prod(s_shape)) if s_shape else 1
        B_total = int(np.prod(B_shape)) if B_shape else 1
        single_shape = tuple(image.shape[image.ndim - single_ndim:])
        flat = image.reshape(B_total, S_total, *single_shape)

        outs = []
        for b in range(B_total):
            frame_outs = []
            for s in range(S_total):
                frame_outs.append(self._adjoint_single(flat[b, s], s))
            outs.append(torch.stack(frame_outs, dim=0))
        stacked = torch.stack(outs, dim=0)  # (B, S, n_coils, *grid_shape)

        if self.smaps is not None:
            n_coils = int(self.smaps.shape[0])
        else:
            n_coils = int(stacked.shape[2])
        return stacked.reshape(*B_shape, *s_shape, n_coils, *self.grid_shape)

    def _adjoint_single(self, image: torch.Tensor, s_flat_idx: int = 0):
        """Single-frame adjoint.  Returns ``(n_coils, *grid_shape)``."""
        src_device = image.device
        comp_device = self.device if self.device is not None else src_device
        dtype = image.dtype

        mask_s, _ = self._stack_mask(s_flat_idx)
        real_dtype = dtype.to_real() if dtype.is_complex else dtype
        mask_s = mask_s.to(comp_device, dtype=real_dtype)
        image_d = image.to(comp_device)

        if self.smaps is not None:
            smaps = self.smaps.to(comp_device, dtype=dtype)
            n_coils = int(smaps.shape[0])
        else:
            n_coils = int(image_d.shape[0])

        output = torch.empty(n_coils, *self.grid_shape, dtype=dtype, device=comp_device)
        padded = torch.zeros(*self.grid_shape, dtype=dtype, device=comp_device)

        for c in range(n_coils):
            coil_img = image_d * smaps[c] if self.smaps is not None else image_d[c]
            padded.zero_()
            padded[self._pad_slices] = coil_img
            kgrid = fft(padded, axes=self.fft_axes)
            output[c] = kgrid * mask_s

        return output.to(src_device)

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

        When ``self.toeplitz`` is True (requires ``density`` to be set),
        uses a pre-computed PSF on ``grid_shape`` (built lazily on first
        call).  Otherwise falls back to ``adjoint(forward(x))``.
        """
        if self.toeplitz:
            if self._toep_op is None:
                from .._toep._grog_toep import GrogToeplitzOp
                self._toep_op = GrogToeplitzOp(self, device=self.device)
            return self._toep_op(image)
        return self.adjoint(self.forward(image))

    # ------------------------------------------------------------------
    # Batch helpers for decorators
    # ------------------------------------------------------------------
    def _mask_ifft_crop_batch(
        self,
        batch_kspace_grid: torch.Tensor,
        s_flat_idx: int = 0,
    ) -> torch.Tensor:
        """Apply mask, ONE batched IFFT, center-crop.

        Parameters
        ----------
        batch_kspace_grid : torch.Tensor
            ``(B, *grid_shape)`` complex gridded k-space (already density-
            compensated).
        s_flat_idx : int
            Flattened stack-element index.

        Returns
        -------
        torch.Tensor
            ``(B, *image_shape)`` complex.
        """
        B = batch_kspace_grid.shape[0]
        src_device = batch_kspace_grid.device
        comp_device = self.device if self.device is not None else src_device
        dtype = batch_kspace_grid.dtype

        mask_s, _ = self._stack_mask(s_flat_idx)
        real_dtype = dtype.to_real() if dtype.is_complex else dtype
        mask_s = mask_s.to(comp_device, dtype=real_dtype)

        kgrid = batch_kspace_grid.to(comp_device) * mask_s.unsqueeze(0)
        full_imgs = ifft(kgrid, axes=self.fft_axes)      # (B, *grid_shape)
        imgs = resize(full_imgs, (B, *self.image_shape))
        return imgs.to(src_device)

    def _fft_pad_mask_batch(
        self,
        batch_imgs: torch.Tensor,
        s_flat_idx: int = 0,
    ) -> torch.Tensor:
        """ONE batched FFT, zero-pad, apply mask.

        Parameters
        ----------
        batch_imgs : torch.Tensor
            ``(B, *image_shape)`` complex.
        s_flat_idx : int
            Flattened stack-element index.

        Returns
        -------
        torch.Tensor
            ``(B, *grid_shape)`` complex, masked.
        """
        B = batch_imgs.shape[0]
        src_device = batch_imgs.device
        comp_device = self.device if self.device is not None else src_device
        dtype = batch_imgs.dtype

        mask_s, _ = self._stack_mask(s_flat_idx)
        real_dtype = dtype.to_real() if dtype.is_complex else dtype
        mask_s = mask_s.to(comp_device, dtype=real_dtype)

        imgs_d = batch_imgs.to(comp_device)
        padded = torch.zeros(B, *self.grid_shape, dtype=dtype, device=comp_device)
        padded[(slice(None), *self._pad_slices)] = imgs_d
        kgrid = fft(padded, axes=self.fft_axes)           # (B, *grid_shape)
        return (kgrid * mask_s.unsqueeze(0)).to(src_device)
