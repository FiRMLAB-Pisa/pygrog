"""Torch autograd wrapper for SparseFFT — enables gradient-based recon."""

__all__ = ["SparseFFTFunction", "sparse_fft_forward", "sparse_fft_adjoint"]

import torch
from torch.autograd import Function

from ..operator._sparse_fft import SparseFFT


class SparseFFTFunction(Function):
    """Autograd function wrapping SparseFFT adjoint (k-space -> image)."""

    @staticmethod
    def forward(ctx, kspace: torch.Tensor, op: SparseFFT) -> torch.Tensor:
        ctx.op = op
        return op.forward(kspace)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return ctx.op.adjoint(grad_output), None


def sparse_fft_forward(kspace: torch.Tensor, op: SparseFFT) -> torch.Tensor:
    """Differentiable sparse k-space -> image."""
    return SparseFFTFunction.apply(kspace, op)


class _SparseFFTAdjFunction(Function):
    """Autograd function wrapping SparseFFT forward (image -> k-space)."""

    @staticmethod
    def forward(ctx, image: torch.Tensor, op: SparseFFT) -> torch.Tensor:
        ctx.op = op
        return op.adjoint(image)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return ctx.op.forward(grad_output), None


def sparse_fft_adjoint(image: torch.Tensor, op: SparseFFT) -> torch.Tensor:
    """Differentiable image -> sparse k-space."""
    return _SparseFFTAdjFunction.apply(image, op)
