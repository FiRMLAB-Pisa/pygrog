#!/usr/bin/env python3
"""Unified benchmark runner for PyGROG vs FINUFFT/CUFINUFFT.

This script is designed to run on CPU-only laptops and on GPU servers
without code changes. GPU benchmarks are skipped automatically when the
required hardware/backend is unavailable.
"""

from __future__ import annotations

import argparse
import json
import math
import traceback
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Suppress mrinufft's noisy trajectory-rescaling UserWarning.
warnings.filterwarnings(
    "ignore",
    message=".*[Ss]amples will be rescaled.*",
    category=UserWarning,
)

import numpy as np
import torch

from benchmark import profile

from mrinufft import get_operator

from pygrog.calib import GrogInterpolator
from pygrog.operator import SparseFFT

DATASET_FILES = {
    "kspace": "kspace.npy",
    "smaps": "smaps.npy",
    "basis": "basis.npy",
    "trajectory": "trajectory.npy",
    "dcf": "dcf.npy",
}

# Default synthetic size sweep from very small to near-MRF scale.
DEFAULT_SCALING_RATIOS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7]


@dataclass
class BenchmarkConfig:
    shape: tuple[int, ...]
    n_coils: int
    max_coeff: int | None
    n_spokes: int
    n_readout: int
    repeats: int
    warmup: int
    gpu_device: int
    require_cufinufft: bool
    data_dir: str
    scaling_ratios: list[float]
    no_gpu: bool = False


@dataclass
class BenchmarkInputs:
    shape: tuple[int, ...]
    smaps: np.ndarray
    basis_kt: np.ndarray
    kspace_tcns: np.ndarray
    samples: np.ndarray
    density: np.ndarray
    calib_image: np.ndarray
    source: str
    metadata: dict[str, Any]


def _coeff_from_frames(images_tyx: np.ndarray, basis_kt: np.ndarray) -> np.ndarray:
    flat = images_tyx.reshape(images_tyx.shape[0], -1)
    coeff = basis_kt @ flat
    return coeff.reshape(basis_kt.shape[0], *images_tyx.shape[1:]).astype(np.complex64)


def _rss(images_cyx: np.ndarray) -> np.ndarray:
    return np.sqrt((np.abs(images_cyx) ** 2).sum(axis=0))


def _prepare_real_inputs(cfg: BenchmarkConfig, data_dir: Path) -> BenchmarkInputs:
    required = {key: data_dir / filename for key, filename in DATASET_FILES.items()}
    missing = [str(p) for p in required.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing benchmark dataset files in benchmark/data: " + ", ".join(missing)
        )

    kspace_tcns = np.load(required["kspace"]).astype(np.complex64)
    trajectory = np.load(required["trajectory"]).astype(np.float32)
    dcf = np.load(required["dcf"]).astype(np.float32)
    smaps = np.load(required["smaps"]).astype(np.complex64)
    shape = tuple(int(x) for x in smaps.shape[1:])

    if kspace_tcns.ndim != 4:
        raise ValueError(
            f"Expected kspace.npy to have shape (T, C, spokes, readout); got {kspace_tcns.shape}"
        )
    if trajectory.ndim != 4 or trajectory.shape[-1] != len(shape):
        raise ValueError(
            f"Expected trajectory.npy to have shape (T, spokes, readout, {len(shape)}); got {trajectory.shape}"
        )
    if dcf.ndim != 3:
        raise ValueError(
            f"Expected dcf.npy to have shape (T, spokes, readout); got {dcf.shape}"
        )
    if (
        kspace_tcns.shape[0] != trajectory.shape[0]
        or kspace_tcns.shape[0] != dcf.shape[0]
    ):
        raise ValueError(
            "kspace, trajectory, and dcf frame counts do not match: "
            f"{kspace_tcns.shape[0]}, {trajectory.shape[0]}, {dcf.shape[0]}"
        )

    total_frames = kspace_tcns.shape[0]
    n_frames = total_frames
    kspace_tcns = kspace_tcns[:n_frames]
    trajectory = trajectory[:n_frames]
    dcf = dcf[:n_frames]

    basis = np.load(required["basis"]).astype(np.complex64)
    if basis.ndim != 2:
        raise ValueError(f"Expected basis.npy to be 2D, got shape {basis.shape}")

    if basis.shape[0] == total_frames:
        total_coeff = basis.shape[1]
        if cfg.max_coeff is None:
            n_coeff = total_coeff
        else:
            n_coeff = min(max(1, cfg.max_coeff), total_coeff)
        basis_kt = basis[:n_frames, :n_coeff].T
    elif basis.shape[1] == total_frames:
        total_coeff = basis.shape[0]
        if cfg.max_coeff is None:
            n_coeff = total_coeff
        else:
            n_coeff = min(max(1, cfg.max_coeff), total_coeff)
        basis_kt = basis[:n_coeff, :n_frames]
    else:
        raise ValueError(
            f"Unexpected basis.npy shape {basis.shape}; could not align with {total_frames} frames"
        )

    sample_ref = trajectory[0]
    density_ref = dcf[0].reshape(-1)
    variable_traj = not np.allclose(
        trajectory[: min(8, n_frames)], sample_ref[None, ...], atol=1e-6, rtol=1e-5
    )

    return BenchmarkInputs(
        shape=shape,
        smaps=smaps,
        basis_kt=basis_kt.astype(np.complex64),
        kspace_tcns=kspace_tcns,
        samples=sample_ref,
        density=density_ref,
        calib_image=np.ones(shape, dtype=np.float32),
        source="real",
        metadata={
            "data_dir": str(data_dir),
            "dataset_files": DATASET_FILES,
            "total_frames": int(total_frames),
            "used_frames": int(n_frames),
            "total_coeff": int(total_coeff),
            "used_coeff": int(n_coeff),
            "trajectory_varies_across_frames": bool(variable_traj),
        },
    )


def _make_nufft_operator(
    backend: str,
    samples: np.ndarray,
    shape: tuple[int, int],
    smaps: np.ndarray,
    density: np.ndarray,
):
    density_arr = np.asarray(density)
    if density_arr.ndim > 1:
        density_arr = density_arr.reshape(-1)

    return get_operator(backend)(
        samples=samples,
        shape=shape,
        n_coils=smaps.shape[0],
        smaps=smaps,
        density=density_arr,
        squeeze_dims=True,
    )


def _make_grog(
    shape: tuple[int, ...],
    samples: np.ndarray,
    smaps: np.ndarray,
    calib_image: np.ndarray,
) -> tuple[GrogInterpolator, SparseFFT, dict[str, Any]]:
    coords = (samples * np.asarray(shape, dtype=np.float32)).astype(np.float32)

    grog = GrogInterpolator(shape=shape, coords=coords, oversamp=1.25, kernel_width=2, image_shape=shape)

    coil_calib = smaps * calib_image[None, ...]
    fft_axes = tuple(range(-calib_image.ndim, 0))
    calib_cart = np.fft.fftshift(
        np.fft.fftn(np.fft.ifftshift(coil_calib, axes=fft_axes), axes=fft_axes),
        axes=fft_axes,
    ).astype(np.complex64)

    grog.calc_interp_table(calib_cart, lamda=0.01, precision=1)

    fft_plan = grog.fft_plan(image_shape=shape)
    sparse_fft = SparseFFT(plan=fft_plan, smaps=torch.as_tensor(smaps))

    weights = np.asarray(grog.metadata().weights, dtype=np.float32)
    sqrt_weights = np.sqrt(weights).ravel()
    prep = {
        "coords_shape": list(coords.shape),
        "weights_shape": list(weights.shape),
    }
    return grog, sparse_fft, {"sqrt_weights": sqrt_weights, "prep": prep}


def _grog_interpolate_all(
    grog: GrogInterpolator,
    kspace_tcns: np.ndarray,
    sqrt_weights: np.ndarray,
) -> np.ndarray:
    # Important for benchmarking throughput on many frames:
    # use the low-level sparse kernel directly to avoid per-frame
    # tensor conversions and gc.collect() inside the public wrapper.
    kspace_t = torch.as_tensor(kspace_tcns, dtype=torch.complex64)
    sqrt_w_t = torch.as_tensor(sqrt_weights, dtype=torch.float32)

    sparse_frames = []
    for t in range(kspace_t.shape[0]):
        frame_sparse = grog._sparse_full(kspace_t[t])
        frame_sparse = frame_sparse * sqrt_w_t[: frame_sparse.shape[-1]].to(
            frame_sparse.device
        )
        sparse_frames.append(frame_sparse)

    return torch.stack(sparse_frames, dim=0).cpu().numpy()


def _grog_adjoint_from_sparse(
    sparse_tcn: np.ndarray,
    sparse_fft: SparseFFT,
    mode: str,
    gpu_device: int,
) -> np.ndarray:
    imgs = []
    use_cuda = mode in {"gpu-full", "gpu-dual"}
    for t in range(sparse_tcn.shape[0]):
        sparse = torch.as_tensor(sparse_tcn[t], dtype=torch.complex64)
        if use_cuda and mode == "gpu-full":
            sparse = sparse.to(f"cuda:{gpu_device}")
        img = sparse_fft.forward(sparse)
        imgs.append(img.detach().cpu().numpy())
    return np.stack(imgs, axis=0)


def _grog_forward_from_images(
    images_t: np.ndarray,
    sparse_fft: SparseFFT,
    mode: str,
    gpu_device: int,
) -> np.ndarray:
    ksps = []
    use_cuda = mode in {"gpu-full", "gpu-dual"}
    for t in range(images_t.shape[0]):
        img = torch.as_tensor(images_t[t], dtype=torch.complex64)
        if use_cuda and mode == "gpu-full":
            img = img.to(f"cuda:{gpu_device}")
        ksp = sparse_fft.adjoint(img)
        ksps.append(ksp.detach().cpu().numpy())
    return np.stack(ksps, axis=0)


def _nufft_adjoint_all(op, kspace_tcns: np.ndarray) -> np.ndarray:
    imgs = []
    for t in range(kspace_tcns.shape[0]):
        kspace_t = kspace_tcns[t]
        kspace_flat = kspace_t.reshape(kspace_t.shape[0], -1)
        imgs.append(op.adj_op(kspace_flat))
    return np.stack(imgs, axis=0)


def _nufft_forward_all(
    op, images_t: np.ndarray, sample_shape: tuple[int, ...] | None = None
) -> np.ndarray:
    ksps = []
    for t in range(images_t.shape[0]):
        kspace = op.op(images_t[t])
        if sample_shape is not None and getattr(kspace, "ndim", 0) == 2:
            kspace = kspace.reshape(kspace.shape[0], *sample_shape)
        ksps.append(kspace)
    return np.stack(ksps, axis=0)


# ---------------------------------------------------------------------------
# Subspace (MRF) operators — correct algorithm: loop over n_coeff (K << T).
#
#   Adjoint: for i in range(K):
#               y_i = Σ_t  phi[i,t]^*  ·  y_t        (weight k-space / sparse)
#               alpha[i] = E^H(y_i)                   (one NUFFT / SparseFFT)
#
#   Forward: for i in range(K):
#               k_i = E(alpha[i])                     (one NUFFT / SparseFFT)
#               y_t += phi[i,t] · k_i   for all t
#
# K=5 transforms instead of T=500 — 100× fewer IFFT/NUFFT calls.
# Peak memory: O(T · n_coils · n_sparse)  +  O(K · spatial) for GROG;
#              O(n_coils · N)              +  O(K · spatial) for NUFFT.
# ---------------------------------------------------------------------------


def _grog_subspace_adjoint(
    grog: GrogInterpolator,
    kspace_tcns: np.ndarray,
    sqrt_weights: np.ndarray,
    sparse_fft: SparseFFT,
    basis_kt: np.ndarray,  # (n_coeff, n_frames)
    mode: str = "cpu",
    gpu_device: int = 0,
) -> np.ndarray:
    """Subspace adjoint (E Φ)^H y → alpha (n_coeff, *shape).

    Double loop: T GROG interps (can't avoid) + K SparseFFT calls.
    For each frame t, immediately accumulates phi[i,t]^* * sparse_t
    into K pre-allocated buffers — no (T, n_coils, n_sparse) stack in RAM.
    Peak memory: (K+1) × (n_coils, n_sparse) ≈ tens of MB.
    """
    n_coeff, n_frames = basis_kt.shape
    use_cuda = mode in {"gpu-full", "gpu-dual"}
    kspace_t = torch.as_tensor(kspace_tcns, dtype=torch.complex64)
    sqrt_w_t = torch.as_tensor(sqrt_weights, dtype=torch.float32)
    phi = torch.as_tensor(basis_kt, dtype=torch.complex64)  # (K, T)

    # K accumulators — shape allocated on first frame
    accum: list[torch.Tensor | None] = [None] * n_coeff

    for t in range(n_frames):
        sparse_t = grog._sparse_full(kspace_t[t]).cpu()  # (n_coils, n_sparse)
        sparse_t = sparse_t * sqrt_w_t[: sparse_t.shape[-1]]
        for i in range(n_coeff):
            contrib = phi[i, t].conj() * sparse_t  # scalar × tensor
            accum[i] = contrib if accum[i] is None else accum[i] + contrib

    coeff_imgs: list[torch.Tensor] = []
    for i in range(n_coeff):
        y_i = accum[i]
        assert y_i is not None
        if use_cuda:
            y_i = y_i.to(f"cuda:{gpu_device}")
        img_i = sparse_fft.forward(y_i)  # (*shape,)
        coeff_imgs.append(img_i.detach().cpu())

    return torch.stack(coeff_imgs, dim=0).numpy()  # (K, *shape)


def _nufft_subspace_adjoint(
    op,
    kspace_tcns: np.ndarray,
    basis_kt: np.ndarray,  # (n_coeff, n_frames)
) -> np.ndarray:
    """Subspace adjoint (E Φ)^H y → alpha (n_coeff, *shape).

    For each coeff i, weights all frames' k-space by phi[i,:]^* and applies
    a single NUFFT adjoint (using the fixed reference trajectory stored in op).
    Total: K NUFFT adjoint calls.
    """
    n_coeff, n_frames = basis_kt.shape
    n_coils = kspace_tcns.shape[1]
    # (T, n_coils, n_spokes, n_readout) → (T, n_coils, N)
    y = kspace_tcns.reshape(n_frames, n_coils, -1).astype(np.complex64)

    coeff_imgs: list[np.ndarray] = []
    for i in range(n_coeff):
        phi_i = basis_kt[i].conj()  # (T,) complex
        # Weighted sum over frames: (T, 1, 1) * (T, n_coils, N) → (n_coils, N)
        y_i = (phi_i[:, None, None] * y).sum(0)
        img_i = np.asarray(op.adj_op(y_i))  # (*shape,)
        coeff_imgs.append(img_i)

    return np.stack(coeff_imgs, axis=0)  # (K, *shape)


def _grog_subspace_forward(
    sparse_fft: SparseFFT,
    coeff: np.ndarray,  # (n_coeff, *shape)
    basis_kt: np.ndarray,  # (n_coeff, n_frames)
    mode: str = "cpu",
    gpu_device: int = 0,
) -> np.ndarray:
    """Subspace forward E Φ alpha → (n_coeff, n_coils, n_sparse).

    For each coeff i applies one SparseFFT adjoint (image → sparse Cartesian
    k-space).  Returns the per-coefficient sparse outputs (K, n_coils, n_sparse)
    rather than the full (T, n_coils, n_sparse) frame stack: the T-frame
    synthesis (K×T scalar multiplications) is negligible vs K FFT calls.
    Total: K SparseFFT calls.  Peak memory: K × (n_coils, n_sparse) ≈ tens of MB.
    """
    n_coeff = basis_kt.shape[0]
    use_cuda = mode in {"gpu-full", "gpu-dual"}
    coeff_t = torch.as_tensor(coeff, dtype=torch.complex64)

    ksps: list[torch.Tensor] = []
    for i in range(n_coeff):
        img_i = coeff_t[i]
        if use_cuda:
            img_i = img_i.to(f"cuda:{gpu_device}")
        ksp_i = sparse_fft.adjoint(img_i).cpu()  # (n_coils, n_sparse)
        ksps.append(ksp_i)

    return torch.stack(ksps, dim=0).numpy()  # (K, n_coils, n_sparse)


def _nufft_subspace_forward(
    op,
    coeff: np.ndarray,  # (n_coeff, *shape)
    basis_kt: np.ndarray,  # (n_coeff, n_frames)
    sample_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Subspace forward E Φ alpha → (n_coeff, n_coils, n_samples).

    For each coeff i applies one NUFFT forward call (using the fixed reference
    trajectory in op).  Returns per-coeff k-spaces (K, n_coils, N) — the
    T-frame synthesis (K×T scalar mults) is negligible vs K NUFFT calls.
    Total: K NUFFT forward calls.
    """
    n_coeff = basis_kt.shape[0]
    ksps: list[np.ndarray] = []
    for i in range(n_coeff):
        ksp_i = np.asarray(op.op(coeff[i]))  # (n_coils, N)
        if sample_shape is not None and getattr(ksp_i, "ndim", 0) == 2:
            ksp_i = ksp_i.reshape(ksp_i.shape[0], *sample_shape)
        ksps.append(ksp_i)
    return np.stack(ksps, axis=0)  # (K, n_coils, n_samples)


def _safe_backend_available(backend: str) -> tuple[bool, str | None]:
    try:
        _ = get_operator(backend)
        return True, None
    except Exception as exc:
        return False, str(exc)


def _serialize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in metrics.items():
        if isinstance(v, (np.floating, np.integer)):
            out[k] = float(v)
        elif isinstance(v, list):
            out[k] = [float(x) for x in v]
        else:
            out[k] = v
    return out


def _extract_vram_gb(metrics: dict[str, Any]) -> float:
    return float(
        metrics.get("peak_gpu_mem_gb_nvml")
        or metrics.get("peak_gpu_mem_gb_torch")
        or metrics.get("peak_gpu_mem_gb_cupy")
        or 0.0
    )


def _combine_preprocess(
    plan_m: dict[str, Any],
    interp_m: dict[str, Any],
    *,
    include_vram: bool,
) -> dict[str, Any]:
    plan = {
        "runtime_sec": float(plan_m["runtime_mean_sec"]),
        "ram_gb": float(plan_m.get("peak_ram_gb", 0.0)),
        "vram_gb": float(_extract_vram_gb(plan_m)) if include_vram else 0.0,
    }
    interp = {
        "runtime_sec": float(interp_m["runtime_mean_sec"]),
        "ram_gb": float(interp_m.get("peak_ram_gb", 0.0)),
        "vram_gb": float(_extract_vram_gb(interp_m)) if include_vram else 0.0,
    }
    return {
        "planning": plan,
        "interpolation": interp,
        "runtime_sec": float(plan["runtime_sec"] + interp["runtime_sec"]),
        "ram_gb": float(max(plan["ram_gb"], interp["ram_gb"])),
        "vram_gb": float(max(plan["vram_gb"], interp["vram_gb"])),
    }


def _make_synthetic_case(
    *,
    samples_per_frame: int,
    shape: tuple[int, ...],
    n_coils: int,
    n_frames: int,
    n_readout: int,
    ndim: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    samples_per_frame = max(1, int(samples_per_frame))
    n_readout = max(8, int(n_readout))
    n_spokes = math.ceil(samples_per_frame / n_readout)

    # Build a structured radial-like trajectory so synthetic scaling resembles
    # real non-Cartesian workloads better than fully i.i.d. random points.
    radii = np.linspace(-0.5, 0.5, n_readout, dtype=np.float32)
    directions = rng.standard_normal(size=(n_spokes, ndim)).astype(np.float32)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True) + 1e-8
    trajectory = directions[:, None, :] * radii[None, :, None]
    trajectory = trajectory.astype(np.float32)
    dcf = np.ones((n_spokes, n_readout), dtype=np.float32)

    kspace = (
        rng.standard_normal(size=(n_frames, n_coils, n_spokes, n_readout)).astype(
            np.float32
        )
        + 1j
        * rng.standard_normal(size=(n_frames, n_coils, n_spokes, n_readout)).astype(
            np.float32
        )
    ).astype(np.complex64)
    smaps = (
        rng.standard_normal(size=(n_coils, *shape)).astype(np.float32)
        + 1j * rng.standard_normal(size=(n_coils, *shape)).astype(np.float32)
    ).astype(np.complex64)
    smaps /= np.sqrt(np.sum(np.abs(smaps) ** 2, axis=0, keepdims=True) + 1e-6)

    return {
        "shape": shape,
        "smaps": smaps,
        "samples": trajectory,
        "density": dcf.reshape(-1),
        "kspace": kspace,
        "calib_image": np.ones(shape, dtype=np.float32),
        "samples_per_frame": int(n_spokes * n_readout),
    }


def _benchmark_scaling_case(
    *,
    label: str,
    kind: str,
    case: dict[str, Any],
    cfg: BenchmarkConfig,
    cufinufft_ok: bool,
) -> dict[str, Any]:
    shape = tuple(int(x) for x in case["shape"])
    smaps = np.asarray(case["smaps"], dtype=np.complex64)
    samples = np.asarray(case["samples"], dtype=np.float32)
    density = np.asarray(case["density"], dtype=np.float32)
    kspace_tcns = np.asarray(case["kspace"], dtype=np.complex64)
    calib_image = np.asarray(case["calib_image"], dtype=np.float32)

    nufft_finufft = _make_nufft_operator("finufft", samples, shape, smaps, density)

    _, plan_cpu_m = profile(
        _make_grog,
        shape,
        samples,
        smaps,
        calib_image,
        warmup=0,
        repeat=max(1, cfg.repeats),
        gpu_device=None,
    )
    grog, sparse_fft_cpu, grog_aux = _make_grog(shape, samples, smaps, calib_image)

    sparse_tcns_cpu, interp_cpu_m = profile(
        _grog_interpolate_all,
        grog,
        kspace_tcns,
        grog_aux["sqrt_weights"],
        warmup=cfg.warmup,
        repeat=cfg.repeats,
        gpu_device=None,
    )

    prep_cpu = _combine_preprocess(plan_cpu_m, interp_cpu_m, include_vram=False)

    cuda_available = torch.cuda.is_available() and not cfg.no_gpu

    prep_gpu = None
    if cuda_available:
        _, plan_gpu_m = profile(
            _make_grog,
            shape,
            samples,
            smaps,
            calib_image,
            warmup=0,
            repeat=max(1, cfg.repeats),
            gpu_device=cfg.gpu_device,
        )
        _, interp_gpu_m = profile(
            _grog_interpolate_all,
            grog,
            kspace_tcns,
            grog_aux["sqrt_weights"],
            warmup=cfg.warmup,
            repeat=cfg.repeats,
            gpu_device=cfg.gpu_device,
        )
        prep_gpu = _combine_preprocess(plan_gpu_m, interp_gpu_m, include_vram=True)

    nufft_adj_cpu, nufft_adj_cpu_m = profile(
        _nufft_adjoint_all,
        nufft_finufft,
        kspace_tcns,
        warmup=cfg.warmup,
        repeat=cfg.repeats,
        gpu_device=None,
    )
    _, nufft_fwd_cpu_m = profile(
        _nufft_forward_all,
        nufft_finufft,
        np.asarray(nufft_adj_cpu),
        samples.shape[:-1],
        warmup=cfg.warmup,
        repeat=cfg.repeats,
        gpu_device=None,
    )
    _, grog_adj_cpu_m = profile(
        _grog_adjoint_from_sparse,
        sparse_tcns_cpu,
        sparse_fft_cpu,
        "cpu",
        cfg.gpu_device,
        warmup=cfg.warmup,
        repeat=cfg.repeats,
        gpu_device=None,
    )
    _, grog_fwd_cpu_m = profile(
        _grog_forward_from_images,
        np.asarray(nufft_adj_cpu),
        sparse_fft_cpu,
        "cpu",
        cfg.gpu_device,
        warmup=cfg.warmup,
        repeat=cfg.repeats,
        gpu_device=None,
    )

    nufft_adj_gpu_m = None
    nufft_fwd_gpu_m = None
    grog_adj_gpu_m = None
    grog_fwd_gpu_m = None
    if cuda_available and cufinufft_ok:
        nufft_cuf = _make_nufft_operator("cufinufft", samples, shape, smaps, density)
        _, nufft_adj_gpu_m = profile(
            _nufft_adjoint_all,
            nufft_cuf,
            kspace_tcns,
            warmup=cfg.warmup,
            repeat=cfg.repeats,
            gpu_device=cfg.gpu_device,
        )
        _, nufft_fwd_gpu_m = profile(
            _nufft_forward_all,
            nufft_cuf,
            np.asarray(nufft_adj_cpu),
            samples.shape[:-1],
            warmup=cfg.warmup,
            repeat=cfg.repeats,
            gpu_device=cfg.gpu_device,
        )

        sparse_fft_gpu = SparseFFT(
            plan=grog.fft_plan(image_shape=shape),
            smaps=torch.as_tensor(smaps).to(f"cuda:{cfg.gpu_device}"),
            device=f"cuda:{cfg.gpu_device}",
        )
        _, grog_adj_gpu_m = profile(
            _grog_adjoint_from_sparse,
            sparse_tcns_cpu,
            sparse_fft_gpu,
            "gpu-full",
            cfg.gpu_device,
            warmup=cfg.warmup,
            repeat=cfg.repeats,
            gpu_device=cfg.gpu_device,
        )
        _, grog_fwd_gpu_m = profile(
            _grog_forward_from_images,
            np.asarray(nufft_adj_cpu),
            sparse_fft_gpu,
            "gpu-full",
            cfg.gpu_device,
            warmup=cfg.warmup,
            repeat=cfg.repeats,
            gpu_device=cfg.gpu_device,
        )

    def _pack(
        metrics: dict[str, Any] | None, *, is_gpu: bool
    ) -> dict[str, float | None]:
        if metrics is None:
            return {"runtime_sec": None, "ram_gb": None, "vram_gb": None}
        return {
            "runtime_sec": float(metrics["runtime_mean_sec"]),
            "ram_gb": float(metrics.get("peak_ram_gb", 0.0)),
            "vram_gb": float(_extract_vram_gb(metrics)) if is_gpu else 0.0,
        }

    return {
        "label": label,
        "kind": kind,
        "samples_per_frame": int(case["samples_per_frame"]),
        "shape": list(shape),
        "n_coils": int(smaps.shape[0]),
        "preprocessing": {
            "cpu": prep_cpu,
            "gpu": prep_gpu,
        },
        "linop": {
            "forward": {
                "finufft_cpu": _pack(nufft_fwd_cpu_m, is_gpu=False),
                "grog_cpu": _pack(grog_fwd_cpu_m, is_gpu=False),
                "cufinufft_gpu": _pack(nufft_fwd_gpu_m, is_gpu=True),
                "grog_gpu": _pack(grog_fwd_gpu_m, is_gpu=True),
            },
            "adjoint": {
                "finufft_cpu": _pack(nufft_adj_cpu_m, is_gpu=False),
                "grog_cpu": _pack(grog_adj_cpu_m, is_gpu=False),
                "cufinufft_gpu": _pack(nufft_adj_gpu_m, is_gpu=True),
                "grog_gpu": _pack(grog_adj_gpu_m, is_gpu=True),
            },
        },
    }


def _build_scaling_suite(
    cfg: BenchmarkConfig,
    inputs: BenchmarkInputs,
    cufinufft_ok: bool,
) -> dict[str, Any]:
    rng = np.random.default_rng(12345)

    real_samples_per_frame = int(np.prod(inputs.samples.shape[:-1]))
    ndim_synth = len(inputs.shape)
    n_frames_synth = max(1, min(2, int(inputs.kspace_tcns.shape[0])))

    cases = []
    for ratio in sorted({float(r) for r in cfg.scaling_ratios if float(r) > 0.0}):
        target_samples = int(max(64, round(real_samples_per_frame * ratio)))
        synth_case = _make_synthetic_case(
            samples_per_frame=target_samples,
            shape=tuple(inputs.shape),
            n_coils=int(inputs.smaps.shape[0]),
            n_frames=n_frames_synth,
            n_readout=int(cfg.n_readout),
            ndim=ndim_synth,
            rng=rng,
        )
        cases.append(
            _benchmark_scaling_case(
                label=f"Synth-{target_samples // 1000}k",
                kind="synthetic",
                case=synth_case,
                cfg=cfg,
                cufinufft_ok=cufinufft_ok,
            )
        )

    synth_match_case = _make_synthetic_case(
        samples_per_frame=real_samples_per_frame,
        shape=tuple(inputs.shape),
        n_coils=int(inputs.smaps.shape[0]),
        n_frames=n_frames_synth,
        n_readout=int(cfg.n_readout),
        ndim=ndim_synth,
        rng=rng,
    )
    cases.append(
        _benchmark_scaling_case(
            label="Synth-MRF-size",
            kind="synthetic_mrf_match",
            case=synth_match_case,
            cfg=cfg,
            cufinufft_ok=cufinufft_ok,
        )
    )

    real_case = {
        "shape": inputs.shape,
        "smaps": inputs.smaps,
        "samples": inputs.samples,
        "density": inputs.density,
        "kspace": inputs.kspace_tcns[:n_frames_synth],
        "calib_image": inputs.calib_image,
        "samples_per_frame": real_samples_per_frame,
    }
    cases.append(
        _benchmark_scaling_case(
            label="MRF-real",
            kind="real_mrf",
            case=real_case,
            cfg=cfg,
            cufinufft_ok=cufinufft_ok,
        )
    )

    return {
        "ratios": [float(r) for r in cfg.scaling_ratios],
        "real_samples_per_frame": real_samples_per_frame,
        "scaling_frames": n_frames_synth,
        "cases": cases,
    }


def run(cfg: BenchmarkConfig, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = _prepare_real_inputs(cfg, Path(cfg.data_dir))

    results: dict[str, Any] = {
        "config": asdict(cfg),
        "data_source": inputs.source,
        "input": {
            "shape": list(inputs.shape),
            "n_coils": int(inputs.smaps.shape[0]),
            "n_frames": int(inputs.kspace_tcns.shape[0]),
            "n_coeff": int(inputs.basis_kt.shape[0]),
            **inputs.metadata,
        },
        "environment": {
            "torch_cuda_available": bool(torch.cuda.is_available()),
            "torch_cuda_device_count": (
                int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
            ),
        },
        "steps": {},
    }

    finufft_ok, finufft_err = _safe_backend_available("finufft")
    if not finufft_ok:
        raise RuntimeError(
            f"FINUFFT backend is required for this benchmark: {finufft_err}"
        )
    cufinufft_ok, cufinufft_err = _safe_backend_available("cufinufft")

    nufft_sim = _make_nufft_operator(
        "finufft", inputs.samples, inputs.shape, inputs.smaps, inputs.density
    )
    basis_kt = inputs.basis_kt
    kspace_tcns = inputs.kspace_tcns.astype(np.complex64)

    # GROG plan creation
    _cuda_available = torch.cuda.is_available() and not cfg.no_gpu
    _, plan_metrics = profile(
        _make_grog,
        inputs.shape,
        inputs.samples,
        inputs.smaps,
        inputs.calib_image,
        warmup=0,
        repeat=max(1, cfg.repeats),
        gpu_device=cfg.gpu_device if _cuda_available else None,
    )
    grog, sparse_fft_cpu, grog_aux = _make_grog(
        inputs.shape,
        inputs.samples,
        inputs.smaps,
        inputs.calib_image,
    )
    results["steps"]["grog_plan_creation"] = _serialize_metrics(plan_metrics)
    results["steps"]["grog_plan_creation"]["prep"] = grog_aux["prep"]

    # Subspace coefficient comparison: NUFFT vs GROG.
    # Both accumulate (n_coeff, *shape) by looping over frames — no (n_frames, *shape) stack.
    coeff_nufft, _ = profile(
        _nufft_subspace_adjoint,
        nufft_sim,
        kspace_tcns,
        basis_kt,
        warmup=cfg.warmup,
        repeat=1,
        gpu_device=None,
    )
    coeff_grog_cpu, _ = profile(
        _grog_subspace_adjoint,
        grog,
        kspace_tcns,
        grog_aux["sqrt_weights"],
        sparse_fft_cpu,
        basis_kt,
        "cpu",
        cfg.gpu_device,
        warmup=cfg.warmup,
        repeat=1,
        gpu_device=None,
    )

    coeff_nufft = np.asarray(coeff_nufft)
    coeff_grog_cpu = np.asarray(coeff_grog_cpu)

    coeff_rel = float(
        np.linalg.norm(coeff_nufft - coeff_grog_cpu) / (np.linalg.norm(coeff_nufft) + 1e-8)
    )
    coeff_corr = float(
        np.corrcoef(np.abs(coeff_nufft).ravel(), np.abs(coeff_grog_cpu).ravel())[0, 1]
    )

    np.save(output_dir / "coeff_nufft.npy", coeff_nufft)
    np.save(output_dir / "coeff_grog.npy", coeff_grog_cpu)
    results["steps"]["subspace_comparison"] = {
        "rel_l2_error": coeff_rel,
        "abs_corrcoef": coeff_corr,
    }

    # Runtime + memory comparisons — subspace adjoint/forward
    # Adjoint: (E Φ)^H y → alpha (n_coeff, *shape)
    # Forward: E Φ alpha → per-frame k-space
    # CPU
    _, nufft_adj_cpu_m = profile(
        _nufft_subspace_adjoint,
        nufft_sim,
        kspace_tcns,
        basis_kt,
        warmup=cfg.warmup,
        repeat=cfg.repeats,
        gpu_device=None,
    )
    _, nufft_fwd_cpu_m = profile(
        _nufft_subspace_forward,
        nufft_sim,
        coeff_nufft,
        basis_kt,
        inputs.samples.shape[:-1],
        warmup=cfg.warmup,
        repeat=cfg.repeats,
        gpu_device=None,
    )

    _, grog_adj_cpu_m = profile(
        _grog_subspace_adjoint,
        grog,
        kspace_tcns,
        grog_aux["sqrt_weights"],
        sparse_fft_cpu,
        basis_kt,
        "cpu",
        cfg.gpu_device,
        warmup=cfg.warmup,
        repeat=cfg.repeats,
        gpu_device=None,
    )
    _, grog_fwd_cpu_m = profile(
        _grog_subspace_forward,
        sparse_fft_cpu,
        coeff_nufft,
        basis_kt,
        "cpu",
        cfg.gpu_device,
        warmup=cfg.warmup,
        repeat=cfg.repeats,
        gpu_device=None,
    )

    results["steps"]["runtime_cpu"] = {
        "nufft_finufft_adjoint": _serialize_metrics(nufft_adj_cpu_m),
        "nufft_finufft_forward": _serialize_metrics(nufft_fwd_cpu_m),
        "grog_adjoint": _serialize_metrics(grog_adj_cpu_m),
        "grog_forward": _serialize_metrics(grog_fwd_cpu_m),
    }

    # GPU full and dual-stream modes
    gpu_results: dict[str, Any] = {}
    _gpu_sparse_fft: SparseFFT | None = None
    _gpu_nufft = None
    if _cuda_available:
        try:
            sparse_fft_gpu_full = SparseFFT(
                plan=grog.fft_plan(image_shape=inputs.shape),
                smaps=torch.as_tensor(inputs.smaps).to(f"cuda:{cfg.gpu_device}"),
                device=f"cuda:{cfg.gpu_device}",
            )
            _gpu_sparse_fft = sparse_fft_gpu_full
            _, grog_adj_gpu_full_m = profile(
                _grog_subspace_adjoint,
                grog,
                kspace_tcns,
                grog_aux["sqrt_weights"],
                sparse_fft_gpu_full,
                basis_kt,
                "gpu-full",
                cfg.gpu_device,
                warmup=cfg.warmup,
                repeat=cfg.repeats,
                gpu_device=cfg.gpu_device,
            )
            _, grog_fwd_gpu_full_m = profile(
                _grog_subspace_forward,
                sparse_fft_gpu_full,
                coeff_nufft,
                basis_kt,
                "gpu-full",
                cfg.gpu_device,
                warmup=cfg.warmup,
                repeat=cfg.repeats,
                gpu_device=cfg.gpu_device,
            )
            gpu_results["grog_full_gpu"] = {
                "grog_adjoint": _serialize_metrics(grog_adj_gpu_full_m),
                "grog_forward": _serialize_metrics(grog_fwd_gpu_full_m),
            }

            sparse_fft_gpu_dual = SparseFFT(
                plan=grog.fft_plan(image_shape=inputs.shape),
                smaps=torch.as_tensor(inputs.smaps),
                device=f"cuda:{cfg.gpu_device}",
            )
            _, grog_adj_gpu_dual_m = profile(
                _grog_subspace_adjoint,
                grog,
                kspace_tcns,
                grog_aux["sqrt_weights"],
                sparse_fft_gpu_dual,
                basis_kt,
                "gpu-dual",
                cfg.gpu_device,
                warmup=cfg.warmup,
                repeat=cfg.repeats,
                gpu_device=cfg.gpu_device,
            )
            _, grog_fwd_gpu_dual_m = profile(
                _grog_subspace_forward,
                sparse_fft_gpu_dual,
                coeff_nufft,
                basis_kt,
                "gpu-dual",
                cfg.gpu_device,
                warmup=cfg.warmup,
                repeat=cfg.repeats,
                gpu_device=cfg.gpu_device,
            )
            gpu_results["grog_dual_stream_gpu"] = {
                "grog_adjoint": _serialize_metrics(grog_adj_gpu_dual_m),
                "grog_forward": _serialize_metrics(grog_fwd_gpu_dual_m),
            }
        except Exception as exc:
            gpu_results["grog_gpu_skipped"] = {"reason": str(exc)}

    if _cuda_available and cufinufft_ok:
        try:
            nufft_gpu = _make_nufft_operator(
                "cufinufft", inputs.samples, inputs.shape, inputs.smaps, inputs.density
            )
            _gpu_nufft = nufft_gpu
            _, nufft_adj_gpu_m = profile(
                _nufft_subspace_adjoint,
                nufft_gpu,
                kspace_tcns,
                basis_kt,
                warmup=cfg.warmup,
                repeat=cfg.repeats,
                gpu_device=cfg.gpu_device,
            )
            _, nufft_fwd_gpu_m = profile(
                _nufft_subspace_forward,
                nufft_gpu,
                coeff_nufft,
                basis_kt,
                inputs.samples.shape[:-1],
                warmup=cfg.warmup,
                repeat=cfg.repeats,
                gpu_device=cfg.gpu_device,
            )
            gpu_results["nufft_cufinufft_gpu"] = {
                "nufft_adjoint": _serialize_metrics(nufft_adj_gpu_m),
                "nufft_forward": _serialize_metrics(nufft_fwd_gpu_m),
            }
        except Exception as exc:
            if cfg.require_cufinufft:
                raise RuntimeError(
                    "CUFINUFFT benchmark is required but failed to run: " f"{exc}"
                ) from exc
            gpu_results["nufft_cufinufft_gpu_skipped"] = {"reason": str(exc)}
    else:
        if cfg.no_gpu:
            reason = "GPU skipped (--no-gpu)"
        elif not torch.cuda.is_available():
            reason = "CUDA not available"
        else:
            reason = cufinufft_err
        if cfg.require_cufinufft:
            raise RuntimeError(
                "CUFINUFFT benchmark is required but unavailable: " f"{reason}"
            )
        gpu_results["nufft_cufinufft_gpu_skipped"] = {"reason": reason}

    # Keep GPU comparisons fair: only report GPU GROG if GPU NUFFT is available.
    if "nufft_cufinufft_gpu" not in gpu_results and (
        "grog_full_gpu" in gpu_results or "grog_dual_stream_gpu" in gpu_results
    ):
        grog_full = gpu_results.pop("grog_full_gpu", None)
        grog_dual = gpu_results.pop("grog_dual_stream_gpu", None)
        gpu_results["grog_gpu_skipped"] = {
            "reason": (
                "Skipped for parity because CUFINUFFT GPU results are unavailable."
            ),
            "suppressed_grog_full_gpu": bool(grog_full is not None),
            "suppressed_grog_dual_stream_gpu": bool(grog_dual is not None),
        }

    results["steps"]["runtime_gpu"] = gpu_results

    # CUDA subspace comparison (only when GPU NUFFT + GROG both available)
    if _gpu_sparse_fft is not None and _gpu_nufft is not None:
        try:
            coeff_nufft_cuda = np.asarray(
                _nufft_subspace_adjoint(_gpu_nufft, kspace_tcns, basis_kt)
            )
            coeff_grog_cuda = _grog_subspace_adjoint(
                grog, kspace_tcns, grog_aux["sqrt_weights"],
                _gpu_sparse_fft, basis_kt, "gpu-full", cfg.gpu_device
            )
            np.save(output_dir / "coeff_nufft_cuda.npy", coeff_nufft_cuda)
            np.save(output_dir / "coeff_grog_cuda.npy", coeff_grog_cuda)
            coeff_cuda_rel = float(
                np.linalg.norm(coeff_nufft_cuda - coeff_grog_cuda)
                / (np.linalg.norm(coeff_nufft_cuda) + 1e-8)
            )
            coeff_cuda_corr = float(
                np.corrcoef(
                    np.abs(coeff_nufft_cuda).ravel(), np.abs(coeff_grog_cuda).ravel()
                )[0, 1]
            )
            results["steps"]["subspace_comparison"]["cuda"] = {
                "rel_l2_error": coeff_cuda_rel,
                "abs_corrcoef": coeff_cuda_corr,
            }
        except Exception as exc:
            results["steps"]["subspace_comparison"]["cuda_skipped"] = {
                "reason": str(exc)
            }

    results["scaling"] = _build_scaling_suite(cfg, inputs, cufinufft_ok)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark/results"))
    parser.add_argument("--data-dir", type=Path, default=Path("benchmark/data"))
    parser.add_argument("--shape", type=int, nargs="+", default=(220, 220, 220))
    parser.add_argument("--n-coils", type=int, default=8)
    parser.add_argument(
        "--max-coeff",
        type=int,
        default=None,
        help="Optional cap on number of basis coefficients to use. Default: use full basis rank.",
    )
    parser.add_argument("--n-spokes", type=int, default=48)
    parser.add_argument("--n-readout", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        default=False,
        help="Skip all GPU benchmarks even when CUDA is available.",
    )
    parser.add_argument(
        "--require-cufinufft",
        action="store_true",
        help="Fail if CUFINUFFT GPU NUFFT benchmark cannot run.",
    )
    parser.add_argument(
        "--scaling-ratios",
        type=str,
        default=",".join(str(v) for v in DEFAULT_SCALING_RATIOS),
        help=(
            "Comma-separated synthetic problem-size ratios relative to MRF samples/frame. "
            "A synthetic MRF-sized case and the real MRF case are appended automatically."
        ),
    )
    return parser.parse_args()


def _parse_scaling_ratios(raw: str) -> list[float]:
    vals: list[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        v = float(token)
        if v > 0.0:
            vals.append(v)
    if not vals:
        raise ValueError("--scaling-ratios must include at least one positive number")
    return vals


def main() -> None:
    args = parse_args()
    scaling_ratios = _parse_scaling_ratios(args.scaling_ratios)
    cfg = BenchmarkConfig(
        shape=tuple(args.shape),
        n_coils=args.n_coils,
        max_coeff=args.max_coeff,
        n_spokes=args.n_spokes,
        n_readout=args.n_readout,
        repeats=args.repeats,
        warmup=args.warmup,
        gpu_device=args.gpu_device,
        require_cufinufft=args.require_cufinufft,
        data_dir=str(args.data_dir),
        scaling_ratios=scaling_ratios,
        no_gpu=args.no_gpu,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        results = run(cfg, output_dir)
        with (output_dir / "results.json").open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved benchmark results to {output_dir / 'results.json'}")
    except Exception as exc:
        err = {
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "config": asdict(cfg),
        }
        with (output_dir / "results_error.json").open("w", encoding="utf-8") as f:
            json.dump(err, f, indent=2)
        raise


if __name__ == "__main__":
    main()
