#!/usr/bin/env python3
"""Unified benchmark runner for PyGROG vs FINUFFT/CUFINUFFT.

This script is designed to run on CPU-only laptops and on GPU servers
without code changes. GPU benchmarks are skipped automatically when the
required hardware/backend is unavailable.
"""

from __future__ import annotations

import argparse
import json
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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


@dataclass
class BenchmarkConfig:
    shape: tuple[int, ...]
    n_coils: int
    n_frames: int
    n_coeff: int
    n_spokes: int
    n_readout: int
    repeats: int
    warmup: int
    gpu_device: int
    data_dir: str


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
    required = {
        key: data_dir / filename for key, filename in DATASET_FILES.items()
    }
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
        raise ValueError(f"Expected kspace.npy to have shape (T, C, spokes, readout); got {kspace_tcns.shape}")
    if trajectory.ndim != 4 or trajectory.shape[-1] != len(shape):
        raise ValueError(
            f"Expected trajectory.npy to have shape (T, spokes, readout, {len(shape)}); got {trajectory.shape}"
        )
    if dcf.ndim != 3:
        raise ValueError(f"Expected dcf.npy to have shape (T, spokes, readout); got {dcf.shape}")
    if kspace_tcns.shape[0] != trajectory.shape[0] or kspace_tcns.shape[0] != dcf.shape[0]:
        raise ValueError(
            "kspace, trajectory, and dcf frame counts do not match: "
            f"{kspace_tcns.shape[0]}, {trajectory.shape[0]}, {dcf.shape[0]}"
        )

    total_frames = kspace_tcns.shape[0]
    n_frames = min(cfg.n_frames, total_frames)
    kspace_tcns = kspace_tcns[:n_frames]
    trajectory = trajectory[:n_frames]
    dcf = dcf[:n_frames]

    basis = np.load(required["basis"]).astype(np.complex64)
    if basis.ndim != 2:
        raise ValueError(f"Expected basis.npy to be 2D, got shape {basis.shape}")
    if basis.shape[0] == total_frames:
        basis_kt = basis[:n_frames, : cfg.n_coeff].T
    elif basis.shape[1] == total_frames:
        basis_kt = basis[: cfg.n_coeff, :n_frames]
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

    grog = GrogInterpolator(shape=shape, coords=coords, image_shape=shape)

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
            "torch_cuda_device_count": int(torch.cuda.device_count())
            if torch.cuda.is_available()
            else 0,
        },
        "steps": {},
    }

    finufft_ok, finufft_err = _safe_backend_available("finufft")
    if not finufft_ok:
        raise RuntimeError(f"FINUFFT backend is required for this benchmark: {finufft_err}")

    nufft_sim = _make_nufft_operator(
        "finufft", inputs.samples, inputs.shape, inputs.smaps, inputs.density
    )
    basis_kt = inputs.basis_kt
    kspace_tcns = inputs.kspace_tcns.astype(np.complex64)

    # GROG plan creation
    _, plan_metrics = profile(
        _make_grog,
        inputs.shape,
        inputs.samples,
        inputs.smaps,
        inputs.calib_image,
        warmup=0,
        repeat=max(1, cfg.repeats),
        gpu_device=cfg.gpu_device if torch.cuda.is_available() else None,
    )
    grog, sparse_fft_cpu, grog_aux = _make_grog(
        inputs.shape,
        inputs.samples,
        inputs.smaps,
        inputs.calib_image,
    )
    results["steps"]["grog_plan_creation"] = _serialize_metrics(plan_metrics)
    results["steps"]["grog_plan_creation"]["prep"] = grog_aux["prep"]

    # GROG interpolation runtime
    sparse_tcns, interp_metrics = profile(
        _grog_interpolate_all,
        grog,
        kspace_tcns,
        grog_aux["sqrt_weights"],
        warmup=cfg.warmup,
        repeat=cfg.repeats,
        gpu_device=cfg.gpu_device if torch.cuda.is_available() else None,
    )
    results["steps"]["grog_interpolation"] = _serialize_metrics(interp_metrics)

    # Subspace coefficient comparison: NUFFT adjoint vs GROG (interp + SparseFFT forward)
    imgs_nufft, _ = profile(
        _nufft_adjoint_all,
        nufft_sim,
        kspace_tcns,
        warmup=cfg.warmup,
        repeat=1,
        gpu_device=None,
    )
    imgs_grog_cpu, _ = profile(
        _grog_adjoint_from_sparse,
        sparse_tcns,
        sparse_fft_cpu,
        "cpu",
        cfg.gpu_device,
        warmup=cfg.warmup,
        repeat=1,
        gpu_device=None,
    )

    coeff_nufft = _coeff_from_frames(np.asarray(imgs_nufft), basis_kt)
    coeff_grog = _coeff_from_frames(np.asarray(imgs_grog_cpu), basis_kt)

    coeff_rel = float(
        np.linalg.norm(coeff_nufft - coeff_grog)
        / (np.linalg.norm(coeff_nufft) + 1e-8)
    )
    coeff_corr = float(
        np.corrcoef(np.abs(coeff_nufft).ravel(), np.abs(coeff_grog).ravel())[0, 1]
    )

    np.save(output_dir / "coeff_nufft.npy", coeff_nufft)
    np.save(output_dir / "coeff_grog.npy", coeff_grog)
    results["steps"]["subspace_comparison"] = {
        "rel_l2_error": coeff_rel,
        "abs_corrcoef": coeff_corr,
    }

    # Runtime + memory comparisons
    # CPU
    nufft_adj_cpu, nufft_adj_cpu_m = profile(
        _nufft_adjoint_all,
        nufft_sim,
        kspace_tcns,
        warmup=cfg.warmup,
        repeat=cfg.repeats,
        gpu_device=None,
    )
    _, nufft_fwd_cpu_m = profile(
        _nufft_forward_all,
        nufft_sim,
        np.asarray(nufft_adj_cpu),
        inputs.samples.shape[:-1],
        warmup=cfg.warmup,
        repeat=cfg.repeats,
        gpu_device=None,
    )

    _, grog_adj_cpu_m = profile(
        _grog_adjoint_from_sparse,
        sparse_tcns,
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

    results["steps"]["runtime_cpu"] = {
        "nufft_finufft_adjoint": _serialize_metrics(nufft_adj_cpu_m),
        "nufft_finufft_forward": _serialize_metrics(nufft_fwd_cpu_m),
        "grog_adjoint": _serialize_metrics(grog_adj_cpu_m),
        "grog_forward": _serialize_metrics(grog_fwd_cpu_m),
    }

    # GPU full and dual-stream modes
    gpu_results: dict[str, Any] = {}
    if torch.cuda.is_available():
        try:
            sparse_fft_gpu_full = SparseFFT(
                plan=grog.fft_plan(image_shape=inputs.shape),
                smaps=torch.as_tensor(inputs.smaps).to(f"cuda:{cfg.gpu_device}"),
                device=f"cuda:{cfg.gpu_device}",
            )
            _, grog_adj_gpu_full_m = profile(
                _grog_adjoint_from_sparse,
                sparse_tcns,
                sparse_fft_gpu_full,
                "gpu-full",
                cfg.gpu_device,
                warmup=cfg.warmup,
                repeat=cfg.repeats,
                gpu_device=cfg.gpu_device,
            )
            _, grog_fwd_gpu_full_m = profile(
                _grog_forward_from_images,
                np.asarray(nufft_adj_cpu),
                sparse_fft_gpu_full,
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
                _grog_adjoint_from_sparse,
                sparse_tcns,
                sparse_fft_gpu_dual,
                "gpu-dual",
                cfg.gpu_device,
                warmup=cfg.warmup,
                repeat=cfg.repeats,
                gpu_device=cfg.gpu_device,
            )
            _, grog_fwd_gpu_dual_m = profile(
                _grog_forward_from_images,
                np.asarray(nufft_adj_cpu),
                sparse_fft_gpu_dual,
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

    cufinufft_ok, cufinufft_err = _safe_backend_available("cufinufft")
    if torch.cuda.is_available() and cufinufft_ok:
        try:
            nufft_gpu = _make_nufft_operator(
                "cufinufft", inputs.samples, inputs.shape, inputs.smaps, inputs.density
            )
            _, nufft_adj_gpu_m = profile(
                _nufft_adjoint_all,
                nufft_gpu,
                kspace_tcns,
                warmup=cfg.warmup,
                repeat=cfg.repeats,
                gpu_device=cfg.gpu_device,
            )
            _, nufft_fwd_gpu_m = profile(
                _nufft_forward_all,
                nufft_gpu,
                np.asarray(nufft_adj_cpu),
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
            gpu_results["nufft_cufinufft_gpu_skipped"] = {"reason": str(exc)}
    else:
        reason = "CUDA not available" if not torch.cuda.is_available() else cufinufft_err
        gpu_results["nufft_cufinufft_gpu_skipped"] = {"reason": reason}

    results["steps"]["runtime_gpu"] = gpu_results

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark/results"))
    parser.add_argument("--data-dir", type=Path, default=Path("benchmark/data"))
    parser.add_argument("--shape", type=int, nargs="+", default=(160, 160))
    parser.add_argument("--n-coils", type=int, default=8)
    parser.add_argument("--n-frames", type=int, default=64)
    parser.add_argument("--n-coeff", type=int, default=5)
    parser.add_argument("--n-spokes", type=int, default=48)
    parser.add_argument("--n-readout", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--gpu-device", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = BenchmarkConfig(
        shape=tuple(args.shape),
        n_coils=args.n_coils,
        n_frames=args.n_frames,
        n_coeff=args.n_coeff,
        n_spokes=args.n_spokes,
        n_readout=args.n_readout,
        repeats=args.repeats,
        warmup=args.warmup,
        gpu_device=args.gpu_device,
        data_dir=str(args.data_dir),
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
