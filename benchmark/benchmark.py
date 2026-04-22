"""Benchmark profiling helpers for runtime and memory usage."""

from __future__ import annotations

__all__ = ["profile"]

import statistics
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

try:
    import psutil
except Exception:  # pragma: no cover - optional dependency
    psutil = None

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None

try:
    import cupy as cp
except Exception:  # pragma: no cover - optional dependency
    cp = None

try:
    import pynvml
except Exception:  # pragma: no cover - optional dependency
    pynvml = None


@dataclass
class _Peaks:
    cpu_percent: float = 0.0
    ram_bytes: int = 0
    gpu_mem_bytes: int = 0


def _sync_cuda() -> None:
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
    if cp is not None:
        try:
            cp.cuda.runtime.deviceSynchronize()
        except Exception:
            pass


def _monitor_resources(
    stop_event: threading.Event,
    peaks: _Peaks,
    sample_interval_sec: float,
    gpu_device: int | None,
) -> None:
    proc = psutil.Process() if psutil is not None else None
    if proc is not None:
        proc.cpu_percent(interval=None)

    nvml_handle = None
    if gpu_device is not None and pynvml is not None:
        try:
            pynvml.nvmlInit()
            nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_device)
        except Exception:
            nvml_handle = None

    while not stop_event.is_set():
        if proc is not None:
            cpu = proc.cpu_percent(interval=None)
            rss = proc.memory_info().rss
            peaks.cpu_percent = max(peaks.cpu_percent, cpu)
            peaks.ram_bytes = max(peaks.ram_bytes, rss)

        if nvml_handle is not None:
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(nvml_handle)
                peaks.gpu_mem_bytes = max(peaks.gpu_mem_bytes, mem.used)
            except Exception:
                pass

        time.sleep(sample_interval_sec)

    if nvml_handle is not None:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def profile(
    func: Callable[..., Any],
    *args: Any,
    warmup: int = 1,
    repeat: int = 3,
    sample_interval_sec: float = 0.05,
    gpu_device: int | None = None,
    sync_cuda: bool = True,
    **kwargs: Any,
) -> tuple[Any, dict[str, Any]]:
    """Profile function runtime and peak memory.

    Parameters
    ----------
    func : Callable
        Function to profile.
    warmup : int, optional
        Number of untimed warmup runs.
    repeat : int, optional
        Number of timed repetitions.
    sample_interval_sec : float, optional
        Polling interval for process/GPU memory sampling.
    gpu_device : int | None, optional
        GPU index for NVML memory sampling.
    sync_cuda : bool, optional
        Synchronize CUDA before and after each timed call.

    Returns
    -------
    output : Any
        Output from the final timed run.
    metrics : dict
        Runtime summary and memory peaks.
    """
    warmup = max(0, int(warmup))
    repeat = max(1, int(repeat))

    for _ in range(warmup):
        if sync_cuda:
            _sync_cuda()
        func(*args, **kwargs)
        if sync_cuda:
            _sync_cuda()

    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    peaks = _Peaks()
    stop_event = threading.Event()
    thread = threading.Thread(
        target=_monitor_resources,
        args=(stop_event, peaks, sample_interval_sec, gpu_device),
        daemon=True,
    )
    thread.start()

    runtimes = []
    output = None
    try:
        for _ in range(repeat):
            if sync_cuda:
                _sync_cuda()
            t0 = time.perf_counter()
            output = func(*args, **kwargs)
            if sync_cuda:
                _sync_cuda()
            runtimes.append(time.perf_counter() - t0)
    finally:
        stop_event.set()
        thread.join(timeout=2.0)

    torch_peak_bytes = None
    if torch is not None and torch.cuda.is_available():
        try:
            torch_peak_bytes = int(torch.cuda.max_memory_allocated())
        except Exception:
            torch_peak_bytes = None

    cupy_peak_bytes = None
    if cp is not None:
        try:
            cupy_peak_bytes = int(cp.get_default_memory_pool().used_bytes())
        except Exception:
            cupy_peak_bytes = None

    metrics = {
        "repeat": repeat,
        "warmup": warmup,
        "runtimes_sec": runtimes,
        "runtime_mean_sec": float(statistics.mean(runtimes)),
        "runtime_std_sec": float(statistics.pstdev(runtimes) if len(runtimes) > 1 else 0.0),
        "peak_cpu_percent": float(peaks.cpu_percent),
        "peak_ram_gb": float(peaks.ram_bytes / (1024**3)),
        "peak_gpu_mem_gb_nvml": float(peaks.gpu_mem_bytes / (1024**3)) if peaks.gpu_mem_bytes else None,
        "peak_gpu_mem_gb_torch": float(torch_peak_bytes / (1024**3)) if torch_peak_bytes is not None else None,
        "peak_gpu_mem_gb_cupy": float(cupy_peak_bytes / (1024**3)) if cupy_peak_bytes is not None else None,
    }
    return output, metrics