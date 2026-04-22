#!/usr/bin/env python3
"""Run full benchmark pipeline: download (if needed), benchmark, and plotting."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REQUIRED_DATA_FILES = [
    "basis.npy",
    "smaps.npy",
    "trajectory.npy",
    "dcf.npy",
    "kspace.npy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("benchmark/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark/results"))

    parser.add_argument("--record", type=str, help="Zenodo record ID for data download.")
    parser.add_argument("--doi", type=str, help="Zenodo DOI for data download.")

    parser.add_argument("--shape", type=int, nargs="+", default=[160, 160])
    parser.add_argument("--n-coils", type=int, default=8)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on frames. Default: use all frames from dataset.",
    )
    parser.add_argument(
        "--max-coeff",
        type=int,
        default=None,
        help="Optional cap on basis coefficients. Default: use full basis rank.",
    )
    parser.add_argument("--n-spokes", type=int, default=48)
    parser.add_argument("--n-readout", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--gpu-device", type=int, default=0)

    return parser.parse_args()


def _missing_data_files(data_dir: Path) -> list[Path]:
    return [data_dir / name for name in REQUIRED_DATA_FILES if not (data_dir / name).exists()]


def _run(cmd: list[str]) -> None:
    print("$", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = (repo_root / args.data_dir).resolve() if not args.data_dir.is_absolute() else args.data_dir
    output_dir = (
        (repo_root / args.output_dir).resolve()
        if not args.output_dir.is_absolute()
        else args.output_dir
    )

    missing = _missing_data_files(data_dir)
    if missing:
        if not args.record and not args.doi:
            missing_str = ", ".join(str(p) for p in missing)
            raise SystemExit(
                "Missing benchmark data files: "
                f"{missing_str}. Provide --record or --doi to download dataset."
            )

        download_cmd = [
            sys.executable,
            str(repo_root / "benchmark" / "download_data.py"),
            "--data-dir",
            str(data_dir),
        ]
        if args.doi:
            download_cmd += ["--doi", args.doi]
        else:
            download_cmd += ["--record", args.record]
        _run(download_cmd)

    benchmark_cmd = [
        sys.executable,
        str(repo_root / "benchmark" / "run_benchmarks.py"),
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(output_dir),
        "--shape",
        *[str(v) for v in args.shape],
        "--n-coils",
        str(args.n_coils),
        "--n-spokes",
        str(args.n_spokes),
        "--n-readout",
        str(args.n_readout),
        "--repeats",
        str(args.repeats),
        "--warmup",
        str(args.warmup),
        "--gpu-device",
        str(args.gpu_device),
    ]
    if args.max_frames is not None:
        benchmark_cmd += ["--max-frames", str(args.max_frames)]
    if args.max_coeff is not None:
        benchmark_cmd += ["--max-coeff", str(args.max_coeff)]
    _run(benchmark_cmd)

    plot_cmd = [
        sys.executable,
        str(repo_root / "benchmark" / "plot_benchmarks.py"),
        "--results-json",
        str(output_dir / "results.json"),
        "--output-dir",
        str(output_dir),
    ]
    _run(plot_cmd)

    print(f"Pipeline completed. Outputs written to {output_dir}")


if __name__ == "__main__":
    main()
