#!/usr/bin/env python3
"""Download benchmark data from Zenodo into benchmark/data."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

EXPECTED_FILES = {
    "compressed_data.npy",
    "coilsens.npy",
    "basis.npy",
    "metadata.npy",
    "traj_grp48_inacc1.mat",
    "dictionary.mat",
    "phi.mat",
}

OUTPUT_FILES = {
    "smaps": "smaps.npy",
    "basis": "basis.npy",
    "trajectory": "trajectory.npy",
    "dcf": "dcf.npy",
    "kspace": "kspace.npy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--record",
        type=str,
        help="Zenodo record id, e.g. 1234567.",
    )
    parser.add_argument(
        "--doi",
        type=str,
        help="Zenodo DOI, e.g. 10.5281/zenodo.1234567.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("benchmark/data"),
        help="Destination folder for benchmark data.",
    )
    return parser.parse_args()


def _zenodo_command() -> list[str]:
    cli = shutil.which("zenodo_get")
    if cli is not None:
        return [cli]
    return [sys.executable, "-m", "zenodo_get"]


def _normalize_dataset(tmp_path: Path, data_dir: Path) -> int:
    compressed = np.load(tmp_path / "compressed_data.npy").astype(np.complex64)
    metadata = np.load(tmp_path / "metadata.npy", allow_pickle=True).item()
    smaps = np.load(tmp_path / "coilsens.npy").astype(np.complex64)
    basis = np.load(tmp_path / "basis.npy").astype(np.complex64)

    kspace = compressed.swapaxes(0, 2)[0].swapaxes(0, 1).astype(np.complex64)
    trajectory = metadata["coords"].swapaxes(0, 2)[0].astype(np.float32)
    dcf = metadata["weights"].swapaxes(0, 2)[0].astype(np.float32)

    if trajectory.ndim == 5 and trajectory.shape[0] == 1:
        trajectory = trajectory[0]
    if dcf.ndim == 4 and dcf.shape[0] == 1:
        dcf = dcf[0]

    np.save(data_dir / OUTPUT_FILES["kspace"], kspace)
    np.save(data_dir / OUTPUT_FILES["trajectory"], trajectory)
    np.save(data_dir / OUTPUT_FILES["dcf"], dcf)
    np.save(data_dir / OUTPUT_FILES["smaps"], smaps)
    np.save(data_dir / OUTPUT_FILES["basis"], basis)
    return len(OUTPUT_FILES)


def main() -> None:
    args = parse_args()
    if not args.record and not args.doi:
        raise ValueError("Provide --record or --doi.")

    identifier = args.doi if args.doi else args.record
    data_dir = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="zenodo_benchmark_") as tmp:
        tmp_path = Path(tmp)
        cmd = [*_zenodo_command(), str(identifier), "-o", str(tmp_path)]
        subprocess.run(cmd, check=True)

        found = 0
        for src in tmp_path.rglob("*"):
            if not src.is_file():
                continue
            if src.name in EXPECTED_FILES:
                found += 1

        if found == 0:
            raise RuntimeError(
                "Zenodo download completed but expected benchmark files were not found."
            )

        normalized = _normalize_dataset(tmp_path, data_dir)

    print(
        f"Downloaded {found} legacy benchmark files and wrote {normalized} active data files into {data_dir}"
    )


if __name__ == "__main__":
    main()
