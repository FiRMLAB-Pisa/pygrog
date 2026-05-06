#!/usr/bin/env python
"""Repair a Linux wheel with auditwheel, excluding PyTorch's shared libs.

auditwheel's default behaviour bundles every non-system .so that the
extension links against.  For a PyTorch extension that includes
``libtorch_cpu.so`` (~500 MB), which pushes the wheel far past PyPI's
100 MB per-file limit.  Since ``torch`` is a declared runtime dependency
the libraries are already present in every user's environment, so there is
no need to vendor them into the wheel.

Usage (called by cibuildwheel via CIBW_REPAIR_WHEEL_COMMAND_LINUX):

    python scripts/repair_wheel.py {wheel} {dest_dir}
"""

import re
import subprocess
import sys
from pathlib import Path

import torch


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(f"Usage: {sys.argv[0]} <wheel> <dest_dir>")

    wheel, dest_dir = sys.argv[1], sys.argv[2]

    torch_lib_dir = Path(torch.__file__).parent / "lib"
    if not torch_lib_dir.is_dir():
        sys.exit(
            f"PyTorch lib directory not found: {torch_lib_dir}\n"
            "Ensure torch is installed in the build environment."
        )

    # Match shared-library filenames: libfoo.so or libfoo.so.1.2.3
    _so_re = re.compile(r"\.so(\.\d+)*$")

    excludes = [
        arg
        for name in (entry.name for entry in torch_lib_dir.iterdir())
        if _so_re.search(name)
        for arg in ("--exclude", name)
    ]

    cmd = ["auditwheel", "repair", *excludes, "-w", dest_dir, wheel]
    # Command arguments are generated locally from trusted inputs in CI.
    subprocess.check_call(cmd)  # noqa: S603


if __name__ == "__main__":
    main()
