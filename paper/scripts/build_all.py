"""Generate the four poster figures (and any prerequisites).

Usage::

    # Default — generate the 4 poster-targeted figures.
    python pygrog/paper/scripts/build_all.py

    # Also re-render the per-feature component figures used by fig_features.
    python pygrog/paper/scripts/build_all.py --components

    # Pick specific scripts.
    python pygrog/paper/scripts/build_all.py --only fig_orc.py fig_benchmarks.py

Each script runs in a fresh subprocess; failures don't contaminate others.
Exit code is non-zero if any script fails.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent

# Final poster figures (what goes on the poster).
POSTER_SCRIPTS = [
    "fig_package.py",
    "fig_features.py",  # auto-runs the components below if missing.
    "fig_mrf_qualitative.py",
    "fig_benchmarks.py",
]

# Per-feature components used to build fig_features.
COMPONENT_SCRIPTS = [
    "fig_brain_recon.py",
    "fig_orc.py",
    "fig_subspace.py",
    "fig_solvers.py",
    "fig_toeplitz.py",
    "fig_interop.py",
]

ALL_SCRIPTS = POSTER_SCRIPTS + COMPONENT_SCRIPTS


def _run(name: str) -> tuple[int, float]:
    path = SCRIPTS_DIR / name
    print(f"\n[build_all] >>> {name}", flush=True)
    t0 = time.perf_counter()
    rc = subprocess.run([sys.executable, str(path)]).returncode  # noqa: S603
    dt = time.perf_counter() - t0
    print(
        f"[build_all] <<< {name}   {'OK ' if rc == 0 else 'FAIL'}   {dt:.1f} s",
        flush=True,
    )
    return rc, dt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--components",
        action="store_true",
        help="Also re-render the per-feature component figures.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="SCRIPT",
        help="Run only the named scripts (e.g. fig_orc.py).",
    )
    args = parser.parse_args()

    if args.only:
        unknown = [t for t in args.only if t not in ALL_SCRIPTS]
        if unknown:
            print(f"[build_all] unknown scripts: {unknown}", file=sys.stderr)
            return 2
        targets = list(args.only)
    else:
        targets = list(POSTER_SCRIPTS)
        if args.components:
            for s in COMPONENT_SCRIPTS:
                if s not in targets:
                    targets.append(s)

    failures: list[str] = []
    timings: dict[str, float] = {}
    for name in targets:
        rc, dt = _run(name)
        timings[name] = dt
        if rc != 0:
            failures.append(name)

    print("\n[build_all] Summary")
    for name in targets:
        mark = "FAIL" if name in failures else "ok  "
        print(f"  [{mark}] {name:30s}  {timings[name]:6.1f} s")
    if failures:
        print(f"\n[build_all] {len(failures)} script(s) failed.")
        return 1
    print("\n[build_all] All requested figures generated successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
