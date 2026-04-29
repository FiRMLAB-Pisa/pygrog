# PyGROG poster figures

Standalone scripts that generate poster-ready PNGs for the PyGROG project.

## Poster set (4 figures)

| File | Purpose |
| --- | --- |
| `figures/fig_package.png`         | Package-architecture diagram |
| `figures/fig_features.png`        | 2×3 composite — features at a glance |
| `figures/fig_mrf_qualitative.png` | In-vivo MRF: PyGROG vs FINUFFT subspace coefficients |
| `figures/fig_benchmarks.png`      | Runtime + peak-memory bars from `benchmark/results/results.json` |

`fig_features.png` tiles six smaller per-feature panels (basic recon, ORC,
subspace, solvers, Toeplitz, interop). Those panels are also produced
standalone by `scripts/fig_brain_recon.py`, `fig_orc.py`, `fig_subspace.py`,
`fig_solvers.py`, `fig_toeplitz.py`, `fig_interop.py` — kept as reusable
building blocks.

## Regenerate

```bash
# Default — generate the 4 poster figures (auto-runs component figs as needed).
conda run -n pygrog --no-capture-output python pygrog/paper/scripts/build_all.py

# Also re-render the 6 per-feature components.
conda run -n pygrog --no-capture-output python pygrog/paper/scripts/build_all.py --components

# Run a single script.
conda run -n pygrog --no-capture-output python pygrog/paper/scripts/fig_orc.py
```

## Layout

```
paper/
├── README.md
├── figures/                # generated PNGs (300 dpi, transparent background)
└── scripts/
    ├── _common.py          # shared style + helpers
    ├── build_all.py
    ├── fig_package.py            \  poster
    ├── fig_features.py           |   set
    ├── fig_mrf_qualitative.py    |   (4)
    ├── fig_benchmarks.py         /
    ├── fig_brain_recon.py        \  components
    ├── fig_orc.py                |   (tiled by
    ├── fig_subspace.py           |    fig_features)
    ├── fig_solvers.py            |
    ├── fig_toeplitz.py           |
    └── fig_interop.py            /
```

## Data sources

* **Component figures** download the BrainWeb T1 phantom on first run via
  [`brainweb-dl`](https://github.com/brainweb-dl).
* **`fig_mrf_qualitative`** loads cached arrays from
  `pygrog/benchmark/results/`. Pass `--rerun` to recompute from the in-vivo
  MRF dataset under `pygrog/benchmark/data/`.
* **`fig_benchmarks`** re-renders from
  `pygrog/benchmark/results/results.json` only — refreshing the underlying
  numbers is done separately via `pygrog/benchmark/run_benchmarks.py`.

## Style

All figures use a shared poster style (≥14 pt fonts, transparent background,
`viridis` magnitudes, `RdBu_r` differences) defined in `scripts/_common.py`.
