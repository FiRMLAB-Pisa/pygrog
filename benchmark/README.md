# Benchmark

This folder includes a benchmark workflow targeting:

- MRF subspace coefficient comparison between NUFFT and GROG.
- Runtime comparison in forward and adjoint modes.
- CPU and GPU backends (`finufft` and `cufinufft`).
- GROG plan creation and interpolation profiling.
- Memory footprint summary (RAM and VRAM when available).
- GROG GPU full mode vs dual-stream mode.

Folder layout:

- `benchmark/data`: active MRF benchmark inputs only.
  - `smaps.npy`
  - `basis.npy`
  - `trajectory.npy`
  - `dcf.npy`
  - `kspace.npy`
- `benchmark/legacy`: deprecated scripts and older result/cache artifacts.
- `benchmark/run_benchmarks.py`: benchmark runner.
- `benchmark/plot_benchmarks.py`: figure generation from JSON outputs.

## Environment

Use the existing conda environment:

```bash
conda activate pygrog
```

Install/refresh dependencies in editable mode:

```bash
pip install --no-build-isolation -e ".[dev,gpu]"
```

On Linux GPU servers, also ensure `cufinufft` is installed:

```bash
pip install cufinufft
```

## Download Data (Zenodo)

Use `zenodo_get` (installed through dev dependencies) to fetch the legacy MRF dataset and normalize it into the five active files above:

```bash
python benchmark/download_data.py --record <ZENODO_RECORD_ID>
```

or

```bash
python benchmark/download_data.py --doi <ZENODO_DOI>
```

## Run Benchmarks

From the repository root:

```bash
python benchmark/run_benchmarks.py \
  --output-dir benchmark/results \
  --warmup 1 \
  --repeats 5
```

This runner uses the real legacy MRF benchmark data from `benchmark/data`.
By default it uses all available frames and full basis rank from the dataset.
Use `--max-frames` and `--max-coeff` only if you want to cap problem size.

For preprocessing and linop scaling figures, synthetic sizes are generated from
`--scaling-ratios` and then the final two points are always:
1) synthetic case matched to MRF samples/frame, and
2) real MRF case.

Default synthetic ratios are:
`0.01,0.02,0.05,0.1,0.2,0.4,0.7`
(plus the auto-appended `Synth-MRF-size` and `MRF-real` points).

## Full Pipeline (Single Script)

From repository root, run download-if-needed + benchmark + plotting in one command:

```bash
python scripts/run_benchmark_pipeline.py \
  --output-dir benchmark/results \
  --warmup 1 \
  --repeats 5
```

If `benchmark/data` is missing required files, provide Zenodo source:

```bash
python scripts/run_benchmark_pipeline.py --record <ZENODO_RECORD_ID>
```

or

```bash
python scripts/run_benchmark_pipeline.py --doi <ZENODO_DOI>
```

On machines without a working CUDA stack, GPU benchmarks are skipped automatically.

## Generate Figures

```bash
python benchmark/plot_benchmarks.py \
  --results-json benchmark/results/results.json \
  --output-dir benchmark/results
```

Generated files:

- `benchmark/results/figure_preprocessing.png`
- `benchmark/results/figure_linop.png`
- `benchmark/results/figure_coeffs.png`
- `benchmark/results/figure_grog_views.png`

## Notes for A40 Server Runs

- Keep the same commands; only increase `--repeats` if needed.
- Ensure CUDA toolkit/driver compatibility with PyTorch and `cufinufft`.
- Optionally pin `--gpu-device` for multi-GPU servers.
