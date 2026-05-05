# Developing PyGROG

## Development environment

We recommend using a conda environment with an editable install:

```bash
git clone https://github.com/FiRMLAB-Pisa/pygrog
cd pygrog
conda create -n pygrog python=3.12
conda activate pygrog

# 1. Install PyTorch and build tools first (--no-build-isolation requires them)
pip install torch numpy
pip install scikit-build-core setuptools-scm

# 2. Install the C++ toolchain (Linux)
sudo ./scripts/install_build_deps.sh      # CPU only
# sudo ./scripts/install_build_deps.sh --cuda  # with CUDA

# 3. Editable install including all dev extras
pip install --no-build-isolation -e ".[dev]"
```

See [Installation → Build from source](../install.md#build-from-source) for
platform-specific toolchain instructions (macOS, Windows, CUDA versions).

## Running tests

```bash
cd pygrog
pytest tests/
```

For coverage:

```bash
pytest --cov=pygrog --cov-report=html tests/
```

## Writing documentation

Documentation is hosted at
<https://firnlab-pisa.github.io/pygrog/>.

Build locally with:

```bash
bash scripts/build_docs.sh --clean
```

Serve and preview:

```bash
bash scripts/build_docs.sh --serve
```

Then open <http://localhost:8000> in your browser.

## Code style

PyGROG uses **Ruff-only** formatting and linting:

```bash
ruff format .
ruff check .
```

`ruff check` is configured with `fix = true`, so safe auto-fixes are applied
automatically.

A pre-commit hook is provided for convenience:

```bash
pre-commit install
```
