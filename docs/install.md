# Installation

## Requirements

- Python ≥ 3.10 (3.10 – 3.14 supported)
- PyTorch ≥ 2.0

---

## Quick install (recommended)

Pre-compiled CPU wheels are published to PyPI for Linux (x86\_64, aarch64),
macOS (Intel and Apple Silicon), and Windows (x86\_64):

```bash
pip install pygrog
```

Verify the installation:

```bash
python -c "import pygrog; print('pygrog', pygrog.__version__, '-- OK')"
```

---

## CUDA wheels

CUDA wheels are attached as assets to every
[GitHub Release](https://github.com/FiRMLAB-Pisa/pygrog/releases) and can
be installed with `--find-links`:

| CUDA version | PyTorch index label |
|---|---|
| 12.6 | `cu126` |
| 12.8 | `cu128` |
| 13.0 | `cu130` |

```bash
pip install "pygrog==<version>+<cu>" \
  --find-links https://github.com/FiRMLAB-Pisa/pygrog/releases/expanded_assets/v<version>
```

---

## Development install

Clone the repository and install in editable mode together with all optional
dependencies:

```bash
git clone https://github.com/FiRMLAB-Pisa/pygrog
cd pygrog
pip install --no-build-isolation -e ".[dev]"
```

The `[dev]` extra pulls in testing, linting, and documentation tools (see
[`pyproject.toml`](https://github.com/FiRMLAB-Pisa/pygrog/blob/main/pyproject.toml)).

### Building the C++ extension manually

The C++ extension (`_pygrog_torch`) is compiled automatically by
`pip install`.  If compilation fails, PyGROG falls back to JIT compilation
on first use.  To force a manual rebuild:

```bash
python setup.py build_ext --inplace
```

See [Building C++ Extensions](build.md) for full details on CUDA, SIMD flags,
and cross-compilation.
