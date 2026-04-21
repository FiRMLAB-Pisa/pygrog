# Developing PyGROG

## Development environment

We recommend using a virtual environment (e.g., conda) and installing
PyGROG with all optional dependencies:

```bash
git clone https://github.com/FiRMLAB-Pisa/pygrog
cd pygrog
conda create -n pygrog python=3.12
conda activate pygrog
pip install --no-build-isolation -e ".[dev]"
```

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
cd pygrog/docs
pip install -r requirements.txt
make html
```

View the result:

```bash
python -m http.server --directory _build/html 8000
```

Then open <http://localhost:8000> in your browser.

## Code style

PyGROG enforces **black** formatting and **ruff** linting:

```bash
black .
ruff --check --fix .
ruff --check .
```

A pre-commit hook is provided for convenience:

```bash
pre-commit install
```
