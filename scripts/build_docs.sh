#!/usr/bin/env bash
# build_docs.sh — Build the PyGROG Sphinx documentation.
#
# Usage:
#   ./scripts/build_docs.sh          # build HTML into docs/_build/html/
#   ./scripts/build_docs.sh --clean  # remove generated/ and _build/ first
#   ./scripts/build_docs.sh --serve  # build then serve on localhost:8000
#
# Requirements (install once):
#   pip install pygrog[dev]
#   # or: pip install -r docs/requirements.txt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DOCS_DIR="$REPO_ROOT/docs"
BUILD_DIR="$DOCS_DIR/_build/html"

CLEAN=false
SERVE=false

for arg in "$@"; do
    case "$arg" in
        --clean) CLEAN=true ;;
        --serve) SERVE=true ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

if $CLEAN; then
    echo "==> Cleaning docs/_build/ and docs/generated/ ..."
    rm -rf "$DOCS_DIR/_build" "$DOCS_DIR/generated"
fi

echo "==> Building HTML documentation ..."
python -m sphinx "$DOCS_DIR" "$BUILD_DIR" -b html -W --keep-going

echo ""
echo "Documentation built successfully."
echo "Output: $BUILD_DIR/index.html"

if $SERVE; then
    echo ""
    echo "==> Serving at http://localhost:8000 (Ctrl-C to stop) ..."
    python -m http.server --directory "$BUILD_DIR" 8000
fi
