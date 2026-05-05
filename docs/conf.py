"""
Configuration file for the Sphinx documentation builder.

Full list of options:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path().resolve()))
sys.path.insert(0, str(Path("../src").resolve()))

# torchvision CPU wheels fail at import time when the torchvision C extension
# has not been compiled with the required ops (e.g. torchvision::nms).
# deepinv imports torchvision at the top level, so this would break autosummary
# and autodoc even though deepinv itself is only *optionally* required by pygrog.
# We try to import torchvision; if it raises, we install a lightweight mock so
# that deepinv (and therefore the whole pygrog package) can be imported for docs.
# sphinx-gallery notebooks that exercise deepinv functionality run with
# abort_on_example_error=False and will skip gracefully if torchvision is broken.
if "torchvision" not in sys.modules:
    try:
        import torchvision  # noqa: F401
    except (RuntimeError, ImportError):
        sys.modules["torchvision"] = MagicMock()


def _invalidate_gallery_cache_if_thumb_missing() -> None:
    """Drop stale Sphinx-Gallery md5 files when thumbs are missing.

    Sphinx-Gallery may skip an example when its ``.md5`` cache exists,
    but backreference generation still requires the thumbnail file.
    If a thumbnail was removed from ``docs/generated`` while the ``.md5``
    persisted, the docs build can fail with a missing-thumb error.
    """
    docs_dir = Path(__file__).resolve().parent
    examples_dir = (docs_dir / "../examples").resolve()
    gallery_dir = (docs_dir / "generated/autoexamples").resolve()
    thumbs_dir = gallery_dir / "images" / "thumb"

    if not examples_dir.exists() or not gallery_dir.exists():
        return

    for example in sorted(examples_dir.glob("example*_*.py")):
        stem = example.stem
        md5_file = gallery_dir / f"{example.name}.md5"
        thumb_file = thumbs_dir / f"sphx_glr_{stem}_thumb.png"
        if md5_file.exists() and not thumb_file.exists():
            md5_file.unlink()


_invalidate_gallery_cache_if_thumb_missing()

# -- Project information -----------------------------------------------------

project = "PyGROG"
copyright = "2024, PyGROG contributors"  # noqa: A001
author = "PyGROG contributors"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx_copybutton",
    "sphinx.ext.duration",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_gallery.gen_gallery",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "api/index.rst",
    "build.md",
    "installation_guide.md",
]

suppress_warnings = ["myst.xref_missing", "ref.ref", "docutils"]

# MyST (Markdown) settings
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "amsmath"]
myst_heading_anchors = 3

# generate autosummary even if no references
autosummary_generate = True
autodoc_inherit_docstrings = True
autodoc_member_order = "bysource"
autodoc_typehints = "none"

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_private_with_doc = False
napoleon_use_admonition_for_references = True

pygments_style = "sphinx"
highlight_language = "python"

# -- Sphinx Gallery ----------------------------------------------------------

sphinx_gallery_conf = {
    "doc_module": "pygrog",
    "backreferences_dir": "generated/gallery_backreferences",
    "reference_url": {"pygrog": None},
    "examples_dirs": ["../examples/"],
    "gallery_dirs": ["generated/autoexamples"],
    "filename_pattern": r"/example\d*_",
    "ignore_pattern": r"(__init__|conftest|fast_binning)\.py",
    "nested_sections": True,
    "within_subsection_order": "FileNameSortKey",
    "first_notebook_cell": "!pip install pygrog[dev]",
    "abort_on_example_error": False,
}

# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "mrinufft": ("https://mind-inria.github.io/mri-nufft/", None),
}

# -- HTML output -------------------------------------------------------------

html_theme = "sphinx_book_theme"
html_title = "PyGROG"
html_theme_options = {
    "repository_url": "https://github.com/FiRMLAB-Pisa/pygrog",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "use_download_button": True,
    "home_page_in_toc": True,
}
