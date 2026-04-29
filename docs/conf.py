"""
Configuration file for the Sphinx documentation builder.

Full list of options:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path().resolve()))
sys.path.insert(0, str(Path("../src").resolve()))

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

suppress_warnings = ["myst.xref_missing"]

# MyST (Markdown) settings
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "amsmath"]
myst_heading_anchors = 3

# generate autosummary even if no references
autosummary_generate = True
autodoc_inherit_docstrings = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_private_with_doc = True
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
    "filename_pattern": "/example_",
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
