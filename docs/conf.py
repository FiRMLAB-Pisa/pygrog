# Configuration file for Sphinx documentation builder.

project = "PyGROG"
copyright = "2024, PyGROG contributors"
author = "PyGROG contributors"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "sphinx_book_theme"
html_title = "PyGROG"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

autodoc_typehints = "description"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
