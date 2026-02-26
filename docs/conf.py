import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(".."))

project = "NEO_JAX"
author = "UW Plasma"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosectionlabel",
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
    "sphinx_design",
]

autosectionlabel_prefix_document = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_static_path = ["_static"]

bibtex_bibfiles = ["refs.bib"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", {}),
    "numpy": ("https://numpy.org/doc/stable", {}),
}

# MathJax config can be extended later for macros.
mathjax3_config = {
    "tex": {"macros": {"bm": ["\\boldsymbol{#1}", 1]}},
}
