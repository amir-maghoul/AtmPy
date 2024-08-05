# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import sys
import os

# import sphinx-rtd-theme

sys.path.insert(0, os.path.abspath("./atmpy"))

project = "Atmpy"
copyright = "2024, Amir Maghoul"
author = "Amir Maghoul"
release = "0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

master_doc = "index"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.inheritance_diagram",
    "autoapi.extension",
]

autoapi_dirs = ["../atmpy"]
autoapi_type = "python"

#
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme_path = ["_themes"]
html_static_path = ["_static"]

html_theme = "sphinx_rtd_theme"
# html_theme_path = [sphinx_theme.get_html_theme_path('stanford-theme')]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
#
# html_theme = 'alabaster'
html_static_path = ["_static"]
