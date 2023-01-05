# Configuration file for the Sphinx documentation builder.
import os
import sys
sys.path.append(os.path.abspath('../..'))

# -- Project information

project = 'FFPACK'
copyright = '2022, Dongping Zhu'
author = 'Dongping Zhu'

release = '0.3.0'
version = '0.3.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'myst_parser',
    'nbsphinx',
]

source_suffix = ['.rst', '.md']

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = "footnote"

def skip(app, what, name, obj, would_skip, options):
    if name == "__init__":
        return False
    return would_skip

def setup(app):
    app.connect("autodoc-skip-member", skip)
