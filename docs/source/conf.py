# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

import os
import sys
import jupytext

sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------

project = 'prose'
copyright = '2020, Lionel Garcia'
author = 'Lionel Garcia'

extensions = [
    "sphinx.ext.autodoc", 
    "sphinx.ext.napoleon", 
    'sphinx.ext.autosummary', 
    ]

master_doc = 'index'

exclude_patterns = []
source_suffix = {'.rst': 'restructuredtext'}

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"

html_static_path = ["_static"]
html_css_files = ["css/style.css"]
html_style = "css/style.css"

pygments_style = "friendly"

html_theme_options = {
}

napoleon_numpy_docstring = True
napoleon_use_param = False

# html_favicon = "_static/specphot logo small.png"

autodoc_member_order = 'bysource'

rst_prolog = """
.. |prose| replace:: *prose*
.. _photutils: https://photutils.readthedocs.io/en/stable/
.. _scikit-image: https://scikit-image.org/
"""

import inspect
from os import path
from prose import blocks, Block

for name, obj in inspect.getmembers(blocks):
    if inspect.isclass(obj):
        if issubclass(obj, Block):
            block_doc = obj.doc()
            if block_doc is not None:
                filename = path.join("./guide/api/blocks", "{}.rst".format(name))
                with open(filename, "w") as f:
                    f.write("{}\n{}".format(name, "-"*len(name)))
                    f.write("\n\n{}".format(block_doc))
                    f.write("\n\n.. autoclass:: prose.blocks.{}\n\t:members:".format(name))


exec(open('./convert_notebooks.py').read())