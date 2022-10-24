# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

import os
import sys

sys.path.insert(0, os.path.abspath("../prose"))


# -- Project information -----------------------------------------------------

project = 'prose'
copyright = '2022, Lionel Garcia'
author = 'Lionel Garcia'

extensions = [
    'sphinx.ext.autodoc', 
    'sphinx.ext.napoleon', 
    'sphinx.ext.autosummary',
    'sphinx_copybutton', 
    'nbsphinx',
    'jupyter_sphinx',
    'myst_parser'
    ]

autodoc_typehints = 'signature'
autoclass_content = 'both'

master_doc = 'index'
exclude_patterns = ['_build', '**.ipynb_checkpoints']
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

nbsphinx_allow_errors = True

# -- Options for HTML output -------------------------------------------------

html_title = "prose"
html_theme = "furo"
html_static_path = ['static']
templates_path = ["templates"]
html_css_files = ['custom.css']


rst_prolog = """
.. |prose| replace:: *prose*
.. _photutils: https://photutils.readthedocs.io/en/stable/
.. _scikit-image: https://scikit-image.org/

.. role:: blockread
.. |read| replace:: :blockread:`read`

.. role:: blockwrite
.. |write| replace:: :blockwrite:`write`

.. role:: blockmodify
.. |modify| replace:: :blockmodify:`modify data`

"""

import prose

prose.CONFIG.update_builtins()

 
# Making all_blocks.rst
# ---------------------
import inspect
import sys
from glob import glob
import prose
from prose import Block

files = glob("../../prose/blocks/*.py")
classes = []

for f in files:
    module = f.split("../../")[1].rstrip(".py").replace("/", ".")
    if module in sys.modules:
        for cls_name, cls_obj in inspect.getmembers(sys.modules[module]):
            if  cls_name.split(".")[-1][0] != "_":
                if inspect.isclass(cls_obj):
                    if issubclass(cls_obj, Block):
                        if cls_obj.__module__ == module:
                            classes.append(f"{module}.{cls_name}")

classes = sorted(list(set(list(classes))), key=lambda x: x.split(".")[-1].lower())
_all_blocks = "\n".join([f"\t~{cl}" for cl in classes])
all_blocks = f"""

.. currentmodule:: prose

.. autosummary::
   :toctree: generated
   :template: blocksum.rst
   :nosignatures:

   {_all_blocks}

"""

open("rst/all_blocks.rst", "w").write(all_blocks)