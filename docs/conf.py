# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = "prose"
copyright = "2023, Lionel Garcia"
author = "Lionel Garcia"


# -- General configuration ---------------------------------------------------


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "bronzivka"
html_static_path = ["_static"]

# Title
# get version number from pyproject.toml
# --------------------------------------
import toml

pyproject = toml.load("../pyproject.toml")
version = pyproject["tool"]["poetry"]["version"]
html_short_title = "prose"
html_title = f"{html_short_title}"

root_doc = "index"

use_repository_button = True

myst_enable_extensions = [
    "dollarmath",
]

nb_execution_mode = "auto"
nb_execution_raise_on_error = True

autodoc_typehints = "signature"
autoclass_content = "both"

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

.. _JAX: https://jax.readthedocs.io/en/latest/

"""

html_css_files = ["style.css"]

# Making all_blocks.rst
# ---------------------
import inspect
import sys
from glob import glob

import prose
from prose import Block

files = glob("../prose/blocks/*.py")
classes = []
for f in files:
    module = f.split("../")[1].replace(".py", "").replace("/", ".")
    if module in sys.modules:
        for cls_name, cls_obj in inspect.getmembers(sys.modules[module]):
            if cls_name.split(".")[-1][0] != "_":
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

open("md/all_blocks.rst", "w").write(all_blocks)

# check if blocks are tested
# --------------------------
# import os

# from prose.core.block import is_tested
# from prose.utils import get_all_blocks

# os.chdir("..")
# tested = []
# tested.append("# Tested blocks\n")
# tested.append("| Block | Tested |")
# tested.append("| ----- | ------ |")
# blocks = get_all_blocks()
# blocks = sorted(blocks, key=lambda block: block.__name__.lower())

# for block in blocks:
#     _is = is_tested(block.__name__)
#     tested.append(
#         f" | [`{block.__name__}`]({block.__module__}.{block.__name__}) | {'✅' if _is else '❌'} |"
#     )
# os.chdir("docs")

# open("./tested_blocks.md", "w").write("\n".join(tested))
