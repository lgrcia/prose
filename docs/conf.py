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

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_nb",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
]


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Mine
# -------------------
html_title = "prose"

source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".myst": "myst-nb",
}

root_doc = "index"

html_theme_options = {
    "repository_url": "https://github.com/lgrcia/prose",
    "use_repository_button": True,
}

nb_render_image_options = {"align": "center"}

myst_enable_extensions = [
    "dollarmath",
]

templates_path = ["_templates"]
nb_execution_mode = "auto"
nb_execution_raise_on_error = True

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

autodoc_typehints = "signature"
autoclass_content = "both"

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
