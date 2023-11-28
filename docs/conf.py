import toml

project = "prose"
copyright = "2023, Lionel Garcia"
author = "Lionel Garcia"

extensions = [
    "myst_nb",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


html_theme = "sphinx_book_theme"
html_static_path = ["_static"]

pyproject = toml.load("../pyproject.toml")
version = pyproject["tool"]["poetry"]["version"]
html_short_title = "prose"
html_title = f"{html_short_title}"

# html_logo = "_static/prose3.png"

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
