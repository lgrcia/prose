# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] tags=[]
# # Fits manager

# %%
from prose import FitsManager, Telescope
from prose import tutorials

# %% [markdown]
# Astronomical observations often generate highly disorganised fits images folders. To know the content of these files, file names can be used but have their limitations. At the end it is not rare to start opening these files to acces the information in their headers.
#
# To solve this issue, prose features the `FitsManager` object, a conveniant tool to ease the sorting process.

# %% [markdown]
# ## Generating fake fits
#
# To demonstrate the use of the FITS manager, lets' generate a set of fake data from telescope `A` and `B`, defined with

# %%
_ = Telescope(dict(name="A"))
_ = Telescope(dict(name="B"))

# %% [markdown]
# Images will be located in a single folder, featuring different sizes, filters and associated calibration files, with no way to distinguish them from their file names

# %%
destination = "./fake_observations"
tutorials.disorganised_folder(destination)

# %% [markdown]
# ## The Fits Manager object

# %% [markdown]
# To dig into these disorganised folder, let's instantiate a `FitsManager` object

# %%
fm = FitsManager(destination)
fm

# %% [markdown]
# The keywords of all images have been parsed and associated with different telescopes. The advantage is that specific keywords from specific telescopes are recognized and standardized to common namings. This is usefull to define telescope agnostic pipelines (see for example the [photometry tutotial](./photometry.ipynb)).

# %% [markdown]
# ## Picking an observation
#
# From there let say we want to keep the files from an observation using its `id`

# %%
files = fm.observation_files(1)

# %% [markdown]
# flats with the right filter have been kept, as well as darks

# %% [markdown]
# ### Telescope specific keywords

# %% [markdown]
# The information retained by `FitsManager` was taken from images headers. To know which keywords to use, we had to register telescopes `A` and `B` with a dictionary. Whenever their names appear in a fits header, their dictionary is loaded to read their header keywords.
#
# Since we just specified the telescope names all the rest is default. For example the filter is taken from the keyword `FILTER` and the image type from `IMAGETYP`, knowing that `IMAGETYP=light` is a light (a.k.a science) frame. These keywords can be set in more details when registering the telescope.
#
# For more details, chcek the `Telescope` object

# %% tags=[]
# hidden
from shutil import rmtree

rmtree(destination)
