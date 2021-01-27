# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Fits manager

# %%
from prose import FitsManager, Telescope
from prose import tutorials

# %% [markdown]
# Astronomical observations often generate highly disorganised fits images folders. To know the content of these files, file names can be used but have their limitations. At the end it is not rare to start opening these files to acces the information in their headers.
#
# To solve this organisation problem, prose features the `FitsManager` object, a conveniant tool to ease the sorting process.

# %% [markdown]
# ## Generating fake fits
#
# Lets' generate a set of fake images all stored in the same folder but with different attributes and no way to distinguish them from their file names. These data will be taken from telescope `A` and `B` , let's define them:

# %%
_ = Telescope(dict(name="A"))
_ = Telescope(dict(name="B"))

# %% [markdown]
# We will now simulate some images from `A` and `B` all located in a single folder, featuring different sizes, filters and associated calibration:

# %%
destination = "./fake_observations"
tutorials.disorganised_folder(destination)

# %% [markdown]
# ## The Fits Manager object

# %% [markdown]
# To dig into these disorganised data, we instantiate a `FitsManager` on the folder and see its content

# %%
fm = FitsManager(destination)
fm.observations

# %% [markdown]
# As we can see the `FitsManager` object does a great job in splitting our fits into convient categories. We created some calibration files as well that can be seen with

# %%
fm.calib

# %%
fm.describe("dimensions", "flip", "type")

# %% [markdown]
# ## Picking an observation
#
# From there let say we want to keep the observation indexed `1` from the observations table:

# %%
fm.set_observation(1)
fm.calib

# %% [markdown]
# We now have our observation isolated. We recognised for example that flats with the right filter have been kept. To get some specific files:

# %%
fm.darks, fm.flats, fm.images

# %% [markdown]
# ## Some more details
#
# ### Restoring original state 

# %% [markdown]
# When using `set_observation` the `FitsManager` object has been replaced inplace. However, the original information is not lost and can be restored whith

# %%
fm.restore()
fm.observations

# %% [markdown]
# ### Telescope specific keywords

# %% [markdown]
# The information retained by `FitsManager` was taken from images headers. To know which keywords to use, we had to register telescopes `A` and `B` with a dictionary. Whenever their names appear in a fits header, their disctionary is loaded to read their header keywords.
#
# Since we just specified the telescope names all the rest is default. For example the filter is taken from the keyword `FILTER` and the image type from `IMAGETYP`, knowing that `IMAGETYP=light` is a light (a.k.a science) frame. All this can be set in more details when registering the telescope and make prose work with any instrument.
#
# for more details, chcek the `Telescope` object

# %%
# hidden
from shutil import rmtree

rmtree(destination)
