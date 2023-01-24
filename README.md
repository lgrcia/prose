# prose

<p align="center">
    <img src="docs/source/static/prose_illustration.png" width="350">
</p>

<p align="center">
  A python package to build image processing pipelines. Built for Astronomy
  <br>
  <p align="center">
    <a href="https://github.com/lgrcia/prose">
      <img src="https://img.shields.io/badge/github-lgrcia/prose-blue.svg?style=flat" alt="github"/>
    </a>
    <a href="">
      <img src="https://img.shields.io/badge/license-MIT-lightgray.svg?style=flat" alt="license"/>
    </a>
    <a href="https://arxiv.org/abs/2111.02814">
      <img src="https://img.shields.io/badge/paper-yellow.svg?style=flat" alt="paper"/>
    </a>
    <a href="https://prose.readthedocs.io">
      <img src="https://img.shields.io/badge/documentation-black.svg?style=flat" alt="documentation"/>
    </a>
  </p>
</p>

 *prose* is a Python package to build image processing pipelines, built for Astronomy and using only pip packages 📦. Beyond featuring the blocks to build pipelines from scratch, it provides pre-implemented ones to perform common tasks such as automated calibration, reduction and photometry.

*powered by*
<p align="center">
  <a href="https://www.astropy.org/">
  <img src="https://docs.astropy.org/en/stable/_static/astropy_banner.svg" height=50/>
  </a>
  <a href="https://photutils.readthedocs.io">
  <img src="https://photutils.readthedocs.io/en/stable/_static/photutils_banner.svg" height=50/>
  </a>
  <p align="center">

## Example

Here is a quick example pipeline to characterize the point-spread-function (PSF) of an example image


```python
from prose import Sequence, blocks
from prose.tutorials import example_image
import matplotlib.pyplot as plt

# getting the example image
image = example_image()

sequence = Sequence([
    blocks.SegmentedPeaks(),  # stars detection
    blocks.Cutouts(size=21),  # cutouts extraction
    blocks.MedianPSF(),       # PSF building
    blocks.psf.Moffat2D(),    # PSF modeling
])

sequence.run(image)

# plotting
image.show()           # detected stars
image.plot_psf_model() # PSF model
```

While being run on a single image, a Sequence is designed to be run on list of images (paths) and provides the architecture to build powerful pipelines. For more details check [Quickstart](https://prose.readthedocs.io/en/latest/notebooks/quickstart.html) and [What is a pipeline?](https://prose.readthedocs.io/en/latest/rst/core.html)

## Default pipelines
 *prose* features default pipelines to perform common tasks like:

```python

from prose.pipeline import Calibration, AperturePhotometry

destination = "reduced_folder"

reduction = Calibration(darks=[...], flats=[...])
reduction.run(images, destination)

photometry = AperturePhotometry(calib.images, calib.stack)
photometry.run(calib.phot)

```

However, the package is designed to avoid pre-implemented black-boxes, in favor of transparent pipelines. For a practical illustration of that, check our [Photometry tutorial](https://prose.readthedocs.io/en/latest/notebooks/photometry.html).

## Installation

### latest

*prose* is written for python 3 and can be installed from [pypi](https://pypi.org/project/prose/) with:

```shell
pip install prose
```

To install it through conda (recommended, within a fresh environment):

```shell
conda install numpy scipy tensorflow netcdf4 numba

# then 

pip install prose
```

### dev

clone the repo

```shell
git clone https://github.com/lgrcia/prose.git
```

install locally (if within conda, same environment setup as above)

```
pip install -e {path_to_repo}
```


## Helping us

We are interested in seeing how you use prose, as well as helping creating blocks you need. Do not hesitate to reach us out! ☎️

<p align="center">
    <img src="docs/source/static/lookatit.png" width="150">
</p>
