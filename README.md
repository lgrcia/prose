# prose

<p align="center">

  <img width="450" src="https://github.com/lgrcia/prose/blob/master/docs/source/prose_illustration.png">
  <br>  
  <br>
  A python framework to build FITS images pipelines.
  <br>
  <p align="center">
    <a href="https://github.com/lgrcia/prose">
      <img src="https://img.shields.io/badge/github-lgrcia/prose-blue.svg?style=flat" alt="github"/>
    </a>
    <a href="https://lgrcia.github.io/prose-docs">
      <img src="https://img.shields.io/badge/read-thedoc-black.svg?style=flat" alt="read the doc"/>
    </a>
    <a href="">
      <img src="https://img.shields.io/badge/license-MIT-lightgray.svg?style=flat" alt="license"/>
    </a>
  </p>
</p>

 *prose* is a Python tool to build pipelines dedicated to astronomical images processing (all based on pip packages 📦). Beyond providing all the blocks to do so, it features default pipelines to perform common tasks such as automated calibration, reduction and photometry.

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
    blocks.Moffat2D(),        # PSF modeling
])

sequence.run([image])
```

For more details check [Quickstart](https://lgrcia.github.io/prose/build/html/notebooks/quickstart.html).

## Default pipelines


```python

from prose.pipeline import Calibration, AperturePhotometry

destination = "reduced_folder"

reduction = Calibration(images=[...], flats=[...])
reduction.run(destination)

photometry = AperturePhotometry(destination)
photometry.run()

```

## Installation

prose is written for python 3 and can be installed from pypi with:

```shell
pip install prose
```

To install it through conda, once in your newly created environment, go with:


```shell
conda install numpy scipy tensorflow netcdf4 numba

# then 

pip install prose
```