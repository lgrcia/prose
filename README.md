# prose

<p align="center" style="margin-bottom:-50px">
    <img src="docs/_static/prose3.png" width="450">
</p>

<p align="center">
  Modular image processing pipelines for Astronomy
  <br>
  <p align="center">
    <a href="https://github.com/lgrcia/prose">
      <img src="https://img.shields.io/badge/github-lgrcia/prose-03A487.svg?style=flat" alt="github"/>
    </a>
    <a href="LICENCE">
      <img src="https://img.shields.io/badge/license-MIT-lightgray.svg?style=flat" alt="license"/>
    </a>
    <a href="https://arxiv.org/abs/2111.02814">
      <img src="https://img.shields.io/badge/paper-B166A9.svg?style=flat" alt="paper"/>
    </a>
    <a href="https://prose.readthedocs.io/en/3.0.0">
      <img src="https://img.shields.io/badge/documentation-black.svg?style=flat" alt="documentation"/>
    </a>
  </p>
</p>

 *prose* is a Python package to build image processing pipelines for Astronomy. Beyond featuring the blocks to build pipelines from scratch, it provides pre-implemented ones to perform common tasks such as automated calibration, reduction and photometry.

*powered by [astropy](https://www.astropy.org/) and [photutils](https://photutils.readthedocs.io)*!

## Example

Here is a quick example pipeline to characterize the point-spread-function (PSF) of an example image


```python
import matplotlib.pyplot as plt
from prose import Sequence, blocks
from prose.simulations import example_image

# getting the example image
image = example_image()

sequence = Sequence(
    [
        blocks.PointSourceDetection(),  # stars detection
        blocks.Cutouts(shape=21),  # cutouts extraction
        blocks.MedianEPSF(),  # PSF building
        blocks.Moffat2D(),  # PSF modeling
    ]
)

sequence.run(image)

# plotting
image.show()  # detected stars

# effective PSF parameters
image.epsf.params
```

While being run on a single image, a Sequence is designed to be run on list of images (paths) and provides the architecture to build powerful pipelines. For more details check [Quickstart](https://prose.readthedocs.io/en/latest/ipynb/quickstart.html) and [What is a pipeline?](https://prose.readthedocs.io/en/latest/ipynb/core.html)

## Installation

### latest

*prose* is written for python 3 and can be installed from [pypi](https://pypi.org/project/prose/) with:

```shell
pip install prose
```

For the latest version 

```shell
pip install 'prose @ git+https://github.com/lgrcia/prose'
```

## Contributions
See our [contributions guidelines](docs/CONTRIBUTING.md)

## Attribution

If you find `prose` useful for your research, cite [Garcia et. al 2022](https://ui.adsabs.harvard.edu/abs/2022MNRAS.509.4817G). The BibTeX entry for the paper is:
```
@ARTICLE{prose,
       author = {{Garcia}, Lionel J. and {Timmermans}, Mathilde and {Pozuelos}, Francisco J. and {Ducrot}, Elsa and {Gillon}, Micha{\"e}l and {Delrez}, Laetitia and {Wells}, Robert D. and {Jehin}, Emmanu{\"e}l},
        title = "{PROSE: a PYTHON framework for modular astronomical images processing}",
      journal = {\mnras},
     keywords = {instrumentation: detectors, methods: data analysis, planetary systems, Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - Earth and Planetary Astrophysics},
         year = 2022,
        month = feb,
       volume = {509},
       number = {4},
        pages = {4817-4828},
          doi = {10.1093/mnras/stab3113},
archivePrefix = {arXiv},
       eprint = {2111.02814},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2022MNRAS.509.4817G},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

and read about how to cite the dependencies of your sequences [here](https://prose.readthedocs.io/en/latest/ipynb/acknowledgement.html).

