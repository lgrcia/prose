# ePSF


The task of the ePSF blocks is to build and model the [effective PSF](https://photutils.readthedocs.io/en/stable/epsf.html) on an [Image](prose.Image).


## ePSF building blocks

The following blocks can be used to build an effective PSF from a set of stars. Once an effective PSF constructed, it is stored as an [Image](prose.Image) in `Image.epsf`.


```{eval-rst}
.. currentmodule:: prose.blocks.psf

.. autosummary::
   :template: blocksum.rst
   :nosignatures:

   MedianEPSF

```

## ePSF modeling blocks

```{eval-rst}
.. autosummary::
   :template: blocksum.rst
   :nosignatures:

   Gaussian2D
   Moffat2D
   JAXGaussian2D
   JAXMoffat2D

```