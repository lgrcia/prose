.. _phots-structure:

Data products: ``.phots``
========================

|prose| stores and retrieves data products in `phots` files. `phots` files are fits files and contain at least:

- a primary header with global information about the observation
- photometry for all stars at all apertures

For example, using the `astropy.io.fits` to show its content : 

.. code-block:: python

    from astropy.io import fits

    hdu = fits.open("./Callisto_20190527_Sp1744-5834_I+z_photometry.phots")
    hdu.info()

.. code-block::

    Filename: .../Callisto_20190527_Sp1744-5834_I+z_photometry.phots
    No.    Name      Ver    Type      Cards   Dimensions   Format
    0  PRIMARY       1 PrimaryHDU       6   ()      
    1  PHOTOMETRY    1 ImageHDU         9   (25, 1561, 6)   float64   
    2  STARS         1 ImageHDU         8   (2, 1561)   float64   
    3  APERTURES     1 ImageHDU         7   (6,)   float64   
    4  FWHM          1 ImageHDU         7   (25,)   float64   
    5  SKY           1 ImageHDU         7   (25,)   float64   
    6  DX            1 ImageHDU         7   (25,)   float64   
    7  DY            1 ImageHDU         7   (25,)   float64   
    8  AIRMASS       1 ImageHDU         7   (25,)   float64   
    9  JD            1 ImageHDU         7   (25,)   float64   
    10  LIGHTCURVES    1 ImageHDU         9   (25, 1561, 6)   float64   
    11  LIGHTCURVES ERRORS    1 ImageHDU         9   (25, 1561, 6)   float64

.. note::

    Be careful: Dimensions here are in reverse orders of what ``numpy.shape()`` would give

Header
-------

The header (`PRIMARY` hdu) contains all necessary informatio, to identify the observation. When produced with |prose|, it contains the header of the stack image, itself being the header of the reference image used for alignment during reduction). 


Data HDUs
---------

The main HDUs in this `phots` file are:

- ``JD`` : Julian dates of the observations
- ``PHOTOMETRY`` : photometry (fluxes), whith shape ``(n_apertures, n_stars, n_images)``
- ``LIGHTCURVES`` : differential photometry, whith shape ``(n_apertures, n_stars, n_images)``
- ``LIGHTCURVES ERRORS`` : their errors, whith shape ``(n_apertures, n_stars, n_images)`` 
- ``STARS`` : stars positions with shape ``(n_stars, 2)``
- ``APERTURES`` : apertures given in pixels, shape is of course ``(n_apertures)``

Other one-dimensional HDUs often correspond to time-series recorded along abservation (with shape ``(n_images)``) such as :

- ``FWHM``
- ``DX``
- ... etc






