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

    Filename: Callisto_20190527_Sp1744-5834_I+z_photometry.phots
    No.    Name             Ver    Type      Cards   Dimensions   Format
    0  PRIMARY              1 PrimaryHDU     147  ()      
    1  PHOTOMETRY           1 ImageHDU       9    (92, 332, 40)   float64   
    2  PHOTOMETRY ERRORS    1 ImageHDU       9    (92, 332, 40)   float64   
    3  STARS                1 ImageHDU       8    (2, 332)   float64   
    4  TIME SERIES          1 BinTableHDU    37   92R x 14C   ...
    5  APERTURES AREA       1 ImageHDU       8    (92, 40)   float64   
    6  ANNULUS AREA         1 ImageHDU       7    (92,)   float64   
    7  ANNULUS SKY          1 ImageHDU       8    (92, 332)   float64   
    8  LIGHTCURVES          1 ImageHDU       9    (92, 332, 40)   float64   
    9  LIGHTCURVES ERRORS   1 ImageHDU       9    (92, 332, 40)   float64   
    10  COMPARISON STARS    1 ImageHDU       8    (30, 40)   int64   
    11  ARTIFICIAL LCS      1 ImageHDU       8    (92, 40)   float64  

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

Other HDUs often correspond to additional data recorded during reduction and photometry. One particularly useful HDU is the :code:`TIME SERIES` containing data recorded along time. This HDU is a BinTableHDU and can be read in the following:

.. code-block:: python

    from astropy.table import Table

    dataframe = Table(hdus["TIME SERIES"].data)

Usually the Photometry object contains attributes to access these data, for example in :code:`Photometry.data` data-frame.


