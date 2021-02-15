Observation
===========

|prose| :py:class:`~prose.Observation` class holds methods to manipulate photometric products

.. currentmodule:: prose.Observation


Quick start
-----------

.. rubric:: Cleaning

The philosophy of prose it to extract photometry from all images with no prior assumption about their quality. You can decide to exclude some measurements by using the following methods:

.. autosummary::
   :nosignatures:

    where
    sigma_clip
    keep_good_stars
    

.. rubric:: Differential photometry

.. autosummary::
   :nosignatures:

    diff
    broeg2005

.. rubric:: Plotting

.. autosummary::
   :nosignatures:

    show_stars
    show_gaia
    show_tic
    show_cutout
    plot_comps_lcs
    plot_psf_fit

.. rubric:: Modeling

.. autosummary::
   :nosignatures:

    polynomial
    transit
    trend
    best_polynomial

Complete API
------------
.. autoclass:: prose.Observation
    :members:
    :inherited-members:
