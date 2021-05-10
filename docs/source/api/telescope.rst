Telescope
=========

``FITS`` headers can change from one observatory to another. To analyse data from any telescope, |prose| makes use of the :py:class:`~prose.Telescope` object which stores header translations in a dictionnary, for example:

.. code-block:: python3

    from prose import Telescope

    telescope_dict = dict(

        # name and alternatives
        name="OMM",
        names=["OMM", "OMM-1.6m"],
        
        # Keywords translation
        keyword_object="OBJECT",
        keyword_image_type="OBJECT",
        keyword_light_images="TOI",
        keyword_dark_images="dark",
        keyword_flat_images="flat",
        keyword_bias_images="bias",
        keyword_observation_date="DATE-OBS",
        keyword_exposure_time="EXPOSURE",
        keyword_filter="FILTER",
        keyword_observatory="TELESCOP",
        keyword_jd="", # empty means computed from date
        keyword_ra="RA",
        keyword_dec="DEC",
        keyword_flip="PIERSIDE",
        keyword_fwhm="FWHM",
        
        # units
        ra_unit="hourangle",

        # Telescope specs
        trimming=[5, 5], # pixels
        read_noise=10, # ADU
        gain=1.02, # ADU/e-
        altitude=2000, # m
        diameter=100, # cm
        pixel_scale=0.4598573, # arcsec
        latlong=[45.455556, -71.152778], # deg
    )

    Telescope(telescope_dict)

Once a new telescope is instantiated its dictionary is permanantly saved by prose and automatically used whenever the telescope name is encountered in a fits header. Saved telescopesare located in ``~/.prose`` as yaml files.

API
---
.. autoclass:: prose.Telescope
    :members: