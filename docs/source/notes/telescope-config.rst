.. _telescope-config:

Telescope setting
=================

``fits`` headers can change from one observatory to another. |prose| uses dictionnaries to analyse any telescope data with the following format:

.. code-block:: yaml

    name: "SSO"

    # Keywords translation
    keyword_object: "OBJECT"
    keyword_image_type: "IMAGETYP"
    keyword_light_images: "light"
    keyword_dark_images: "dark"
    keyword_flat_images: "flat"
    keyword_bias_images: "bias"     
    keyword_observation_date: "DATE-OBS"
    keyword_exposure_time: "EXPTIME"
    keyword_filter: "FILTER"
    keyword_observatory: "OBSERVAT"
    keyword_julian_date: "JD"
    keyword_ra: "RA"    
    keyword_dec: "DEC"  
    keyword_flip: "PIERSIDE"
    keyword_fwhm: "FWHM"

    # Telescope specs
    trimming: [8, 22] # pixels
    read_noise: 10 # ADU
    gain: 1.02 # ADU/e-
    altitude: 2000 # m
    diameter: 100 # cm
    pixel_scale: 0.33 # arcsec
    latlong: : [24.6275, 70.4044] # deg

These dictionaries are used by the ``Telescope`` object used by |prose| which can be instantiated from:

- a **python** ``dict``:

    .. code-block:: python

        telescope_dict = {
            keyword_object: "OBJECT"
            keyword_image_type: "IMAGETYP"
            ...
            gain: 10
            altitude: 1500
            diameter: 80
        }

        telescope = Telescope(telescope_dict)


- a ``.yaml`` file with a structure similar to the one shown above:

    .. code-block:: yaml

        keyword_object: "OBJECT"
        keyword_image_type: "IMAGETYP"
        ...
        gain: 10
        altitude: 1500
        diameter: 80

    and then

    .. code-block:: python

        telescope = Telescope("path_to/my_telescope.yaml")

When working with a new telescope, this operation needs to be done only **once**, after which the telescope dictionnary is saved and automatically used whenever the telescope name is encountered in a fits header. See :ref:`reduction` for a use case.