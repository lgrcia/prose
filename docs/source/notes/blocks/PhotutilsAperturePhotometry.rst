PhotutilsAperturePhotometry
---------------------------

Aperture photometry using the :code:`CircularAperture` and :code:`CircularAnnulus` of photutils_ with a wide range of apertures. By default annulus goes from 5 fwhm to 8 fwhm and apertures from 0.1 to 10 times the fwhm with 0.25 steps (leading to 40 apertures).

The error (e.g. in ADU) is then computed following:

.. math::

   \sigma = \sqrt{S + (A_p + \frac{A_p}{A_n})(b + r^2 + \frac{gain^2}{2}) + scint }


.. image:: images/aperture_phot.png
   :align: center
   :width: 110px

with :math:`S` the flux (ADU) within an aperture of area :math:`A_p`, :math:`b` the background flux (ADU) within an annulus of area :math:`A_n`, :math:`r` the read-noise (ADU) and :math:`scint` is a scintillation term expressed as:


.. math::

   scint = \frac{S_fd^{2/3} airmass^{7/4} h}{16T}

with :math:`S_f` a scintillation factor, :math:`d` the aperture diameter (m), :math:`h` the altitude (m) and :math:`T` the exposure time.

The positions of individual stars are taken from :code:`Image.stars_coords` so one of the detection block should be used, placed before this one.

.. autoclass:: prose.blocks.PhotutilsAperturePhotometry
	:members: