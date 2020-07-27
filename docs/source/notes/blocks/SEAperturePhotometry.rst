SEAperturePhotometry
--------------------

Aperture photometry using `sep <https://sep.readthedocs.io>`_, a python wrapper around the C Source Extractor.

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

.. autoclass:: prose.blocks.SEAperturePhotometry
	:members: