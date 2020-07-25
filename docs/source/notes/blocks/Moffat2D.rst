Moffat2D
--------

An elliptical 2D Moffat model expressed as

.. math::   

   f(x, y|A, x_0, y_0, \sigma_x, \sigma_y, 	heta, eta, b) = rac{A}{\left(1 + rac{x'-x'_0}{\sigma_x^2} + rac{y'-y'_0}{\sigma_y^2}ight)^eta} + b

.. math::

   	ext{with}\quad egin{gather*}
   x' = xcos(	heta) + ysin(	heta) \
   y' = -xsin(	heta) + ycos(	heta)
   \end{gather*}

is fitted from an effective psf. :code:`scipy.optimize.minimize` is used to minimize :math:`\chi ^2` from data. Initial parameters are found using the moments of the `effective psf <https://photutils.readthedocs.io/en/stable/epsf.html>`_. 

.. autoclass:: prose.blocks.Moffat2D
	:members: