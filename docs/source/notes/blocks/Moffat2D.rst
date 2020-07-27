Moffat2D
--------

An elliptical 2D Moffat model expressed as

.. math::   

   f(x, y|A, x_0, y_0, \sigma_x, \sigma_y, \theta, \beta, b) = \frac{A}{\left(1 + \frac{x'-x'_0}{\sigma_x^2} + \frac{y'-y'_0}{\sigma_y^2}\right)^\beta} + b

.. math::

   \text{with}\quad \begin{gather*}
   x' = xcos(\theta) + ysin(\theta) \\
   y' = -xsin(\theta) + ycos(\theta)
   \end{gather*}

is fitted from an effective psf. :code:`scipy.optimize.minimize` is used to minimize :math:`\chi ^2` from data. Initial parameters are found using the moments of the `effective psf <https://photutils.readthedocs.io/en/stable/epsf.html>`_. 

.. autoclass:: prose.blocks.Moffat2D
	:members: