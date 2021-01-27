:orphan:
XYShift
-------

**xyshift** is originaly the **TRAPHOT** (M. Gillon) method to align consecutive images.

*Principle*: Let's consider two images, ``im`` and ``ref``. We want to know the shift between ``im`` and ``ref`` knowing the position of the stars in these two images. We suppose that there is N stars in ``im`` and ``ref`` and that their position are given by

.. math::

   \boldsymbol{S^{im}} = \begin{bmatrix}
   x^{im}_0 & y^{im}_0 \\
   x^{im}_1 & y^{im}_1 \\
   . & . \\
   x^{im}_N & y^{im}_N
   \end{bmatrix} \quad and \quad 
   \boldsymbol{S^{ref}} = \begin{bmatrix}
   x^{ref}_0 & y^{ref}_0 \\
   x^{ref}_1 & y^{ref}_1 \\
   . & . \\
   x^{ref}_N & y^{ref}_N
   \end{bmatrix}


:math:`x_i, y_i` being the coordinates of the star :math:`i`

If we want to know, let's say, the :math:`x` shift between `im` and `ref` we just have to compute :math:`x^{im}_0 - x^{ref}_0`, or to be more accurate, by considering all the stars, the mean shift

.. math::

    \Delta x = \frac{1}{N}\sum_{0}^{N} x^{im}_i - x^{ref}_i


This is possible because :math:`x^{im}_i`  and :math:`x^{ref}_i` are the position of the same identified star :math:`i` (:math:`N` of them in each image)

Identifying stars in the sky, for example using their position with respect to constellations or by trying to match catalogs (as with astrometry.net) is possible with a single image, but is time-consuming when applied to a complete set of images. Unfortunately most of the time detected stars do not hold the same index :math:`i` (index is attributed by the star detection algorithm) and their number might differ.

The goal of **xyshift** is to check all the possible shifts from one star with respect to all the others, and to identify the most common shifts between stars of two different images.

.. image:: images/xyshift_principle.png

.. autoclass:: prose.blocks.XYShift
	:members: