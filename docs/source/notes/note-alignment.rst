.. _note-alignment:

Alignment methods
-----------------


.. code-block:: python

    from specphot import utils, reduction
    import numpy as np


``xyshift``
^^^^^^^^^^^

**xyshift** is originaly the **TRAPHOT** (M. Gillon) method to align consecutive images. Here is a description of this method as implement in **specphot**


.. code-block:: python

    reference = "/Users/lionelgarcia/Data/fake_20180512_cambridge_server/Sp2019-5816/Sp2019-5816-S001-R001-C967-I+z.fts"
    image = "/Users/lionelgarcia/Data/fake_20180512_cambridge_server/Sp2019-5816/Sp2019-5816-S001-R001-C993-I+z.fts"


Principle
"""""""""

Let's consider two images, ``im`` and ``ref``. We want to know the shift between ``im`` and ``ref`` knowing the position of the stars in these two images. We suppose that there is N stars in ``im`` and ``ref`` and that their position are given by

.. math::

    s^{im}_{i,j} = \begin{bmatrix}
    x^{im}_0 & y^{im}_0 \\
    x^{im}_1 & y^{im}_1 \\
    . & . \\
    x^{im}_N & y^{im}_N
    \end{bmatrix} \quad and \quad 
    s^{ref}_{i,j} = \begin{bmatrix}
    x^{ref}_0 & y^{ref}_0 \\
    x^{ref}_1 & y^{ref}_1 \\
    . & . \\
    x^{ref}_N & y^{ref}_N
    \end{bmatrix}


:math:`x_i, y_i` being the coordinates of the star :math:`i`

If we want to know, let's say, the :math:`x` shift between `im` and `ref` we just have to compute :math:`x^{im}_0 - x^{ref}_0`, or to be more accurate, by considering all the stars, the mean shift

.. math::

    \Delta x = \frac{1}{N}\sum_{0}^{N} x^{im}_i - x^{ref}_i


This is possible because :math:`x^{im}_i`  and :math:`x^{ref}_i` are the position of the same identified star :math:`i`.

Identifying stars in the sky, for example using their position with respect to constellations or by trying to match catalogs (as with astrometry.net) is possible with a single image, but is time-consuming when applied to a complete set of images. Unfortunately most of the time, detected stars do not hold the same index :math:`i` (index is attributed by the star detection algorithm).

The goal of **xyshift** is to check all the possible shifts from one star with respect to all the others, and to identify the most common shifts between stars of two different images.



.. image:: ../images/alignement/xyshift_principle.png



xyshift takes as input the positions of the stars in the two images with the shape:

.. math::

    s_{i,j} = \begin{bmatrix}
    x_0 & y_0 \\
    x_1 & y_1 \\
    . & . \\
    x_N & y_N
    \end{bmatrix}


:math:`x_i, y_i` being the coordinates of the star :math:`i`


.. code-block:: python

    ref_stars_pos = utils.daofindstars(reference, n_stars=60)
    im_stars_pos = utils.daofindstars(image, n_stars=60)


A cleaning can be applied so that stars too close one from the other are merged as one. The parameter `tolerance`, given in pixel, indicates the minimal distance between stars and can be set when using ``utils.clean_stars_positions``. ``xyshift`` by default do not apply any cleaning

xyshift starts by computing the matrices :math:`\Delta x` (resp. :math:`\Delta y`) where:

.. math::

    \Delta x_{i,j} = x^{im}_j - x^{ref}_i \quad i.e. \quad \Delta x_{N\times N} = \begin{bmatrix}
    x^{im}_0 - x^{ref}_0 & x^{im}_1 - x^{ref}_0 & . & x^{im}_N - x^{ref}_0 \\
    x^{im}_0 - x^{ref}_1 & x^{im}_1 - x^{ref}_1 & . & x^{im}_N - x^{ref}_1 \\
    x^{im}_0 - x^{ref}_2 & x^{im}_1 - x^{ref}_2 & . & x^{im}_N - x^{ref}_2 \\
    . & . & . &  . \\
    . & . & . &  . \\
    x^{im}_0 - x^{ref}_{N-1} & x^{im}_1 - x^{ref}_{N-1} & . & x^{im}_N - x^{ref}_{N-1} \\
    x^{im}_0 - x^{ref}_N & x^{im}_1 - x^{ref}_N & . & x^{im}_N - x^{ref}_N \\
    \end{bmatrix}


This matrix contains the diference between all the :math:`x^{im}_{i}` of the stars in the image with the :math:`x^{ref}_{i}` of the stars in the reference image (same for :math:`y`).

To optimize the process, :math:`\Delta x` and :math:`\Delta y` are flatten so that


.. code-block:: python

    delta_x = np.array([ref_stars_pos[:, 0] - v for v in im_stars_pos[:, 0]]).flatten()
    delta_y = np.array([ref_stars_pos[:, 1] - v for v in im_stars_pos[:, 1]]).flatten()


The goal is then to find the most common :math:`x^{im} - x^{ref}` values, meaning that several pair of stars will have undergo the same shift from one image to another reference image

To do that, **xyshift** computes the differences between all the :math:`\Delta x_{i,j}` (resp. :math:`\Delta y_{i,j}`) into a new matrix, again flattened for optimisation


.. code-block:: python

    delta_x_compare = []
    for i, dxi in enumerate(delta_x):
        dcxi = dxi - delta_x
        dcxi[i] = np.inf
        delta_x_compare.append(dcxi)

    delta_y_compare = []
    for i, dyi in enumerate(delta_y):
        dcyi = dyi - delta_y
        dcyi[i] = np.inf
        delta_y_compare.append(dcyi)


These flattened 3D matrices, that we will call :math:`\Delta^{\Delta} x` (resp. :math:`\Delta^{\Delta} y`),  are equivalent to:

.. math::

    \Delta^{\Delta} x_{i,j,k} = \Delta x_{i,j} - \Delta x_{k} 

.. math::

    i.e. \quad \Delta^{\Delta} x_{N\times N\times(N\times N)} = \begin{bmatrix}
    \Delta x_{0,0} - \Delta x_{k} & \Delta x_{1,0} - \Delta x_{k} & . & \Delta x_{N,0} - \Delta x_{k}\\
    \Delta x_{0,1} - \Delta x_{k} & \Delta x_{1,1} - \Delta x_{k} & . & \Delta x_{N,1} - \Delta x_{k}\\
    \Delta x_{0,2} - \Delta x_{k} & \Delta x_{1,2} - \Delta x_{k} & . & \Delta x_{N,2} - \Delta x_{k}\\
    . & . & . &  . \\
    . & . & . &  . \\
    \Delta x_{0,N-1} - \Delta x_{k} & \Delta x_{1,N-1} - \Delta x_{k} & . & \Delta x_{N,N-1} - \Delta x_{k}\\ \\
    \Delta x_{0,N} - \Delta x_{k} & \Delta x_{1,N} - \Delta x_{k} & . & \Delta x_{N,N} - \Delta x_{k}\\ \\
    \end{bmatrix}


when :math:`\Delta^{\Delta} x` and :math:`\Delta^{\Delta} y` are built, **xyshift** checks the differences that looks alike by looking at the differences very close from one to the other


.. code-block:: python

    tolerance = 1.5

    tests = [
        np.logical_and(np.abs(dxc) < tolerance, np.abs(dyc) < tolerance)
        for dxc, dyc in zip(delta_x_compare, delta_y_compare)
    ]


and counts how many of them are similar


.. code-block:: python

    num = np.array([np.count_nonzero(test) for test in tests])


Finally it takes the one with the highest occurence


.. code-block:: python

    max_count_num_i = int(np.argmax(num))
    max_nums_ids = np.argwhere(num == num[max_count_num_i]).flatten()
    dxs = np.array([delta_x[np.where(tests[i])] for i in max_nums_ids])
    dys = np.array([delta_y[np.where(tests[i])] for i in max_nums_ids])

    np.array([np.mean(dxs), np.mean(dys)])

    -> array([ 2.12482829, -0.78454003])

