NNCentroid
----------

A Convolutional Neural Network trained to locate the center of a star from a cutout image. Lead to an accurate and very fast method for centroiding (mean square error is 0.03 pixels)

.. figure:: images/nn_centroid.png
   :align: center
   :width: 380px

   Comparison (distance from true center) with :code:`photutils.centroids.2dg_centroid`, which performs a 2D Gaussian fit, on a simulated star.

.. autoclass:: prose.blocks.NNCentroid
	:members: