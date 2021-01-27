:orphan:
SEAperturePhotometry
--------------------


A fast detection algorithm which:

- segment the image in blobs with pixels above a certain threshold
- compute the centroids of individual blobs as stars positions

.. image:: images/segmentation.png
   :align: center
   :height: 260px

This method should be used when speed is required over accuracy. Uses scikit-image_.


.. autoclass:: prose.blocks.SEAperturePhotometry
	:members: