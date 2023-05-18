Detection
---------

.. image:: ../_static/detection.png
   :align: center
   :height: 230px

The task of the detection blocks is to detect and set :py:class:`~prose.core.source.Sources` in an :py:class:`~prose.Image` (written in the ``sources`` attribute of the image).

Available blocks
^^^^^^^^^^^^^^^^
.. currentmodule:: prose.blocks.detection

.. autosummary::
   :template: blocksum.rst
   :nosignatures:

   AutoSourceDetection
   PointSourceDetection   
   DAOFindStars
   SEDetection
   TraceDetection