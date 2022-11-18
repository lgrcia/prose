Blocks
======

.. image:: ../static/block.png
   :align: center
   :width: 200px


A ``Block`` is a single unit of processing acting on the ``Image`` object, reading and writing its attributes. Blocks documentation include the following information:

- |read|: the ``Image`` attributes read by the ``Block``
- |write|: the ``Image`` attributes written by the ``Block``
- |modify|: indicates that the ``Image.data`` is modified by the ``Block``

Detection
---------

.. currentmodule:: prose.blocks.detection

.. image:: ../static/detection.png
   :align: center
   :height: 230px

.. autosummary::
   :toctree: generated
   :template: blocksum.rst
   :nosignatures:

   SegmentedPeaks
   SEDetection
   DAOFindStars
   

PSF
---

.. currentmodule:: prose.blocks.psf

.. autosummary::
   :toctree: generated
   :template: blocksum.rst
   :nosignatures:

   Cutouts
   MedianPSF
   FWHM
   Gaussian2D
   Moffat2D
   FastGaussian

Alignment, Centroiding
---------------------------

.. currentmodule:: prose.blocks

.. image:: ../static/matching.png
   :align: center
   :height: 220px


.. autosummary::
   :toctree: generated
   :template: blocksum.rst
   :nosignatures:

   XYShift
   Twirl
   AffineTransform
   Cutout2D

.. currentmodule:: prose.blocks.centroids

.. autosummary::
   :toctree: generated
   :template: blocksum.rst
   :nosignatures:

   COM
   Quadratic
   Gaussian2D
   BalletCentroid

Photometry
----------

.. currentmodule:: prose.blocks

.. image:: ../static/photometry.png
   :align: center
   :width: 220px


.. autosummary::
   :toctree: generated
   :template: blocksum.rst
   :nosignatures:

   PhotutilsAperturePhotometry
   PhotutilsPSFPhotometry
   SEAperturePhotometry
   
Utils
-----

.. image:: ../static/utils.png
   :align: center
   :height: 190px
   
.. autosummary::
   :toctree: generated
   :template: blocksum.rst
   :nosignatures:

   Calibration
   Cutouts
   Flip
   ImageBuffer
   Pass
   SaveReduced
   Set
   Stack
   Trim
   XArray

All
---

.. include:: all_blocks.rst
