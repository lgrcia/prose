Blocks
======

.. currentmodule:: prose.blocks


A ``Block`` is a single unit of processing acting on the ``Image`` object, reading and writing its attributes.

Detection
---------

.. image:: _static/detection.png
   :align: center
   :height: 230px

.. autosummary::
   :toctree: generated
   :nosignatures:

   SegmentedPeaks
   SEDetection
   DAOFindStars
   

Alignment, Centroiding, PSF
---------------------------

.. image:: _static/matching.png
   :align: center
   :height: 220px


.. autosummary::
   :toctree: generated
   :nosignatures:

   XYShift
   Twirl
   Align
   AffineTransform
   Cutout2D
   AstroAlignShift
   Gaussian2D
   Moffat2D
   FastGaussian

Photometry
----------

.. image:: _static/photometry.png
   :align: center
   :width: 220px


.. autosummary::
   :toctree: generated
   :nosignatures:

   PhotutilsAperturePhotometry
   PhotutilsPSFPhotometry
   SEAperturePhotometry
   
Utils
-----

.. image:: _static/utils.png
   :align: center
   :height: 190px
   
.. autosummary::
   :toctree: generated
   :nosignatures:

   Align
   Calibration
   Cutouts
   Flip
   ImageBuffer
   Pass
   SavePhot
   SaveReduced
   Set
   Stack
   Trim
   Video
   Plot
   XArray
