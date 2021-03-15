Blocks
======

.. currentmodule:: prose.blocks

A ``Block`` is a single unit of processing acting on the ``Image`` object, reading and writing its attributes.

.. image:: _static/block.png
   :align: center
   :width: 220px

   

Detection
---------

.. autosummary::
   :toctree: generated
   :nosignatures:

   SegmentedPeaks
   SEDetection
   DAOFindStars
   

Alignment, Centroiding, PSF
---------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   XYShift
   Align
   AstroAlignShift
   Gaussian2D
   Moffat2D
   FastGaussian

Photometry
----------

.. autosummary::
   :toctree: generated
   :nosignatures:

   PhotutilsAperturePhotometry
   PhotutilsPSFPhotometry
   SEAperturePhotometry
   
Utils
-----
   
.. autosummary::
   :toctree: generated
   :nosignatures:

   Align
   Calibration
   CleanCosmics
   Cutouts
   Flip
   ImageBuffer
   Pass
   RemoveBackground
   SavePhot
   SaveReduced
   Set
   Stack
   StackStd
   Trim
   Video
   Plot
