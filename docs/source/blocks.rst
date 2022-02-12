Blocks
======

.. currentmodule:: prose.blocks

.. image:: _static/block.png
   :align: center
   :width: 200px


A ``Block`` is a single unit of processing acting on the ``Image`` object, reading and writing its attributes. Blocks documentation include the following information:

- |read|: the ``Image`` attributes read by the ``Block``
- |write|: the ``Image`` attributes written by the ``Block``
- |modify|: indicates that the ``Image.data`` is modified by the ``Block``

Detection
---------

.. image:: _static/detection.png
   :align: center
   :height: 230px

.. autosummary::
   :toctree: generated
   :template: autosum.rst
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
   :template: autosum.rst
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
   :template: autosum.rst
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
   :template: autosum.rst
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
   Video
   Plot
   XArray
