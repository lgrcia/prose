.. _blocks:

Blocks
#######

:py:class:`~Block` objects are dedicated to image analysis and/or modification and are aranged into :py:class:`~Unit`. The following diagram shows how an :py:class:`~Image` object is sequentially processed through the individual blocks of a unit. An example is provided `here <modular-reduction>`_

.. figure:: ../tutorials/modular-reduction/unit_structure.png
   :align: center
   :height: 350

   :py:class:`~prose.Unit` hooks and data flow


A :py:class:`~Block` can write and read the properties of an :py:class:`~Image` object, providing a conveniant way to pass information from block to block.

Here is a list of all blocks available in |prose|:

.. currentmodule:: prose.blocks


.. rubric:: Detection

.. autosummary::
   :nosignatures:

    DAOFindStars
    SegmentedPeaks
    FindPeaks


.. rubric:: Alignment, registration and centroiding

.. autosummary::
   :nosignatures:

    XYShift
    AstroAlignShift
    NNCentroid

.. rubric:: Characterization

.. autosummary::
   :nosignatures:

    Gaussian2D
    Moffat2D

.. rubric:: Photometry

.. autosummary::
   :nosignatures:

    ForcedAperturePhotometry
    MovingAperturePhotometry

