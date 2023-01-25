API
===


.. currentmodule:: prose

Image processing
----------------

Core objects to deal with astronomical image representation and processing. See the `fits manager`_ or quickstart_ tutorials.

.. _quickstart: ../notebooks/quickstart.ipynb
.. _`fits manager`: ../notebooks/fits_manager.ipynb


.. autosummary::
   :nosignatures:
   :template: class.rst

   Telescope
   FitsManager
   Image
   FITSImage
   Block
   Sequence

Sources
-------

Objects to represent sources in astronomical images. See the sources_ tutorial.

.. _sources: ../ipynb/sources.ipynb

.. currentmodule:: prose.core.source

.. autosummary::
   :nosignatures:
   :template: class.rst

   PointSource
   ExtendedSource
   TraceSource

Other
---------

.. currentmodule:: prose


.. autosummary::
   :nosignatures:
   :template: class.rst

   fluxes.ApertureFluxes
   Observation