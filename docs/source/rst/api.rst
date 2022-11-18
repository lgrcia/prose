.. _api:

API
===

.. currentmodule:: prose

Image processing
----------------

Core objects to deal with astronomical image representation and processing. See the `fits manager`_ or quickstart_ tutorials.

.. _quickstart: ../notebooks/quickstart.ipynb
.. _`fits manager`: ../notebooks/fits_manager.ipynb


.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: class.rst

   Telescope
   FitsManager
   Image
   Block
   Sequence

Sources
-------

Objects to represent sources in astronomical images. See the sources_ tutorial.

.. _sources: ../notebooks/sources.ipynb

.. currentmodule:: prose.core.source

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: class.rst

   PointSource
   ExtendedSource
   TraceSource

Other
---------

.. currentmodule:: prose


.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: class.rst

   fluxes.ApertureFluxes
   Observation