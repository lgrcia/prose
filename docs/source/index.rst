.. prose documentation master file, created by
   sphinx-quickstart on Mon Apr 27 15:15:45 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

prose
=====

A framework for FITS processing pipelines in python. Built for Astronomy, |prose| features pipelines to perform common tasks (such as automated calibration, reduction and photometry) and make building custom ones easy.

.. image:: prose.png
   :width: 450
   :align: center


.. toctree::
   :caption: References
   :maxdepth: 1

   guide/installation
   guide/citing
   guide/api

.. toctree::
   :caption: Tutorials
   :maxdepth: 1

   tutorials/reduction/reduction.ipynb
   tutorials/modular-reduction/custom_pipeline.ipynb


.. toctree::
   :caption: Notes
   :maxdepth: 1

   notes/phot.ipynb
   notes/telescope-config
   notes/blocks