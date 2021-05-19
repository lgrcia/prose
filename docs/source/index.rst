.. prose documentation master file, created by
   sphinx-quickstart on Mon Apr 27 15:15:45 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

prose
=====

A python framework to process FITS images. 

.. image:: prose.png
   :width: 450
   :align: center

.. image:: https://img.shields.io/badge/github-lgrcia/prose-blue.svg?style=flat
    :target: https://github.com/lgrcia/prose
.. image:: https://img.shields.io/badge/read-thedoc-black.svg?style=flat
    :target: https://prose.readthedocs.io/en/latest/
.. image:: https://img.shields.io/badge/license-MIT-lightgray.svg?style=flat

Built for Astronomy, |prose| features pipelines to perform common tasks (such as automated calibration, reduction and photometry) and makes building custom ones easy.


.. toctree::
   :caption: References
   :maxdepth: 1

   installation
   citing
   api
   core

.. toctree::
   :caption: Tutorials
   :maxdepth: 1

   notebooks/fits_manager.ipynb
   notebooks/reduction.ipynb
   notebooks/manual_reduction.ipynb
   notebooks/custom_pipeline.ipynb
   notebooks/modeling.ipynb
   notebooks/reports.ipynb
   
.. toctree::
   :caption: Notes
   :maxdepth: 1

   notebooks/phot.ipynb
   notebooks/extra.ipynb