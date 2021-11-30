.. prose documentation master file, created by
   sphinx-quickstart on Mon Apr 27 15:15:45 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


.. role:: underline
    :class: underline


prose
=====

.. image:: prose_illustration.png
   :width: 500
   :align: center

.. image:: https://img.shields.io/badge/github-lgrcia/prose-blue.svg?style=flat
    :target: https://github.com/lgrcia/prose
.. image:: https://img.shields.io/badge/read-thedoc-black.svg?style=flat
    :target: https://prose.readthedocs.io/en/latest/
.. image:: https://img.shields.io/badge/license-MIT-lightgray.svg?style=flat

|prose| is a tool to build pipelines dedicated to astronomical images processing, :underline:`only based on pip installable dependencies` (e.g. no IRAF, Sextractor or Astrometry.net install needed 🎉). It features default pipelines to perform common tasks (such as automated calibration, reduction and photometry) and makes building custom ones easy.


.. toctree::
   :caption: References
   :maxdepth: 1

   installation
   core
   blocks
   api
   

.. toctree::
   :caption: Tutorials
   :maxdepth: 1

   notebooks/fits_manager.ipynb
   notebooks/reduction.ipynb
   notebooks/manual_reduction.ipynb
   notebooks/custom_pipeline.ipynb
   notebooks/modeling.ipynb
   notebooks/reports.ipynb
   notebooks/neb.ipynb
   
.. toctree::
   :caption: Notes
   :maxdepth: 1

   notebooks/phot.ipynb
   notebooks/extra.ipynb