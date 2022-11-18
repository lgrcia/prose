.. prose documentation master file, created by
   sphinx-quickstart on Mon Apr 27 15:15:45 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


.. role:: underline
    :class: underline

   
.. image:: static/prose_illustration.png
   :width: 350
   :align: center
   
.. raw:: html

   <p align="center">
   A python framework to build FITS images pipelines.
   <br>
   <p align="center">
      <a href="https://github.com/lgrcia/prose">
         <img src="https://img.shields.io/badge/github-lgrcia/prose-blue.svg?style=flat" alt="github"/>
      </a>
      <a href="">
         <img src="https://img.shields.io/badge/license-MIT-lightgray.svg?style=flat" alt="license"/>
      </a>
      <a href="https://arxiv.org/abs/2111.02814">
         <img src="https://img.shields.io/badge/paper-yellow.svg?style=flat" alt="paper"/>
      </a>
      <a href="https://lgrcia.github.io/prose-docs">
         <img src="https://img.shields.io/badge/documentation-black.svg?style=flat" alt="documentation"/>
      </a>
   </p>
   </p>


|prose| is a Python tool to build pipelines dedicated to astronomical images processing (all based on pip packages 📦). Beyond providing all the blocks to do so, it features default pipelines to perform common tasks such as automated calibration, reduction and photometry.


.. toctree::
   :caption: References
   :maxdepth: 2

   rst/installation.md
   notebooks/quickstart.ipynb
   rst/core
   notebooks/sources.ipynb
   rst/blocks
   rst/api
   notebooks/acknowledgement.ipynb
   

.. toctree::
   :caption: Tutorials
   :maxdepth: 1

   notebooks/fits_manager.ipynb
   notebooks/photometry.ipynb
   notebooks/reports.ipynb
   notebooks/custom_block.ipynb
   notebooks/custom_block_2.ipynb
   notebooks/modeling.ipynb
   notebooks/archival.ipynb
   notebooks/catalogs.ipynb

.. toctree::
   :caption: Case studies 🔭
   :maxdepth: 1

   notebooks/diagnostic_video.ipynb
   notebooks/hiaka_occultation.ipynb
   
.. toctree::
   :caption: Notes
   :maxdepth: 1

   notebooks/phot.ipynb
   notebooks/extra.ipynb

.. image:: static/lookatit.png
   :width: 150
   :align: center