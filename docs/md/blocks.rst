Blocks
======

.. image:: ../_static/block.png
   :align: center
   :width: 200px


A ``Block`` is a single unit of processing acting on the ``Image`` object, reading and writing its attributes. Blocks documentation include the following information:

- |read|: the ``Image`` attributes read by the ``Block``
- |write|: the ``Image`` attributes written by the ``Block``
- |modify|: indicates that the ``Image.data`` is modified by the ``Block``

Blocks categories
-----------------

.. toctree::
   :maxdepth: 1

   detection_blocks.rst
   psf_blocks
   geometry_blocks
   centroiding_blocks
   photometry_blocks
   utils_blocks.rst

All Blocks
----------

.. include:: all_blocks.rst
