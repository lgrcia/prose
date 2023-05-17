Blocks
======

.. image:: ../_static/block.png
   :align: center
   :width: 200px


A ``Block`` is a single unit of processing acting on the ``Image`` object, reading and writing its attributes. Blocks documentation include the following information:

- |read|: the ``Image`` attributes read by the ``Block``
- |write|: the ``Image`` attributes written by the ``Block``
- |modify|: indicates that the ``Image.data`` is modified by the ``Block``

blocks categories
-----------------

.. toctree::
   :maxdepth: 0

   detection_blocks.rst
   psf_blocks.rst
   geometry_blocks.rst
   centoiding_blocks.rst
   photometry_blocks.rst
   utils_blocks.rst
   all_blocks.rst