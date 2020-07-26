.. _reduction:

Basic reduction
===============

In this tutorial you will learn how to process a complete night of raw data from any telescope with some basic reduction tools provided by |prose|.

Example data
^^^^^^^^^^^^
you can follow this tutorial on your own data or generate a synthetic dataset with

.. code-block:: python3

    from prose.datasets import generate_prose_reduction_datatset

    fits_folder = generate_prose_reduction_datatset("./tutorial_dataset")

Telescope setting
^^^^^^^^^^^^^^^^^

We start by setting up the telescope information we need for the reduction, for example some fits keywords that are specific to this observatory plus few specs:

.. code-block:: python3

    from prose import Telescope

    Telescope({
        "name": "fake_telescope",
        "trimming": (0, 0),
        "latlong": [24.6275, 70.4044]
    })


.. parsed-literal::

    Telescope 'fake_telescope' saved

This has to be done **only once** and saves this telescope settings for any future use (whenever its name appears in fits headers). More details are given in the :ref:`telescope settings <telescope-config>` note.

Folder exploration
^^^^^^^^^^^^^^^^^^

The first thing we want to do is to see what is contained within our folder. For that we instantiate a :py:class:`~prose.FitsManager` object on our folder to describe its content

.. code-block:: python3

    from prose import FitsManager

    fm = FitsManager(fits_folder, depth=2)
    fm.describe("calib")

.. parsed-literal::

    ╒════════════╤════════════════╤════════╤══════════╤══════════════╤══════════╤════════════╕
    │ date       │ telescope      │ type   │ target   │ dimensions   │ filter   │   quantity │
    ╞════════════╪════════════════╪════════╪══════════╪══════════════╪══════════╪════════════╡
    │ 2020-02-29 │ fake_telescope │ bias   │          │ 80x80        │          │          1 │
    ├────────────┼────────────────┼────────┼──────────┼──────────────┼──────────┼────────────┤
    │ 2020-02-29 │ fake_telescope │ dark   │          │ 80x80        │          │          1 │
    ├────────────┼────────────────┼────────┼──────────┼──────────────┼──────────┼────────────┤
    │ 2020-02-29 │ fake_telescope │ flat   │          │ 80x80        │ I+z      │          3 │
    ├────────────┼────────────────┼────────┼──────────┼──────────────┼──────────┼────────────┤
    │ 2020-02-29 │ fake_telescope │ light  │ prose    │ 80x80        │ I+z      │         80 │
    ╘════════════╧════════════════╧════════╧══════════╧══════════════╧══════════╧════════════╛

We have 80 images of the *prose* target together with some calibration files

Reduction and Photometry
^^^^^^^^^^^^^^^^^^^^^^^^

The ``Reduction`` part is then simply

.. code-block:: python3
    
    from prose import Reduction

    reduction = Reduction(fm)
    reduction.run()

.. parsed-literal::

    RUN Reduction: 100%|█████████████████████████| 80/80 [00:05<00:00, 13.98files/s]

a ``fake_telescope_20200229_prose_I+z`` folder has been created in which we can now run the ``Photometry``

.. code-block:: python3

    from prose import Photometry

    photometry = Photometry(reduction.destination)
    photometry.run()


.. parsed-literal::

    INFO detected stars: 7
    INFO global psf FWHM: 2.10 (pixels)
    RUN Photometric extraction: 100%|████████████| 80/80 [00:05<00:00, 15.28files/s]

Here is the content of the reduced folder:

::

    fake_telescope_20200229_prose_I+z/
      │ 
      ├── fake_telescope_20200229_prose_I+z.phots
      ├── fake_telescope_20200229_prose_I+z_stack.fits
      ├── fake_telescope_20200229_prose_I+z_movie.gif
      ├── fake-C001-002020-03-01T00:00:00.000_reduced.fits
      ├── fake-C001-002020-03-01T00:01:00.000_reduced.fits
      ├── fake-C001-002020-03-01T00:02:00.000_reduced.fits
      └── ...

It contains all reduced images and a stack fits of the night as well as a ``phots`` file containing all extracted fluxes (see :ref:`data products description <phots-structure>`). It also contains a small gif of the night 

.. figure:: fake_telescope_20200229_prose_I+z_movie.gif
   :align: center
   :width: 200

   *fake_telescope_20200229_prose_I+z_movie.gif*


We can now load this folder into a ``PhotProducts`` object and proceed with further analaysis (e.g. in the :ref:`next turorial <photometry-analysis>`.