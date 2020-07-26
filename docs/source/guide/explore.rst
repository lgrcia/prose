.. _explore:

Fits manager
--------------

|prose| provides a :py:class:`~prose.FitsManager` class to deal with fits folder
exploration, mainly to identify and retrieve calibration files
associated with specific observations. This class is heavily used is handy to deal with unorganized fits folders (e.g. where there is no separation between science and calibration images)

A FitsManager object can be created with

.. code:: python

    from prose.io import FitsManager
    
    fm = FitsManager("test_folder", depth=5)


.. parsed-literal::

    100%|██████████| 1528/1528 [00:17<00:00, 87.24it/s]


The ``depth`` parameter specifies how deep you want to explore in term
of sub-folders. Then ``describe`` provides table visualisation of what
is contained within your folder

.. code:: python

    # To see all individual observations
    fm.describe("obs")


.. parsed-literal::

    ╒═════════╤════════════╤══════════════╤═════════════╤══════════╤════════════╕
    │   index │ date       │ telescope    │ target      │ filter   │   quantity │
    ╞═════════╪════════════╪══════════════╪═════════════╪══════════╪════════════╡
    │       0 │ 2018-12-21 │ SPECULOOS-IO │ Sp0553-7133 │ I+z      │       1000 │
    ├─────────┼────────────┼──────────────┼─────────────┼──────────┼────────────┤
    │       1 │ 2019-09-22 │ SPECULOOS-IO │ Sp0111-4908 │ I+z      │        263 │
    ├─────────┼────────────┼──────────────┼─────────────┼──────────┼────────────┤
    │       2 │ 2019-09-22 │ SPECULOOS-IO │ Sp1945-2557 │ I+z      │        126 │
    ╘═════════╧════════════╧══════════════╧═════════════╧══════════╧════════════╛


.. code:: python

    # To see all observations together with calibration files
    
    fm.describe("calib")
    
    # You can see all individual files with:
    # fm.describe("files")


.. parsed-literal::

    ╒════════════╤════════════════════╤════════╤═════════════╤══════════════╤══════════╤════════════╕
    │ date       │ telescope          │ type   │ target      │ dimensions   │ filter   │   quantity │
    ╞════════════╪════════════════════╪════════╪═════════════╪══════════════╪══════════╪════════════╡
    │ 2018-12-18 │ SPECULOOS-IO       │ bias   │             │ 2048x2088    │          │          9 │
    ├────────────┼────────────────────┼────────┼─────────────┼──────────────┼──────────┼────────────┤
    │ 2018-12-18 │ SPECULOOS-IO       │ dark   │             │ 2048x2088    │          │         27 │
    ├────────────┼────────────────────┼────────┼─────────────┼──────────────┼──────────┼────────────┤
    │ 2018-12-21 │ SPECULOOS-IO       │ flat   │             │ 2048x2088    │ I+z      │          7 │
    ├────────────┼────────────────────┼────────┼─────────────┼──────────────┼──────────┼────────────┤
    │ 2018-12-21 │ SPECULOOS-IO       │ light  │ Sp0553-7133 │ 2048x2088    │ I+z      │       1000 │
    ├────────────┼────────────────────┼────────┼─────────────┼──────────────┼──────────┼────────────┤
    │ 2019-09-22 │ SPECULOOS-IO       │ bias   │             │ 2048x2088    │          │          9 │
    ├────────────┼────────────────────┼────────┼─────────────┼──────────────┼──────────┼────────────┤
    │ 2019-09-22 │ SPECULOOS-IO       │ dark   │             │ 2048x2088    │          │         27 │
    ├────────────┼────────────────────┼────────┼─────────────┼──────────────┼──────────┼────────────┤
    │ 2019-09-22 │ SPECULOOS-IO       │ flat   │             │ 2048x2088    │ I+z      │         14 │
    ├────────────┼────────────────────┼────────┼─────────────┼──────────────┼──────────┼────────────┤
    │ 2019-09-22 │ SPECULOOS-IO       │ light  │ Sp0111-4908 │ 2048x2088    │ I+z      │        263 │
    ├────────────┼────────────────────┼────────┼─────────────┼──────────────┼──────────┼────────────┤
    │ 2019-09-22 │ SPECULOOS-IO       │ light  │ Sp1945-2557 │ 2048x2088    │ I+z      │        126 │
    ├────────────┼────────────────────┼────────┼─────────────┼──────────────┼──────────┼────────────┤
    │ 2019-12-20 │ SPECULOOS-GANYMEDE │ bias   │             │ 2048x2088    │          │          9 │
    ├────────────┼────────────────────┼────────┼─────────────┼──────────────┼──────────┼────────────┤
    │ 2019-12-20 │ SPECULOOS-GANYMEDE │ dark   │             │ 2048x2088    │          │         27 │
    ├────────────┼────────────────────┼────────┼─────────────┼──────────────┼──────────┼────────────┤
    │ 2019-12-20 │ SPECULOOS-GANYMEDE │ flat   │             │ 2048x2088    │ I+z      │         10 │
    ╘════════════╧════════════════════╧════════╧═════════════╧══════════════╧══════════╧════════════╛


you can then filter your observations and keep calibration files needed
for your analysis

.. code:: python

    fm.keep(target="Sp0111-4908", calibration=True)
    
    fm.describe("calib")


.. parsed-literal::

    ╒════════════╤══════════════╤════════╤═════════════╤══════════════╤══════════╤════════════╕
    │ date       │ telescope    │ type   │ target      │ dimensions   │ filter   │   quantity │
    ╞════════════╪══════════════╪════════╪═════════════╪══════════════╪══════════╪════════════╡
    │ 2019-09-22 │ SPECULOOS-IO │ bias   │             │ 2048x2088    │          │          9 │
    ├────────────┼──────────────┼────────┼─────────────┼──────────────┼──────────┼────────────┤
    │ 2019-09-22 │ SPECULOOS-IO │ dark   │             │ 2048x2088    │          │         27 │
    ├────────────┼──────────────┼────────┼─────────────┼──────────────┼──────────┼────────────┤
    │ 2019-09-22 │ SPECULOOS-IO │ flat   │             │ 2048x2088    │ I+z      │         14 │
    ├────────────┼──────────────┼────────┼─────────────┼──────────────┼──────────┼────────────┤
    │ 2019-09-22 │ SPECULOOS-IO │ light  │ Sp0111-4908 │ 2048x2088    │ I+z      │        263 │
    ╘════════════╧══════════════╧════════╧═════════════╧══════════════╧══════════╧════════════╛


Specific paths can then be retrieved with

.. code:: python
    
    im_science = fm.get("light")
    im_dark = fm.get("dark")
    im_flat = fm.get("flat")


Index file
==========

Every time a folder is explored with :py:class:`~prose.FitsManager`, an index file is created. When dealing with large folders, the keyword :code:`index=True` can be used avoid re-analyzing folder content and save time. Using the example above we would do:

.. code:: python

    fm = FitsManager("test_folder", index=True)