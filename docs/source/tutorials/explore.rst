.. _explore:

Fits explorer
--------------

|prose| provides a FitsManager class to deal with fits folder
exploration, mainly to identify and retrieve calibration files
associated with specific observations

A FitsManager object can be created with

.. code:: ipython3

    from prose.io import FitsManager
    
    fm = FitsManager("/Users/lionelgarcia/Data/test/test_deepness", deepness=5)


.. parsed-literal::

    100%|██████████| 1528/1528 [00:17<00:00, 87.24it/s]


The ``deepness`` parameters specify how deep you want to explore in term
of subfolders. Then the ``describe`` provide table visualisation of what
is contained within your folder

.. code:: ipython3

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


.. code:: ipython3

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

.. code:: ipython3

    fm.keep(target="Sp0111-4908", keep_closest_calibration=True)
    
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


Specific paths can then be retrieved

.. code:: ipython3
    
    science_im_paths fm.get("light")
    dark_im_paths = fm.get("dark")
    flat_im_paths fm.get("flat")


