
.. TODO: update lc image

.. _photometry-analysis:

Photometry analysis
===================

In this tutorial we will see how to use and analyse photometric products from |prose|. As an example we will work from the products of the :ref:`previous tutorial <reduction>` which led to the reduced folder ``fake_telescope_20200229_prose_I+z``

Let's instantiate a :py:class:`~prose.PhotProducts`  object containing all we need for this analysis and show the detected stars

.. code:: ipython3

    from prose import Photometry
    
    phot = Photometry("./fake_telescope_20200229_prose_I+z")
    phot.show_stars()


.. image:: stars_before_lc.png
   :align: center
   :width: 300

If target was not specified in the reduction process, we need to specify it before producing our differential Photometry.

.. code:: ipython3

    phot.target_id = 1
    phot.Broeg2005()
    phot.lc.plot()


.. image:: lc.png
   :align: center
   :width: 450

We used the Broeg 2005 algorithm and ended by plotting our light-curve. ``phot.lc`` contains a :py:class:`~prose.LightCurve` object providing convenient methods for light-curves data manipulation and plotting.

We can check the comparison stars

.. code:: ipython3

    phot.show_stars()

.. image:: stars_after_lc.png
   :align: center
   :width: 300

and continue with further visualisation or analysis. All available plotting methods are described in the :ref:`quick-ref` and in details in :py:class:`~prose.PhotProducts`.