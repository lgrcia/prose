.. _photometry-analysis:

Photometry analysis
===================

In this tutorial we will see how to use and analyse photometric products from |prose|. As in the :ref:`last tutorial <reduction>`  we will study an observation of Quatar2-b exoplanet transit observed from the Trappist-North telescope, but this time with a bit more consequent products, to produce a denser light curve (92 images). These products are available in TODO (10MB).

Let's instantiate a :py:class:`~prose.Photometry`  object containing all we need for this analysis and show the detected stars

.. code:: ipython3

    from prose import Photometry
    
    phot = Photometry("lighter_quatar2b_dataset_reduced")
    phot.show_stars()


.. image:: output_0_0.png
   :align: center

If target was not specified in the reduction process, we need to specify it before producing our differential Photometry.

.. code:: ipython3

    phot.target["id"] = 5
    phot.Broeg2005()
    phot.lc.plot()


.. image:: output_1_0.png
   :align: center

We used the Broeg 2005 algorithm and ended by plotting our nice transit light-curve. ``phot.lc`` contains a :py:class:`~prose.LightCurve` object providing convenient methods for light-curves data manipulation and plotting.

We can check the comparison stars

.. code:: ipython3

    phot.show_stars()

.. image:: output_2_0.png
   :align: center

and continue with further visualisation or analysis. All available plotting methods are described in the :ref:`quick-ref` and in details in :py:class:`~prose.Photometry`.