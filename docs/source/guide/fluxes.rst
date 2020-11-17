.. _fluxes:

The flues module from prose contains fluxes representation and manipulation functions. It exposes a Flux object which can be stacked with other fluxes, binned, folded, masked and so on. 

Beyond flux and time, the Flux object contains time-series that are likely to be used for their analysis (such as flags or other instrumental measurements) and that will undergo same transformation as the flux (such as binning, folding, masking or stacking)

Basic operations
================
Let's work on a synthetic example:


Binning
-------
Masking
-------
Folding
-------


Advanced usage
==============
Stacking
--------
Ensemble photometry
-------------------
Linear modeling
---------------
Altought prose is not meant to be a modeling tool, the Flux object provides a quick and conveniant method to model flux against a polynomial of embedded time-series.


No, wrong answer! Just work on a 3D xarray!
