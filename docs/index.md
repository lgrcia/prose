# prose

```{image} _static/prose3.png
:width: 420px
:align: center
```

```{warning} 
This is prose **3.0.0**, a version still under construction. [Latest doc here](https://prose.readthedocs.io/en/latest/)
```
+++

A Python package to build image processing pipelines for Astronomy. Beyond featuring the blocks to build pipelines from scratch, it provides pre-implemented ones to perform common tasks such as automated calibration, reduction and photometry.

```{admonition} Where to start?
:class: tip 
🌌 [Install](md/installation.md) prose and read about its [core objects](ipynb/core.ipynb).

📦 Explore the library of pre-implemented [blocks](md/blocks.rst)

✨ Obtain a light curve from raw images by following the [Basic Photometry tutorial](ipynb/photometry.ipynb)
```

```{toctree}
:maxdepth: 0
:caption: Get started

md/installation
ipynb/quickstart
ipynb/core
```

```{toctree}
:maxdepth: 0
:caption: Tutorials

ipynb/fitsmanager
ipynb/photometry
ipynb/customblock
ipynb/catalogs
```


```{toctree}
:maxdepth: 0
:caption: Case studies

ipynb/casestudies/transit.ipynb
ipynb/casestudies/hiaka.ipynb
ipynb/casestudies/comet.ipynb
ipynb/casestudies/satellite.ipynb
```

```{toctree}
:maxdepth: 0
:caption: Reference

ipynb/sources
md/blocks
md/api
```