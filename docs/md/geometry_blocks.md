# Alignment & Geometry


```{image} ../_static/matching.png
:align: center
:height: 220px
```

The task of the alignment and geometry blocks is to compute and apply geometric transformations to the [Image](prose.Image) `data` and `sources`. For this purpose, an [Image](prose.Image) contains a `transform` attrtibute, that corresponds to a [scikit-image](https://scikit-image.org/) [`AffineTransform`](https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.AffineTransform). 

## Available blocks

```{eval-rst}

.. autosummary::
   :template: blocksum.rst
   :nosignatures:

   ~prose.blocks.geometry.Trim
   ~prose.blocks.geometry.Cutouts
   ~prose.blocks.alignment.Align
   ~prose.blocks.alignment.AlignReferenceSources
   ~prose.blocks.alignment.AlignReferenceWCS

```

Usually, the `transform` object stores the geometric transform between the [Image](prose.Image) and a reference one, so that it is not directly related to the physical [WCS](https://docs.astropy.org/en/stable/wcs/) of the image (which transforms to physical sky coordinates).