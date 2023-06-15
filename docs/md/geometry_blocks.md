# Alignment & Geometry


```{image} ../_static/matching.png
:align: center
:height: 220px
```

The task of the alignment and geometry blocks is to compute and apply geometric transformations to the [Image](prose.Image) `data` and `sources`. For this purpose, an [Image](prose.Image) contains a `transform` attrtibute, that corresponds to a [scikit-image](https://scikit-image.org/) [`AffineTransform`](https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.AffineTransform). 

## Transform computation blocks

```{admonition} FAQ: Why computing the transform without applying it?
:class: note

To control when to apply the transform  and to which data. For example, transforming an image `data` to a common reference (e.g. with interpolation) could be only wanted at the end of a sequence in order to build a stack image.
````

## Other geometry blocks

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