from ..core import Block
import numpy as np
from skimage.transform import warp
from skimage.transform import AffineTransform as skAffineTransform
from astropy.nddata import Cutout2D as _Cutout2D


class Cutout2D(Block):
    """
    Align an image to a reference image using :code:`astropy.nddata.Cutout2D`
    Parameters
    ----------
    reference : np.,darray
        reference image on which alignment is done
    """

    def __init__(self, reference_image, **kwargs):
        super().__init__(**kwargs)
        self.ref_shape = np.array(reference_image.shape)
        self.ref_center = self.ref_shape[::-1] / 2

    def run(self, image):
        shift = np.array([image.header["DX"], image.header["DY"]])

        aligned_image = _Cutout2D(
                        image.data,
                        self.ref_center-shift.astype("int"),
                        self.ref_shape,
                        mode="partial",
                        fill_value=np.mean(image.data),
                        wcs=image.wcs
                    )
        image.data = aligned_image.data
        image.stars_coords += shift

    def citations(self, image):
        return "astropy", "numpy"


class AffineTransform(Block):
    """
    Align an image to a reference

    """

    def __init__(self, stars=True, data=True, inverse=False, fill="median", **kwargs):
        super().__init__(**kwargs)
        self.data = data
        self.stars = stars
        self.inverse = inverse
        if fill == "median":
            self.fill_function = lambda im: np.median(im.data)

    def run(self, image, **kwargs):
        if "transform" not in image.__dict__:
            if "TWROT" in image.header:
                image.transform = skAffineTransform(
                    rotation=image.header["TWROT"],
                    translation=(image.header["TWTRANSX"], image.header["TWTRANSY"]),
                    scale=(image.header["TWSCALEX"], image.header["TWSCALEX"])
                )
            else:
                raise AssertionError("Could not find transformation matrix")

        transform = image.transform

        if self.inverse:
            transform = transform.inverse

        if self.data:
            image.data = warp(image.data, transform.inverse, cval=self.fill_function(image))

        if self.stars:
            image.stars_coords = transform(image.stars_coords)

    def citations(self, image):
        return "astropy", "numpy"