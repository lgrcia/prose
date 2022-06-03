from .. import Block
import numpy as np
from skimage.transform import warp
from skimage.transform import AffineTransform as skAffineTransform
from astropy.nddata import Cutout2D as _Cutout2D
from ..utils import register_args, nan_gaussian_filter


class Cutout2D(Block):
    """
    TODO: change name of this... not explicit
    Align an image to a reference image using ``astropy.nddata.Cutout2D``

    Parameters
    ----------
    reference : np.ndarray
        reference image on which alignment is done
    """

    # TODO should take shape as input not an image
    def __init__(self, reference_image, **kwargs):
        super().__init__(**kwargs)
        self.ref_shape = np.array(reference_image.shape)
        self.ref_center = self.ref_shape[::-1] / 2

    def run(self, image):
        # TODO this is shitty, should use image.dx, image.dy
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
    Apply an affine transformation to image and/or stars

    |read|
    
    - rotation : ``Image.header['TWROT']``
    - translation : ``Image.header['TWTRANSX']``, ``Image.header['TWTRANSY']``
    - scale : ``Image.header['TWSCALEX']``, ``Image.header['TWSCALEY']``


    |write|
    
    - ``Image.transform``
    - ``Image.inverse``
    - ``Image.stars_coords``

    Parameters
    ----------
    stars : bool, optional
        whether to apply transform to ``Image.stars_coords``, by default True
    data : bool, optional
        whether to apply transform to ``Image.data``, by default True
    inverse : bool, optional
        whether to apply inverse transform, by default False
    output_shape : tuple-like, optional
        shape of the transformed image. By default None, conserving the orignial shape

    """

    @register_args
    def __init__(self, stars=True, data=True, inverse=False, output_shape=None, **kwargs):
        super().__init__(**kwargs)
        self.data = data
        self.stars = stars
        self.inverse = inverse
        self.output_shape = output_shape

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
            try:
                #image.data[image.data<0] = np.nan
                #image.data = nan_gaussian_filter(image.data, sigma=1.)
                image.data = warp(
                    image.data, 
                    transform.inverse, 
                    cval=np.nanmedian(image.data), 
                    output_shape=self.output_shape
                )
            except np.linalg.LinAlgError:
                image.discard = True

        if self.stars:
            try:
                image.stars_coords = transform(image.stars_coords)
            except np.linalg.LinAlgError:
                image.discard = True

    def citations(self, image):
        return "astropy", "numpy"