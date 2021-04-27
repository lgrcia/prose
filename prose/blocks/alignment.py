from ..core import Block
from skimage.transform import warp
from skimage.transform import AffineTransform as skAffineTransform


class AffineTransform(Block):
    """
    Align an image to a reference

    """

    def __init__(self, stars_only=False, **kwargs):
        super().__init__(**kwargs)
        self.stars_only = stars_only

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

        if not self.stars_only:
            transform = image.transform.inverse
            image.data = warp(image.data, transform)
        image.stars_coords = image.transform(image.stars_coords)

    def citations(self, image):
        return "astropy", "numpy"