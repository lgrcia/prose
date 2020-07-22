from prose._blocks.base import Block
from astropy.nddata import Cutout2D
import numpy as np


class Alignment(Block):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Align(Alignment):
    """
    Align an image to a reference image using :code:`astropy.nddata.Cutout2D`
    """

    def __init__(self, reference=1 / 2, **kwargs):
        super().__init__(**kwargs)
        self.reference = reference

        self.ref_shape = None
        self.ref_center = None

    def initialize(self, fits_manager):
        reference_frame = int(self.reference * len(fits_manager.files))
        reference_image_path = fits_manager.files[reference_frame]
        reference_image = fits_manager.trim(reference_image_path)

        self.ref_shape = np.array(reference_image.shape)
        self.ref_center = self.ref_shape[::-1] / 2

    def run(self, image):
        shift = np.array([image.header["DX"], image.header["DY"]])
        aligned_image = Cutout2D(
                        image.data,
                        self.ref_center-shift.astype("int"),
                        self.ref_shape,
                        mode="partial",
                        fill_value=np.mean(image.data),
                        wcs=image.wcs
                    )
        image.data = aligned_image.data
        image.wcs = aligned_image.wcs